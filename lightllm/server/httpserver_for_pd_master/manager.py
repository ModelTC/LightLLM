import sys
import zmq
import zmq.asyncio
import asyncio
import uvloop
import rpyc
import time
import hashlib
import datetime
import aiohttp
import ujson as json

asyncio.set_event_loop_policy(uvloop.EventLoopPolicy())
from typing import Union, List, Tuple, Dict
from ..io_struct import FinishStatus
from ..pd_io_struct import PD_Client_Obj, UpKVStatus
from ..sampling_params import SamplingParams
from ..multimodal_params import MultimodalParams
from ..req_id_generator import ReqIDGenerator
from fastapi import Request
from lightllm.utils.log_utils import init_logger
from lightllm.server.metrics.manager import MetricClient
from lightllm.utils.statics_utils import MovingAverage

logger = init_logger(__name__)


class HttpServerManagerForPDMaster:
    def __init__(
        self,
        args,
        metric_port,
    ):
        self.args = args
        self.metric_client = MetricClient(metric_port)
        self.id_gen = ReqIDGenerator()
        self.prefill_nodes: List[PD_Client_Obj] = []
        self.decode_nodes: List[PD_Client_Obj] = []
        self.url_to_pd_nodes: Dict[str, PD_Client_Obj] = {}

        self.id_to_event: Dict[int, asyncio.Event] = {}
        self.session = None
        self.first_time_costs = MovingAverage()
        self.create_session_costs = MovingAverage()
        return

    async def register_pd(self, pd_info_json):
        pd_client = PD_Client_Obj(**pd_info_json)
        self.url_to_pd_nodes[pd_client.client_ip_port] = pd_client
        if pd_client.mode == "prefill":
            self.prefill_nodes = [e for e in self.prefill_nodes if e.client_ip_port != pd_client.client_ip_port]
            self.prefill_nodes.append(pd_client)
        elif pd_client.mode == "decode":
            self.decode_nodes = [e for e in self.decode_nodes if e.client_ip_port != pd_client.client_ip_port]
            self.decode_nodes.append(pd_client)
        else:
            assert False

        logger.info(f"mode: {pd_client.mode} url: {pd_client.client_ip_port} registed")
        return

    async def remove_pd(self, pd_info_json):
        pd_client = PD_Client_Obj(**pd_info_json)
        try:
            del self.url_to_pd_nodes[pd_client.client_ip_port]
        except:
            pass
        self.prefill_nodes = [e for e in self.prefill_nodes if e.client_ip_port != pd_client.client_ip_port]
        self.decode_nodes = [e for e in self.decode_nodes if e.client_ip_port != pd_client.client_ip_port]
        logger.info(f"mode: {pd_client.mode} url: {pd_client.client_ip_port} removed")
        return

    async def update_req_status(self, upkv_status: UpKVStatus):
        try:
            event = self.id_to_event[upkv_status.group_request_id]
            event.upkv_status = upkv_status
            event.set()
            del self.id_to_event[upkv_status.group_request_id]
        except:
            pass
        return

    def tokens(self, prompt: str):
        # to do
        raise NotImplementedError("tokens is not implements")

    async def select_p_d_node(
        self, prompt: Union[str, List[int]], sampling_params: SamplingParams, multimodal_params: MultimodalParams
    ) -> Tuple[PD_Client_Obj, PD_Client_Obj]:
        import random

        p_node = random.choice(self.prefill_nodes)
        d_node = random.choice(self.decode_nodes)
        return p_node, d_node

    async def generate(
        self,
        prompt: Union[str, List[int]],
        sampling_params: SamplingParams,
        multimodal_params: MultimodalParams,
        request: Request,
    ) -> Tuple[int, str, dict, FinishStatus]:
        start_time = time.time()
        group_request_id = self.id_gen.generate_id()
        try:
            sampling_params.group_request_id = group_request_id
            # 记录请求到达的相关信息
            await self._log_req_header(request, group_request_id)
            # 监控
            self.metric_client.counter_inc("lightllm_request_count")
            self.metric_client.histogram_observe("lightllm_request_max_new_tokens", sampling_params.max_new_tokens)

            p_node, d_node = await self.select_p_d_node(prompt, sampling_params, multimodal_params)

            results_generator = self._wait_to_token_package(
                p_node,
                d_node,
                start_time,
                prompt,
                sampling_params,
                multimodal_params,
                request,
            )
            async for sub_req_id, request_output, metadata, finish_status in results_generator:
                yield sub_req_id, request_output, metadata, finish_status
        finally:
            await self.remove_req(group_request_id)
        return

    async def _log_req_header(self, request: Request, group_request_id: int):
        x_request_id = request.headers.get("X-Request-Id", "")
        x_session_id = request.headers.get("X-Session-Id", "")
        format_in_time = datetime.datetime.fromtimestamp(time.time()).strftime("%Y-%m-%d %H:%M:%S")
        logger.info(
            f"recieved req X-Request-Id:{x_request_id} "
            f"X-Session-Id:{x_session_id} start_time:{format_in_time} "
            f"lightllm_req_id:{group_request_id} "
        )
        return

    async def _to_req_info(
        self, prompt: Union[str, List[int]], sampling_params: SamplingParams, multimodal_params: MultimodalParams
    ):
        req = {
            "inputs": prompt,
            "parameters": sampling_params.to_origin_dict(),
            "multimodal_params": multimodal_params.to_origin_dict(),
        }
        return req

    async def fetch_stream(
        self,
        p_node: PD_Client_Obj,
        d_node: PD_Client_Obj,
        prompt: Union[str, List[int]],
        sampling_params: SamplingParams,
        multimodal_params: MultimodalParams,
    ):
        group_request_id = sampling_params.group_request_id
        event = asyncio.Event()
        self.id_to_event[group_request_id] = event
        # 初始化连接池
        if self.session is None:
            self.session = aiohttp.ClientSession(connector=aiohttp.TCPConnector(limit=2000, verify_ssl=False))
            await self.session.__aenter__()

        d_start_args = d_node.start_args
        decode_node_dict = {
            "node_id": d_start_args["pd_node_id"],
            "ip": d_start_args["host"],
            "rpyc_port": d_start_args["pd_decode_rpyc_port"],
            "max_new_tokens": sampling_params.max_new_tokens - 1,
        }

        try:
            old_max_new_tokens = sampling_params.max_new_tokens
            sampling_params.max_new_tokens = 1
            sampling_params.move_kv_to_decode_node = decode_node_dict if old_max_new_tokens != 1 else None
            sampling_params.suggested_dp_index = None

            req = await self._to_req_info(prompt, sampling_params, multimodal_params)
            create_start_time = time.time()
            async with self.session.post(p_node.to_llm_url(), json=req) as response:
                self.create_session_costs.add((time.time() - create_start_time) * 1000)
                if response.status == 200:
                    async for line in response.content:
                        line = line.decode("utf-8").strip()
                        if line.startswith("data:"):
                            data = line[len("data:") :].strip()
                            sub_req_id, request_output, metadata, finish_status = json.loads(data)
                            if old_max_new_tokens != 1:
                                finish_status = FinishStatus.NO_FINISH
                            else:
                                finish_status = FinishStatus(finish_status)
                            # 得到 p 节点返回的 prompt_ids 信息
                            if metadata.get("prompt_ids", None) is not None:
                                prompt_ids = metadata.get("prompt_ids")
                                prompt_ids.append(metadata.get("id"))
                            yield sub_req_id, request_output, metadata, finish_status
                else:
                    logger.error(f"fetch_stream error: {response.status}")
                    raise Exception(f"group_req_id {group_request_id} connection error: {response}")

            # 如果只需要一个输出 token，prefill 完就直接结束掉吧
            if old_max_new_tokens == 1:
                return

            try:
                await asyncio.wait_for(event.wait(), timeout=60)
            except asyncio.TimeoutError:
                logger.warning(f"group_request_id: {group_request_id} time out err")
                raise Exception("server is busy")
                # raise Exception(f"group_request_id: {group_request_id} time out err, maybe kv move get questions")

            sampling_params.move_kv_to_decode_node = None
            sampling_params.max_new_tokens = old_max_new_tokens - 1
            sampling_params.suggested_dp_index = event.upkv_status.dp_index

            req = await self._to_req_info(prompt_ids, sampling_params, multimodal_params)
            async with self.session.post(d_node.to_llm_url(), json=req) as response:
                if response.status == 200:
                    async for line in response.content:
                        line = line.decode("utf-8").strip()
                        if line.startswith("data:"):
                            data = line[len("data:") :].strip()
                            sub_req_id, request_output, metadata, finish_status = json.loads(data)
                            yield sub_req_id, request_output, metadata, FinishStatus(finish_status)
                else:
                    logger.error(f"fetch_stream error: {response.status}")
                    raise Exception(f"group_req_id {group_request_id} connection error: {response}")
        finally:
            await self.remove_req(group_request_id)
        return

    async def _wait_to_token_package(
        self,
        p_node: PD_Client_Obj,
        d_node: PD_Client_Obj,
        start_time: float,
        prompt: str,
        sampling_params: SamplingParams,
        multimodal_params: MultimodalParams,
        request: Request,
    ):
        out_token_counter = 0
        first_token_cost_ms = sys.float_info.max
        group_request_id = sampling_params.group_request_id
        unfinished_count = sampling_params.best_of
        is_first_token = True

        async for sub_req_id, out_str, metadata, finish_status in self.fetch_stream(
            p_node, d_node, prompt, sampling_params, multimodal_params
        ):
            if await request.is_disconnected():
                await self.abort(group_request_id)
                raise Exception(f"req_id {group_request_id} disconnected")
            prompt_tokens = metadata["prompt_tokens"]
            out_token_counter += 1
            if is_first_token:
                first_token_cost_ms = (time.time() - start_time) * 1000
                is_first_token = False
                self.first_time_costs.add(first_token_cost_ms)

            yield sub_req_id, out_str, metadata, finish_status
            if finish_status.is_finished():
                unfinished_count -= 1
            if unfinished_count == 0:
                break

        total_cost_time_ms = (time.time() - start_time) * 1000
        mean_per_token_cost_time_ms = (total_cost_time_ms - first_token_cost_ms) / out_token_counter
        x_request_id = request.headers.get("X-Request-Id", "")
        x_session_id = request.headers.get("X-Session-Id", "")
        prompt_cache_len = metadata.pop("prompt_cache_len", 0)
        prompt_cache_ratio = prompt_cache_len / prompt_tokens
        format_start_time = datetime.datetime.fromtimestamp(start_time).strftime("%Y-%m-%d %H:%M:%S")
        logger.info(
            f"X-Request-Id:{x_request_id} "
            f"X-Session-Id:{x_session_id} start_time:{format_start_time} "
            f"lightllm_req_id:{group_request_id} first_token_cost:{first_token_cost_ms}ms "
            f"total_cost_time:{total_cost_time_ms}ms,out_token_counter:{out_token_counter} "
            f"mean_per_token_cost_time: {mean_per_token_cost_time_ms}ms "
            f"prompt_token_num:{prompt_tokens} "
            f"prompt_cache_len:{prompt_cache_len} "
            f"prompt_cache_ratio:{prompt_cache_ratio} "
        )
        self.metric_client.histogram_observe("lightllm_request_inference_duration", total_cost_time_ms / 1000.0)
        self.metric_client.histogram_observe(
            "lightllm_request_mean_time_per_token_duration", mean_per_token_cost_time_ms / 1000.0
        )
        self.metric_client.histogram_observe("lightllm_request_first_token_duration", first_token_cost_ms / 1000.0)
        self.metric_client.histogram_observe("lightllm_request_generated_tokens", out_token_counter)
        self.metric_client.counter_inc("lightllm_request_success")
        return

    async def abort(self, group_request_id):
        logger.warning(f"aborted group_request_id {group_request_id}")
        try:
            del self.id_to_event[group_request_id]
        except:
            pass
        return

    async def remove_req(self, group_request_id):
        try:
            del self.id_to_event[group_request_id]
        except:
            pass

    async def handle_loop(self):
        while True:
            # 可以做一个定时任务
            await asyncio.sleep(20)
            logger.info(f"mean first cost: {self.first_time_costs.average()} ms")
            logger.info(f"create_session_costs: {self.create_session_costs.average()} ms")
        return
