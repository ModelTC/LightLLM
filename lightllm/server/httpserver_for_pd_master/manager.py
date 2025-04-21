import sys
import asyncio
import uvloop
import time
import datetime
import ujson as json
import pickle

asyncio.set_event_loop_policy(uvloop.EventLoopPolicy())
from typing import Union, List, Tuple, Dict
from lightllm.server.core.objs import FinishStatus
from ..pd_io_struct import PD_Client_Obj, UpKVStatus, ObjType
from lightllm.server.core.objs import SamplingParams
from ..multimodal_params import MultimodalParams
from ..tokenizer import get_tokenizer
from ..req_id_generator import ReqIDGenerator, convert_sub_id_to_group_id
from fastapi import Request
from lightllm.utils.log_utils import init_logger
from lightllm.server.metrics.manager import MetricClient
from lightllm.utils.statics_utils import MovingAverage
from lightllm.server.httpserver.manager import AsyncQueue
from lightllm.utils.error_utils import ServerBusyError

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

        self.req_id_to_out_inf: Dict[int, ReqStatus] = {}
        self.infos_queues = None  # 这个需要延迟初始化，否则使用的loop不对

        self.tokenizer = get_tokenizer(args.model_dir, args.tokenizer_mode, trust_remote_code=args.trust_remote_code)

        self.first_time_costs = MovingAverage()
        self.per_token_costs = MovingAverage()
        return

    async def register_pd(self, pd_info_json, websocket):
        pd_client = PD_Client_Obj(**pd_info_json)
        pd_client.websocket = websocket
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


    def tokens(self, prompt, multimodal_params, samping_params: SamplingParams, kwargs=None):
        kwargs = {} if kwargs is None else kwargs
        prompt_ids = self.tokenizer.encode(prompt, None, **kwargs)
        image_tokens = 0
        img_count = 0
        audio_tokens = 0
        audio_count = 0
        for img in multimodal_params.images:
            img_count += 1
            self.tokenizer.init_imageitem_extral_params(img, multimodal_params, samping_params)
            image_tokens += self.tokenizer.get_image_token_length(img)
        for audio in multimodal_params.audios:
            audio_count += 1
            self.tokenizer.init_audioitem_extral_params(audio, multimodal_params, samping_params)
            audio_tokens += self.tokenizer.get_audio_token_length(audio)
        return len(prompt_ids) + image_tokens + img_count + audio_tokens + audio_count

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
    ):
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

        except BaseException as e:
            logger.error(f"has exception {str(e)}")
            await self.abort(group_request_id)
            raise e

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
        request: Request,
    ):
        group_request_id = sampling_params.group_request_id

        req_status = ReqStatus(group_request_id, p_node, d_node)
        self.req_id_to_out_inf[group_request_id] = req_status

        up_status_event = req_status.up_status_event

        p_start_args = p_node.start_args
        prefill_node_dict = {
            "node_id": p_start_args["pd_node_id"],
            "ip": p_start_args["host"],
            "rpyc_port": p_start_args["pd_remote_prefill_port"],
            "max_new_tokens": sampling_params.max_new_tokens - 1,
            "pd_master_node_id": self.args.pd_node_id,
        }

        sampling_params.move_kv_to_decode_node.initialize(prefill_node_dict)
        sampling_params.suggested_dp_index = -1

        await d_node.websocket.send_bytes(pickle.dumps((ObjType.REQ, (prompt, sampling_params, multimodal_params))))

        while True:
            await req_status.wait_to_ready()
            if await request.is_disconnected():
                raise Exception(f"req_id {group_request_id} disconnected")
            if await req_status.can_read(self.req_id_to_out_inf):
                token_list = await req_status.pop_all_tokens()
                for sub_req_id, request_output, metadata, finish_status in token_list:
                    yield sub_req_id, request_output, metadata, finish_status

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
            p_node, d_node, prompt, sampling_params, multimodal_params, request
        ):
            if await request.is_disconnected():
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
        self.per_token_costs.add(mean_per_token_cost_time_ms)
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
            req_status = self.req_id_to_out_inf[group_request_id]
            del self.req_id_to_out_inf[group_request_id]
        except:
            pass

        try:
            await req_status.d_node.websocket.send_bytes(pickle.dumps((ObjType.ABORT, group_request_id)))
        except:
            pass

        return

    async def remove_req(self, group_request_id):
        try:
            del self.req_id_to_out_inf[group_request_id]
        except:
            pass

    async def timer_log(self):
        while True:
            await asyncio.sleep(30)
            self.first_time_costs.print_log("mean first cost")
            self.per_token_costs.print_log("mean per token cost")

    async def put_to_handle_queue(self, obj):
        await self.infos_queues.put(obj)

    async def handle_loop(self):
        self.infos_queues = AsyncQueue()
        asyncio.create_task(self.timer_log())

        use_config_server = self.args.config_server_host and self.args.config_server_port

        if use_config_server:
            from lightllm.server.httpserver_for_pd_master.register_loop import register_loop

            asyncio.create_task(register_loop(self))

        while True:
            objs = await self.infos_queues.wait_to_get_all_data()

            try:
                for obj in objs:
                    if obj[0] == ObjType.TOKEN_PACKS:
                        for sub_req_id, text, metadata, finish_status in obj[1]:
                            finish_status: FinishStatus = finish_status
                            group_req_id = convert_sub_id_to_group_id(sub_req_id)
                            try:
                                req_status: ReqStatus = self.req_id_to_out_inf[group_req_id]
                                async with req_status.lock:
                                    req_status.out_token_info_list.append((sub_req_id, text, metadata, finish_status))
                                    req_status.event.set()
                            except:
                                pass
                    else:
                        logger.error(f"recevie error obj {obj}")
            except BaseException as e:
                logger.exception(str(e))
        return


class ReqStatus:
    def __init__(self, req_id, p_node, d_node) -> None:
        self.req_id = req_id
        self.lock = asyncio.Lock()
        self.event = asyncio.Event()
        self.up_status_event = asyncio.Event()
        self.out_token_info_list: List[Tuple[int, str, dict, FinishStatus]] = []
        self.p_node: PD_Client_Obj = p_node
        self.d_node: PD_Client_Obj = d_node

    async def wait_to_ready(self):
        try:
            await asyncio.wait_for(self.event.wait(), timeout=5)
        except asyncio.TimeoutError:
            pass

    async def can_read(self, req_id_to_out_inf):
        async with self.lock:
            self.event.clear()
            assert self.req_id in req_id_to_out_inf, f"error state req_id {self.req_id}"
            if len(self.out_token_info_list) == 0:
                return False
            else:
                return True

    async def pop_all_tokens(self):
        async with self.lock:
            ans = self.out_token_info_list.copy()
            self.out_token_info_list.clear()
        return ans
