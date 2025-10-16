import sys
import asyncio
import uvloop
import time
import datetime
import ujson as json
import pickle

asyncio.set_event_loop_policy(uvloop.EventLoopPolicy())
from typing import Union, List, Tuple, Dict, Optional
from lightllm.server.core.objs import FinishStatus
from ..pd_io_struct import PD_Client_Obj, UpKVStatus, NixlUpKVStatus, ObjType, NodeRole, NIXLDecodeNodeInfo
from lightllm.server.core.objs import SamplingParams, StartArgs
from ..multimodal_params import MultimodalParams
from ..tokenizer import get_tokenizer
from ..req_id_generator import ReqIDGenerator, convert_sub_id_to_group_id
from fastapi import Request
from lightllm.utils.log_utils import init_logger
from lightllm.server.metrics.manager import MetricClient
from lightllm.utils.statics_utils import MovingAverage
from lightllm.server.httpserver.manager import AsyncQueue
from lightllm.utils.error_utils import ServerBusyError
from .pd_selector import create_selector

logger = init_logger(__name__)


class HttpServerManagerForPDMaster:
    def __init__(
        self,
        args: StartArgs,
        metric_port: int,
    ):
        self.args = args
        self.metric_client = MetricClient(metric_port)
        self.id_gen = ReqIDGenerator()

        self.pd_manager = PDManager(args)

        self.req_id_to_out_inf: Dict[int, ReqStatus] = {}
        self.infos_queues = None  # 这个需要延迟初始化，否则使用的loop不对

        self.tokenizer = get_tokenizer(args.model_dir, args.tokenizer_mode, trust_remote_code=args.trust_remote_code)

        self.first_time_costs = MovingAverage()
        self.per_token_costs = MovingAverage()
        return

    async def register_pd(self, pd_info_json, websocket):
        self.pd_manager.register_pd(pd_info_json, websocket)
        return

    async def remove_pd(self, pd_info_json):
        self.pd_manager.remove_pd(pd_info_json)
        return

    async def update_req_status(self, upkv_status: Union[UpKVStatus, NixlUpKVStatus]):
        try:
            group_request_id = convert_sub_id_to_group_id(upkv_status.group_request_id)
            up_status_event = self.req_id_to_out_inf[group_request_id].up_status_event
            up_status_event.upkv_status = upkv_status
            up_status_event.set()
        except:
            pass
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
        return self.pd_manager.select_p_d_node(prompt, sampling_params, multimodal_params)

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

            if not p_node or not d_node:
                logger.error(f"{group_request_id}: No p_node or d_node found")
                raise Exception(f"{group_request_id}: No p_node or d_node found")

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
            try:
                await self.abort(group_request_id, p_node=p_node, d_node=d_node)
            except:
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
        sampling_params.pd_master_node_id.initialize(self.args.pd_node_id)

        req_status = ReqStatus(group_request_id, p_node, d_node)
        self.req_id_to_out_inf[group_request_id] = req_status

        up_status_event = req_status.up_status_event

        d_start_args = d_node.start_args
        decode_node_dict = {
            "node_id": d_start_args["pd_node_id"],
            "ip": d_start_args["host"],
            "rpyc_port": d_start_args["pd_decode_rpyc_port"],
            "max_new_tokens": sampling_params.max_new_tokens - 1,
        }

        old_max_new_tokens = sampling_params.max_new_tokens
        sampling_params.max_new_tokens = 1
        sampling_params.move_kv_to_decode_node.initialize(decode_node_dict if old_max_new_tokens != 1 else None)
        sampling_params.suggested_dp_index = -1

        await p_node.websocket.send_bytes(pickle.dumps((ObjType.REQ, (prompt, sampling_params, multimodal_params))))

        while True:
            await req_status.wait_to_ready()
            if await request.is_disconnected():
                raise Exception(f"req_id {group_request_id} disconnected")

            if await req_status.can_read(self.req_id_to_out_inf):
                token_list = await req_status.pop_all_tokens()
                for sub_req_id, request_output, metadata, finish_status in token_list:
                    if old_max_new_tokens != 1:
                        finish_status = FinishStatus(FinishStatus.NO_FINISH)
                    else:
                        finish_status = FinishStatus(FinishStatus.FINISHED_LENGTH)
                    # 得到 p 节点返回的 prompt_ids 信息
                    if metadata.get("prompt_ids", None) is not None:
                        prompt_ids = metadata.get("prompt_ids")
                        prompt_ids.append(metadata.get("id"))
                    yield sub_req_id, request_output, metadata, finish_status
                break

        # 如果只需要一个输出 token，prefill 完就直接结束掉吧
        if old_max_new_tokens == 1:
            return

        try:
            await asyncio.wait_for(up_status_event.wait(), timeout=60)
        except asyncio.TimeoutError:
            logger.warning(f"group_request_id: {group_request_id} kv move time out err, server is busy now.")
            raise ServerBusyError()

        sampling_params.move_kv_to_decode_node.initialize(None)
        sampling_params.max_new_tokens = old_max_new_tokens - 1
        upkv_status: UpKVStatus = up_status_event.upkv_status
        sampling_params.suggested_dp_index = upkv_status.dp_index

        await d_node.websocket.send_bytes(
            pickle.dumps((ObjType.REQ, (prompt_ids, sampling_params, MultimodalParams())))
        )

        while True:
            await req_status.wait_to_ready()
            if await request.is_disconnected():
                raise Exception(f"req_id {group_request_id} disconnected")
            if await req_status.can_read(self.req_id_to_out_inf):
                token_list = await req_status.pop_all_tokens()
                for sub_req_id, request_output, metadata, finish_status in token_list:
                    yield sub_req_id, request_output, metadata, finish_status

        return

    async def fetch_nixl_stream(
        self,
        p_node: PD_Client_Obj,
        d_node: PD_Client_Obj,
        prompt: Union[str, List[int]],
        sampling_params: SamplingParams,
        multimodal_params: MultimodalParams,
        request: Request,
    ):
        group_request_id = sampling_params.group_request_id
        sampling_params.pd_master_node_id.initialize(self.args.pd_node_id)

        req_status = ReqStatus(group_request_id, p_node, d_node)
        self.req_id_to_out_inf[group_request_id] = req_status

        up_status_event = req_status.up_status_event
        nixl_np_up_prompt_ids_event = req_status.nixl_np_up_prompt_ids_event

        old_max_new_tokens = sampling_params.max_new_tokens
        sampling_params.max_new_tokens = 1
        await p_node.websocket.send_bytes(pickle.dumps((ObjType.REQ, (prompt, sampling_params, multimodal_params))))

        try:
            await asyncio.wait_for(nixl_np_up_prompt_ids_event.wait(), timeout=60)
        except asyncio.TimeoutError:
            logger.warning(f"group_request_id: {group_request_id} wait np up prompt ids time out")
            raise ServerBusyError()

        if await request.is_disconnected():
            raise Exception(f"req_id {group_request_id} disconnected")

        prompt_ids = nixl_np_up_prompt_ids_event.prompt_ids
        logger.info(f"group_request_id: {group_request_id} get np up prompt ids len {len(prompt_ids)}")

        sampling_params.max_new_tokens = old_max_new_tokens
        await d_node.websocket.send_bytes(
            pickle.dumps((ObjType.REQ, (prompt_ids, sampling_params, MultimodalParams())))
        )

        try:
            await asyncio.wait_for(up_status_event.wait(), timeout=60)
        except asyncio.TimeoutError:
            logger.warning(f"group_request_id: {group_request_id} kv move time out err, server is busy now.")
            raise ServerBusyError()

        # 将 decode 节点上报的当前请求使用的decode节点的信息下发给 p 节点，这样 p 节点才知道将 kv 传输给那个 d 节点。
        upkv_status: NixlUpKVStatus = up_status_event.upkv_status
        nixl_params: bytes = upkv_status.nixl_params
        decode_node_info: NIXLDecodeNodeInfo = pickle.loads(nixl_params)
        await p_node.websocket.send_bytes(
            pickle.dumps((ObjType.NIXL_REQ_DECODE_NODE_INFO, group_request_id, decode_node_info))
        )

        first_token_gen = False
        while True:
            await req_status.wait_to_ready()
            if await request.is_disconnected():
                raise Exception(f"req_id {group_request_id} disconnected")
            if await req_status.can_read(self.req_id_to_out_inf):
                token_list = await req_status.pop_all_tokens()
                for sub_req_id, request_output, metadata, finish_status in token_list:
                    output_index = metadata.get("count_output_tokens")
                    # 因为 nixl 的 prefill 和 decode 节点都有可能上报首token，所以需要做一下过滤。
                    if output_index == 1:
                        if first_token_gen is False:
                            first_token_gen = True
                            node_run_mode = metadata.pop("node_mode", None)
                            if node_run_mode == "nixl_prefill":
                                if old_max_new_tokens != 1 and finish_status.is_finished_length():
                                    finish_status = FinishStatus(FinishStatus.NO_FINISH)
                            yield sub_req_id, request_output, metadata, finish_status
                        else:
                            continue
                    else:
                        yield sub_req_id, request_output, metadata, finish_status

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
        if sampling_params.disable_prompt_cache:
            assert False, "pd mode dont support set disable_prompt_cache to True"

        out_token_counter = 0
        first_token_cost_ms = float("inf")
        group_request_id = sampling_params.group_request_id
        unfinished_count = sampling_params.best_of
        is_first_token = True

        client_mode: NodeRole = NodeRole(d_node.mode)

        fetch_stream = self.fetch_nixl_stream if client_mode.is_NP_or_ND() else self.fetch_stream

        async for sub_req_id, out_str, metadata, finish_status in fetch_stream(
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

    async def abort(
        self, group_request_id, p_node: Optional[PD_Client_Obj] = None, d_node: Optional[PD_Client_Obj] = None
    ):
        logger.warning(f"aborted group_request_id {group_request_id}")

        try:
            req_status = self.req_id_to_out_inf[group_request_id]
            del self.req_id_to_out_inf[group_request_id]
            p_node = req_status.p_node
            d_node = req_status.d_node
        except:
            pass

        try:
            await p_node.websocket.send_bytes(pickle.dumps((ObjType.ABORT, group_request_id)))
        except:
            pass

        try:
            await d_node.websocket.send_bytes(pickle.dumps((ObjType.ABORT, group_request_id)))
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
                        token_list, node_load_info = obj[1], obj[2]
                        self.pd_manager.update_node_load_info(node_load_info)

                        for sub_req_id, text, metadata, finish_status in token_list:
                            finish_status: FinishStatus = finish_status
                            group_req_id = convert_sub_id_to_group_id(sub_req_id)
                            try:
                                req_status: ReqStatus = self.req_id_to_out_inf[group_req_id]
                                async with req_status.lock:
                                    req_status.out_token_info_list.append((sub_req_id, text, metadata, finish_status))
                                    req_status.event.set()
                            except:
                                pass
                    elif obj[0] == ObjType.NIXL_UPLOAD_NP_PROMPT_IDS:
                        _, group_req_id, prompt_ids = obj
                        try:
                            req_status: ReqStatus = self.req_id_to_out_inf[group_req_id]
                            async with req_status.lock:
                                req_status.nixl_np_up_prompt_ids_event.prompt_ids = prompt_ids
                                req_status.nixl_np_up_prompt_ids_event.set()
                        except:
                            logger.error(
                                f"NIXL_UPLOAD_NP_PROMPT_IDS fail find req status for group_req_id: {group_req_id}"
                            )
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
        self.nixl_np_up_prompt_ids_event = asyncio.Event()
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


class PDManager:
    def __init__(self, args: StartArgs):
        self.args: StartArgs = args
        self.prefill_nodes: List[PD_Client_Obj] = []
        self.decode_nodes: List[PD_Client_Obj] = []
        self.url_to_pd_nodes: Dict[str, PD_Client_Obj] = {}
        self.selector = create_selector(args.select_p_d_node_strategy, self)
        return

    def register_pd(self, pd_info_json, websocket):
        pd_client = PD_Client_Obj(**pd_info_json)
        pd_client.websocket = websocket
        self.url_to_pd_nodes[pd_client.client_ip_port] = pd_client

        if pd_client.mode in ["prefill", "nixl_prefill"]:
            self.prefill_nodes = [e for e in self.prefill_nodes if e.client_ip_port != pd_client.client_ip_port]
            self.prefill_nodes.append(pd_client)
        elif pd_client.mode in ["decode", "nixl_decode"]:
            self.decode_nodes = [e for e in self.decode_nodes if e.client_ip_port != pd_client.client_ip_port]
            self.decode_nodes.append(pd_client)
        else:
            assert False, f"mode must in ['prefill', 'decode'], but get {pd_client.mode}"

        self.selector.update_nodes(self.prefill_nodes, self.decode_nodes)

        logger.info(f"mode: {pd_client.mode} url: {pd_client.client_ip_port} registed")
        return

    def remove_pd(self, pd_info_json):
        pd_client = PD_Client_Obj(**pd_info_json)

        self.url_to_pd_nodes.pop(pd_client.client_ip_port, None)
        self.prefill_nodes = [e for e in self.prefill_nodes if e.client_ip_port != pd_client.client_ip_port]
        self.decode_nodes = [e for e in self.decode_nodes if e.client_ip_port != pd_client.client_ip_port]

        self.selector.update_nodes(self.prefill_nodes, self.decode_nodes)

        logger.info(f"mode: {pd_client.mode} url: {pd_client.client_ip_port} removed")
        return

    def update_node_load_info(self, load_info: Optional[dict]):
        """更新节点负载信息
        load_info: 节点负载信息字典，内容格式如下，可以为 None
        {
        "total_token_usage_rate": xxxx,
        "client_ip_port": xxxx,
        }
        """
        try:
            if load_info is None:
                return
            client_ip_port = load_info["client_ip_port"]
            total_token_usage_rate = load_info["total_token_usage_rate"]
            pd_client = self.url_to_pd_nodes.get(client_ip_port)
            pd_client.run_status.total_token_usage_rate = total_token_usage_rate
        except BaseException as e:
            logger.warning(f"udpate node load info failed, load_info: {load_info} error: {str(e)}")
        return

    def select_p_d_node(
        self, prompt: Union[str, List[int]], sampling_params: SamplingParams, multimodal_params: MultimodalParams
    ) -> Tuple[PD_Client_Obj, PD_Client_Obj]:
        p_node, d_node = self.selector.select_p_d_node(prompt, sampling_params, multimodal_params)
        return p_node, d_node
