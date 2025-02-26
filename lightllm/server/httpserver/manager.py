import sys
import zmq
import zmq.asyncio
import asyncio
import uvloop
import rpyc
import time
import hashlib
import datetime
import websockets
import pickle
import ujson as json

asyncio.set_event_loop_policy(uvloop.EventLoopPolicy())
from typing import Union, List, Tuple, Dict
from ..tokenizer import get_tokenizer
from ..pd_io_struct import NodeRole, ObjType
from ..embed_cache.utils import get_shm_name_data, create_shm
from ..multimodal_params import MultimodalParams
from ..req_id_generator import ReqIDGenerator
from .async_queue import AsyncQueue
from lightllm.server.core.objs import Req, FinishStatus
from lightllm.server.core.objs import SamplingParams
from lightllm.server.core.objs.io_objs import GroupReqObjs
from fastapi import Request
from lightllm.server.core.objs.shm_req_manager import ShmReqManager
from lightllm.utils.log_utils import init_logger
from lightllm.utils.envs_utils import get_env_start_args
from lightllm.server.metrics.manager import MetricClient
from lightllm.utils.statics_utils import MovingAverage
from lightllm.utils.net_utils import get_hostname_ip
from lightllm.utils.config_utils import get_vocab_size

logger = init_logger(__name__)


class HttpServerManager:
    def __init__(
        self,
        args,
        router_port,
        cache_port,
        detokenization_pub_port,
        visual_port,
        metric_port,
        enable_multimodal,
    ):
        self.args = args
        context = zmq.asyncio.Context(2)
        self.send_to_router = context.socket(zmq.PUSH)
        self.send_to_router.connect(f"{args.zmq_mode}127.0.0.1:{router_port}")

        self.enable_multimodal = enable_multimodal
        if self.enable_multimodal:
            self.cache_client = rpyc.connect("localhost", cache_port)
            self.send_to_visual = context.socket(zmq.PUSH)
            self.send_to_visual.connect(f"{args.zmq_mode}127.0.0.1:{visual_port}")

        self.shm_req_manager = ShmReqManager()

        self.recv_from_detokenization = context.socket(zmq.SUB)
        self.recv_from_detokenization.connect(f"{args.zmq_mode}127.0.0.1:{detokenization_pub_port}")
        self.recv_from_detokenization.setsockopt(zmq.SUBSCRIBE, b"")

        self.tokenizer = get_tokenizer(args.model_dir, args.tokenizer_mode, trust_remote_code=args.trust_remote_code)

        self.req_id_to_out_inf: Dict[int, ReqStatus] = {}  # value type (out_str, metadata, finished, event)
        self.forwarding_queue: AsyncQueue = None  # p d 分离模式使用的转发队列, 需要延迟初始化

        self.max_req_total_len = args.max_req_total_len
        self.metric_client = MetricClient(metric_port)

        self.pd_mode: NodeRole = NodeRole(self.args.run_mode)
        assert self.pd_mode in [NodeRole.P, NodeRole.D, NodeRole.NORMAL]
        self.id_gen = ReqIDGenerator()
        self.first_time_costs = MovingAverage()
        self.per_token_costs = MovingAverage()
        # 有的模型的vocab size 读取tokenizer和config.json中不一致
        self.vocab_size = max(get_vocab_size(args.model_dir), self.tokenizer.vocab_size)

        return

    # connect cache server, calculate md5, alloc resource, return uuid
    async def _alloc_resource(self, data, num):
        md5sum = hashlib.md5(data).hexdigest()
        wait_time = 1
        while True:
            record = self.cache_client.root.alloc(md5sum, num)
            # hit or new
            if record:
                uid = record["id"]
                if not self.cache_client.root.get_item_data(uid):
                    create_shm(get_shm_name_data(uid), data)
                    self.cache_client.root.set_item_data(uid)
                return record
            # cache full
            else:
                await asyncio.sleep(wait_time)
                wait_time = min(wait_time + 2, 9)

    async def _alloc_multimodal_resources(self, multimodal_params: MultimodalParams):
        # 只有 P 和 NORMAL 节点需要真的管理多模态资源
        if self.pd_mode.is_P_or_NORMAL():
            for img in multimodal_params.images:
                record = await self._alloc_resource(img.read(), self.tokenizer.get_image_token_length(img))
                img.uuid = record["id"]
                img.token_id = record["token_id"]
                img.token_num = record["token_num"]
        return

    async def _release_multimodal_resources(self, multimodal_params: MultimodalParams):
        # 只有 P 和 NORMAL 节点需要真的管理多模态资源
        if self.pd_mode.is_P_or_NORMAL():
            if multimodal_params is not None:
                for img in multimodal_params.images:
                    if img.uuid is not None:
                        self.cache_client.root.release(img.uuid)
                        # 将 uuid 等 赋值为 None, 防止因为abort等异常情况造成重复释放异常
                        img.uuid = None
                        img.token_id = None
                        img.token_num = None
        return

    def tokens(self, prompt, kwargs=None):
        kwargs = {} if kwargs is None else kwargs
        prompt_ids = self.tokenizer.encode(prompt, None, **kwargs)
        return len(prompt_ids)

    async def generate(
        self,
        prompt: Union[str, List[int]],
        sampling_params: SamplingParams,
        multimodal_params: MultimodalParams,
        request: Request,
    ) -> Tuple[int, str, dict, FinishStatus]:
        start_time = time.time()
        # 请求的 id 可以由外部传入，也可以由内部生成，但是由外部传入的时候，要自己保证全局唯一性
        # 否则会造成异常问题。目前限制 NORMAL 模式都使用内部id替换， P 和 D 模式按需设置
        if self.pd_mode == NodeRole.NORMAL:
            group_request_id = self.id_gen.generate_id()
            sampling_params.group_request_id = group_request_id
        elif self.pd_mode == NodeRole.P or self.pd_mode == NodeRole.D:
            assert sampling_params.group_request_id is not None, "p d mode, group_request_id must be setting"
            group_request_id = sampling_params.group_request_id
        else:
            assert False, "dead code path"

        try:
            if self.pd_mode.is_P_or_NORMAL():
                multimodal_params.verify_and_preload()

            # 记录请求到达的相关信息
            await self._log_req_header(request, group_request_id)
            # 监控
            self.metric_client.counter_inc("lightllm_request_count")

            prompt_ids = await self._encode(
                prompt, multimodal_params, add_special_tokens=sampling_params.add_special_tokens
            )
            prompt_tokens = len(prompt_ids)
            # 监控
            self.metric_client.histogram_observe("lightllm_request_input_length", prompt_tokens)
            self.metric_client.histogram_observe("lightllm_request_max_new_tokens", sampling_params.max_new_tokens)
            verify_time_begin = time.time()
            prompt_ids = await self._check_and_repair_length(prompt_ids, sampling_params)
            verify_time_end = time.time()
            self.metric_client.histogram_observe(
                "lightllm_request_validation_duration", verify_time_end - verify_time_begin
            )

            # 申请资源并存储
            alloced_req_indexes = []
            while len(alloced_req_indexes) < sampling_params.n:
                alloc_req_index = await self.shm_req_manager.async_alloc_req_index()
                while alloc_req_index is None:
                    await asyncio.sleep(0.1)
                    alloc_req_index = await self.shm_req_manager.async_alloc_req_index()
                alloced_req_indexes.append(alloc_req_index)
            req_objs = []
            for i, req_index in enumerate(alloced_req_indexes):
                req_obj = await self.shm_req_manager.async_get_req_obj_by_index(req_index)
                req_obj.init(
                    group_request_id + i,
                    prompt_ids,
                    sampling_params,
                    self.tokenizer,
                    chunked_prefill_size=self.args.chunked_prefill_size,
                )
                req_objs.append(req_obj)

            req_status = ReqStatus(group_request_id, multimodal_params, req_objs, start_time)
            self.req_id_to_out_inf[group_request_id] = req_status

            # 将请求转发给其他节点
            await self.transfer_to_next_module(req_status.group_req_objs)

            results_generator = self._wait_to_token_package(
                start_time, prompt_ids, group_request_id, sampling_params, req_status, request
            )
            async for sub_req_id, request_output, metadata, finish_status in results_generator:
                # p d 模式下，将 token 数据放入到转发队列中
                if self.pd_mode.is_P_or_D() and sub_req_id >= 0:
                    await self.forwarding_queue.put((sub_req_id, request_output, metadata, finish_status))
                else:
                    yield sub_req_id, request_output, metadata, finish_status

        except Exception as e:
            logger.error(f"group_request_id: {group_request_id} has exception {str(e)}")
            await self.abort(group_request_id)
            raise e
        return

    async def _log_req_header(self, request: Request, group_request_id: int):

        x_request_id = request.headers.get("X-Request-Id", "") if request is not None else ""
        x_session_id = request.headers.get("X-Session-Id", "") if request is not None else ""

        format_in_time = datetime.datetime.fromtimestamp(time.time()).strftime("%Y-%m-%d %H:%M:%S")
        logger.info(
            f"recieved req X-Request-Id:{x_request_id} "
            f"X-Session-Id:{x_session_id} start_time:{format_in_time} "
            f"lightllm_req_id:{group_request_id} "
        )
        return

    async def _encode(
        self, prompt: Union[str, List[int]], multimodal_params: MultimodalParams, add_special_tokens: bool
    ):
        if isinstance(prompt, str):
            if self.enable_multimodal:
                assert len(multimodal_params.images) <= self.args.cache_capacity, "too many images!"
                await self._alloc_multimodal_resources(multimodal_params)
                prompt_ids = self.tokenizer.encode(prompt, multimodal_params, add_special_tokens=add_special_tokens)
            else:
                prompt_ids = self.tokenizer.encode(prompt, add_special_tokens=add_special_tokens)
            return prompt_ids

        # 这里的校验对多模态不是很充分, to do
        if all(isinstance(e, int) for e in prompt):
            if not self.enable_multimodal:
                if all(e < self.vocab_size for e in prompt):
                    return prompt
                else:
                    raise ValueError("prompt List[int] format contain id > vocab_size")
            else:
                return prompt
        else:
            raise ValueError(f"prompt format error, get type{type(prompt)}")
        return

    async def _check_and_repair_length(self, prompt_ids: List[int], sampling_params: SamplingParams):
        prompt_tokens = len(prompt_ids)
        if prompt_tokens + sampling_params.max_new_tokens > self.max_req_total_len:
            # use long_truncation_mode to truncate long input len req.
            if self.args.long_truncation_mode is None:
                raise ValueError(
                    f"the input prompt token len {prompt_tokens} + max_new_tokens \
                        {sampling_params.max_new_tokens} > {self.max_req_total_len}"
                )
            elif self.args.long_truncation_mode == "head":
                prompt_ids = prompt_ids[-(self.max_req_total_len - sampling_params.max_new_tokens) :]
            elif self.args.long_truncation_mode == "center":
                req_input_len = self.max_req_total_len - sampling_params.max_new_tokens
                prompt_ids = prompt_ids[0 : req_input_len // 2] + prompt_ids[-(req_input_len - req_input_len // 2) :]
                prompt_tokens = len(prompt_ids)
                assert prompt_tokens == req_input_len
            else:
                assert False, "error args"

        # last repaired
        req_total_len = len(prompt_ids) + sampling_params.max_new_tokens
        if req_total_len > self.max_req_total_len:
            raise ValueError(
                f"the req total len (input len + output len) is too long > max_req_total_len:{self.max_req_total_len}"
            )

        return prompt_ids

    async def transfer_to_next_module(
        self,
        group_req_objs: GroupReqObjs,
    ):
        if self.pd_mode == NodeRole.P:
            if self.enable_multimodal:
                self.send_to_visual.send_pyobj(
                    group_req_objs.to_group_req_index(),
                    protocol=pickle.HIGHEST_PROTOCOL,
                )
            else:
                self.send_to_router.send_pyobj(
                    group_req_objs.to_group_req_index(),
                    protocol=pickle.HIGHEST_PROTOCOL,
                )
            return

        if self.pd_mode == NodeRole.D:
            # 在 D 模式下，不需要传输真的多模态参数，因为其已经被 P 处理好了, 传输一个空的即可
            self.send_to_router.send_pyobj(
                group_req_objs.to_group_req_index(),
                protocol=pickle.HIGHEST_PROTOCOL,
            )
            return

        if self.pd_mode == NodeRole.NORMAL:
            if self.enable_multimodal:
                self.send_to_visual.send_pyobj(
                    group_req_objs.to_group_req_index(),
                    protocol=pickle.HIGHEST_PROTOCOL,
                )
            else:
                self.send_to_router.send_pyobj(
                    group_req_objs.to_group_req_index(),
                    protocol=pickle.HIGHEST_PROTOCOL,
                )
            return

        assert False, "dead code path"
        return

    async def _wait_to_token_package(
        self,
        start_time,
        prompt_ids: List[int],
        group_request_id: int,
        sampling_params: SamplingParams,
        req_status: "ReqStatus",
        request: Request,
    ):

        event = req_status.event
        unfinished_count = sampling_params.best_of
        out_token_counter = 0
        first_token_cost_ms = sys.float_info.max
        prompt_tokens = len(prompt_ids)
        is_first_token = True

        while True:
            try:
                await asyncio.wait_for(event.wait(), timeout=5)
            except asyncio.TimeoutError:
                pass

            if request is not None and await request.is_disconnected():
                await self.abort(group_request_id)
                raise Exception(f"req_id {group_request_id} disconnected")

            async with req_status.lock:
                event.clear()
                if len(req_status.out_token_info_list) == 0:
                    continue

                for sub_req_id, out_str, metadata, finish_status in req_status.out_token_info_list:
                    # pd master 节点需要这个做统计信息， 所以放在元数据中返回给 pd master 节点
                    metadata["prompt_tokens"] = prompt_tokens
                    # p 节点返回 prompt_ids 信息，防止 d 节点重新 encode
                    if self.pd_mode == NodeRole.P and is_first_token:
                        metadata["prompt_ids"] = prompt_ids

                    prompt_cache_len = metadata.pop("prompt_cache_len", 0)
                    if is_first_token:
                        first_token_cost_ms = (time.time() - start_time) * 1000
                        is_first_token = False
                        self.first_time_costs.add(first_token_cost_ms)

                    out_token_counter += 1

                    yield sub_req_id, out_str, metadata, finish_status
                    # 如果有子请求完成，就更新计数
                    if finish_status.is_finished():
                        unfinished_count -= 1

                    # 所有子请求完成后，就删除占用的资源
                    if unfinished_count == 0:
                        total_cost_time_ms = (time.time() - start_time) * 1000
                        mean_per_token_cost_time_ms = (total_cost_time_ms - first_token_cost_ms) / out_token_counter
                        self.per_token_costs.add(mean_per_token_cost_time_ms)
                        x_request_id = request.headers.get("X-Request-Id", "") if request is not None else ""
                        x_session_id = request.headers.get("X-Session-Id", "") if request is not None else ""

                        prompt_cache_ratio = prompt_cache_len / prompt_tokens
                        self.metric_client.histogram_observe("lightllm_cache_length", prompt_cache_len)
                        self.metric_client.histogram_observe("lightllm_cache_ratio", prompt_cache_ratio)
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
                        self.metric_client.histogram_observe(
                            "lightllm_request_inference_duration", total_cost_time_ms / 1000.0
                        )
                        self.metric_client.histogram_observe(
                            "lightllm_request_mean_time_per_token_duration", mean_per_token_cost_time_ms / 1000.0
                        )
                        self.metric_client.histogram_observe(
                            "lightllm_request_first_token_duration", first_token_cost_ms / 1000.0
                        )
                        self.metric_client.histogram_observe("lightllm_request_generated_tokens", out_token_counter)
                        self.metric_client.counter_inc("lightllm_request_success")

                        return
                req_status.out_token_info_list.clear()
        return

    async def abort(self, group_req_id: int):
        if group_req_id in self.req_id_to_out_inf:
            req_status = self.req_id_to_out_inf[group_req_id]
            group_req_objs: GroupReqObjs = req_status.group_req_objs
            for req in group_req_objs.shm_req_objs:
                req.is_aborted = True
            logger.warning(f"aborted group_request_id {group_req_objs.group_req_id}")
        else:
            logger.warning("aborted group_request_id not exist")
        return

    async def recycle_resource_loop(self):
        pre_time_mark = time.time()

        while True:

            try:
                await asyncio.wait_for(self.recycle_event.wait(), timeout=0.02)
            except asyncio.TimeoutError:
                pass
            self.recycle_event.clear()

            # 清理已经处理完的可以删除的请求
            release_req_status: List[ReqStatus] = []
            for req_status in self.req_id_to_out_inf.values():
                if req_status.can_release():
                    release_req_status.append(req_status)

            for req_status in release_req_status:
                self.req_id_to_out_inf.pop(req_status.group_req_objs.group_req_id, None)
                for req in req_status.group_req_objs.shm_req_objs:
                    await self.shm_req_manager.async_put_back_req_obj(req)
                    await self.shm_req_manager.async_release_req_index(req.index_in_shm_mem)
                await self._release_multimodal_resources(req_status.group_req_objs.multimodal_params)

            # 先保留这个关键得日志，用于方便定位重构中的问题。
            if time.time() - pre_time_mark > 20:
                pre_time_mark = time.time()
                for req_status in self.req_id_to_out_inf.values():
                    logger.info(
                        f"left req id {req_status.group_req_objs.group_req_id}"
                        f"can release {req_status.group_req_objs.shm_req_objs[0].can_released_mark} "
                        f"refcount {req_status.group_req_objs.shm_req_objs[0].ref_count}"
                    )
        return

    async def handle_loop(self):
        self.recycle_event = asyncio.Event()
        asyncio.create_task(self.recycle_resource_loop())

        if self.pd_mode.is_P_or_D():
            self.forwarding_queue = AsyncQueue()
            asyncio.create_task(self.pd_handle_loop())

        while True:
            try:
                await asyncio.wait_for(self.recv_from_detokenization.recv_pyobj(), timeout=0.05)
            except asyncio.TimeoutError:
                pass

            for req_status in self.req_id_to_out_inf.values():
                token_list = []
                for req in req_status.group_req_objs.shm_req_objs:
                    req_id = req.request_id
                    if not req.out_tokens_queue.is_empty():

                        text, src_index, special, count_output_tokens = req.out_tokens_queue.peek()
                        req.cumlogprob += float(req.shm_logprobs.arr[src_index])
                        metadata = {
                            "id": int(req.shm_prompt_ids.arr[src_index]),
                            "logprob": float(req.shm_logprobs.arr[src_index]),
                            "cumlogprob": float(req.cumlogprob) / count_output_tokens,
                            "special": special,
                            "count_output_tokens": count_output_tokens,
                            "prompt_cache_len": req.prompt_cache_len,
                        }
                        if self.args.return_all_prompt_logprobs:
                            metadata.update(req.get_all_prompt_metadata())
                        if self.args.use_reward_model:
                            metadata["score"] = float(req.reward_score)

                        req.out_tokens_queue.pop_no_ret()

                        if req.finish_token_index != src_index:
                            token_list.append((req_id, text, metadata, FinishStatus()))
                        else:
                            finish_status = FinishStatus(req.finish_status.status)
                            token_list.append((req_id, text, metadata, finish_status))

                async with req_status.lock:
                    req_status.out_token_info_list.extend(token_list)
                    req_status.event.set()

            self.recycle_event.set()
        return

    async def pd_handle_loop(self):
        asyncio.create_task(self.timer_log())
        if self.pd_mode not in [NodeRole.P, NodeRole.D]:
            return

        self.host_ip = get_hostname_ip()
        if self.host_ip is None:
            self.host_ip = self.args.host

        while True:
            forwarding_tokens_task = None
            try:
                uri = f"ws://{self.args.pd_master_ip}:{self.args.pd_master_port}/pd_register"
                async with websockets.connect(uri, max_queue=(2048 * 1024, 2048 * 1023)) as websocket:
                    import socket

                    sock = websocket.transport.get_extra_info("socket")
                    sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)

                    args_dict = vars(self.args)
                    args_dict["host"] = self.host_ip
                    # 发送注册信息
                    regist_json = {
                        "node_id": self.args.pd_node_id,
                        "client_ip_port": f"{self.host_ip}:{self.args.port}",
                        "mode": self.pd_mode.value,
                        "start_args": args_dict,
                    }

                    await websocket.send(json.dumps(regist_json))
                    logger.info(f"Sent registration JSON: {regist_json}")

                    # 转发token的task
                    async def up_tokens_to_pd_master(forwarding_queue: AsyncQueue, websocket):
                        while True:
                            handle_list = await forwarding_queue.wait_to_get_all_data()
                            if len(handle_list) != 0:
                                await websocket.send(pickle.dumps((ObjType.TOKEN_PACKS, handle_list)))
                        return

                    forwarding_tokens_task = asyncio.create_task(
                        up_tokens_to_pd_master(self.forwarding_queue, websocket)
                    )

                    while True:
                        recv_bytes = await websocket.recv()
                        obj = pickle.loads(recv_bytes)
                        if obj[0] == ObjType.REQ:
                            prompt, sampling_params, multimodal_params = obj[1]

                            # 触发推理的task
                            async def pd_process_generate(
                                manager: "HttpServerManager", prompt, sampling_params, multimodal_params
                            ):
                                try:
                                    async for _, _, _, _ in manager.generate(
                                        prompt, sampling_params, multimodal_params, None
                                    ):
                                        pass
                                except BaseException as e:
                                    logger.error(str(e))

                            asyncio.create_task(pd_process_generate(self, prompt, sampling_params, multimodal_params))

                        elif obj[0] == ObjType.ABORT:
                            group_req_id = obj[1]
                            await self.abort(group_req_id)
                        else:
                            logger.error(f"recevie error obj {str(obj)}")

            except Exception as e:
                logger.error("connetion to pd_master has error")
                logger.exception(str(e))
                if forwarding_tokens_task is not None:
                    forwarding_tokens_task.cancel()
                await asyncio.sleep(10)
                await self.forwarding_queue.get_all_data()
                logger.info("reconnection to pd_master")

    async def timer_log(self):
        while True:
            await asyncio.sleep(30)
            self.first_time_costs.print_log("mean first cost")
            self.per_token_costs.print_log("mean per token cost")


class ReqStatus:
    def __init__(self, group_request_id, multimodal_params, req_objs: List[Req], start_time) -> None:
        self.lock = asyncio.Lock()
        self.event = asyncio.Event()
        self.group_req_objs = GroupReqObjs(
            group_req_id=group_request_id,
            multimodal_params=multimodal_params,
            shm_req_objs=req_objs,
            time_mark=start_time,
        )
        self.out_token_info_list = []

    def can_release(self):
        for req in self.group_req_objs.shm_req_objs:
            if not req.can_release():
                return False
        return True
