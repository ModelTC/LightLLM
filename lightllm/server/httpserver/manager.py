import sys
import zmq
import zmq.asyncio
import asyncio
import uvloop
import rpyc
import time
import copy
import hashlib
import datetime
import pickle
from frozendict import frozendict

asyncio.set_event_loop_policy(uvloop.EventLoopPolicy())
from typing import Union, List, Tuple, Dict, Optional
from fastapi import Request
from ..tokenizer import get_tokenizer
from ..pd_io_struct import NodeRole
from ..embed_cache.utils import get_shm_name_data, create_shm
from ..multimodal_params import AudioItem, MultimodalParams, ImageItem
from ..req_id_generator import ReqIDGenerator
from .async_queue import AsyncQueue
from lightllm.server.core.objs import Req, FinishStatus
from lightllm.server.core.objs import SamplingParams
from lightllm.server.core.objs.out_token_circlequeue import LIGHTLLM_OUT_TOKEN_QUEUE_SIZE
from lightllm.server.core.objs.io_objs import GroupReqObjs
from lightllm.server.core.objs.shm_req_manager import ShmReqManager
from lightllm.server.core.objs.atomic_array_lock import AtomicShmArrayLock, AsyncLock, AtomicLockItem
from lightllm.server.router.dynamic_prompt.shared_arr import SharedInt
from lightllm.utils.log_utils import init_logger
from lightllm.server.metrics.manager import MetricClient
from lightllm.utils.statics_utils import MovingAverage
from lightllm.utils.config_utils import get_vocab_size
from lightllm.utils.envs_utils import get_unique_server_name
from rpyc.utils.classic import obtain

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

        self.multinode_req_manager = None
        self.nnodes = args.nnodes
        self._shm_lock_pool = AtomicShmArrayLock(f"{get_unique_server_name()}_lightllm_resource_lock", 1)
        self._resource_lock = AsyncLock(self._shm_lock_pool.get_lock_context(0))
        self.node_rank = args.node_rank
        self.disable_abort = args.nnodes > 1 and args.dp == 1  # mulitnode dp=1 mode, disable abort
        self.is_multinode_tp = args.dp == 1 and args.nnodes > 1
        self.is_multinode_tp_master = args.dp == 1 and args.nnodes > 1 and args.node_rank == 0
        self.is_multinode_tp_slave = args.dp == 1 and args.nnodes > 1 and args.node_rank > 0
        if self.is_multinode_tp:
            if args.node_rank == 0:
                self.multinode_req_manager = []
                for child_ip in args.child_ips:
                    context = zmq.asyncio.Context(2)
                    self.multinode_req_manager.append(context.socket(zmq.PUSH))
                    self.multinode_req_manager[-1].connect(f"tcp://{child_ip}:{args.multinode_httpmanager_port}")
                    logger.info(
                        f"HttpServerManager connected to child node at {child_ip}:{args.multinode_httpmanager_port}"
                    )
            else:
                context = zmq.asyncio.Context(2)
                self.multinode_req_manager = context.socket(zmq.PULL)
                self.multinode_req_manager.bind(f"tcp://*:{args.multinode_httpmanager_port}")
                logger.info(
                    f"HttpServerManager listening for child node requests on *:{args.multinode_httpmanager_port}"
                )

        self.enable_multimodal = enable_multimodal
        if self.enable_multimodal:
            self.cache_client = rpyc.connect("localhost", cache_port, config={"allow_pickle": True})
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

        # The timemark of the latest inference(prefill/decode) which is used to check the health status of the system.
        # If the timemark is not updated for a pre-set time, a prob request will be sent to the backend.
        self.latest_success_infer_time_mark = SharedInt(f"{get_unique_server_name()}_latest_success_infer_time_mark")
        self.latest_success_infer_time_mark.set_value(int(time.time()))
        return

    async def _alloc_resource(self, items, md5sums, token_nums, datas):

        while True:
            records = obtain(self.cache_client.root.alloc(md5sums, token_nums))

            if records is None:
                await asyncio.sleep(0.1)
                continue

            uid_list = []
            for item, rec in zip(items, records):
                item.uuid = rec["id"]
                item.token_id = rec["token_id"]
                item.token_num = rec["token_num"]
                uid_list.append(rec["id"])

            ready_flags = obtain(self.cache_client.root.get_items_data(uid_list))
            update_data_ids = []

            for uid, ready, data in zip(uid_list, ready_flags, datas):
                if not ready:
                    create_shm(get_shm_name_data(uid), data)
                    update_data_ids.append(uid)

            if update_data_ids:
                self.cache_client.root.set_items_data(update_data_ids)
            return

    async def _alloc_multimodal_resources(self, multimodal_params: MultimodalParams, sampling_params: SamplingParams):
        # 只有 P 和 NORMAL 节点需要真的管理多模态资源
        if self.pd_mode.is_P_or_NORMAL():
            # 这里的锁是为了 防止多个含有多张图片的请求 同时申请的record数量 大于cache_capacity，从而造成死锁的问题。
            # 如果不加任何锁，假如请求1和请求2都有6张图片，而cache_capacity为10，
            # 那么如果某一时刻shm中存在请求1的5张图和请求2的5张图，将会资源竞争产生死锁。
            async with self._resource_lock:
                items, md5sums, tokens_nums, datas = [], [], [], []
                for img in multimodal_params.images:
                    self.tokenizer.init_imageitem_extral_params(img, multimodal_params, sampling_params)
                    data = img.read()
                    # must after init_imageitem_extral_params
                    token_num = self.tokenizer.get_image_token_length(img)
                    md5sum = hashlib.md5(data).hexdigest() + "_" + str(hash(frozendict(img.extra_params)))
                    md5sums.append(md5sum)
                    tokens_nums.append(token_num)
                    datas.append(data)
                    items.append(img)
                for audio in multimodal_params.audios:
                    self.tokenizer.init_audioitem_extral_params(audio, multimodal_params, sampling_params)
                    data = audio.read()
                    token_num = self.tokenizer.get_audio_token_length(audio)
                    md5sum = hashlib.md5(data).hexdigest() + "_" + str(hash(frozendict(audio.extra_params)))
                    md5sums.append(md5sum)
                    tokens_nums.append(token_num)
                    datas.append(data)
                    items.append(audio)

                await self._alloc_resource(items, md5sums, tokens_nums, datas)
        return

    async def _release_multimodal_resources(self, multimodal_params: MultimodalParams):
        # 只有 P 和 NORMAL 节点需要真的管理多模态资源
        if self.pd_mode.is_P_or_NORMAL():
            if multimodal_params is not None:
                ids_to_release = []
                for img in multimodal_params.images:
                    if img.uuid is not None:
                        ids_to_release.append(img.uuid)
                        # 将 uuid 等 赋值为 None, 防止因为abort等异常情况造成重复释放异常
                        img.uuid = None
                        img.token_id = None
                        img.token_num = None
                for audio in multimodal_params.audios:
                    if audio.uuid is not None:
                        ids_to_release.append(audio.uuid)
                        # 将 uuid 等 赋值为 None, 防止因为abort等异常情况造成重复释放异常
                        audio.uuid = None
                        audio.token_id = None
                        audio.token_num = None
                if ids_to_release:
                    self.cache_client.root.release(ids_to_release)
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

    async def loop_for_request(self):
        assert self.args.node_rank > 0
        while True:
            (
                prompt,
                sampling_params,
                multimodal_params,
            ) = await self.multinode_req_manager.recv_pyobj()
            results_generator = self.generate(prompt, sampling_params, multimodal_params, None)

            async def generate_wrapper(results_generator):
                async for _, _, _, _ in results_generator:
                    pass

            asyncio.create_task(generate_wrapper(results_generator))
        return

    def alloc_req_id(self, sampling_params, is_health_req: bool = False):
        # 请求的 id 可以由外部传入，也可以由内部生成，但是由外部传入的时候，要自己保证全局唯一性
        # 否则会造成异常问题。目前限制 NORMAL 模式都使用内部id替换， P 和 D 模式按需设置
        # health 请求 request_id 为负数，直接返回
        if is_health_req:
            return sampling_params.group_request_id
        if self.pd_mode == NodeRole.NORMAL:
            if not self.is_multinode_tp:
                group_request_id = self.id_gen.generate_id()
            else:
                if self.node_rank == 0:
                    group_request_id = self.id_gen.generate_id()
                else:
                    assert sampling_params.group_request_id != -1
                    group_request_id = sampling_params.group_request_id
            sampling_params.group_request_id = group_request_id
        elif self.pd_mode == NodeRole.P or self.pd_mode == NodeRole.D:
            assert sampling_params.group_request_id is not None, "p d mode, group_request_id must be setting"
            group_request_id = sampling_params.group_request_id
        else:
            assert False, "dead code path"
        return group_request_id

    async def generate(
        self,
        prompt: Union[str, List[int]],
        sampling_params: SamplingParams,
        multimodal_params: MultimodalParams,
        request: Request,
        is_health_req: bool = False,
    ) -> Tuple[int, str, dict, FinishStatus]:
        start_time = time.time()
        request_headers = request.headers if request is not None else {}
        group_request_id = self.alloc_req_id(sampling_params, is_health_req)

        try:
            original_multimodal_params = None
            if self.is_multinode_tp_master:
                original_multimodal_params = copy.deepcopy(multimodal_params)

            if self.pd_mode.is_P_or_NORMAL():
                await multimodal_params.verify_and_preload(request)

            # 记录请求到达的相关信息
            await self._log_req_header(request_headers, group_request_id)
            # 监控

            prompt_ids = await self._encode(prompt, multimodal_params, sampling_params)
            prompt_tokens = len(prompt_ids)
            # 监控
            if group_request_id > 0:
                self.metric_client.counter_inc("lightllm_request_count")
                self.metric_client.histogram_observe("lightllm_request_input_length", prompt_tokens)
                self.metric_client.histogram_observe("lightllm_request_max_new_tokens", sampling_params.max_new_tokens)
            prompt_ids = await self._check_and_repair_length(prompt_ids, sampling_params)

            # 申请资源并存储
            alloced_req_indexes = []
            while len(alloced_req_indexes) < sampling_params.n:
                alloc_req_index = await self.shm_req_manager.async_alloc_req_index()
                sleep_time = 0.1
                while alloc_req_index is None:
                    await asyncio.sleep(sleep_time)
                    sleep_time *= 1.1
                    sleep_time = min(1, sleep_time)

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

            await self.transfer_to_next_module_or_node(
                prompt, sampling_params, original_multimodal_params, req_status.group_req_objs
            )

            results_generator = self._wait_to_token_package(
                start_time,
                prompt_ids,
                group_request_id,
                sampling_params,
                req_status,
                request,
            )

            # 计算输入 token 使用量统计
            image_tokens, audio_tokens = self._count_multimodal_tokens(multimodal_params)
            text_tokens = len(prompt_ids) - (image_tokens + audio_tokens)
            input_usage = {
                "input_text_tokens": text_tokens,
                "input_audio_tokens": audio_tokens,
                "input_image_tokens": image_tokens,
            }

            is_first_gen_token = True
            async for sub_req_id, request_output, metadata, finish_status in results_generator:
                # 只有第一个生成的 token 的 metadata 中包含 input_usage
                if is_first_gen_token:
                    metadata["input_usage"] = input_usage
                    is_first_gen_token = False

                yield sub_req_id, request_output, metadata, finish_status

        except Exception as e:
            logger.error(f"group_request_id: {group_request_id} has exception {str(e)}")
            # error need to release multimodel resources.
            # 对于还没有形成正式请求对象管理的多模态资源，需要单独自己释放
            # 已经放入到 req_id_to_out_inf 中的请求对象，由统一的回收循环
            # 进行回收。
            if group_request_id not in self.req_id_to_out_inf:
                await self._release_multimodal_resources(multimodal_params)
            await self.abort(group_request_id)
            raise e
        return

    def _count_multimodal_tokens(self, multimodal_params: MultimodalParams) -> Tuple[int, int]:
        image_tokens = 0
        audio_tokens = 0

        if self.enable_multimodal and self.pd_mode.is_P_or_NORMAL() and multimodal_params is not None:
            for img in multimodal_params.images:
                if img.token_num is not None:
                    image_tokens += img.token_num
            for audio in multimodal_params.audios:
                if audio.token_num is not None:
                    audio_tokens += audio.token_num

        return image_tokens, audio_tokens

    async def _log_req_header(self, request_headers, group_request_id: int):

        x_request_id = request_headers.get("X-Request-Id", "")
        x_session_id = request_headers.get("X-Session-Id", "")

        format_in_time = datetime.datetime.fromtimestamp(time.time()).strftime("%Y-%m-%d %H:%M:%S")
        logger.info(
            f"recieved req X-Request-Id:{x_request_id} "
            f"X-Session-Id:{x_session_id} start_time:{format_in_time} "
            f"lightllm_req_id:{group_request_id} "
        )
        return

    async def _encode(
        self, prompt: Union[str, List[int]], multimodal_params: MultimodalParams, sampling_params: SamplingParams
    ):
        if isinstance(prompt, str):
            if self.enable_multimodal:
                assert (
                    len(multimodal_params.images + multimodal_params.audios) <= self.args.cache_capacity
                ), "too many multimodal items!"
                if multimodal_params.audios:
                    assert self.args.enable_multimodal_audio, "audio multimodal not enabled"
                await self._alloc_multimodal_resources(multimodal_params, sampling_params)
                prompt_ids = self.tokenizer.encode(
                    prompt, multimodal_params, add_special_tokens=sampling_params.add_special_tokens
                )
            else:
                prompt_ids = self.tokenizer.encode(prompt, add_special_tokens=sampling_params.add_special_tokens)
            return prompt_ids

        # 这里的校验对多模态不是很充分, to do
        if all(isinstance(e, int) for e in prompt):
            if not self.enable_multimodal and not self.pd_mode.is_D():
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
        if not prompt_ids:
            raise ValueError("prompt_ids is empty")
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

    async def transfer_to_next_module_or_node(
        self,
        prompt: str,
        sampling_params: SamplingParams,
        original_multimodal_params: MultimodalParams,
        group_req_objs: Optional[GroupReqObjs] = None,
    ):
        # 多节点纯tp 运行模式下，master 节点需要将请求转发给slave节点.
        if self.is_multinode_tp_master:
            for sender in self.multinode_req_manager:
                sender.send_pyobj(
                    (prompt, sampling_params, original_multimodal_params),
                    protocol=pickle.HIGHEST_PROTOCOL,
                )

        await self.transfer_to_next_module(group_req_objs)
        return

    async def transfer_to_next_module(
        self,
        group_req_objs: Optional[GroupReqObjs] = None,
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

            if not self.disable_abort and request is not None and await request.is_disconnected():
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

                    # update inference timemark
                    self.latest_success_infer_time_mark.set_value(int(time.time()))

                    yield sub_req_id, out_str, metadata, finish_status
                    # 如果有子请求完成，就更新计数
                    if finish_status.is_finished():
                        unfinished_count -= 1

                    if unfinished_count == 0:
                        total_cost_time_ms = (time.time() - start_time) * 1000
                        mean_per_token_cost_time_ms = (total_cost_time_ms - first_token_cost_ms) / out_token_counter
                        self.per_token_costs.add(mean_per_token_cost_time_ms)
                        x_request_id = request.headers.get("X-Request-Id", "") if request is not None else ""
                        x_session_id = request.headers.get("X-Session-Id", "") if request is not None else ""
                        prompt_cache_ratio = prompt_cache_len / prompt_tokens

                        mtp_avg_token_per_step = out_token_counter / max(
                            (out_token_counter - metadata["mtp_accepted_token_num"]), 1
                        )
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
                            f"mtp_avg_token_per_step:{mtp_avg_token_per_step} "
                        )
                        if group_request_id < 0:
                            # health 探测请求，不记录日志和监控
                            return
                        self.metric_client.histogram_observe("lightllm_cache_length", prompt_cache_len)
                        self.metric_client.histogram_observe("lightllm_cache_ratio", prompt_cache_ratio)
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
        req_status: ReqStatus = self.req_id_to_out_inf.get(group_req_id, None)
        if req_status is None:
            logger.warning(f"aborted group_request_id {group_req_id} not exist")
            return

        group_req_objs: GroupReqObjs = req_status.group_req_objs
        for req in group_req_objs.shm_req_objs:
            req.is_aborted = True
        logger.warning(f"aborted group_request_id {group_req_objs.group_req_id}")
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
            for group_req_id_ in list(self.req_id_to_out_inf.keys()):
                req_status: ReqStatus = self.req_id_to_out_inf.get(group_req_id_, None)
                if req_status is not None and req_status.can_release():
                    release_req_status.append(req_status)

            for req_status in release_req_status:
                self.req_id_to_out_inf.pop(req_status.group_req_objs.group_req_id, None)
                for req in req_status.group_req_objs.shm_req_objs:
                    await self.shm_req_manager.async_put_back_req_obj(req)
                    await self.shm_req_manager.async_release_req_index(req.index_in_shm_mem)
                await self._release_multimodal_resources(req_status.group_req_objs.multimodal_params)

            # 先保留这个关键得日志，用于方便定位重构中的问题。
            if time.time() - pre_time_mark > 120:
                pre_time_mark = time.time()
                for group_req_id_ in list(self.req_id_to_out_inf.keys()):
                    req_status: ReqStatus = self.req_id_to_out_inf.get(group_req_id_, None)
                    if req_status is None:
                        continue

                    logger.info(
                        f"left req id {req_status.group_req_objs.group_req_id}"
                        f"can release {req_status.group_req_objs.shm_req_objs[0].can_released_mark} "
                        f"refcount {req_status.group_req_objs.shm_req_objs[0].ref_count}"
                    )
        return

    async def handle_loop(self):
        self.recycle_event = asyncio.Event()
        asyncio.create_task(self.recycle_resource_loop())

        # 多节点tp模式下的slave节点，需要开启一个协程task用来接收
        # master 转发过来的请求对象。
        if self.is_multinode_tp_slave:
            asyncio.create_task(self.loop_for_request())

        if self.pd_mode.is_P_or_D():
            from lightllm.server.httpserver.pd_loop import pd_handle_loop

            asyncio.create_task(pd_handle_loop(self))

        while True:
            try:
                await asyncio.wait_for(self.recv_from_detokenization.recv_pyobj(), timeout=0.05)
            except asyncio.TimeoutError:
                pass

            try:
                for group_req_id_ in list(self.req_id_to_out_inf.keys()):
                    req_status = self.req_id_to_out_inf.get(group_req_id_, None)
                    if req_status is None:
                        continue

                    token_list = []
                    for req in req_status.group_req_objs.shm_req_objs:
                        req_id = req.request_id
                        read_token_count = 1
                        if req.out_tokens_queue.is_full():
                            read_token_count = LIGHTLLM_OUT_TOKEN_QUEUE_SIZE

                        for _ in range(read_token_count):
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
                                    "mtp_accepted_token_num": req.mtp_accepted_token_num,
                                }
                                if self.args.return_all_prompt_logprobs:
                                    metadata.update(req.get_all_prompt_metadata())
                                if self.args.use_reward_model:
                                    metadata["score"] = float(req.reward_score)

                                req.out_tokens_queue.pop_no_ret()

                                finished_token_index = (
                                    req.stop_str_matched_token_index if req.stop_str_matched else req.finish_token_index
                                )

                                if finished_token_index != src_index:
                                    token_list.append((req_id, text, metadata, FinishStatus()))
                                else:
                                    if req.stop_str_matched:
                                        finish_status = FinishStatus(FinishStatus.FINISHED_STOP)
                                    else:
                                        finish_status = FinishStatus(req.finish_status.status)

                                    token_list.append((req_id, text, metadata, finish_status))
                            else:
                                break

                    async with req_status.lock:
                        req_status.out_token_info_list.extend(token_list)
                        req_status.event.set()
            except BaseException as e:
                logger.exception(str(e))
                raise e

            self.recycle_event.set()
        return


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
