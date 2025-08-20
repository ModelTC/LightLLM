import sys
import zmq
import zmq.asyncio
import asyncio
import uvloop
import rpyc
import time
import json
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
from ..embed_cache.utils import get_shm_name_data, create_shm, afs_embed_exists
from ..multimodal_params import AudioItem, MultimodalParams, ImageItem
from ..req_id_generator import ReqIDGenerator
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


class HttpServerManagerForVisualOnly:
    def __init__(self, args, cache_port, visual_port, metric_port):
        self.args = args
        context = zmq.asyncio.Context(2)

        self._shm_lock_pool = AtomicShmArrayLock(f"{get_unique_server_name()}_lightllm_resource_lock", 1)
        self._resource_lock = AsyncLock(self._shm_lock_pool.get_lock_context(0))
        self.cache_client = rpyc.connect("localhost", cache_port, config={"allow_pickle": True})
        self.send_to_visual = context.socket(zmq.PUSH)
        self.send_to_visual.connect(f"{args.zmq_mode}127.0.0.1:{visual_port}")
        self.shm_req_manager = ShmReqManager()
        self.tokenizer = get_tokenizer(args.model_dir, args.tokenizer_mode, trust_remote_code=args.trust_remote_code)
        self.req_id_to_out_inf: Dict[int, ReqStatus] = {}  # value type (out_str, metadata, finished, event)
        self.max_req_total_len = args.max_req_total_len
        self.id_gen = ReqIDGenerator()
        self.metric_client = MetricClient(metric_port)
        # 有的模型的vocab size 读取tokenizer和config.json中不一致
        self.vocab_size = max(get_vocab_size(args.model_dir), self.tokenizer.vocab_size)
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
                md5sum = "{}_{}".format(
                    hashlib.md5(data).hexdigest(),
                    hashlib.md5(pickle.dumps(img.extra_params, protocol=4)).hexdigest(),
                )
                md5sums.append(md5sum)
                tokens_nums.append(token_num)
                datas.append(data)
                items.append(img)
            for audio in multimodal_params.audios:
                self.tokenizer.init_audioitem_extral_params(audio, multimodal_params, sampling_params)
                data = audio.read()
                token_num = self.tokenizer.get_audio_token_length(audio)
                md5sum = "{}_{}".format(
                    hashlib.md5(data).hexdigest(),
                    hashlib.md5(pickle.dumps(audio.extra_params, protocol=4)).hexdigest(),
                )
                md5sums.append(md5sum)
                tokens_nums.append(token_num)
                datas.append(data)
                items.append(audio)

            await self._alloc_resource(items, md5sums, tokens_nums, datas)
        return

    async def _release_multimodal_resources(self, multimodal_params: MultimodalParams):
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

    def tokens(self, multimodal_params, samping_params: SamplingParams, kwargs=None):
        kwargs = {} if kwargs is None else kwargs
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
        return image_tokens + img_count + audio_tokens + audio_count

    async def loop_for_request(self):
        assert self.args.node_rank > 0
        while True:
            (
                sampling_params,
                multimodal_params,
            ) = await self.multinode_req_manager.recv_pyobj()
            results_generator = self.generate(sampling_params, multimodal_params, None)

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
        group_request_id = self.id_gen.generate_id()

        sampling_params.group_request_id = group_request_id
        return group_request_id

    async def generate(
        self,
        sampling_params: SamplingParams,
        multimodal_params: MultimodalParams,
        request: Request,
        is_health_req: bool = False,
    ) -> Tuple[int, str, dict, FinishStatus]:
        start_time = time.time()
        request_headers = request.headers if request is not None else {}
        group_request_id = self.alloc_req_id(sampling_params, is_health_req)

        try:
            await multimodal_params.verify_and_preload(request)

            # 记录请求到达的相关信息
            await self._log_req_header(request_headers, group_request_id)
            assert (
                len(multimodal_params.images + multimodal_params.audios) <= self.args.cache_capacity
            ), "too many multimodal items!"
            await self._alloc_multimodal_resources(multimodal_params, sampling_params)

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
                    #  随便写的，后面改掉
                    [21456],
                    sampling_params,
                    self.tokenizer,
                    chunked_prefill_size=self.args.chunked_prefill_size,
                )
                req_objs.append(req_obj)

            req_status = ReqStatus(group_request_id, multimodal_params, req_objs, start_time)
            self.req_id_to_out_inf[group_request_id] = req_status

            await self.transfer_to_visual(req_status.group_req_objs)

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

    async def transfer_to_visual(
        self,
        group_req_objs: Optional[GroupReqObjs] = None,
    ):
        await self.send_to_visual.send_pyobj(
            group_req_objs.to_group_req_index(),
            protocol=pickle.HIGHEST_PROTOCOL,
        )
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
        self.finished = False

    def can_release(self):
        for req in self.group_req_objs.shm_req_objs:
            if not req.can_release():
                return False
        return True
