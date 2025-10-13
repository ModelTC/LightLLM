import zmq
import time
import zmq.asyncio
import asyncio
import uvloop
import rpyc
import socket
import pickle
import hashlib
import datetime
import inspect
from fastapi import Request
from ..tokenizer import get_tokenizer
import setproctitle
from typing import List
from lightllm.server.core.objs.io_objs.group_req import GroupReqIndexes
from lightllm.server.embed_cache.utils import get_shm_name_data, create_shm
from lightllm.server.core.objs import ShmReqManager
from lightllm.server.core.objs import SamplingParams
from lightllm.server.core.objs import Req, FinishStatus
from typing import Union, Tuple, Dict, Optional
from ..req_id_generator import ReqIDGenerator
from lightllm.server.core.objs.io_objs import GroupReqObjs
from lightllm.server.embed_cache.impl.memory_cache_with_redis import MemoryCacheWithRedis

asyncio.set_event_loop_policy(uvloop.EventLoopPolicy())
from lightllm.server.multimodal_params import MultimodalParams, ImageItem
from .model_infer.model_rpc import start_model_process, VisualModelRpcClient
from lightllm.utils.log_utils import init_logger
from lightllm.utils.graceful_utils import graceful_registry
from lightllm.utils.process_check import start_parent_check_thread
from lightllm.utils.envs_utils import get_unique_server_name
from rpyc.utils.classic import obtain


logger = init_logger(__name__)


class VisualManager:
    def __init__(
        self,
        args,
        next_module_port,
        visual_port,
        cache_port,
        visual_model_rpc_ports,
    ):
        self.args = args
        self.remote_vit = args.enable_remote_vit or args.run_mode == "visual"
        self.cache_port = cache_port
        self.visual_port = visual_port
        self.next_module_port = next_module_port
        self.waiting_reqs: List[GroupReqIndexes] = []
        self.infer_batch_size = args.visual_infer_batch_size
        self.trust_remote_code = args.trust_remote_code
        self.visual_model_rpc_ports = visual_model_rpc_ports
        self.shm_req_manager = ShmReqManager()
        self._setup_connections()

    def _setup_connections(self):
        context = zmq.Context(2)
        if self.remote_vit:
            self.vit_receiver = context.socket(zmq.PULL)
            self.vit_receiver.bind(f"tcp://*:{self.args.remote_vit_port}")
        else:
            self.vit_receiver = context.socket(zmq.PULL)
            self.vit_receiver.bind(f"{self.args.zmq_mode}127.0.0.1:{self.visual_port}")
            self.send_to_next_module = context.socket(zmq.PUSH)  # router or audio server (if --enable_multimodal_audio)
            self.send_to_next_module.connect(f"{self.args.zmq_mode}127.0.0.1:{self.next_module_port}")
        self.cache_client = rpyc.connect("localhost", self.cache_port, config={"allow_pickle": True})
        self.cache_client._channel.stream.sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)

    async def wait_to_model_ready(self):
        visual_dp = self.args.visual_dp
        visual_tp = self.args.visual_tp
        self.model_rpcs: List[List[VisualModelRpcClient]] = [[] for _ in range(visual_dp)]

        for dp_rank_id in range(visual_dp):
            tp_ports_each_dp = self.visual_model_rpc_ports[dp_rank_id]
            for tp_rank_id in range(visual_tp):
                device_id = self.args.visual_gpu_ids[dp_rank_id * visual_tp + tp_rank_id]
                rpc_model = await start_model_process(
                    port=tp_ports_each_dp[tp_rank_id], vit_tp=visual_tp, device_id=device_id
                )
                self.model_rpcs[dp_rank_id].append(rpc_model)

        init_model_ret = []
        for dp_rank_id in range(visual_dp):  # async init model process
            for tp_rank_id in range(visual_tp):
                kvargs = {
                    "tp_rank_id": tp_rank_id,
                    "dp_rank_id": dp_rank_id,
                }
                init_model_ret.append(self.model_rpcs[dp_rank_id][tp_rank_id].init_model(kvargs))
        await asyncio.gather(*init_model_ret)
        return

    async def infer_imgs(self, images: List[ImageItem]):
        if len(images) == 0:
            return

        tasks = []
        for vit_dp_rank in range(self.args.visual_dp):
            assigned_images = [images[i] for i in range(vit_dp_rank, len(images), self.args.visual_dp)]
            if assigned_images:
                for vit_tp_rank in range(self.args.visual_tp):
                    task = asyncio.create_task(self.model_rpcs[vit_dp_rank][vit_tp_rank].encode(assigned_images))
                    tasks.append(task)
        await asyncio.gather(*tasks)
        return

    async def loop_for_fwd(self):
        while True:
            if len(self.waiting_reqs) == 0:
                await asyncio.sleep(0.01)  # 10ms
            else:
                processing_group_reqs = []
                images_need_infer = []
                while len(self.waiting_reqs) > 0:
                    group_req_indexes = self.waiting_reqs.pop(0)
                    shm_req = self.shm_req_manager.get_req_obj_by_index(group_req_indexes.shm_req_indexes[0])
                    is_aborted = shm_req.is_aborted
                    disable_prompt_cache = shm_req.sample_params.disable_prompt_cache
                    self.shm_req_manager.put_back_req_obj(shm_req)
                    if is_aborted:
                        # 因为连接断开 aborted 掉的请求也需要传输到后续的模块进行处理
                        # 因为采用 shm 来映射所有的 req 对象以后，引用管理情况复杂了
                        # 需要一些一致的流程来保证不出现异步问题。
                        self.send_to_next_module.send_pyobj(group_req_indexes, protocol=pickle.HIGHEST_PROTOCOL)
                        continue

                    multimodal_params = group_req_indexes.multimodal_params

                    img_uuids = [img.uuid for img in multimodal_params.images]
                    # disable prompt cache通常用来测试，需要也去掉image cache的影响
                    if disable_prompt_cache:
                        ready_image = [False] * len(img_uuids)
                    else:
                        ready_image = obtain(self.cache_client.root.get_items_embed(img_uuids))

                    for img, ready in zip(multimodal_params.images, ready_image):
                        if not ready:
                            images_need_infer.append(img)

                        if len(images_need_infer) == self.infer_batch_size:
                            await self.infer_imgs(images_need_infer)
                            images_need_infer = []
                            for _group_req_indexes in processing_group_reqs:
                                self.send_to_next_module.send_pyobj(
                                    _group_req_indexes, protocol=pickle.HIGHEST_PROTOCOL
                                )
                            processing_group_reqs = []

                    if len(images_need_infer) == 0:
                        self.send_to_next_module.send_pyobj(group_req_indexes, protocol=pickle.HIGHEST_PROTOCOL)
                    else:
                        processing_group_reqs.append(group_req_indexes)

                if len(images_need_infer) > 0:
                    await self.infer_imgs(images_need_infer)
                    for _group_req_indexes in processing_group_reqs:
                        self.send_to_next_module.send_pyobj(_group_req_indexes, protocol=pickle.HIGHEST_PROTOCOL)
                    processing_group_reqs = []
                    images_need_infer = []

    async def _recv_reqs(self):
        if self.remote_vit:
            recv_req: GroupReqIndexes = self.vit_receiver.recv_pyobj(zmq.NOBLOCK)
            # recv_req.multimodal_params.images[:]= [
            #     img for img in recv_req.multimodal_params.images
            #     if not self.cache_client.root.get_item_embed(img.uuid)  # embed已存在的被丢弃 , ref +1
            # ]
            logger.info(f"Receive req {recv_req.group_req_id}, image_count:{len(recv_req.multimodal_params.images)}")
            uuids = [img.uuid for img in recv_req.multimodal_params.images]
            already_embed = await asyncio.to_thread(self.cache_client.root.get_items_embed, uuids)
            if all(already_embed):
                return None

            uuids = []
            token_nums = []
            datas = []
            for img, embed in zip(recv_req.multimodal_params.images, already_embed):
                if not embed:
                    uuids.append(img.uuid)
                    token_nums.append(img.token_num)
                    datas.append(img._preload_data)
                    img.free()
            while True:
                records = await asyncio.to_thread(self.cache_client.root.alloc, uuids, token_nums)
                if records is not None:
                    break
                await asyncio.sleep(0.01)
            ready_flags = obtain(self.cache_client.root.get_items_data(uuids))
            update_data_ids = []

            for uid, ready, data in zip(uuids, ready_flags, datas):
                if not ready:
                    create_shm(get_shm_name_data(uid), data)
                    update_data_ids.append(uid)

            if update_data_ids:
                await asyncio.to_thread(self.cache_client.root.set_items_data, update_data_ids)
            return recv_req
        else:
            return self.vit_receiver.recv_pyobj(zmq.NOBLOCK)

    async def loop_for_netio_req(self):
        if not hasattr(self, "visual_recv_max_count"):
            self.visual_recv_max_count = 64

        while True:
            try:
                for _ in range(self.visual_recv_max_count):
                    recv_req: GroupReqIndexes = await self._recv_reqs()
                    if recv_req is None:
                        continue
                    if isinstance(recv_req, GroupReqIndexes):
                        self.waiting_reqs.append(recv_req)
                    else:
                        assert False, f"Error Req Inf {recv_req}"
                    await asyncio.sleep(0)
                self.visual_recv_max_count = min(int(self.visual_recv_max_count * 1.3), 256)
            except zmq.ZMQError:
                # 当队列已经开始清空的时候，将一次接受数量下调
                self.visual_recv_max_count = 64
            except Exception as e:
                logger.exception(f"Error in loop_for_netio_req: {e}")
                raise e
            await asyncio.sleep(0.01)

    # code for visual only mode
    async def loop_for_fwd_visual_only(self):
        while True:
            if len(self.waiting_reqs) == 0:
                await asyncio.sleep(0.01)  # 10ms
            else:
                images_need_infer = []

                while len(self.waiting_reqs) > 0:
                    visual_req = self.waiting_reqs.pop(0)

                    for img in visual_req.multimodal_params.images:
                        images_need_infer.append(img)

                        if len(images_need_infer) == self.infer_batch_size:
                            await self.infer_imgs(images_need_infer)
                            images_need_infer = []

                    if len(images_need_infer) > 0:
                        await self.infer_imgs(images_need_infer)
                        images_need_infer = []
                    # 在这里release这个image，ref-1
                    logger.info(f"req-id {visual_req.group_req_id} has been release ok")

    def clean_up(self):
        for model_rpc in self.model_rpcs:
            model_rpc.rpc_server_process.kill()
        for model_rpc in self.model_rpcs:
            model_rpc.rpc_server_process.join()
        return


def create_forward_loop(args, visualserver: VisualManager, loop: asyncio.AbstractEventLoop):
    if args.run_mode == "visual":
        from .register_loop import register_loop

        loop.create_task(visualserver.loop_for_fwd_visual_only())
        loop.create_task(register_loop(args))
    else:
        loop.create_task(visualserver.loop_for_fwd())
    return


def start_visual_process(args, next_module_port, visual_port, cache_port, model_rpc_ports, pipe_writer):
    # 注册graceful 退出的处理
    graceful_registry(inspect.currentframe().f_code.co_name)
    setproctitle.setproctitle(f"lightllm::{get_unique_server_name()}::visual_server")
    start_parent_check_thread()
    try:
        visualserver = VisualManager(args, next_module_port, visual_port, cache_port, model_rpc_ports)
        asyncio.run(visualserver.wait_to_model_ready())
    except Exception as e:
        logger.exception(str(e))
        visualserver.clean_up()
        raise e

    pipe_writer.send("init ok")

    def handle_exception(loop, context):
        logger.exception(f"VisualServer Caught exception: {str(context)}")

    loop = asyncio.new_event_loop()
    loop.set_exception_handler(handle_exception)
    asyncio.set_event_loop(loop)
    create_forward_loop(args, visualserver, loop)
    loop.run_until_complete(visualserver.loop_for_netio_req())
    return
