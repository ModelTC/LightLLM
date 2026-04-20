import zmq
import zmq.asyncio
import asyncio
import uvloop
import rpyc
import socket
import pickle
import inspect
import setproctitle
import threading
import collections
from typing import List
from lightllm.server.core.objs.io_objs.group_req import GroupReqIndexes
from lightllm.server.core.objs import ShmReqManager, StartArgs

asyncio.set_event_loop_policy(uvloop.EventLoopPolicy())
from lightllm.server.multimodal_params import MultimodalParams, ImageItem
from .model_infer import start_model_process, VisualModelRpcClient
from lightllm.common.basemodel.attention_vit.create_utils import init_vit_att_backend
from lightllm.utils.log_utils import init_logger
from lightllm.utils.graceful_utils import graceful_registry
from lightllm.utils.process_check import start_parent_check_thread
from lightllm.utils.envs_utils import get_unique_server_name
from rpyc.utils.classic import obtain


logger = init_logger(__name__)


class VisualManager:
    def __init__(
        self,
        args: StartArgs,
    ):
        self.args = args
        context = zmq.Context(2)
        enable_audio = not args.disable_audio
        if enable_audio:
            self.send_to_next_module = context.socket(zmq.PUSH)
            self.send_to_next_module.connect(f"{args.zmq_mode}127.0.0.1:{args.audio_port}")
        else:
            if args.enable_cpu_cache:
                self.send_to_next_module = context.socket(zmq.PUSH)
                self.send_to_next_module.connect(f"{args.zmq_mode}127.0.0.1:{args.multi_level_kv_cache_port}")
            else:
                self.send_to_next_module = context.socket(zmq.PUSH)
                self.send_to_next_module.connect(f"{args.zmq_mode}127.0.0.1:{args.router_port}")

        self.zmq_recv_socket = context.socket(zmq.PULL)
        self.zmq_recv_socket.bind(f"{args.zmq_mode}127.0.0.1:{args.visual_port}")
        self.cache_client = rpyc.connect("localhost", args.cache_port, config={"allow_pickle": True})
        self.cache_client._channel.stream.sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        self.model_weightdir = args.model_dir
        self.vit_dp = args.visual_dp
        self.vit_tp = args.visual_tp
        # image 最大推理 batch size
        self.infer_batch_size = args.visual_infer_batch_size
        self.send_batch_size = args.visual_send_batch_size
        self.shm_req_manager = ShmReqManager()
        self.lock = asyncio.Lock()

    async def wait_to_model_ready(self):

        self.model_rpcs: List[List[VisualModelRpcClient]] = [[] for _ in range(self.vit_dp)]
        self.vit_attn_backend = init_vit_att_backend(index=0)
        for dp_rank_id in range(self.vit_dp):
            for tp_rank_id in range(self.vit_tp):

                rpc_model = await start_model_process()
                self.model_rpcs[dp_rank_id].append(rpc_model)

        init_model_ret = []
        for dp_rank_id in range(self.vit_dp):  # async init model process
            for tp_rank_id in range(self.vit_tp):
                device_id = self.args.visual_gpu_ids[dp_rank_id * self.vit_tp + tp_rank_id]
                kvargs = {
                    "weight_dir": self.model_weightdir,
                    "device_id": device_id,
                    "vit_tp": self.vit_tp,
                    "cache_port": self.args.cache_port,
                    "tp_rank_id": tp_rank_id,
                    "dp_rank_id": dp_rank_id,
                    "data_type": self.args.data_type,
                    "visual_nccl_port": self.args.visual_nccl_ports[dp_rank_id],
                    "quant_type": self.args.vit_quant_type,
                    "quant_cfg": self.args.vit_quant_cfg,
                    "max_batch_size": max(self.infer_batch_size // self.vit_dp, 1),
                    "vit_attn_backend": self.vit_attn_backend,
                }
                init_model_ret.append(self.model_rpcs[dp_rank_id][tp_rank_id].init_model(kvargs))
        await asyncio.gather(*init_model_ret)
        return

    def get_need_infer_images(self, group_req_indexes: GroupReqIndexes) -> List[ImageItem]:
        shm_req = self.shm_req_manager.get_req_obj_by_index(group_req_indexes.shm_req_indexes[0])
        is_aborted = shm_req.is_aborted
        disable_prompt_cache = shm_req.sample_params.disable_prompt_cache
        self.shm_req_manager.put_back_req_obj(shm_req)
        # case 0
        if is_aborted:
            # 因为连接断开 aborted 掉的请求也需要传输到后续的模块进行处理
            # 因为采用 shm 来映射所有的 req 对象以后，引用管理情况复杂了
            # 需要一些一致的流程来保证不出现异步问题。
            return []

        multimodal_params = group_req_indexes.multimodal_params
        img_uuids = [img.uuid for img in multimodal_params.images]
        # disable prompt cache通常用来测试，需要也去掉image cache的影响
        if disable_prompt_cache:
            ready_image = [False] * len(img_uuids)
        else:
            if len(img_uuids) > 0:
                ready_image = obtain(self.cache_client.root.get_items_embed(img_uuids))
            else:
                ready_image = []

        images_need_infer = []
        for img, ready in zip(multimodal_params.images, ready_image):
            if not ready:
                images_need_infer.append(img)

        return images_need_infer

    async def handle_group_indexes(self, group_req_indexes: GroupReqIndexes):
        images_need_infer = self.get_need_infer_images(group_req_indexes)

        if len(images_need_infer) == 0:
            self.send_to_next_module.send_pyobj(group_req_indexes, protocol=pickle.HIGHEST_PROTOCOL)
            return
        else:
            await self.handle_images(images_need_infer)
            self.send_to_next_module.send_pyobj(group_req_indexes, protocol=pickle.HIGHEST_PROTOCOL)
            return

    async def handle_images(self, images_need_infer: List[ImageItem]):
        if not hasattr(self, "cur_dp_index"):
            self.cur_dp_index = 0

        dp_to_handle_images = collections.defaultdict(list)
        for image in images_need_infer:
            self.cur_dp_index += 1
            select_dp = self.cur_dp_index % self.vit_dp
            dp_to_handle_images[select_dp].append((image, threading.Event()))

        taskes = []
        for dp_index in range(self.vit_dp):
            _images = dp_to_handle_images[dp_index]
            if _images:
                taskes.append(
                    self.infer_images(dp_index, images=[e[0] for e in _images], events=[e[1] for e in _images])
                )

        async with self.lock:
            try:
                await asyncio.gather(*taskes)
            except BaseException as e:
                logger.exception(str(e))
                raise e

        # 等待推理通知已经 ok
        for dp_index in range(self.vit_dp):
            _images = dp_to_handle_images[dp_index]
            if _images:
                await asyncio.to_thread(_images[-1][1].wait)
        return

    async def infer_images(self, dp_index: int, images, events):
        taskes = []
        for vit_tp_rank in range(self.vit_tp):
            task = self.model_rpcs[dp_index][vit_tp_rank].run_task(images, events)
            taskes.append(task)
        await asyncio.gather(*taskes)

    async def loop_for_netio_req(self):
        try:
            while True:
                recv_req: GroupReqIndexes = await asyncio.to_thread(self.zmq_recv_socket.recv_pyobj)
                if isinstance(recv_req, GroupReqIndexes):
                    logger.info(
                        f"visual recv req id {recv_req.group_req_id} "
                        f"img count {len(recv_req.multimodal_params.images)}"
                    )
                    asyncio.create_task(self.handle_group_indexes(group_req_indexes=recv_req))
                else:
                    assert False, f"Error Req Inf {recv_req}"
        except Exception as e:
            logger.exception(str(e))

    def clean_up(self):
        return


def start_visual_process(args, pipe_writer):
    import lightllm.utils.rpyc_fix_utils as _

    # 注册graceful 退出的处理
    graceful_registry(inspect.currentframe().f_code.co_name)
    setproctitle.setproctitle(f"lightllm::{get_unique_server_name()}::visual_server")
    start_parent_check_thread()
    try:
        visualserver = VisualManager(args=args)
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
    loop.run_until_complete(visualserver.loop_for_netio_req())
    return
