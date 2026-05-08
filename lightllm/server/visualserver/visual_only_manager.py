import asyncio
import uvloop
import rpyc
import inspect
import setproctitle
import threading
import uuid
import pickle
import websockets
import socket
import sys
import os
import signal
import time
from lightllm.utils.net_utils import get_hostname_ip
from .objs import VIT_Obj
from typing import List
from lightllm.server.core.objs import StartArgs

asyncio.set_event_loop_policy(uvloop.EventLoopPolicy())
from lightllm.server.multimodal_params import MultimodalParams, ImageItem
from .model_infer import start_model_process, VisualModelRpcClient
from lightllm.common.basemodel.attention_vit.create_utils import init_vit_att_backend
from lightllm.utils.log_utils import init_logger
from lightllm.utils.graceful_utils import graceful_registry
from lightllm.utils.process_check import start_parent_check_thread
from lightllm.utils.envs_utils import get_unique_server_name
from rpyc.utils.classic import obtain
from lightllm.server.embed_cache.utils import create_shm, get_shm_name_data, free_shm
from .manager import VisualManager


logger = init_logger(__name__)


class VisualOnlyManager(rpyc.Service):
    def __init__(
        self,
        args: StartArgs,
    ):
        self.args = args
        self.model_weightdir = args.model_dir
        self.vit_dp = args.visual_dp
        assert self.vit_dp == 1
        self.vit_tp = args.visual_tp
        # image 最大推理 batch size
        self.infer_batch_size = args.visual_infer_batch_size
        self.lock = asyncio.Lock()

        self.new_loop = asyncio.new_event_loop()

        def _event_loop():
            asyncio.set_event_loop(self.new_loop)
            self.new_loop.run_forever()

        t = threading.Thread(target=_event_loop, daemon=True)
        t.start()

    async def register_to_config_server_loop(self, args: StartArgs):
        if args.host in ["127.0.0.1", "localhost"]:
            logger.error("remote visual server must specify host ip, can not be localhost or 127.0.0.1")
            # kill father process to trigger graceful exit, avoid orphan process
            os.kill(os.getppid(), signal.SIGTERM)
            sys.exit(-1)

        if args.host in ["0.0.0.0"]:
            host_ip = get_hostname_ip()
        else:
            host_ip = args.host

        while True:
            try:
                uri = f"ws://{args.config_server_host}:{args.config_server_port}/visual_register"
                async with websockets.connect(uri, max_queue=(2048 * 1024, 2048 * 1023)) as websocket:

                    sock = websocket.transport.get_extra_info("socket")
                    sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)

                    vit_obj = VIT_Obj(node_id=args.visual_node_id, host_ip=host_ip, port=args.visual_rpyc_port)

                    await websocket.send(pickle.dumps(vit_obj))
                    logger.info(f"Sent registration vit_obj: {vit_obj}")

                    while True:
                        await websocket.send("heartbeat")
                        await asyncio.sleep(40)

            except Exception as e:
                logger.error("connetion to config_server has error")
                logger.exception(str(e))
                await asyncio.sleep(10)
                logger.info("reconnection to config_server")

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
                    "cache_port": None,  # visual only 模式下不使用 embed cache
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

    async def handle_images(self, images_need_infer: List[ImageItem]):
        await VisualManager.handle_images(self, images_need_infer=images_need_infer)

    async def infer_images(self, dp_index: int, images, events):
        await VisualManager.infer_images(self, dp_index=dp_index, images=images, events=events)

    def clean_up(self):
        return

    def exposed_remote_infer_images(self, images: List[ImageItem], ref_event: threading.Event):
        try:
            images = obtain(images)
            logger.info(f"Received infer_images request with {len(images)} images md5s: {[img.md5 for img in images]}")
            start = time.time()
            # 将 images 的内容写入到 shm 中，这里修改了原始的uuid，主要是在远端的vit
            # 本身不具有 embed cache 的引用保证，则新的唯一标识来进行推理，最终写入的
            # 目标的 md5 一致即可，这样调用端一样可以拿到准确的数据。
            for image in images:
                image.uuid = str(uuid.uuid4())
                create_shm(get_shm_name_data(image.uuid), image.data_bytes)
                del image.data_bytes

            handle = asyncio.run_coroutine_threadsafe(self.handle_images(images_need_infer=images), loop=self.new_loop)

            def _callback(fut):
                if time.time() - start > 0.05:
                    logger.info(
                        f"Finished infer_images for images {[image.md5 for image in images]}"
                        f" with latency {time.time() - start} seconds"
                    )
                ref_event.set()
                # 清理资源
                for image in images:
                    free_shm(get_shm_name_data(image.uuid))
                logger.info(
                    f"Finished infer_images request for images {[image.md5 for image in images]}"
                    f" and cleaned up shm resources"
                )

            handle.add_done_callback(_callback)
        except BaseException as e:
            logger.exception(str(e))
            # 清理资源
            for image in images:
                free_shm(get_shm_name_data(image.uuid))
            raise e
        return


def start_visual_process(args: StartArgs, pipe_writer):
    import lightllm.utils.rpyc_fix_utils as _

    # 注册graceful 退出的处理
    graceful_registry(inspect.currentframe().f_code.co_name)
    setproctitle.setproctitle(f"lightllm::{get_unique_server_name()}::visual_server")
    start_parent_check_thread()

    try:
        visualserver = VisualOnlyManager(args=args)

        def handle_exception(loop, context):
            logger.exception(f"VisualServer Caught exception: {str(context)}")

        visualserver.new_loop.set_exception_handler(handle_exception)

        future = asyncio.run_coroutine_threadsafe(visualserver.wait_to_model_ready(), loop=visualserver.new_loop)
        future.result()

        asyncio.run_coroutine_threadsafe(
            visualserver.register_to_config_server_loop(args=args), loop=visualserver.new_loop
        )

        from .objs import rpyc_config

        t = rpyc.ThreadedServer(visualserver, port=args.visual_rpyc_port, protocol_config=rpyc_config)
    except Exception as e:
        logger.exception(str(e))
        visualserver.clean_up()
        raise e

    pipe_writer.send("init ok")

    t.start()
    return
