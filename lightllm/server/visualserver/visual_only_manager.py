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
import time
from .objs import VIT_Obj
from typing import Any, List, Tuple
from lightllm.server.core.objs import StartArgs

asyncio.set_event_loop_policy(uvloop.EventLoopPolicy())
from lightllm.server.multimodal_params import MultimodalParams, ImageItem
from .model_infer import start_model_process, VisualModelRpcClient
from lightllm.common.basemodel.attention_vit.create_utils import init_vit_att_backend
from lightllm.utils.log_utils import init_logger
from lightllm.utils.graceful_utils import graceful_registry
from lightllm.utils.process_check import start_parent_check_thread
from lightllm.utils.envs_utils import get_unique_server_name
from lightllm.utils.shm_port_args import get_shm_port_args
from lightllm.server.embed_cache.utils import create_shm, get_shm_name_data, free_shm
from .manager import VisualManager
from .objs import VISUAL_ONLY_RPC_HOST, visual_only_rpyc_config
from .rpc_payload import notify_remote_event_async, unpack_image_items
from .rpyc_server import VisualOnlyThreadedServer


logger = init_logger(__name__)


class VisualOnlyManager(rpyc.Service):
    def __init__(
        self,
        args: StartArgs,
    ):
        """Initialize the visual inference service and its worker event loop."""
        self.args = args
        # Proxies for this service must run on the same host because its RPC endpoint is loopback-only.
        self.rpc_host = VISUAL_ONLY_RPC_HOST
        self.model_weightdir = args.model_dir
        self.vit_dp = args.visual_dp
        assert self.vit_dp == 1
        self.vit_tp = args.visual_tp
        # image 最大推理 batch size
        self.infer_batch_size = args.visual_infer_batch_size
        self.lock = asyncio.Lock()

        self.new_loop = asyncio.new_event_loop()

        def _event_loop() -> None:
            asyncio.set_event_loop(self.new_loop)
            self.new_loop.run_forever()

        t = threading.Thread(target=_event_loop, daemon=True)
        t.start()

    async def register_to_config_server_loop(self, args: StartArgs):
        """Register this loopback-only endpoint with the local config service."""
        host_ip = self.rpc_host

        ports = get_shm_port_args()
        while True:
            try:
                uri = f"ws://{args.config_server_host}:{ports.config_server_port}/visual_register"
                async with websockets.connect(uri, max_queue=(2048 * 1024, 2048 * 1023)) as websocket:

                    sock = websocket.transport.get_extra_info("socket")
                    sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)

                    vit_obj = VIT_Obj(node_id=args.visual_node_id, host_ip=host_ip, port=ports.visual_rpyc_port)

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
        visual_nccl_ports = get_shm_port_args().visual_nccl_ports
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
                    "visual_nccl_port": visual_nccl_ports[dp_rank_id],
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

    def exposed_remote_infer_images(self, payload: Tuple[Any, ...], ref_event: Any) -> None:
        """Validate remote image data, stage it in shared memory, and queue inference."""
        images = unpack_image_items(
            payload,
            ImageItem,
            max_count=self.infer_batch_size,
            require_image_data=True,
        )
        created_shm_names = []

        def _cleanup_shared_memory() -> None:
            """Release shared-memory segments created for this request."""
            for shm_name in created_shm_names:
                try:
                    free_shm(shm_name)
                except FileNotFoundError:
                    pass
                except Exception:
                    logger.exception(f"Failed to clean up visual inference shared memory {shm_name}")

        try:
            logger.info(f"Received infer_images request with {len(images)} images md5s: {[img.md5 for img in images]}")
            start = time.time()
            # 将 images 的内容写入到 shm 中，这里修改了原始的uuid，主要是在远端的vit
            # 本身不具有 embed cache 的引用保证，则新的唯一标识来进行推理，最终写入的
            # 目标的 md5 一致即可，这样调用端一样可以拿到准确的数据。
            for image in images:
                image.uuid = str(uuid.uuid4())
                shm_name = get_shm_name_data(image.uuid)
                create_shm(shm_name, image.data_bytes)
                created_shm_names.append(shm_name)
                del image.data_bytes

            handle = asyncio.run_coroutine_threadsafe(self.handle_images(images_need_infer=images), loop=self.new_loop)

            def _callback(fut: Any) -> None:
                """Release request storage before notifying the client asynchronously."""
                if time.time() - start > 0.05:
                    logger.info(
                        f"Finished infer_images for images {[image.md5 for image in images]}"
                        f" with latency {time.time() - start} seconds"
                    )
                try:
                    _cleanup_shared_memory()
                finally:
                    notify_remote_event_async(ref_event)
                logger.info(
                    f"Finished infer_images request for images {[image.md5 for image in images]}"
                    f" and cleaned up shm resources"
                )

            handle.add_done_callback(_callback)
        except BaseException as e:
            logger.exception(str(e))
            _cleanup_shared_memory()
            raise e
        return


def start_visual_process(args: StartArgs, pipe_writer):
    """Initialize the visual inference process and start its loopback RPC server."""
    import lightllm.utils.rpyc_fix_utils as _

    # 注册graceful 退出的处理
    graceful_registry(inspect.currentframe().f_code.co_name)
    setproctitle.setproctitle(f"lightllm::{get_unique_server_name()}::visual_only_server")
    start_parent_check_thread()

    try:
        visualserver = VisualOnlyManager(args=args)

        def handle_exception(loop: asyncio.AbstractEventLoop, context: Any) -> None:
            """Log unhandled errors raised by the visual inference event loop."""
            logger.exception(f"VisualServer Caught exception: {str(context)}")

        visualserver.new_loop.set_exception_handler(handle_exception)

        future = asyncio.run_coroutine_threadsafe(visualserver.wait_to_model_ready(), loop=visualserver.new_loop)
        future.result()

        asyncio.run_coroutine_threadsafe(
            visualserver.register_to_config_server_loop(args=args), loop=visualserver.new_loop
        )

        t = VisualOnlyThreadedServer(
            visualserver,
            hostname=visualserver.rpc_host,
            port=get_shm_port_args().visual_rpyc_port,
            protocol_config=visual_only_rpyc_config,
        )
    except Exception as e:
        logger.exception(str(e))
        visualserver.clean_up()
        raise e

    pipe_writer.send("init ok")

    t.start()
    return
