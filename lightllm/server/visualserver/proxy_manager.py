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
import base64
import httpx
from typing import List
from lightllm.server.core.objs.io_objs.group_req import GroupReqIndexes
from lightllm.server.core.objs import ShmReqManager, StartArgs

asyncio.set_event_loop_policy(uvloop.EventLoopPolicy())
from lightllm.server.multimodal_params import MultimodalParams, ImageItem
from lightllm.utils.log_utils import init_logger
from lightllm.utils.graceful_utils import graceful_registry
from lightllm.utils.process_check import start_parent_check_thread
from lightllm.utils.envs_utils import get_unique_server_name
from rpyc.utils.classic import obtain
from .manager import VisualManager

logger = init_logger(__name__)


class ProxyVisualManager(VisualManager):
    def __init__(
        self,
        args: StartArgs,
    ):
        super().__init__(args)
        assert self.vit_dp == 1 and self.vit_tp == 1

    async def handle_group_indexes(self, group_req_indexes: GroupReqIndexes):
        images_need_infer = self.get_need_infer_images(group_req_indexes)

        # case 1
        if len(images_need_infer) == 0:
            self.send_to_next_module.send_pyobj(group_req_indexes, protocol=pickle.HIGHEST_PROTOCOL)
            return

        # case 2
        dp_to_handle_images = collections.defaultdict(list)
        for image in images_need_infer:
            self.cur_dp_index += 1
            select_dp = self.cur_dp_index % self.vit_dp
            dp_to_handle_images[select_dp].append((image, threading.Event()))

        taskes = []
        for dp_index in range(self.vit_dp):
            _images = dp_to_handle_images[dp_index]
            if _images:
                taskes.extend(self.run_task(dp_index, images=[e[0] for e in _images], events=[e[1] for e in _images]))

        with self.lock:
            await asyncio.gather(*taskes)

        for dp_index in range(self.vit_dp):
            _images = dp_to_handle_images[dp_index]
            if _images:
                await asyncio.to_thread(_images[-1][1].wait)

        self.send_to_next_module.send_pyobj(group_req_indexes, protocol=pickle.HIGHEST_PROTOCOL)
        return

    def run_task(self, dp_index: int, images, events):
        taskes = []
        for vit_tp_rank in range(self.vit_tp):
            task = self.model_rpcs[dp_index][vit_tp_rank].run_task(images, events)
            taskes.append(task)
        return taskes

    async def loop_for_netio_req(self):
        if not hasattr(self, "visual_recv_max_count"):
            self.visual_recv_max_count = 64

        while True:
            try:
                for _ in range(self.visual_recv_max_count):
                    recv_req: GroupReqIndexes = self.zmq_recv_socket.recv_pyobj(zmq.NOBLOCK)
                    if isinstance(recv_req, GroupReqIndexes):
                        logger.info(
                            f"visual recv req id {recv_req.group_req_id} "
                            f"img count {len(recv_req.multimodal_params.images)}"
                        )
                        asyncio.create_task(self.handle_group_indexes(group_req_indexes=recv_req))
                    else:
                        assert False, f"Error Req Inf {recv_req}"
                self.visual_recv_max_count = int(min(self.visual_recv_max_count * 1.3, 256))
            except zmq.ZMQError:
                # 当队列已经开始清空的时候，将一次接受数量下调
                self.visual_recv_max_count = 64
            await asyncio.sleep(0.01)

    async def loop_to_connect_remote_visual_server(self):
        uri = f"http://{self.args.config_server_host}:{self.args.config_server_port}/registered_visual_objects"
        try:
            async with httpx.AsyncClient(timeout=10.0) as client:
                response = await client.get(uri)
                if response.status_code == 200:
                    base64data = response.json()["data"]
                    id_to_vit_obj = pickle.loads(base64.b64decode(base64data))
                    return id_to_vit_obj
                else:
                    logger.error(f"Failed to get VIT instances: {response.status_code}")
                    return None
        except Exception as e:
            logger.exception(f"Error getting VIT instances: {e}")
            return None

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
