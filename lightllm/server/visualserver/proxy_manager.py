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
from typing import List, Dict
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
from .objs import VIT_Obj

logger = init_logger(__name__)


class ProxyVisualManager(VisualManager):
    def __init__(
        self,
        args: StartArgs,
    ):
        super().__init__(args)
        assert self.vit_dp == 1 and self.vit_tp == 1
        self.id_to_rpyc_conn: Dict[str, rpyc.Connection] = {}
        self.conn_lock = threading.Lock()

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

    async def loop_to_connect_remote_visual_server(self):
        while True:
            uri = f"http://{self.args.config_server_host}:{self.args.config_server_port}/registered_visual_objects"
            try:
                async with httpx.AsyncClient(timeout=10.0) as client:
                    response = await client.get(uri)
                    if response.status_code == 200:
                        base64data = response.json()["data"]
                        id_to_vit_obj = pickle.loads(base64.b64decode(base64data))

                        for node_id in list(self.id_to_rpyc_conn.keys()):
                            if node_id not in id_to_vit_obj:
                                with self.conn_lock:
                                    self.id_to_rpyc_conn.pop(node_id).close()

                            for node_id, vit_obj in id_to_vit_obj.items():
                                vit_obj: VIT_Obj = vit_obj
                                if node_id not in self.id_to_rpyc_conn:

                                    def _connect():
                                        conn = rpyc.connect(
                                            vit_obj.host_ip, vit_obj.port, config={"allow_pickle": True}
                                        )
                                        conn._bg_thread = rpyc.BgServingThread(conn)
                                        return conn

                                    try:
                                        with self.conn_lock:
                                            self.id_to_rpyc_conn[node_id] = await asyncio.to_thread(_connect)
                                    except Exception as e:
                                        logger.exception(str(e))
                    else:
                        logger.error(f"Failed to get VIT instances: {response.status_code}")
            except Exception as e:
                logger.exception(f"Error getting VIT instances: {e}")

            # 在没有连接的时候，高频率更新，有的时候降低更新频率
            if len(self.id_to_rpyc_conn) == 0:
                await asyncio.sleep(10)
            else:
                await asyncio.sleep(30)


def start_visual_process(args, pipe_writer):
    import lightllm.utils.rpyc_fix_utils as _

    # 注册graceful 退出的处理
    graceful_registry(inspect.currentframe().f_code.co_name)
    setproctitle.setproctitle(f"lightllm::{get_unique_server_name()}::visual_server")
    start_parent_check_thread()
    try:
        visualserver = ProxyVisualManager(args=args)
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
    loop.create_task(visualserver.loop_to_connect_remote_visual_server())
    loop.run_until_complete(visualserver.loop_for_netio_req())
    return
