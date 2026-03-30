import asyncio
import uvloop
import rpyc
import socket
import pickle
import inspect
import setproctitle
import threading
import base64
import httpx
import random
import copy
from typing import List, Dict, Optional
from lightllm.server.core.objs.io_objs.group_req import GroupReqIndexes
from lightllm.server.core.objs import ShmReqManager, StartArgs
from lightllm.server.embed_cache.embed_cache_client import CpuEmbedCacheClient

asyncio.set_event_loop_policy(uvloop.EventLoopPolicy())
from lightllm.server.embed_cache.afs_utils import SepEmbedHandler
from lightllm.server.multimodal_params import MultimodalParams, ImageItem
from lightllm.utils.log_utils import init_logger
from lightllm.utils.graceful_utils import graceful_registry
from lightllm.utils.process_check import start_parent_check_thread
from lightllm.utils.envs_utils import get_unique_server_name
from rpyc.utils.classic import obtain
from lightllm.server.embed_cache.utils import read_shm, get_shm_name_data
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

        self.cpu_embed_cache_client = CpuEmbedCacheClient(create_meta_data=False, init_shm_data=False, pin_shm=False)

        self.afs_handler = SepEmbedHandler(
            afs_embed_dir=self.args.afs_image_embed_dir,
            redis_host=self.args.config_server_host,
            redis_port=self.args.config_server_visual_redis_port,
            capacity=self.args.afs_embed_capacity,
        )

    async def handle_group_indexes(self, group_req_indexes: GroupReqIndexes):
        images_need_infer = self.get_need_infer_images(group_req_indexes)

        # case 1
        if len(images_need_infer) == 0:
            self.send_to_next_module.send_pyobj(group_req_indexes, protocol=pickle.HIGHEST_PROTOCOL)
            return

        try:

            def _get_not_afs_ready_images():
                readys = self.afs_handler.check_ready([image.md5 for image in images_need_infer])
                not_readys_images = [image for image, ready in zip(images_need_infer, readys) if not ready]
                # 将 images_need_infer 按照 self.infer_batch_size 切分成多个 batch，发送给不同的 visual server 进行推理，\
                # 最后等待所有推理完成后再发送给下一个模块
                images_batches = [
                    not_readys_images[i : i + self.infer_batch_size]
                    for i in range(0, len(not_readys_images), self.infer_batch_size)
                ]
                return images_batches

            images_batches = await asyncio.to_thread(_get_not_afs_ready_images)
            taskes = []

            for images_batch in images_batches:
                conn = self.select_vit_conn()

                def _run_task():
                    self.run_task(conn, images_batch)

                taskes.append(asyncio.to_thread(_run_task))

            if len(taskes) > 0:

                await asyncio.gather(*taskes)
        except Exception as e:
            # mark aborted
            for shm_req_index in group_req_indexes.shm_req_indexes:
                shm_req = self.shm_req_manager.get_req_obj_by_index(shm_req_index)
                shm_req.is_aborted = True
                self.shm_req_manager.put_back_req_obj(shm_req)

            logger.exception(str(e))

        self.send_to_next_module.send_pyobj(group_req_indexes, protocol=pickle.HIGHEST_PROTOCOL)
        return

    def select_vit_conn(self) -> Optional[rpyc.Connection]:
        with self.conn_lock:
            if not self.id_to_rpyc_conn:
                return None
            ids = list(self.id_to_rpyc_conn.keys())
            id = random.choice(ids)
            return self.id_to_rpyc_conn[id]

    def run_task(self, conn: rpyc.Connection, images: List[ImageItem]):
        event = threading.Event()
        # 避免修改原始的 image 对象，主要是为了避免在后续的流程中出现问题，因为后续的流程可能会对 image 对象进行访问，
        # 尤其是一些 cache 的逻辑，如果直接修改了原始的 image 对象，可能会导致一些不可预期的问题。
        images = copy.deepcopy(images)
        # 将 bytes 从 shm 中读取出来，放到 image.data_bytes 中，供远端的 vit 进行推理使用。
        for image in images:
            image.data_bytes = read_shm(get_shm_name_data(image.uuid))
        conn.root.remote_infer_images(images, event)
        event.wait(timeout=600)

        for image in images:
            tensor = self.afs_handler.load(md5=image.md5)
            if tensor is None:
                raise Exception(f"Failed to load tensor from afs for image uuid {image.uuid} with md5 {image.md5}")
            start = image.start_index_in_embed_cache
            end = start + tensor.shape[0]
            assert end - start == image.token_num
            self.cpu_embed_cache_client.cpu_embed_cache_tensor[start:end] = tensor

        self.cache_client.root.set_items_embed([image.uuid for image in images])
        return

    async def loop_to_connect_remote_visual_server(self):
        counter = 0
        error_counter = 0
        while True:
            uri = f"http://{self.args.config_server_host}:{self.args.config_server_port}/registered_visual_objects"
            try:
                async with httpx.AsyncClient(timeout=10.0) as client:
                    response = await client.get(uri)
                    if response.status_code == 200:
                        base64data = response.json()["data"]
                        id_to_vit_obj = pickle.loads(base64.b64decode(base64data))

                        counter += 1
                        if counter % 6 == 0:
                            logger.info(f"Got visual server info from config server: {id_to_vit_obj}")

                        for node_id in list(self.id_to_rpyc_conn.keys()):
                            if node_id not in id_to_vit_obj:
                                logger.info(f"Visual server {node_id} is removed, closing connection")
                                with self.conn_lock:
                                    self.id_to_rpyc_conn.pop(node_id).close()

                        for node_id, vit_obj in id_to_vit_obj.items():
                            vit_obj: VIT_Obj = vit_obj
                            if node_id not in self.id_to_rpyc_conn:

                                def _connect():
                                    from .objs import rpyc_config

                                    conn = rpyc.connect(vit_obj.host_ip, vit_obj.port, config=rpyc_config)
                                    conn._channel.stream.sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
                                    conn._bg_thread = rpyc.BgServingThread(conn)
                                    logger.info(
                                        f"Connected to visual server {node_id} at {vit_obj.host_ip}:{vit_obj.port}"
                                    )
                                    return conn

                                try:
                                    new_conn = await asyncio.to_thread(_connect)
                                    with self.conn_lock:
                                        self.id_to_rpyc_conn[node_id] = new_conn
                                except Exception as e:
                                    logger.exception(str(e))
                    else:
                        logger.error(f"Failed to get VIT instances: {response.status_code}")
            except Exception as e:
                logger.error(f"Error occurred while connecting to config server: {e}")
                error_counter += 1

            if error_counter >= 6:
                logger.error(
                    "Failed to connect to config server for a long time, remove all connections to visual servers"
                )
                error_counter = 0
                try:
                    with self.conn_lock:
                        for node_id, conn in self.id_to_rpyc_conn.items():
                            logger.info(f"Closing connection to visual server {node_id}")
                            conn.close()
                        self.id_to_rpyc_conn.clear()
                except Exception as e:
                    logger.exception(str(e))

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
