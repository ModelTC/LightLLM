import time
import json
import asyncio
import threading
import websockets
import inspect
import pickle

from typing import Dict, Union
from dataclasses import asdict
from lightllm.server.pd_io_struct import UpKVStatus, NixlUpKVStatus
from lightllm.utils.log_utils import init_logger
from lightllm.utils.graceful_utils import graceful_registry
from lightllm.server.pd_io_struct import PD_Master_Obj
import torch.multiprocessing as mp

logger = init_logger(__name__)


class UpStatusManager:
    def __init__(self, args, task_in_queue: mp.SimpleQueue):
        self.args = args
        self.task_queue: mp.SimpleQueue[Union[UpKVStatus, NixlUpKVStatus]] = task_in_queue
        self.daemon_thread = threading.Thread(target=self.thread_loop, daemon=True)
        self.daemon_thread.start()

    def thread_loop(self):
        asyncio.run(self.task_loop())

    async def task_loop(self):

        self.id_to_handle_task: Dict[int, asyncio.Task] = {}
        self.id_to_handle_queue: Dict[int, asyncio.Queue] = {}

        asyncio.create_task(self.dispatch_task_loop())

        while True:
            try:
                from lightllm.server.httpserver.pd_loop import _get_pd_master_objs

                id_to_pd_master_obj = await _get_pd_master_objs(self.args)
                logger.info(f"get pd_master_objs {id_to_pd_master_obj}")

                if id_to_pd_master_obj is not None:
                    for node_id, pd_master_obj in self.id_to_handle_task.items():
                        if node_id not in id_to_pd_master_obj:
                            self.id_to_handle_task[node_id].cancel()
                            self.id_to_handle_task.pop(node_id, None)
                            self.id_to_handle_queue.pop(node_id, None)
                            logger.info(f"up_kv_status_task {pd_master_obj} cancelled")

                    for node_id, pd_master_obj in id_to_pd_master_obj.items():
                        if node_id not in self.id_to_handle_task:
                            self.id_to_handle_queue[node_id] = asyncio.Queue()
                            self.id_to_handle_task[node_id] = asyncio.create_task(self.up_kv_status_task(pd_master_obj))

                await asyncio.sleep(30)

            except Exception as e:
                logger.exception(str(e))
                await asyncio.sleep(10)

    async def dispatch_task_loop(self):
        while True:
            try:
                loop = asyncio.get_event_loop()
                upkv_status: UpKVStatus = await loop.run_in_executor(None, self.task_queue.get)
                if upkv_status.pd_master_node_id in self.id_to_handle_queue:
                    await self.id_to_handle_queue[upkv_status.pd_master_node_id].put(upkv_status)
                else:
                    logger.warning(f"upstatus {upkv_status} no connection to pd_master, drop it")
            except BaseException as e:
                logger.exception(str(e))
                await asyncio.sleep(10)

    async def up_kv_status_task(self, pd_master_obj: PD_Master_Obj):
        while True:
            try:
                uri = f"ws://{pd_master_obj.host_ip_port}/kv_move_status"
                async with websockets.connect(uri) as websocket:
                    import socket

                    sock = websocket.transport.get_extra_info("socket")
                    sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)

                    while True:
                        try:
                            if pd_master_obj.node_id in self.id_to_handle_queue:
                                task_queue = self.id_to_handle_queue[pd_master_obj.node_id]
                                upkv_status: Union[UpKVStatus, NixlUpKVStatus] = await task_queue.get()
                                await websocket.send(pickle.dumps(upkv_status))
                                logger.info(f"up kv status: {upkv_status}")
                            else:
                                await asyncio.sleep(3)
                        except BaseException as e:
                            logger.error(str(e))
                            raise e
            except asyncio.CancelledError:
                logger.info(f"up_kv_status_task {pd_master_obj} cancelled")
                return

            except Exception as e:
                logger.error(f"connetion to pd_master {pd_master_obj} has error: {str(e)}")
                logger.exception(str(e))
                await asyncio.sleep(10)
                logger.info("reconnection to pd_master")


def _init_env(args, task_in_queue: mp.SimpleQueue):
    graceful_registry(inspect.currentframe().f_code.co_name)
    up_kv_manager = UpStatusManager(args, task_in_queue)
    logger.info(f"up kv manager {str(up_kv_manager)} start ok")
    while True:
        time.sleep(666)
    return


def start_up_kv_status_process(args, task_in_queue: mp.SimpleQueue):
    proc = mp.Process(target=_init_env, args=(args, task_in_queue))
    proc.start()
    assert proc.is_alive()
    logger.info("up_kv_status_process start")
    return proc
