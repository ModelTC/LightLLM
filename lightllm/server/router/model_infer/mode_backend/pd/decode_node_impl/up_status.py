import time
import json
import asyncio
import threading
import websockets
import inspect
import pickle
import setproctitle

from typing import Dict
from dataclasses import asdict
from lightllm.server.pd_io_struct import PDAllocationReady
from lightllm.utils.log_utils import init_logger, log_time_ready
from lightllm.utils.graceful_utils import graceful_registry
from lightllm.server.pd_io_struct import PD_Master_Obj
import torch.multiprocessing as mp
from lightllm.utils.envs_utils import get_unique_server_name
from lightllm.utils.process_check import start_parent_check_thread

logger = init_logger(__name__)


class UpStatusManager:
    def __init__(self, args, task_in_queue: mp.SimpleQueue):
        self.args = args
        self.task_queue = task_in_queue
        self.daemon_thread = threading.Thread(target=self.thread_loop, daemon=True)
        self.daemon_thread.start()

    def thread_loop(self):
        asyncio.run(self.task_loop())

    async def task_loop(self):

        self.id_to_handle_task: Dict[int, asyncio.Task] = {}
        self.id_to_handle_queue: Dict[int, asyncio.Queue] = {}

        dispatch_task = None

        while True:
            try:
                from lightllm.server.httpserver.pd_loop import _get_pd_master_objs

                id_to_pd_master_obj = await _get_pd_master_objs(self.args)
                logger.info(f"get pd_master_objs {id_to_pd_master_obj}")

                if id_to_pd_master_obj is not None:
                    for node_id, pd_master_obj in list(self.id_to_handle_task.items()):
                        if node_id not in id_to_pd_master_obj:
                            self.id_to_handle_task[node_id].cancel()
                            self.id_to_handle_task.pop(node_id, None)
                            self.id_to_handle_queue.pop(node_id, None)
                            logger.info(f"up_kv_status_task {pd_master_obj} cancelled")

                    for node_id, pd_master_obj in id_to_pd_master_obj.items():
                        if node_id not in self.id_to_handle_task:
                            self._connect_worker(pd_master_obj)

                if self.args.run_mode == "prefill":
                    for pending in self.id_to_handle_queue.values():
                        for _ in range(pending.qsize()):
                            status = pending.get_nowait()
                            if status.expires_at > time.monotonic():
                                pending.put_nowait(status)
                if dispatch_task is None:
                    # A P sender starts lazily on its first READY. Discover its owner before consuming it.
                    dispatch_task = asyncio.create_task(self.dispatch_task_loop())
                await asyncio.sleep(30)

            except Exception as e:
                logger.exception(str(e))
                await asyncio.sleep(10)

    def _connect_worker(self, pd_master_obj: PD_Master_Obj):
        node_id = pd_master_obj.node_id
        capacity = self.args.running_max_req_size if self.args.run_mode == "prefill" else 0
        self.id_to_handle_queue[node_id] = asyncio.Queue(maxsize=capacity)
        self.id_to_handle_task[node_id] = asyncio.create_task(self.up_kv_status_task(pd_master_obj))

    async def dispatch_task_loop(self):
        while True:
            try:
                loop = asyncio.get_event_loop()
                upkv_status = await loop.run_in_executor(None, self.task_queue.get)
                if isinstance(upkv_status, PDAllocationReady) and upkv_status.expires_at <= time.monotonic():
                    continue
                node_id = upkv_status.pd_master_node_id
                if isinstance(upkv_status, PDAllocationReady) and node_id not in self.id_to_handle_queue:
                    # The P HTTP worker can register with a new master before our periodic refresh.
                    from lightllm.server.httpserver.pd_loop import _get_pd_master_objs

                    workers = await _get_pd_master_objs(self.args)
                    if workers and node_id in workers and node_id not in self.id_to_handle_queue:
                        self._connect_worker(workers[node_id])
                if node_id in self.id_to_handle_queue:
                    try:
                        self.id_to_handle_queue[node_id].put_nowait(upkv_status)
                    except asyncio.QueueFull:
                        if log_time_ready("pd_worker_ready_queue_full", 10):
                            logger.warning(f"readiness queue full for master {node_id}")
                else:
                    logger.warning(f"upstatus {upkv_status} no connection to pd_master, drop it")
            except BaseException as e:
                logger.exception(str(e))
                await asyncio.sleep(10)

    async def up_kv_status_task(self, pd_master_obj: PD_Master_Obj):
        pending_status = None
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
                                if pending_status is None:
                                    pending_status = await task_queue.get()
                                if (
                                    isinstance(pending_status, PDAllocationReady)
                                    and pending_status.expires_at <= time.monotonic()
                                ):
                                    pending_status = None
                                    continue
                                await websocket.send(pickle.dumps(pending_status))
                                # KV readiness is reported once per request, not once per transfer page.
                                logger.debug(f"up kv status: {pending_status}")
                                pending_status = None
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
    start_parent_check_thread()
    graceful_registry(inspect.currentframe().f_code.co_name)
    setproctitle.setproctitle(f"lightllm::{get_unique_server_name()}::pd_up_kv_status")
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
