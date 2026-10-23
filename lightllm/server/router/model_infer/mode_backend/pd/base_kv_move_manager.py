import time
import os
import threading
import queue
import torch.multiprocessing as mp
from typing import List, Dict, Union, Callable, Optional
from lightllm.utils.log_utils import init_logger
from lightllm.server.pd_io_struct import (
    PDAbortReq,
    PDChunckedTransTask,
    PDChunckedTransTaskGroup,
    PDChunckedTransTaskRet,
)
from lightllm.server.core.objs import StartArgs
from lightllm.server.core.objs.shm_objs_io_buffer import ShmObjsIOBuffer
from .trans_process_obj import KVTransProcess
from lightllm.utils.error_utils import log_exception

logger = init_logger(__name__)

_MAX_TRANS_PROCESS_RESTARTS = 3
_TRANS_PROCESS_RESTART_RESET_SECONDS = 60


class BaseKVMoveManager:
    def __init__(
        self,
        args: StartArgs,
        info_queue: mp.Queue,
        start_trans_process_func: Callable,
        up_status_in_queue: Optional[mp.SimpleQueue] = None,
    ):
        self.args = args
        # args.dp // args.nnodes 在跨机tp的场景下，可能为0
        self.dp_size_in_node = max(1, args.dp // args.nnodes)
        self.node_world_size = args.tp // args.nnodes
        self.dp_world_size = args.tp // args.dp
        # 不支持跨机tp的pd 分离策略
        assert self.dp_world_size <= self.node_world_size

        self.info_queue = info_queue
        self.ret_obj_queue = queue.Queue()
        self.start_trans_process_func = start_trans_process_func
        self.up_status_in_queue = up_status_in_queue

        self.kv_trans_processes: List[KVTransProcess] = [None] * self.node_world_size
        self._trans_process_locks = [threading.Lock() for _ in range(self.node_world_size)]
        self._trans_process_restart_counts = [0] * self.node_world_size
        self._trans_process_last_restart_at: List[Optional[float]] = [None] * self.node_world_size
        for device_id in range(self.node_world_size):
            trans_process = KVTransProcess()
            self.kv_trans_processes[device_id] = trans_process
            assert trans_process.init_all(
                device_id=device_id,
                manager=self,
                start_func=start_trans_process_func,
                up_status_in_queue=up_status_in_queue,
            )

        for trans_process in self.kv_trans_processes:
            if not trans_process.wait_until_ready():
                raise RuntimeError(f"KV trans module for device {trans_process.device_id} failed to initialize")

        for trans_process in self.kv_trans_processes:
            threading.Thread(target=self.task_ret_handle_loop, args=(trans_process,), daemon=True).start()

        # 通过 io buffer 将命令写入到推理进程中
        self.shm_pd_trans_io_buffer = ShmObjsIOBuffer(tail_str="pd")

        for func in [self.task_dispatcher_loop, self.task_ret_upload_loop, self.check_trans_process_loop]:
            threading.Thread(target=func, daemon=True).start()
        return

    # ==================================================================================
    # 主任务循环，接收需要进行kv传输的请求, 转发给 KV_TRANS_PROCESS
    # ==================================================================================

    def task_dispatcher_loop(self):
        raise NotImplementedError()

    def dispatch_trans_task(
        self,
        device_id: int,
        task: Union[PDChunckedTransTask, PDChunckedTransTaskGroup, PDAbortReq],
    ) -> None:
        while True:
            with self._trans_process_locks[device_id]:
                trans_process = self.kv_trans_processes[device_id]
                if trans_process.is_trans_process_health():
                    trans_process.task_in_queue.put(task)
                    return
            time.sleep(0.01)

    # ==================================================================================
    #  将收集到的传输返回信息，批量写回给推理进程，触发其进行相关管理信息的更新。
    # ==================================================================================
    @log_exception
    def task_ret_upload_loop(self):
        while True:
            ret_obj: PDChunckedTransTaskRet = self.ret_obj_queue.get()
            ret_objs: List[PDChunckedTransTaskRet] = [ret_obj]
            ret_objs.extend(self._collect_return_objects())

            while True:
                if self.shm_pd_trans_io_buffer.is_empty():
                    # to do, 这里写入的数量，可能会超过共享管道的大小。
                    self.shm_pd_trans_io_buffer.write_obj(ret_objs)
                    self.shm_pd_trans_io_buffer.set_ready()
                    break
                else:
                    time.sleep(0.01)
                    ret_objs.extend(self._collect_return_objects())

    def _collect_return_objects(self):
        """从队列中收集所有返回对象"""
        ret_objs = []
        try:
            while True:
                ret_obj = self.ret_obj_queue.get_nowait()
                ret_objs.append(ret_obj)
        except queue.Empty:
            pass
        return ret_objs

    # ==================================================================================
    # 处理各个传输任务的返回的信息
    # ==================================================================================
    @log_exception
    def task_ret_handle_loop(self, trans_process: KVTransProcess) -> None:
        while True:
            ret_obj: Optional[PDChunckedTransTaskRet] = trans_process.task_out_queue.get()
            if ret_obj is None:
                return
            self.ret_obj_queue.put(ret_obj)

    def _restart_trans_process(self, device_id: int) -> None:
        with self._trans_process_locks[device_id]:
            old_trans_process = self.kv_trans_processes[device_id]
            if old_trans_process.process is not None:
                if old_trans_process.process.is_alive():
                    old_trans_process.killself()
                old_trans_process.process.join(timeout=5)
                if old_trans_process.process.is_alive():
                    raise RuntimeError(f"KV transfer worker for device {device_id} did not stop before restart")
            try:
                old_trans_process.task_out_queue.put(None)
            except (EOFError, OSError, ValueError):
                logger.warning(f"Could not stop return handler for KV transfer worker on device {device_id}")

            trans_process = KVTransProcess()
            if not trans_process.init_all(
                device_id=device_id,
                manager=self,
                start_func=self.start_trans_process_func,
                up_status_in_queue=self.up_status_in_queue,
            ):
                if trans_process.process is not None and trans_process.process.is_alive():
                    trans_process.killself()
                raise RuntimeError(f"KV transfer worker for device {device_id} failed to restart")

            if not trans_process.wait_until_ready():
                if trans_process.process is not None and trans_process.process.is_alive():
                    trans_process.killself()
                raise RuntimeError(f"KV transfer worker for device {device_id} did not become ready after restart")

            self.kv_trans_processes[device_id] = trans_process
            threading.Thread(target=self.task_ret_handle_loop, args=(trans_process,), daemon=True).start()
            self._trans_process_last_restart_at[device_id] = time.monotonic()
            logger.warning(f"Restarted KV transfer worker for device {device_id}")

    # ==================================================================================
    # 定时检测传输进程的健康状态，出现问题拉崩整个系统触发重启
    # ==================================================================================
    def check_trans_process_loop(self) -> None:
        try:
            while True:
                for device_id in range(self.node_world_size):
                    if not self.kv_trans_processes[device_id].is_trans_process_health():
                        self._trans_process_restart_counts[device_id] += 1
                        if self._trans_process_restart_counts[device_id] > _MAX_TRANS_PROCESS_RESTARTS:
                            raise RuntimeError(f"KV transfer worker for device {device_id} exceeded restart limit")
                        logger.warning(f"KV transfer worker for device {device_id} is unhealthy; restarting")
                        self._restart_trans_process(device_id)
                    elif (
                        self._trans_process_last_restart_at[device_id] is not None
                        and time.monotonic() - self._trans_process_last_restart_at[device_id]
                        >= _TRANS_PROCESS_RESTART_RESET_SECONDS
                    ):
                        self._trans_process_restart_counts[device_id] = 0

                time.sleep(1.0)
        except (BaseException, RuntimeError) as e:
            logger.exception(str(e))

            for device_id in range(self.node_world_size):
                try:
                    self.kv_trans_processes[device_id].killself()
                except Exception:
                    logger.exception(f"Failed to stop KV transfer worker for device {device_id}")

            # The launcher supervises this manager and removes the node on exit.
            os._exit(1)
