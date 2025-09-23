import threading
import psutil
import torch.multiprocessing as mp
from typing import List, Callable, Optional
from dataclasses import dataclass
from lightllm.utils.log_utils import init_logger

logger = init_logger(__name__)


@dataclass
class KVTransProcess:
    process: mp.Process = None
    task_in_queue: mp.Queue = None
    task_out_queue: mp.Queue = None
    device_id: int = None

    def init_all(
        self,
        device_id: int,
        manager,
        start_func: Callable,
        up_status_in_queue: Optional[mp.SimpleQueue],
    ):
        from .base_kv_move_manager import BaseKVMoveManager

        manager: BaseKVMoveManager = manager
        self.device_id = device_id
        self.task_in_queue = mp.Queue()
        self.task_out_queue = mp.Queue()

        try:
            self.process = start_func(
                manager.args,
                device_id,
                self.task_in_queue,
                self.task_out_queue,
                manager.mem_queues,
                up_status_in_queue,
            )
            assert self.task_out_queue.get(timeout=30) == "proc_start"
            assert self.task_out_queue.get(timeout=60) == "get_mem_managers_ok"

            return True

        except Exception as e:
            logger.warning(f"Failed start kv trans process for device {device_id}: {e}")
            logger.exception(str(e))
            return False

    def is_trans_process_health(self):
        try:
            process = psutil.Process(self.process.pid)
            if not (process.is_running() and process.status() != psutil.STATUS_ZOMBIE):
                logger.error(f"kv trans process for device: {self.device_id} dead!!!")
                return False
            else:
                return True
        except:
            return False

    def killself(self):
        self.process.kill()
