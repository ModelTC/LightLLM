import time
import rpyc
import copy
import uuid
import numpy as np
import psutil
import threading
from dataclasses import dataclass
from typing import List, Dict, Union
from lightllm.utils.log_utils import init_logger
import torch.multiprocessing as mp
from lightllm.server.pd_io_struct import NIXLChunckedTransTask
from rpyc.utils.classic import obtain
from ..task_queue import TaskQueue
from lightllm.utils.device_utils import kv_trans_use_p2p
from lightllm.utils.time_utils import TimeChecker
from .prefill_kv_move_manager import PrefillKVMoveManager
from lightllm.utils.net_utils import find_available_port
from ..utils import join_if_alive, clear_queue

logger = init_logger(__name__)


@dataclass
class KVTransProcess:
    process: mp.Process = None
    task_in_queue: mp.Queue = None
    task_out_queue: mp.Queue = None
    device_id: int = None

    def init_all(self, device_id: int, manager: "PrefillKVMoveManager"):
        self.device_id = device_id
        self.device_lock = threading.Lock()
        self.task_in_queue = mp.Queue()
        self.task_out_queue = mp.Queue()

        try:
            from .prefill_trans_process import start_prefill_trans_process

            self.process = start_prefill_trans_process(
                manager.args,
                manager.host_ip,
                device_id,
                self.task_in_queue,
                self.task_out_queue,
                manager.mem_queues,
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
