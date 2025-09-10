import torch
import time
import inspect
import threading
import torch.multiprocessing as mp
import collections
from torch.distributed import TCPStore
from typing import List, Dict, Union
from lightllm.utils.log_utils import init_logger
from lightllm.common.mem_manager import MemoryManager
from lightllm.server.pd_io_struct import NIXLChunckedTransTask
from lightllm.utils.device_utils import kv_trans_use_p2p
from lightllm.utils.graceful_utils import graceful_registry
from lightllm.distributed.pynccl import StatelessP2PProcessGroup, PyNcclCommunicator
from lightllm.server.core.objs import StartArgs


logger = init_logger(__name__)


def start_prefill_trans_process(
    args,
    device_id,
    task_in_queue: mp.Queue,
    task_out_queue: mp.Queue,
    mem_queues: List[mp.Queue],
):
    proc = mp.Process(target=_init_env, args=(args, device_id, task_in_queue, task_out_queue, mem_queues))
    proc.start()
    assert proc.is_alive()
    logger.info(f"prefill trans kv process for device: {device_id} started!")
    return proc


def _init_env(
    args: StartArgs,
    device_id: int,
    task_in_queue: mp.Queue,
    task_out_queue: mp.Queue,
    mem_queues: List[mp.Queue],
):
    torch.backends.cudnn.enabled = False

    try:
        torch.cuda.set_device(device_id)
        graceful_registry(inspect.currentframe().f_code.co_name)

        dp_size_in_node = max(1, args.dp // args.nnodes)
        task_out_queue.put("proc_start")
        mem_managers: List[MemoryManager] = [mem_queue.get(timeout=60) for mem_queue in mem_queues]
        task_out_queue.put("get_mem_managers_ok")

        manager = _PrefillTransModule(args, device_id, task_in_queue, task_out_queue, mem_managers)
        manager.sb = dp_size_in_node

    except Exception as e:
        logger.error(f"Fatal error happened in kv trans process: {e}")
        pass


class _PrefillTransModule:
    def __init__(
        self,
        args: StartArgs,
        device_id: int,
        task_in_queue: mp.Queue,
        task_out_queue: mp.Queue,
        mem_managers: List[mp.Queue],
    ) -> None:
        self.args = args
        self.device_id = device_id
        self.task_in_queue = task_in_queue
        self.task_out_queue = task_out_queue
        self.mem_managers = mem_managers
        self.trans_page_link = collections.deque()
        self._create_nixl_agent()
        return

    def _create_nixl_agent(self):
        from ..nixl_kv_transporter import NixlKVTransporter

        self.nixl_agent = NixlKVTransporter(self.args, self.device_id)

    def handle_task(self, task: NIXLChunckedTransTask):
        # 查看连接是否存在
        pass
