import inspect
import torch.multiprocessing as mp
import time
from typing import List, Dict, Union, Callable
from lightllm.utils.log_utils import init_logger
from lightllm.server.pd_io_struct import NIXLChunckedTransTask
from lightllm.utils.graceful_utils import graceful_registry
from lightllm.server.core.objs import StartArgs
from ..trans_process_obj import KVTransProcess
from ..base_kv_move_manager import BaseKVMoveManager
from lightllm.utils.error_utils import log_exception

logger = init_logger(__name__)


def start_prefill_kv_move_manager_process(args, info_queue: mp.Queue, mem_queues: List[mp.Queue]):
    event = mp.Event()
    proc = mp.Process(target=_init_env, args=(args, info_queue, mem_queues, event))
    proc.start()
    event.wait()
    assert proc.is_alive()
    logger.info("prefill kv move manager process started")
    return


def _init_env(args, info_queue: mp.Queue, mem_queues: List[mp.Queue], event: mp.Event):
    import lightllm.utils.rpyc_fix_utils as _

    # 注册graceful 退出的处理
    graceful_registry(inspect.currentframe().f_code.co_name)

    from .prefill_trans_process import start_prefill_trans_process

    manager = PrefillKVMoveManager(
        args=args, info_queue=info_queue, mem_queues=mem_queues, start_trans_process_func=start_prefill_trans_process
    )
    assert manager is not None
    event.set()
    while True:
        time.sleep(100)
    return


class PrefillKVMoveManager(BaseKVMoveManager):
    def __init__(
        self, args: StartArgs, info_queue: mp.Queue, mem_queues: List[mp.Queue], start_trans_process_func: Callable
    ):
        super().__init__(
            args=args, info_queue=info_queue, mem_queues=mem_queues, start_trans_process_func=start_trans_process_func
        )
        return

    # ==================================================================================
    # 主任务循环，接收需要进行kv传输的请求, 转发给 KV_TRANS_PROCESS
    # ==================================================================================
    @log_exception
    def task_dispatcher_loop(self):
        # 获取任务，并分发给相关卡的处理队列
        while True:
            task: NIXLChunckedTransTask = self.info_queue.get()

            device_id = task.src_device_id
            try:
                trans_process: KVTransProcess = self.kv_trans_processes[device_id]
                trans_process.task_in_queue.put(task)
                logger.info(f"kv move manager dispatch task {task.to_str()} to device {device_id}")
            except BaseException as e:
                logger.exception(str(e))
