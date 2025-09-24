import inspect
import pickle
import torch.multiprocessing as mp
import time
from typing import List, Dict, Optional, Tuple, Union, Callable
from lightllm.utils.log_utils import init_logger
from lightllm.server.pd_io_struct import NIXLChunckedTransTaskGroup, NIXLAbortReq
from lightllm.server.core.objs import StartArgs
from lightllm.utils.graceful_utils import graceful_registry
from ..trans_process_obj import KVTransProcess
from ..base_kv_move_manager import BaseKVMoveManager
from lightllm.utils.error_utils import log_exception

logger = init_logger(__name__)


def start_decode_kv_move_manager_process(args, info_queue: mp.Queue, mem_queues: List[mp.Queue]):
    event = mp.Event()
    proc = mp.Process(target=_init_env, args=(args, info_queue, mem_queues, event))
    proc.start()
    event.wait()
    assert proc.is_alive()
    logger.info("decode kv move manager process started")
    return


def _init_env(args, info_queue: mp.Queue, mem_queues: List[mp.Queue], event: mp.Event):
    import lightllm.utils.rpyc_fix_utils as _

    # 注册graceful 退出的处理
    graceful_registry(inspect.currentframe().f_code.co_name)

    from .up_status import start_up_kv_status_process

    up_status_in_queue = mp.SimpleQueue()
    start_up_kv_status_process(args, up_status_in_queue)

    from .decode_trans_process import start_decode_trans_process

    manager = DecodeKVMoveManager(
        args=args,
        info_queue=info_queue,
        mem_queues=mem_queues,
        start_trans_process_func=start_decode_trans_process,
        up_status_in_queue=up_status_in_queue,
    )
    assert manager is not None
    event.set()
    while True:
        time.sleep(100)
    return


class DecodeKVMoveManager(BaseKVMoveManager):
    def __init__(
        self,
        args: StartArgs,
        info_queue: mp.Queue,
        mem_queues: List[mp.Queue],
        start_trans_process_func: Callable,
        up_status_in_queue: mp.SimpleQueue,
    ):
        super().__init__(
            args=args,
            info_queue=info_queue,
            mem_queues=mem_queues,
            start_trans_process_func=start_trans_process_func,
            up_status_in_queue=up_status_in_queue,
        )
        return

    # ==================================================================================
    # 主任务循环，接收需要进行kv传输的请求, 转发给 KV_TRANS_PROCESS
    # ==================================================================================
    @log_exception
    def task_dispatcher_loop(self):
        # 获取任务，并分发给相关卡的处理队列
        while True:
            task_group: Union[NIXLChunckedTransTaskGroup, NIXLAbortReq] = self.info_queue.get()

            if isinstance(task_group, NIXLChunckedTransTaskGroup):
                device_id = task_group.task_list[0].dst_device_id
            elif isinstance(task_group, NIXLAbortReq):
                device_id = task_group.device_id
            else:
                assert False, f"error obj {task_group}"

            try:
                trans_process: KVTransProcess = self.kv_trans_processes[device_id]
                trans_process.task_in_queue.put(task_group)
                if isinstance(task_group, NIXLChunckedTransTaskGroup):
                    logger.info(
                        f"kv move manager dispatch task group {task_group.task_list[0].to_str()} to device {device_id}"
                    )

            except BaseException as e:
                logger.exception(str(e))
