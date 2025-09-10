import rpyc
import asyncio
import os
import signal
import collections
import dataclasses
import time
import threading
import inspect
from typing import List, Dict, Optional, Tuple, Union
from rpyc import ThreadedServer
from lightllm.utils.log_utils import init_logger
import torch.multiprocessing as mp
from lightllm.server.pd_io_struct import NIXLDecodeKVMoveTask, NIXLStopTransTask, PDTransJoinInfo, PDTransLeaveInfo
from lightllm.server.core.objs import StartArgs
from rpyc import AsyncResult
from lightllm.utils.device_utils import kv_trans_use_p2p
from lightllm.utils.graceful_utils import graceful_registry
from lightllm.utils.envs_utils import get_unique_server_name

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

    manager = DecodeKVMoveManager(args, info_queue, mem_queues)
    t = ThreadedServer(manager, port=args.pd_decode_rpyc_port, protocol_config={"allow_pickle": True})
    threading.Thread(target=lambda: t.start(), daemon=True).start()

    kv_trans_process_check = threading.Thread(target=manager.check_trans_process_loop, daemon=True)
    kv_trans_process_check.start()

    event.set()
    manager.timer_loop()
    return


class DecodeKVMoveManager(rpyc.Service):
    def __init__(self, args: StartArgs, info_queue: mp.Queue, mem_queues: List[mp.Queue]):
        super().__init__()
        self.args = args
        # args.dp // args.nnodes 在跨机tp的场景下，可能为0
        self.dp_size_in_node = max(1, args.dp // args.nnodes)
        self.node_world_size = args.tp // args.nnodes
        self.dp_world_size = args.tp // args.dp
        # 不支持跨机tp的pd 分离策略
        assert self.dp_world_size <= self.node_world_size

        self.info_queue = info_queue
        self.mem_queues = mem_queues

        self.req_id_to_status: Dict[int, _ReqStatus] = {}
        self.req_id_to_status_lock: threading.Lock = threading.Lock()

        from .up_status import start_up_kv_status_process

        self.up_status_in_queue = mp.Queue()
        self.up_status_out_queue = mp.Queue()
        start_up_kv_status_process(self.args, self.up_status_in_queue, self.up_status_out_queue)

        from .decode_trans_obj import KVTransProcess

        self.kv_trans_processes: List[KVTransProcess] = [None] * self.node_world_size
        for device_id in range(self.node_world_size):
            self.kv_trans_processes[device_id] = KVTransProcess()
            assert self.kv_trans_processes[device_id].init_all(device_id, self)

        return

    # ==================================================================================
    # 主任务循环，接收需要进行kv传输的请求进行处理
    # ==================================================================================

    def task_dispatcher_loop(self):
        try:
            # 获取任务，并分发给相关卡的处理队列
            while True:
                task: Union[NIXLDecodeKVMoveTask, NIXLStopTransTask] = self.info_queue.get()
                # 传输任务
                if isinstance(task, NIXLDecodeKVMoveTask):
                    with self.req_id_to_status_lock:
                        if task.request_id not in self.req_id_to_status:
                            self.req_id_to_status[task.request_id] = _ReqStatus(request_id=task.request_id)
                        req_status = self.req_id_to_status[task.request_id]
                        # 当该请求的传输任务没有出现错误的时候，才进行后续的传输任务下发
                        if not req_status.is_error:
                            req_status.chuncked_task_list.append(task)
                            device_id = task.src_device_id
                            try:
                                from .prefill_trans_obj import KVTransProcess

                                trans_process: KVTransProcess = self.kv_trans_processes[device_id]
                                trans_process.task_in_queue.put(task)
                            except BaseException as e:
                                logger.exception(str(e))
                # 停止任务
                elif isinstance(task, NIXLStopTransTask):
                    with self.req_id_to_status_lock:
                        if task.request_id in self.req_id_to_status:
                            self.req_id_to_status.pop(task.request_id, None)
                        else:
                            logger.warning(f"no exist req id {task.request_id} in req_id_to_status")
                else:
                    assert False, f"error task type {type(task)}"

        except (BaseException, RuntimeError) as e:
            logger.exception(str(e))
            raise e
    # ==================================================================================
    # 定时检测传输进程的健康状态，出现问题拉崩整个系统触发重启
    # ==================================================================================

    def check_trans_process_loop(self):
        try:
            while True:
                for device_id in range(self.node_world_size):
                    if not self.kv_trans_processes[device_id].is_trans_process_health():
                        raise Exception(f"device_id {device_id} kv process is unhealth")

                time.sleep(10.0)
        except (BaseException, RuntimeError) as e:
            logger.exception(str(e))

            for device_id in range(self.node_world_size):
                self.kv_trans_processes[device_id].killself()

            # 杀掉当前进程的父进程（router), 触发全局崩溃
            os.kill(os.getppid(), signal.SIGKILL)
            os.kill(os.getpid(), signal.SIGKILL)
            raise e

@dataclasses
class _ReqStatus:
    request_id: int
    is_error: bool = False  # 发生了传输错误
    chuncked_task_list: List[NIXLChunckedTransTask] = dataclasses.field(default_factory=list)  # 该请求下发的所有传输任务
    success_count: int = 0  # 成功传输的块的数量
    failed_count: int = 0  # 失败传输的块的数量
