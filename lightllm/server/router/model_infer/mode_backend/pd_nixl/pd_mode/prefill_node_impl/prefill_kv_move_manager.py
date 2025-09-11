import time
import os
import signal
import threading
import inspect
import queue
from typing import List, Dict, Union
from lightllm.utils.log_utils import init_logger
import torch.multiprocessing as mp
from lightllm.server.pd_io_struct import NIXLChunckedTransTask, ChunckedTransTaskRet
from lightllm.utils.graceful_utils import graceful_registry
from lightllm.server.core.objs import StartArgs
from lightllm.server.router.shm_reqs_io_buffer import ShmReqsIOBuffer

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

    manager = PrefillKVMoveManager(args, info_queue, mem_queues)
    event.set()
    # 阻塞等待子线程退出
    manager.dispatch_task.join()
    return


class PrefillKVMoveManager:
    def __init__(self, args: StartArgs, info_queue: mp.Queue, mem_queues: List[mp.Queue]):
        self.args = args
        # args.dp // args.nnodes 在跨机tp的场景下，可能为0
        self.dp_size_in_node = max(1, args.dp // args.nnodes)
        self.node_world_size = args.tp // args.nnodes
        self.dp_world_size = args.tp // args.dp
        # 不支持跨机tp的pd 分离策略
        assert self.dp_world_size <= self.node_world_size

        self.info_queue = info_queue
        self.mem_queues = mem_queues
        self.ret_obj_queue = queue.Queue()

        from .prefill_trans_obj import KVTransProcess

        self.kv_trans_processes: List[KVTransProcess] = [None] * self.node_world_size
        for device_id in range(self.node_world_size):
            self.kv_trans_processes[device_id] = KVTransProcess()
            assert self.kv_trans_processes[device_id].init_all(device_id, self)
        self.kv_trans_processes_ret_threads: List[threading.Thread] = [None] * self.node_world_size
        for device_id in range(self.node_world_size):
            kv_trans_process = self.kv_trans_processes[device_id]
            self.kv_trans_processes_ret_threads[device_id] = threading.Thread(
                target=self.task_ret_handle_loop, args=(kv_trans_process,), daemon=True
            )
            self.kv_trans_processes_ret_threads[device_id].start()

        # 通过 io buffer 将命令写入到推理进程中
        self.shm_reqs_io_buffer = ShmReqsIOBuffer()
        self.dispatch_task = threading.Thread(target=self.task_dispatcher_loop, daemon=True)
        self.dispatch_task.start()
        self.release_task = threading.Thread(target=self.task_release_loop, daemon=True)
        self.release_task.start()
        self.check_task = threading.Thread(target=self.check_trans_process_loop, daemon=True)
        self.check_task.start()
        return

    # ==================================================================================
    # 主任务循环，接收需要进行kv传输的请求, 转发给 KV_TRANS_PROCESS
    # ==================================================================================

    def task_dispatcher_loop(self):
        try:
            # 获取任务，并分发给相关卡的处理队列
            while True:
                task:NIXLChunckedTransTask = self.info_queue.get()

                device_id = task.src_device_id
                try:
                    from .prefill_trans_obj import KVTransProcess

                    trans_process: KVTransProcess = self.kv_trans_processes[device_id]
                    trans_process.task_in_queue.put(task)
                except BaseException as e:
                    logger.exception(str(e))

        except (BaseException, RuntimeError) as e:
            logger.exception(str(e))
            raise e
        
    # ==================================================================================
    #  将收集到的传输返回信息，批量写回给推理进程，触发其进行相关管理信息的更新。
    # ==================================================================================
    def task_release_loop(self):
        try:
            while True:
                req_objs: List[ChunckedTransTaskRet] = []
                ret_obj: ChunckedTransTaskRet = self.ret_obj_queue.get()
                while True:
                    try:
                        ret_obj = self.ret_obj_queue.get_nowait()
                        req_objs.append(ret_obj)
                    except queue.Empty:
                        break
                
                # 将命令进行写回操作。
                self.shm_reqs_io_buffer.release_reqs(req_ids)
        except (BaseException, RuntimeError) as e:
            logger.exception(str(e))
            raise e

    # ==================================================================================
    # 处理各个传输任务的返回的信息
    # ==================================================================================
    def task_ret_handle_loop(self, trans_process):
        from .prefill_trans_obj import KVTransProcess

        trans_process: KVTransProcess = trans_process

        try:
            while True:
                ret_obj: ChunckedTransTaskRet = trans_process.task_out_queue.get()
                self.ret_obj_queue.put(ret_obj)

        except BaseException as e:
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

