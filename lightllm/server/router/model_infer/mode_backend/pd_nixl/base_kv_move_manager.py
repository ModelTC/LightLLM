import time
import os
import signal
import threading
import queue
import torch.multiprocessing as mp
from typing import List, Dict, Union, Callable, Optional
from lightllm.utils.log_utils import init_logger
from lightllm.server.pd_io_struct import ChunckedTransTaskRet
from lightllm.utils.graceful_utils import graceful_registry
from lightllm.server.core.objs import StartArgs
from lightllm.server.core.objs.shm_objs_io_buffer import ShmObjsIOBuffer
from .trans_process_obj import KVTransProcess

logger = init_logger(__name__)


class BaseKVMoveManager:
    def __init__(self, 
                 args: StartArgs, 
                 info_queue: mp.Queue, 
                 mem_queues: List[mp.Queue],
                 start_trans_process_func:Callable,
                 up_status_in_queue: Optional[mp.SimpleQueue]=None):
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

        self.kv_trans_processes: List[KVTransProcess] = [None] * self.node_world_size
        for device_id in range(self.node_world_size):
            self.kv_trans_processes[device_id] = KVTransProcess()
            assert self.kv_trans_processes[device_id].init_all(device_id=device_id,
                                                               manager=self,
                                                               start_func=start_trans_process_func,
                                                               up_status_in_queue=up_status_in_queue)
            self.kv_trans_processes[device_id].start_ret_handle_thread(func=self.task_ret_handle_loop)

        # 通过 io buffer 将命令写入到推理进程中
        self.shm_nixl_trans_io_buffer = ShmObjsIOBuffer(tail_str="nixl")
        self.dispatch_task = threading.Thread(target=self.task_dispatcher_loop, daemon=True)
        self.dispatch_task.start()
        self.release_task = threading.Thread(target=self.task_ret_upload_loop, daemon=True)
        self.release_task.start()
        self.check_task = threading.Thread(target=self.check_trans_process_loop, daemon=True)
        self.check_task.start()
        return

    # ==================================================================================
    # 主任务循环，接收需要进行kv传输的请求, 转发给 KV_TRANS_PROCESS
    # ==================================================================================

    def task_dispatcher_loop(self):
        raise NotImplementedError()
        
    # ==================================================================================
    #  将收集到的传输返回信息，批量写回给推理进程，触发其进行相关管理信息的更新。
    # ==================================================================================
    def task_ret_upload_loop(self):
        try:
            while True:
                ret_obj: ChunckedTransTaskRet = self.ret_obj_queue.get()
                ret_objs: List[ChunckedTransTaskRet] = [ret_obj]
                ret_objs.extend(self._collect_return_objects())
               
                while True:
                    if self.shm_nixl_trans_io_buffer.is_empty():
                        # to do, 这里写入的数量，可能会超过共享管道的大小。
                        self.shm_nixl_trans_io_buffer.write_obj(ret_objs)
                        self.shm_nixl_trans_io_buffer.set_ready()
                        break
                    else:
                        time.sleep(0.01)
                        ret_objs.extend(self._collect_return_objects())
        
        except (BaseException, RuntimeError) as e:
            logger.exception(str(e))
            raise e
        
    def _collect_return_objects(self):
        """ 从队列中收集所有返回对象 """
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
    def task_ret_handle_loop(self, trans_process):
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

