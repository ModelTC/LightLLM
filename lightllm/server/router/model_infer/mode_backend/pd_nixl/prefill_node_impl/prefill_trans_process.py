import torch
import time
import inspect
import threading
import torch.multiprocessing as mp
import collections
import queue
import pickle
from typing import List, Dict, Union, Deque, Optional
from lightllm.utils.log_utils import init_logger
from lightllm.common.mem_manager import MemoryManager
from lightllm.server.pd_io_struct import NIXLChunckedTransTask, NIXLChunckedTransTaskRet
from lightllm.utils.device_utils import kv_trans_use_p2p
from lightllm.utils.graceful_utils import graceful_registry
from lightllm.server.core.objs import StartArgs
from ..nixl_kv_transporter import NixlKVTransporter
from lightllm.utils.error_utils import log_exception


logger = init_logger(__name__)


def start_prefill_trans_process(
    args,
    device_id,
    task_in_queue: mp.Queue,
    task_out_queue: mp.Queue,
    mem_queues: List[mp.Queue],
    up_status_in_queue: Optional[mp.SimpleQueue] = None
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

        manager = _PrefillTransModule(args=args,
                                      device_id=device_id,
                                      task_in_queue=task_in_queue,
                                      task_out_queue=task_out_queue,
                                      mem_managers=mem_managers)
        manager.transfer_loop()

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
        
        self.transporter = NixlKVTransporter(node_id=self.args.pd_node_id,
                                             tp_idx=device_id,
                                             kv_move_buffer=None)
        self.waiting_dict_lock = threading.Lock()
        self.waiting_dict: Dict[str, NIXLChunckedTransTask] = {}

        self.ready_transfer_queue = queue.Queue()
        self.success_queue = queue.Queue()
        self.failed_queue = queue.Queue()

        self.page_index_queue = queue.Queue()
        for page_index in range(self.args.nixl_pd_kv_page_num):
            self.page_index_queue.put(page_index)

        self.update_status_thread = threading.Thread(target=self.update_task_status_loop, daemon=True)
        self.update_status_thread.start()
        return
    
    @log_exception
    def recv_task_loop(self):
        torch.cuda.set_device(self.device_id)

        while True:
            page_index = self.page_index_queue.get()
            trans_task: NIXLChunckedTransTask = self.task_in_queue.get()
            trans_task.nixl_src_page_index = page_index

            # 初次校验 time out
            if trans_task.time_out():
                trans_task.error_info = "time out"
                self.failed_queue.put(trans_task)
            else:
                self.ready_transfer_queue.put(trans_task)

    
    @log_exception
    def transfer_kv_loop(self):
        torch.cuda.set_device(self.device_id)
        while True:
            trans_task: NIXLChunckedTransTask = self.ready_transfer_queue.get()

            # to do 将kv 数据拷贝到 page 上，然后传输给 decode node，让其进行读取。
            pass
            self.transporter.send_readtask_to_decode_node(peer_name=trans_task.peer_agent_name, trans_task=trans_task)
            trans_task.start_trans_time = time.time()

            with self.waiting_dict_lock:
                self.waiting_dict[trans_task.get_key()] = trans_task
        return
    
    @log_exception
    def update_task_status_loop(
        self,
    ):
        while True:
            if len(self.waiting_dict) == 0:
                time.sleep(0.003)
                continue
            
            # notify update
            notifies_dict = self.transporter.get_new_notifs()
            if notifies_dict:
                for _, _notify_list in notifies_dict.items():
                    for notify in _notify_list:
                        try:
                            notify_obj = pickle.loads(notify)
                        except:
                            notify_obj = None
                    
                    if isinstance(notify_obj, NIXLChunckedTransTaskRet):
                        key = notify_obj.get_key()
                        with self.waiting_dict_lock:
                            trans_task = self.waiting_dict.pop(key, None)
                        
                        if trans_task is not None:
                            trans_task.error_info = notify_obj.error_info
                            if trans_task.error_info is not None:
                                self.failed_queue.put(trans_task)
                            else:
                                self.success_queue.put(trans_task)

            # check time_out update
            with self.waiting_dict_lock:
                iter_keys = list(self.waiting_dict.keys())
            
            for key in iter_keys:
                with self.waiting_dict_lock:
                    trans_task = self.waiting_dict.pop(key, None)
                
                if trans_task is not None and trans_task.time_out():
                    trans_task.error_info = "xfer time out"
                    self.failed_queue.put(trans_task)
                    continue
                
                if trans_task is not None:
                    with self.waiting_dict_lock:
                        self.waiting_dict[trans_task.get_key()] = trans_task


        @log_exception
    
    @log_exception
    def success_loop(self):
        torch.cuda.set_device(self.device_id)
        while True:
            trans_task: NIXLChunckedTransTask = self.success_queue.get()
            # 将数据写回 mem manger

            # 写回后，回收页面
            if trans_task.nixl_src_page_index is not None:
                self.page_index_queue.put(trans_task.nixl_src_page_index)
            
            ret = trans_task.createRetObj()
            self.task_out_queue.put(ret)
    
    @log_exception
    def fail_loop(self):
        torch.cuda.set_device(self.device_id)
        while True:
            trans_task: NIXLChunckedTransTask = self.failed_queue.get()

            # 回收页面
            if trans_task.nixl_src_page_index is not None:
                self.page_index_queue.put(trans_task.nixl_src_page_index)

            self.task_out_queue.put(trans_task.createRetObj())  