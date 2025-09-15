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
from lightllm.server.pd_io_struct import NIXLChunckedTransTask, NIXLChunckedTransTaskGroup, NIXLChunckedTransTaskRet, NixlUpKVStatus
from lightllm.server.pd_io_struct import NIXLDecodeNodeInfo
from lightllm.utils.device_utils import kv_trans_use_p2p
from lightllm.utils.graceful_utils import graceful_registry
from lightllm.server.core.objs import StartArgs
from ..nixl_kv_transporter import NixlKVTransporter
from lightllm.utils.error_utils import log_exception

logger = init_logger(__name__)


def start_decode_trans_process(
    args,
    device_id,
    task_in_queue: mp.Queue,
    task_out_queue: mp.Queue,
    mem_queues: List[mp.Queue],
    up_status_in_queue: Optional[mp.SimpleQueue],
):
    proc = mp.Process(target=_init_env, args=(args, device_id, task_in_queue, task_out_queue, mem_queues, up_status_in_queue))
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
    up_status_in_queue: Optional[mp.SimpleQueue],
):
    torch.backends.cudnn.enabled = False

    try:
        torch.cuda.set_device(device_id)
        graceful_registry(inspect.currentframe().f_code.co_name)

        dp_size_in_node = max(1, args.dp // args.nnodes)
        task_out_queue.put("proc_start")
        mem_managers: List[MemoryManager] = [mem_queue.get(timeout=60) for mem_queue in mem_queues]
        task_out_queue.put("get_mem_managers_ok")

        manager = _DecodeTransModule(args=args, 
                                     device_id=device_id,
                                    task_in_queue=task_in_queue,
                                    task_out_queue=task_out_queue,
                                    mem_managers=mem_managers,
                                    up_status_in_queue=up_status_in_queue)
        manager.transfer_loop()

    except Exception as e:
        logger.error(f"Fatal error happened in kv trans process: {e}")
        pass


class _DecodeTransModule:
    def __init__(
        self,
        args: StartArgs,
        device_id: int,
        task_in_queue: mp.Queue,
        task_out_queue: mp.Queue,
        mem_managers: List[mp.Queue],
        up_status_in_queue: Optional[mp.SimpleQueue]):
        self.args = args
        self.device_id = device_id
        self.task_in_queue = task_in_queue
        self.task_out_queue = task_out_queue
        self.mem_managers = mem_managers
        self.up_status_in_queue = up_status_in_queue
        
        self.transporter = NixlKVTransporter(node_id=self.args.pd_node_id,
                                             tp_idx=device_id,
                                             kv_move_buffer=None)
        self.waiting_dict_lock = threading.Lock()
        self.waiting_dict: Dict[str, NIXLChunckedTransTask] = {}
        self.read_kv_queue = queue.Queue()
        self.update_status_task_list_lock = threading.Lock()
        self.update_status_task_list: List[NIXLChunckedTransTask] = []
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
        while True:
            trans_task_group: NIXLChunckedTransTaskGroup = self.task_in_queue.get()
            task = trans_task_group.task_list[0]
            decode_node_info = NIXLDecodeNodeInfo(
                decode_node_id=self.args.pd_node_id,
                pd_master_node_id=task.pd_master_node_id,
                agent_name=self.transporter.agent_name,
                agent_metadata=self.transporter.agent_metadata,
                num_pages=self.transporter.num_pages,
                page_reg_desc=self.transporter.local_page_mem_desc,
                ready_kv_len=task.start_kv_index,
            )

            up_status = NixlUpKVStatus(
                group_request_id=task.request_id,
                pd_master_node_id=task.pd_master_node_id,
                nixl_params=pickle.dumps(decode_node_info)
            )

            self.up_status_in_queue.put(up_status)

            with self.waiting_dict_lock:
                for task in trans_task_group.task_list:
                    self.waiting_dict[task.get_key()] = task

    @log_exception
    def accept_peer_task_loop(
        self,
    ):
        torch.cuda.set_device(self.device_id)
        while True:
            
            # notify update
            notifies_dict = self.transporter.get_new_notifs()
            if not notifies_dict:
                time.sleep(0.005)
                continue


            for remote_agent_name, _notify_list in notifies_dict.items():
                for notify in _notify_list:
                    try:
                        notify_obj = pickle.loads(notify)
                    except:
                        notify_obj = None
                
                if isinstance(notify_obj, NIXLChunckedTransTask):
                    remote_trans_task = notify_obj
                    key = remote_trans_task.get_key()
                    with self.waiting_dict_lock:
                        local_trans_task = self.waiting_dict.pop(key, None)
                    
                    if local_trans_task is None:
                        self.transporter.send_notify_to_prefill_node(peer_name=remote_agent_name,
                                                                        notify=pickle.dumps(remote_trans_task.createRetObj(has_error=True,
                                                                                                                        error_info="decode node didnot find this task")))
                    else:
                        local_trans_task.nixl_src_page_index = remote_trans_task.nixl_src_page_index
                        self.read_kv_queue.put((remote_agent_name, local_trans_task))
    

    @log_exception
    def read_kv_loop(self):
        while True:

            remote_agent_name, local_trans_task = self.read_kv_queue.get()
            local_trans_task: NIXLChunckedTransTask = local_trans_task
            if local_trans_task.time_out():
                local_trans_task.error_info = "time out"
                self.failed_queue.put(local_trans_task)
                continue
            
            page_index = self.page_index_queue.get()
            local_trans_task.nixl_dst_page_index = page_index

            xfer_handle = self.transporter.read_blocks_paged(peer_name=remote_agent_name, 
                                                trans_task=local_trans_task)
            local_trans_task.xfer_handle = xfer_handle
            local_trans_task.start_trans_time = time.time()
            with self.update_status_task_list_lock:
                self.update_status_task_list.append(local_trans_task)


    @log_exception
    def update_task_status_loop(
        self,
    ):
        while True:
            if len(self.update_status_task_list) == 0:
                time.sleep(0.003)
                continue

            # check xfer state
            with self.update_status_task_list_lock:
                trans_taskes = self.update_status_task_list.copy()
                self.update_status_task_list.clear()
            
            for trans_task in trans_taskes:
                if trans_task.xfer_handle is not None:
                    ret = self.transporter.check_task_status(trans_task=trans_task)
                    if ret == "DONE":
                        self.success_queue.put(trans_task)
                        continue
                    elif ret == "ERR":
                        trans_task.error_info = "xfer error"
                        self.failed_queue.put(trans_task)
                        continue
                
                if trans_task.time_out():
                    trans_task.error_info = "time out"
                    self.failed_queue.put(trans_task)
                    continue

                with self.update_status_task_list_lock:
                    self.update_status_task_list.append(trans_task)
    
    @log_exception
    def success_loop(self):
        torch.cuda.set_device(self.device_id)
        while True:
            trans_task: NIXLChunckedTransTask = self.success_queue.get()
            # 将数据写回 mem manger

            # 写回后，回收页面
            self.page_index_queue.put(trans_task.nixl_dst_page_index)
            ret = trans_task.createRetObj()
            self.task_out_queue.put(ret)
    
    @log_exception
    def fail_loop(self):
        torch.cuda.set_device(self.device_id)
        while True:
            trans_task: NIXLChunckedTransTask = self.failed_queue.get()

            # 回收页面
            self.page_index_queue.put(trans_task.nixl_dst_page_index)
            self.task_out_queue.put(trans_task.createRetObj())