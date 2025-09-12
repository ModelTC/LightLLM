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

        self.page_index_queue = queue.Queue()
        for page_index in range(self.args.nixl_pd_kv_page_num):
            self.page_index_queue.put(page_index)

        self.update_status_thread = threading.Thread(target=self.update_task_status_loop, daemon=True)
        self.update_status_thread.start()
        return

    def transfer_loop(self):
        try:
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

        except BaseException as e:
            logger.exception(str(e))
            raise e

    def trans_kv_loop(
        self,
    ):
        torch.cuda.set_device(self.device_id)
        try:
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
                            local_trans_task = self.waiting_dict.get(key, None)
                        
                        if local_trans_task is None:
                            self.transporter.send_notify_to_prefill_node(peer_name=remote_agent_name,
                                                                            notify=pickle.dumps(remote_trans_task.createRetObj(has_error=True,
                                                                                                                            error_info="decode node didnot find this task")))
                            continue

                        # to do 发起传输
                        page_index = self.page_index_queue.get()
                        local_trans_task.nixl_dst_page_index = page_index
                        local_trans_task.nixl_src_page_index = remote_trans_task.nixl_src_page_index

                        self.transporter.read_blocks_paged(peer_name=remote_agent_name, 
                                                            trans_task=local_trans_task)
                        local_trans_task.start_trans_time = time.time()

        except BaseException as e:
            logger.exception(str(e))
            raise e
        
    def update_task_status_loop(
        self,
    ):
        try:
            while True:
                if len(self.waiting_dict) == 0:
                    time.sleep(0.01)
                    continue

                # check xfer state
                with self.waiting_dict_lock:
                    for key, trans_task in self.waiting_dict.items():


                # check time_out update
                with self.waiting_dict_lock:
                    del_keys = []
                    for key, trans_task in self.waiting_dict.items():
                        if trans_task.time_out():
                            del_keys.append(key)

                    for key in del_keys:
                        trans_task = self.waiting_dict.pop(key, None)
                        if trans_task is not None:
                            self._create_error_ret(trans_task=trans_task, error_info="time out")
                            if trans_task.nixl_dst_page_index is not None:
                                self.page_index_queue.put(trans_task.nixl_dst_page_index)
                            if trans_task.xfer_handle is not None:
                                self.transporter.xxx

        except BaseException as e:
            logger.exception(str(e))
            raise e

    def _create_error_ret(self, trans_task: NIXLChunckedTransTask, error_info=""):
        ret_obj = trans_task.createRetObj(
                has_error=True,
                error_info=error_info
            )
        self.task_out_queue.put(ret_obj)
        logger.error(f"trans error in device id {self.device_id}: info {ret_obj}")
        return

    def _create_success_ret(self, trans_task: NIXLChunckedTransTask):
        ret_obj = trans_task.createRetObj(
            has_error=False,
            error_info=None,
        )
        self.task_out_queue.put(ret_obj)
        logger.info(f"trans success in device id {self.device_id}: info {ret_obj}")
        return