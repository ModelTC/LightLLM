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
    up_status_in_queue: Optional[mp.SimpleQueue] = None,
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
        task_out_queue.put("proc_start")
        mem_managers: List[MemoryManager] = [mem_queue.get(timeout=60) for mem_queue in mem_queues]
        task_out_queue.put("get_mem_managers_ok")

        manager = _PrefillTransModule(
            args=args,
            device_id=device_id,
            task_in_queue=task_in_queue,
            task_out_queue=task_out_queue,
            mem_managers=mem_managers,
        )
        assert manager is not None

        while True:
            time.sleep(100)

    except Exception as e:
        logger.exception(str(e))
        logger.error(f"Fatal error happened in kv trans process: {e}")
        pass


class _PrefillTransModule:
    def __init__(
        self,
        args: StartArgs,
        device_id: int,
        task_in_queue: mp.Queue,
        task_out_queue: mp.Queue,
        mem_managers: List[MemoryManager],
    ) -> None:
        self.args = args
        self.dp_world_size = self.args.tp // self.args.dp
        self.device_id = device_id
        self.task_in_queue = task_in_queue
        self.task_out_queue = task_out_queue
        self.mem_managers = mem_managers

        cur_mem_manager: MemoryManager = self.mem_managers[device_id]
        kv_move_buffer = cur_mem_manager.alloc_paged_kv_move_buffer(
            page_num=self.args.nixl_pd_kv_page_num, page_size=self.args.nixl_pd_kv_page_size
        )
        self.copy_cuda_stream = torch.cuda.Stream()
        self.transporter = NixlKVTransporter(
            node_id=self.args.pd_node_id, tp_idx=device_id, kv_move_buffer=kv_move_buffer
        )
        self.waiting_dict_lock = threading.Lock()
        self.waiting_dict: Dict[str, NIXLChunckedTransTask] = {}

        self.local_copy_kv_queue = queue.Queue()
        self.notify_peer_read_kv_queue = queue.Queue()
        self.success_queue = queue.Queue()
        self.failed_queue = queue.Queue()

        self.page_index_queue = queue.Queue()
        for page_index in range(self.args.nixl_pd_kv_page_num):
            self.page_index_queue.put(page_index)

        for func in [
            self.recv_task_loop,
            self.local_copy_kv_loop,
            self.notify_peer_to_read_kv_loop,
            self.update_task_status_loop,
            self.success_loop,
            self.fail_loop,
        ]:
            threading.Thread(target=func, daemon=True).start()
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
                trans_task.error_info = "time out in recv_task_loop"
                self.failed_queue.put(trans_task)
            else:
                self.local_copy_kv_queue.put(trans_task)

    @log_exception
    def local_copy_kv_loop(self):
        torch.cuda.set_device(self.device_id)
        while True:
            trans_task: NIXLChunckedTransTask = self.local_copy_kv_queue.get()

            # 将kv 数据拷贝到 page 上，然后传输给 decode node，让其进行读取。
            with torch.cuda.stream(stream=self.copy_cuda_stream):
                cur_mem = self.mem_managers[self.device_id]
                cur_mem.write_mem_to_page_kv_move_buffer(
                    trans_task.mem_indexes,
                    page_index=trans_task.nixl_src_page_index,
                    dp_index=trans_task.prefill_dp_index,
                    mem_managers=self.mem_managers,
                    dp_world_size=self.dp_world_size,
                )
                sync_event = torch.cuda.Event()
                sync_event.record()

            self.notify_peer_read_kv_queue.put((sync_event, trans_task))
        return

    @log_exception
    def notify_peer_to_read_kv_loop(self):
        torch.cuda.set_device(self.device_id)
        while True:
            sync_event, trans_task = self.notify_peer_read_kv_queue.get()
            trans_task: NIXLChunckedTransTask = trans_task
            sync_event: torch.cuda.Event = sync_event

            sync_event.synchronize()

            trans_task.start_trans_time = time.time()
            with self.waiting_dict_lock:
                self.waiting_dict[trans_task.get_key()] = trans_task

            try:
                self.transporter.send_readtask_to_decode_node(trans_task=trans_task)
            except BaseException as e:
                logger.error(f"send readtask to decode node failed: {trans_task.to_str()}")
                logger.exception(str(e))
                self.transporter.remove_remote_agent(peer_name=trans_task.decode_agent_name)

                with self.waiting_dict_lock:
                    trans_task = self.waiting_dict.pop(trans_task.get_key(), None)

                if trans_task is not None:
                    trans_task.error_info = f"send readtask to decode node failed: {str(e)}"
                    self.failed_queue.put(trans_task)
                continue

            logger.info(f"send readtask to decode: {trans_task.to_str()}")
        return

    @log_exception
    def update_task_status_loop(
        self,
    ):
        while True:
            if len(self.waiting_dict) == 0:
                time.sleep(0.001)
                continue

            # notify update
            try:
                notifies_dict = self.transporter.get_new_notifs()
            except BaseException as e:
                logger.error(f"get new notifies failed: {str(e)}")
                logger.exception(str(e))
                notifies_dict = {}

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
                            else:
                                logger.warning(f"can not find trans task for ret: {notify_obj}")

            # check time_out update
            self._check_tasks_time_out()

    def _check_tasks_time_out(self):
        with self.waiting_dict_lock:
            keys = list(self.waiting_dict.keys())

        for key in keys:
            with self.waiting_dict_lock:
                trans_task = self.waiting_dict.pop(key, None)

            if trans_task is not None and trans_task.time_out():
                trans_task.error_info = "time out in update_task_status_loop"
                self.failed_queue.put(trans_task)
                continue

            if trans_task is not None:
                with self.waiting_dict_lock:
                    self.waiting_dict[trans_task.get_key()] = trans_task
        return

    @log_exception
    def success_loop(self):
        torch.cuda.set_device(self.device_id)
        while True:
            trans_task: NIXLChunckedTransTask = self.success_queue.get()
            # 写回后，回收页面
            if trans_task.nixl_src_page_index is not None:
                self.page_index_queue.put(trans_task.nixl_src_page_index)

            ret = trans_task.createRetObj()
            ret.first_gen_token_id = None
            ret.first_gen_token_logprob = None
            self.task_out_queue.put(ret)
            logger.info(f"trans task ret success:{ret} cost time: {trans_task.transfer_time()}s")

    @log_exception
    def fail_loop(self):
        torch.cuda.set_device(self.device_id)
        while True:
            trans_task: NIXLChunckedTransTask = self.failed_queue.get()

            # 回收页面
            if trans_task.nixl_src_page_index is not None:
                self.page_index_queue.put(trans_task.nixl_src_page_index)

            ret = trans_task.createRetObj()
            self.task_out_queue.put(ret)
            logger.info(f"trans task ret fail:{ret}")
