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
from lightllm.server.pd_io_struct import (
    NIXLChunckedTransTask,
    NIXLChunckedTransTaskGroup,
    NIXLChunckedTransTaskRet,
    NixlUpKVStatus,
    NIXLAbortReq,
)
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
    proc = mp.Process(
        target=_init_env, args=(args, device_id, task_in_queue, task_out_queue, mem_queues, up_status_in_queue)
    )
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

        task_out_queue.put("proc_start")
        mem_managers: List[MemoryManager] = [mem_queue.get(timeout=60) for mem_queue in mem_queues]
        task_out_queue.put("get_mem_managers_ok")

        manager = _DecodeTransModule(
            args=args,
            device_id=device_id,
            task_in_queue=task_in_queue,
            task_out_queue=task_out_queue,
            mem_managers=mem_managers,
            up_status_in_queue=up_status_in_queue,
        )
        assert manager is not None

        while True:
            time.sleep(100)

    except Exception as e:
        logger.exception(str(e))
        logger.error(f"Fatal error happened in kv trans process: {e}")
        pass


class _DecodeTransModule:
    def __init__(
        self,
        args: StartArgs,
        device_id: int,
        task_in_queue: mp.Queue,
        task_out_queue: mp.Queue,
        mem_managers: List[MemoryManager],
        up_status_in_queue: Optional[mp.SimpleQueue],
    ):
        self.args = args
        self.dp_world_size = self.args.tp // self.args.dp
        self.device_id = device_id
        self.task_in_queue = task_in_queue
        self.task_out_queue = task_out_queue
        self.mem_managers = mem_managers
        self.up_status_in_queue = up_status_in_queue
        cur_mem_manager: MemoryManager = self.mem_managers[device_id]
        kv_move_buffer = cur_mem_manager.alloc_paged_kv_move_buffer(
            page_num=self.args.nixl_pd_kv_page_num, page_size=self.args.nixl_pd_kv_page_size
        )
        self.copy_cuda_stream = torch.cuda.Stream()
        self.transporter = NixlKVTransporter(
            node_id=self.args.pd_node_id, tp_idx=device_id, kv_move_buffer=kv_move_buffer
        )
        self.recv_task_group_queue = queue.Queue()
        self.waiting_dict_lock = threading.Lock()
        self.waiting_dict: Dict[str, NIXLChunckedTransTask] = {}
        self.read_peer_kv_queue = queue.Queue()
        self.update_status_task_queue = queue.Queue()
        self.ready_page_task_queue = queue.Queue()
        self.success_queue = queue.Queue()
        self.failed_queue = queue.Queue()

        self.page_index_queue = queue.Queue()
        for page_index in range(self.args.nixl_pd_kv_page_num):
            self.page_index_queue.put(page_index)

        for func in [
            self.recv_task_loop,
            self.dispatch_task_loop,
            self.accept_peer_task_loop,
            self.read_peer_kv_loop,
            self.update_task_status_loop,
            self.read_page_to_mems_loop,
            self.success_loop,
            self.fail_loop,
        ]:
            threading.Thread(target=func, daemon=True).start()
        return

    @log_exception
    def recv_task_loop(self):
        while True:
            obj: Union[NIXLChunckedTransTaskGroup, NIXLAbortReq] = self.task_in_queue.get()
            if isinstance(obj, NIXLChunckedTransTaskGroup):
                self.recv_task_group_queue.put(obj)
            elif isinstance(obj, NIXLAbortReq):
                self._abort(cmd=obj)
            else:
                assert False, f"recv error obj {obj}"

    def _abort(self, cmd: NIXLAbortReq):
        # check time_out update
        with self.waiting_dict_lock:
            keys = list(self.waiting_dict.keys())

        for key in keys:
            with self.waiting_dict_lock:
                trans_task = self.waiting_dict.pop(key, None)

            if trans_task is not None and trans_task.request_id == cmd.request_id:
                trans_task.error_info = "aborted req"
                self.failed_queue.put(trans_task)
                continue

            if trans_task is not None:
                with self.waiting_dict_lock:
                    self.waiting_dict[trans_task.get_key()] = trans_task
        return

    @log_exception
    def dispatch_task_loop(self):
        while True:
            trans_task_group: NIXLChunckedTransTaskGroup = self.recv_task_group_queue.get()

            with self.waiting_dict_lock:
                for task in trans_task_group.task_list:
                    if task.transfer_kv_num() != 0:
                        self.waiting_dict[task.get_key()] = task
                    else:
                        task.start_trans_time = time.time()
                        self.success_queue.put((None, task))

            # up status
            task = trans_task_group.task_list[0]

            decode_node_info = NIXLDecodeNodeInfo(
                decode_node_id=self.args.pd_node_id,
                pd_master_node_id=task.pd_master_node_id,
                agent_name=self.transporter.agent_name,
                agent_metadata=self.transporter.agent_metadata,
                num_pages=self.transporter.num_pages,
                page_reg_desc=self.transporter.local_page_mem_desc,
                request_id=task.request_id,
                ready_kv_len=task.start_kv_index,
            )

            up_status = NixlUpKVStatus(
                group_request_id=task.request_id,
                pd_master_node_id=task.pd_master_node_id,
                nixl_params=pickle.dumps(decode_node_info),
            )

            self.up_status_in_queue.put(up_status)

    @log_exception
    def accept_peer_task_loop(
        self,
    ):
        torch.cuda.set_device(self.device_id)
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
                for remote_agent_name, _notify_list in notifies_dict.items():
                    for notify in _notify_list:
                        try:
                            notify_obj = pickle.loads(notify)
                        except:
                            notify_obj = None

                        if isinstance(notify_obj, NIXLChunckedTransTask):
                            remote_trans_task = notify_obj
                            key = remote_trans_task.get_key()
                            logger.info(f"recv peer trans task {remote_trans_task.to_str()}")
                            with self.waiting_dict_lock:
                                local_trans_task: NIXLChunckedTransTask = self.waiting_dict.pop(key, None)

                            if local_trans_task is None:
                                remote_trans_task.error_info = "peer not find"
                                try:
                                    self.transporter.send_notify_to_prefill_node(
                                        prefill_agent_name=remote_agent_name,
                                        notify=pickle.dumps(remote_trans_task.createRetObj()),
                                    )
                                except BaseException as e:
                                    logger.error(f"send notify to prefill node failed: {str(e)}")
                                    logger.exception(str(e))
                                    self.transporter.remove_remote_agent(peer_name=remote_agent_name)
                            else:
                                local_trans_task.nixl_src_page_index = remote_trans_task.nixl_src_page_index

                                local_trans_task.prefill_agent_name = remote_trans_task.prefill_agent_name
                                local_trans_task.prefill_agent_metadata = remote_trans_task.prefill_agent_metadata
                                local_trans_task.prefill_num_pages = remote_trans_task.prefill_num_pages
                                local_trans_task.prefill_page_reg_desc = remote_trans_task.prefill_page_reg_desc

                                local_trans_task.first_gen_token_id = remote_trans_task.first_gen_token_id
                                local_trans_task.first_gen_token_logprob = remote_trans_task.first_gen_token_logprob

                                self.read_peer_kv_queue.put(local_trans_task)

            self._check_tasks_time_out()

    def _check_tasks_time_out(self):
        # check time_out update
        with self.waiting_dict_lock:
            keys = list(self.waiting_dict.keys())

        for key in keys:
            with self.waiting_dict_lock:
                trans_task = self.waiting_dict.pop(key, None)

            if trans_task is not None and trans_task.time_out():
                trans_task.error_info = "time out in accept_peer_task_loop"
                self.failed_queue.put(trans_task)
                continue

            if trans_task is not None:
                with self.waiting_dict_lock:
                    self.waiting_dict[trans_task.get_key()] = trans_task
        return

    @log_exception
    def read_peer_kv_loop(self):
        torch.cuda.set_device(self.device_id)
        while True:
            page_index = self.page_index_queue.get()
            local_trans_task = self.read_peer_kv_queue.get()
            local_trans_task: NIXLChunckedTransTask = local_trans_task
            local_trans_task.nixl_dst_page_index = page_index

            if local_trans_task.time_out():
                local_trans_task.error_info = "time out in read_peer_kv_loop"
                self.failed_queue.put(local_trans_task)
                continue

            try:
                xfer_handle = self.transporter.read_blocks_paged(trans_task=local_trans_task)
                local_trans_task.xfer_handle = xfer_handle
                local_trans_task.start_trans_time = time.time()
                self.update_status_task_queue.put(local_trans_task)
            except BaseException as e:
                logger.error(f"read_blocks_paged node failed: {local_trans_task.to_str()}")
                logger.exception(str(e))
                self.transporter.remove_remote_agent(peer_name=local_trans_task.prefill_agent_name)
                local_trans_task.error_info = f"read_blocks_paged failed: {str(e)}"
                self.failed_queue.put(local_trans_task)
                continue

    @log_exception
    def update_task_status_loop(
        self,
    ):
        while True:
            trans_task: NIXLChunckedTransTask = self.update_status_task_queue.get()

            while True:
                ret = self.transporter.check_task_status(trans_task=trans_task)
                if ret == "DONE":
                    self.ready_page_task_queue.put(trans_task)
                    break
                elif ret == "ERR":
                    trans_task.error_info = "xfer error"
                    self.failed_queue.put(trans_task)
                    break
                elif trans_task.time_out():
                    trans_task.error_info = "time out in update_task_status_loop"
                    self.failed_queue.put(trans_task)
                    break

                time.sleep(0.001)

    @log_exception
    def read_page_to_mems_loop(self):
        torch.cuda.set_device(self.device_id)
        while True:
            trans_task: NIXLChunckedTransTask = self.ready_page_task_queue.get()
            # 将数据写回 mem manger
            with torch.cuda.stream(stream=self.copy_cuda_stream):
                cur_mem = self.mem_managers[self.device_id]
                cur_mem.read_page_kv_move_buffer_to_mem(
                    mem_indexes=trans_task.mem_indexes,
                    page_index=trans_task.nixl_dst_page_index,
                    dp_index=trans_task.decode_dp_index,
                    mem_managers=self.mem_managers,
                    dp_world_size=self.dp_world_size,
                )
                sync_event = torch.cuda.Event()
                sync_event.record()

            self.success_queue.put((sync_event, trans_task))
        return

    @log_exception
    def success_loop(self):
        torch.cuda.set_device(self.device_id)
        while True:
            sync_event, trans_task = self.success_queue.get()
            trans_task: NIXLChunckedTransTask = trans_task
            sync_event: Optional[torch.cuda.Event] = sync_event
            # 兼容传输kv 数量为0的时候， sync_event 为 None的情况。
            if sync_event is not None:
                sync_event.synchronize()

            if trans_task.nixl_dst_page_index is not None:
                self.page_index_queue.put(trans_task.nixl_dst_page_index)

            if trans_task.xfer_handle is not None:
                self.transporter.release_xfer_handle(trans_task.xfer_handle)

            ret = trans_task.createRetObj()
            self.task_out_queue.put(ret)
            logger.info(f"trans task ret success:{ret} cost time: {trans_task.transfer_time()} s")

    @log_exception
    def fail_loop(self):
        torch.cuda.set_device(self.device_id)
        while True:
            trans_task: NIXLChunckedTransTask = self.failed_queue.get()

            # 回收页面
            if trans_task.nixl_dst_page_index is not None:
                self.page_index_queue.put(trans_task.nixl_dst_page_index)
            if trans_task.xfer_handle is not None:
                self.transporter.release_xfer_handle(trans_task.xfer_handle)
            ret = trans_task.createRetObj()
            self.task_out_queue.put(ret)
            logger.info(f"trans task ret fail:{ret}")
