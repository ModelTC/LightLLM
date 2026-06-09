import torch
import time
import inspect
import threading
import setproctitle
import torch.multiprocessing as mp
import queue
import pickle
from typing import List, Dict, Optional
from lightllm.utils.log_utils import init_logger
from lightllm.common.kv_cache_mem_manager import MemoryManager
from lightllm.server.pd_io_struct import NIXLChunckedTransTask
from lightllm.utils.graceful_utils import graceful_registry
from lightllm.server.core.objs import StartArgs
from ..nixl_kv_transporter import NixlKVTransporter
from lightllm.utils.error_utils import log_exception
from lightllm.utils.envs_utils import get_unique_server_name


logger = init_logger(__name__)


def start_prefill_trans_process(
    args,
    device_id,
    task_in_queue: mp.Queue,
    task_out_queue: mp.Queue,
    up_status_in_queue: Optional[mp.SimpleQueue] = None,
):
    proc = mp.Process(target=_init_env, args=(args, device_id, task_in_queue, task_out_queue))
    proc.start()
    assert proc.is_alive()
    logger.info(f"prefill trans kv process for device: {device_id} started!")
    return proc


def _init_env(
    args: StartArgs,
    device_id: int,
    task_in_queue: mp.Queue,
    task_out_queue: mp.Queue,
):

    import os

    # prefill source-side page copy and UCX progress are on the request critical path.
    os.environ["CUDA_MPS_CLIENT_PRIORITY"] = "0"

    torch.backends.cudnn.enabled = False
    setproctitle.setproctitle(f"lightllm::{get_unique_server_name()}::nixl_prefill_trans:Device{device_id}")

    try:
        torch.cuda.set_device(device_id)
        graceful_registry(inspect.currentframe().f_code.co_name)
        task_out_queue.put("proc_start")

        # 从共享内存读取所有rank的mem_manager
        node_world_size = args.tp // args.nnodes
        mem_managers: List[MemoryManager] = [
            MemoryManager.loads_from_shm(rank_in_node=rank) for rank in range(node_world_size)
        ]

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
        self.copy_cuda_stream = torch.cuda.Stream(priority=-1)
        self.transporter = NixlKVTransporter(
            node_id=self.args.pd_node_id, tp_idx=device_id, kv_move_buffer=kv_move_buffer
        )
        self.waiting_dict_lock = threading.Lock()
        self.waiting_dict: Dict[str, NIXLChunckedTransTask] = {}

        self.local_copy_kv_queue = queue.Queue()
        self.ready_transfer_queue = queue.Queue()
        self.write_peer_kv_queue = queue.Queue()
        self.success_queue = queue.Queue()
        self.failed_queue = queue.Queue()

        self.page_index_queue = queue.Queue()
        for page_index in range(self.args.nixl_pd_kv_page_num):
            self.page_index_queue.put(page_index)

        # warmup 预先执行一次 kv 写入 page buffer，避免第一次拷贝时出现卡顿。
        self._warmup()

        for func in [
            self.recv_task_loop,
            self.local_copy_kv_loop,
            self.ready_transfer_loop,
            self.accept_decode_write_task_loop,
            self.write_peer_kv_loop,
            self.update_task_status_loop,
            self.success_loop,
            self.fail_loop,
        ]:
            threading.Thread(target=func, daemon=True).start()
        return

    def _warmup(self):
        for dp_index in range(self.args.dp // self.args.nnodes):
            with torch.cuda.stream(stream=self.copy_cuda_stream):
                cur_mem = self.mem_managers[self.device_id]
                cur_mem.write_mem_to_page_kv_move_buffer(
                    mem_indexes=[0],
                    page_index=0,
                    dp_index=dp_index,
                    mem_managers=self.mem_managers,
                    dp_world_size=self.dp_world_size,
                )
                torch.cuda.current_stream().synchronize()
        return

    def _abort(self, request_id: int, error_info: str = "aborted req"):
        aborted_tasks = []
        with self.waiting_dict_lock:
            for key, trans_task in list(self.waiting_dict.items()):
                if trans_task.request_id == request_id:
                    aborted_tasks.append(self.waiting_dict.pop(key))

        for trans_task in aborted_tasks:
            trans_task.error_info = error_info
            self.failed_queue.put(trans_task)
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
                    page_kind=trans_task.page_kind,
                    req_idx=trans_task.req_idx,
                )
                sync_event = torch.cuda.Event()
                sync_event.record()

            self.ready_transfer_queue.put((sync_event, trans_task))
        return

    @log_exception
    def ready_transfer_loop(self):
        torch.cuda.set_device(self.device_id)
        while True:
            sync_event, trans_task = self.ready_transfer_queue.get()
            trans_task: NIXLChunckedTransTask = trans_task
            sync_event: torch.cuda.Event = sync_event
            sync_event.synchronize()
            key = trans_task.get_key()
            try:
                with self.waiting_dict_lock:
                    self.waiting_dict[key] = trans_task
                self.transporter.send_write_request_task_to_decode_node(trans_task)
                logger.info(f"send WRITE request to decode: {key}")
            except BaseException as e:
                with self.waiting_dict_lock:
                    self.waiting_dict.pop(key, None)
                logger.error(f"send WRITE request to decode failed: {trans_task.to_str()}")
                logger.exception(str(e))
                trans_task.error_info = f"send WRITE request to decode failed: {str(e)}"
                self.transporter.remove_remote_agent(peer_name=trans_task.decode_agent_name)
                self.failed_queue.put(trans_task)
                continue
        return

    @log_exception
    def accept_decode_write_task_loop(self):
        while True:
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
                        except BaseException:
                            notify_obj = None

                        if not isinstance(notify_obj, NIXLChunckedTransTask):
                            continue

                        if notify_obj.error_info is not None:
                            logger.warning(f"recv WRITE error from decode: {notify_obj.to_str()}")
                            self._abort(request_id=notify_obj.request_id, error_info=notify_obj.error_info)
                            continue

                        if notify_obj.nixl_write_stage == "ready":
                            key = notify_obj.get_key()
                            with self.waiting_dict_lock:
                                trans_task = self.waiting_dict.pop(key, None)
                            if trans_task is not None:
                                trans_task.nixl_dst_page_index = notify_obj.nixl_dst_page_index
                                self.write_peer_kv_queue.put(trans_task)
                                logger.info(
                                    f"recv WRITE ready from decode request_id={trans_task.request_id} "
                                    f"kv=[{trans_task.start_kv_index},{trans_task.end_kv_index}) "
                                    f"srcpage={trans_task.nixl_src_page_index} dstpage={trans_task.nixl_dst_page_index}"
                                )
                            else:
                                logger.warning(
                                    f"can not find pending WRITE request for ready notify: {notify_obj.to_str()}"
                                )
                                # 发一个error信息回去给 decode 节点，让其可以知道这边有问题了，它可以选择其他清理掉请求。
                                notify_obj.error_info = "can not find pending WRITE request for ready notify"
                                self.transporter.send_error_info_to_decode_node(trans_task=notify_obj)
                            continue
                        else:
                            logger.error(f"ignore unknown WRITE notify stage: {notify_obj.to_str()}")
                            continue

            self._check_tasks_time_out()

            if not notifies_dict:
                time.sleep(0.001)
        return

    def _check_tasks_time_out(self):
        with self.waiting_dict_lock:
            timeout_tasks = []
            for key, trans_task in list(self.waiting_dict.items()):
                if trans_task.time_out():
                    timeout_tasks.append(self.waiting_dict.pop(key))

        for trans_task in timeout_tasks:
            trans_task.error_info = "time out waiting decode WRITE ready"
            self.failed_queue.put(trans_task)
        return

    @log_exception
    def write_peer_kv_loop(self):
        torch.cuda.set_device(self.device_id)
        while True:
            trans_task = self.write_peer_kv_queue.get()
            trans_task: NIXLChunckedTransTask = trans_task

            try:
                xfer_handle = self.transporter.write_blocks_paged(trans_task=trans_task)
                trans_task.xfer_handle = xfer_handle
                trans_task.start_trans_time = time.time()
                with self.waiting_dict_lock:
                    self.waiting_dict[trans_task.get_key()] = trans_task
                logger.info(f"start WRITE to decode node: {trans_task.to_str()}")
                continue
            except BaseException as e:
                logger.error(f"write_blocks_paged failed: {trans_task.to_str()}")
                logger.exception(str(e))
                self.transporter.remove_remote_agent(peer_name=trans_task.decode_agent_name)
                trans_task.error_info = f"write_blocks_paged failed: {str(e)}"
                self.failed_queue.put(trans_task)
                continue
        return

    @log_exception
    def update_task_status_loop(
        self,
    ):
        while True:
            if len(self.waiting_dict) == 0:
                time.sleep(0.001)
                continue

            with self.waiting_dict_lock:
                tasks = list(self.waiting_dict.values())
                for trans_task in tasks:
                    if trans_task.xfer_handle is None:
                        continue

                    # 传输任务状态检查
                    ret = self.transporter.check_task_status(trans_task=trans_task)
                    if ret == "DONE":
                        trans_task = self.waiting_dict.pop(trans_task.get_key(), None)
                        if self.transporter.capture_telemetry:
                            telem = self.transporter.nixl_agent.get_xfer_telemetry(trans_task.xfer_handle)
                            total_us = telem.xferDuration
                            post_us = telem.postDuration
                            backend_us = telem.xferDuration - telem.postDuration
                            nixl_backend = self.transporter.nixl_agent.query_xfer_backend(trans_task.xfer_handle)
                            logger.info(
                                f"write trans task request_id={trans_task.request_id} "
                                f"kv=[{trans_task.start_kv_index},{trans_task.end_kv_index}) "
                                f"src_page={trans_task.nixl_src_page_index} dst_page={trans_task.nixl_dst_page_index} "
                                f"xfer time: {total_us:.3f} us, "
                                f"post time: {post_us:.3f} us, backend time: {backend_us:.3f} us, "
                                f"nixl_backend: {nixl_backend}, total_bytes: {telem.totalBytes}"
                            )
                        self.transporter.send_write_done_task_to_decode_node(trans_task)
                        logger.info(
                            f"send WRITE done nixl notify "
                            f"request_id={trans_task.request_id} "
                            f"kv=[{trans_task.start_kv_index},{trans_task.end_kv_index}) "
                            f"src_page={trans_task.nixl_src_page_index} dst_page={trans_task.nixl_dst_page_index}"
                        )
                        self.success_queue.put(trans_task)
                    elif ret == "ERR":
                        trans_task = self.waiting_dict.pop(trans_task.get_key(), None)
                        trans_task.error_info = "xfer error"
                        self.failed_queue.put(trans_task)
                    elif trans_task.time_out():
                        trans_task = self.waiting_dict.pop(trans_task.get_key(), None)
                        trans_task.error_info = "time out in update_task_status_loop"
                        self.failed_queue.put(trans_task)

            time.sleep(0.001)

    @log_exception
    def success_loop(self):
        torch.cuda.set_device(self.device_id)
        while True:
            trans_task: NIXLChunckedTransTask = self.success_queue.get()
            # 写回后，回收页面
            if trans_task.nixl_src_page_index is not None:
                self.page_index_queue.put(trans_task.nixl_src_page_index)
            if trans_task.xfer_handle is not None:
                self.transporter.release_xfer_handle(trans_task.xfer_handle)

            ret = trans_task.createRetObj()
            ret.first_gen_token_id = None
            ret.first_gen_token_logprob = None
            self.task_out_queue.put(ret)

            if trans_task.start_trans_time is not None:
                logger.info(f"trans task ret success:{ret} cost time: {trans_task.transfer_time()}s")
            else:
                logger.info(f"trans task ret success:{ret}")

    @log_exception
    def fail_loop(self):
        torch.cuda.set_device(self.device_id)
        while True:
            trans_task: NIXLChunckedTransTask = self.failed_queue.get()

            # 回收页面
            if trans_task.nixl_src_page_index is not None:
                self.page_index_queue.put(trans_task.nixl_src_page_index)
            if trans_task.xfer_handle is not None:
                self.transporter.release_xfer_handle(trans_task.xfer_handle)

            ret = trans_task.createRetObj()
            self.task_out_queue.put(ret)
            logger.info(f"trans task ret fail:{ret}")

            if trans_task.error_info is not None:
                self._abort(request_id=trans_task.request_id, error_info=trans_task.error_info)
                self.transporter.send_error_info_to_decode_node(trans_task=trans_task)
