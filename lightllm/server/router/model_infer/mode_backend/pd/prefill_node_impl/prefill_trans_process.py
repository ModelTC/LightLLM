from lightllm.server.router.model_infer.mode_backend.pd.cache_page_io import create_cache_page_io
import torch
import time
import inspect
import threading
import setproctitle
import torch.multiprocessing as mp
import queue
import pickle
from typing import List, Dict, Optional, Union
from lightllm.common.kv_transfer_resources import KVTransferResources
from lightllm.utils.log_utils import init_logger
from lightllm.common.kv_transfer_resources import KVTransferView
from lightllm.server.pd_io_struct import PDAbortReq, PDChunckedTransTask
from lightllm.utils.graceful_utils import graceful_registry
from lightllm.server.core.objs import StartArgs
from ..kv_transporter import create_kv_transporter
from lightllm.utils.error_utils import log_exception
from lightllm.utils.envs_utils import get_unique_server_name
from lightllm.utils.process_check import install_fatal_thread_excepthook, start_parent_check_thread


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
    install_fatal_thread_excepthook()
    start_parent_check_thread()
    import lightllm.utils.rpyc_fix_utils as _

    import os

    # prefill source-side page copy and UCX progress are on the request critical path.
    os.environ["CUDA_MPS_CLIENT_PRIORITY"] = "0"

    torch.backends.cudnn.enabled = False
    setproctitle.setproctitle(f"lightllm::{get_unique_server_name()}::prefill_trans:Device{device_id}")

    try:
        torch.cuda.set_device(device_id)
        graceful_registry(inspect.currentframe().f_code.co_name)
        task_out_queue.put("proc_start")

        # 从共享内存读取所有rank的mem_manager
        node_world_size = args.tp // args.nnodes
        resources = [KVTransferResources.loads_from_shm(rank_in_node=rank) for rank in range(node_world_size)]

        kv_views = [resource.kv for resource in resources]
        state_transfers = [resource.state_transfer for resource in resources]

        task_out_queue.put("get_mem_managers_ok")

        manager = _PrefillTransModule(
            args=args,
            device_id=device_id,
            task_in_queue=task_in_queue,
            task_out_queue=task_out_queue,
            kv_views=kv_views,
            state_transfers=state_transfers,
        )
        assert manager is not None
        task_out_queue.put("module_ready")

        while True:
            time.sleep(100)

    except Exception as e:
        task_out_queue.put("init_failed")
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
        kv_views: List[KVTransferView],
        state_transfers,
    ) -> None:
        self.args = args
        self.dp_world_size = self.args.tp // self.args.dp
        self.device_id = device_id
        self.task_in_queue = task_in_queue
        self.task_out_queue = task_out_queue
        self.kv_views = kv_views
        self.state_transfers = state_transfers

        self.page_io = create_cache_page_io(self.kv_views, self.state_transfers, device_id, self.dp_world_size)
        kv_move_buffer = self.page_io.alloc_buffer(
            page_num=self.args.pd_kv_page_num, page_size=self.args.pd_kv_page_size
        )
        self.copy_cuda_stream = torch.cuda.Stream(priority=-1)
        self.transporter = create_kv_transporter(
            args=self.args,
            node_id=self.args.pd_node_id,
            tp_idx=device_id,
            kv_move_buffer=kv_move_buffer,
        )
        self.waiting_dict_lock = threading.Lock()
        self.waiting_dict: Dict[str, PDChunckedTransTask] = {}

        self.local_copy_kv_queue = queue.Queue()
        self.ready_transfer_queue = queue.Queue()
        self.write_peer_kv_queue = queue.Queue()
        self.success_queue = queue.Queue()
        self.failed_queue = queue.Queue()

        self.page_index_queue = queue.Queue()
        for page_index in range(self.args.pd_kv_page_num):
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
        with torch.cuda.stream(stream=self.copy_cuda_stream):
            self.page_io.warmup(write=True)
            torch.cuda.current_stream().synchronize()

    @log_exception
    def recv_task_loop(self):
        torch.cuda.set_device(self.device_id)

        while True:
            obj: Union[PDChunckedTransTask, PDAbortReq] = self.task_in_queue.get()
            if isinstance(obj, PDAbortReq):
                # Infer 层已经终止该请求，将 abort 命令下传到 P 节点的
                # KV 传输层，尽快结束尚在等待的传输任务。
                self._abort(request_id=obj.request_id)
                continue

            assert isinstance(obj, PDChunckedTransTask), f"unsupported prefill transfer task type: {type(obj)}"

            page_index = self.page_index_queue.get()
            trans_task = obj
            trans_task.src_page_index = page_index

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
            trans_task: PDChunckedTransTask = self.local_copy_kv_queue.get()

            # 将kv 数据拷贝到 page 上，然后传输给 decode node，让其进行读取。
            with torch.cuda.stream(stream=self.copy_cuda_stream):
                trans_task.transfer_nbytes = self.page_io.pack(trans_task)
                sync_event = torch.cuda.Event()
                sync_event.record()

            self.ready_transfer_queue.put((sync_event, trans_task))
        return

    @log_exception
    def ready_transfer_loop(self):
        torch.cuda.set_device(self.device_id)
        while True:
            sync_event, trans_task = self.ready_transfer_queue.get()
            trans_task: PDChunckedTransTask = trans_task
            sync_event: torch.cuda.Event = sync_event
            sync_event.synchronize()
            key = trans_task.get_key()
            try:
                with self.waiting_dict_lock:
                    self.waiting_dict[key] = trans_task
                self.transporter.send_write_request_task_to_decode_node(trans_task)
                if self.args.pd_detail_log:
                    logger.debug(f"send WRITE request to decode: {key}")
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

                        if not isinstance(notify_obj, PDChunckedTransTask):
                            continue

                        if notify_obj.error_info is not None:
                            logger.warning(f"recv WRITE error from decode: {notify_obj.to_str()}")
                            self._abort(request_id=notify_obj.request_id, error_info=notify_obj.error_info)
                            continue

                        if notify_obj.write_stage == "ready":
                            key = notify_obj.get_key()
                            with self.waiting_dict_lock:
                                trans_task = self.waiting_dict.pop(key, None)
                            if trans_task is not None:
                                trans_task.dst_page_index = notify_obj.dst_page_index
                                self.write_peer_kv_queue.put(trans_task)
                                if self.args.pd_detail_log:
                                    logger.debug(
                                        f"recv WRITE ready from decode request_id={trans_task.request_id} "
                                        f"kv=[{trans_task.start_kv_index},{trans_task.end_kv_index}) "
                                        f"srcpage={trans_task.src_page_index} dstpage={trans_task.dst_page_index}"
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
            trans_task: PDChunckedTransTask = trans_task

            try:
                xfer_handle = self.transporter.write_blocks_paged(trans_task=trans_task)
                trans_task.xfer_handle = xfer_handle
                trans_task.start_trans_time = time.time()
                with self.waiting_dict_lock:
                    self.waiting_dict[trans_task.get_key()] = trans_task
                if self.args.pd_detail_log:
                    logger.debug(f"start WRITE to decode node: {trans_task.to_str()}")
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

            self._update_task_status_once()
            time.sleep(0.001)

    def _update_task_status_once(self):
        with self.waiting_dict_lock:
            tasks = list(self.waiting_dict.values())
            for trans_task in tasks:
                if trans_task.xfer_handle is None:
                    continue

                # A native NIXL error must fail this task without terminating the
                # status thread and stranding every transfer queued behind it.
                try:
                    ret = self.transporter.check_task_status(trans_task=trans_task)
                except BaseException as e:
                    failed_task = self.waiting_dict.pop(trans_task.get_key(), None)
                    if failed_task is not None:
                        logger.exception(f"check transfer status failed: {failed_task.to_str()}")
                        failed_task.error_info = f"check transfer status failed: {str(e)}"
                        self.failed_queue.put(failed_task)
                    continue

                if ret == "DONE":
                    completed_task = self.waiting_dict.pop(trans_task.get_key(), None)
                    if completed_task is None:
                        continue
                    if self.transporter.capture_telemetry:
                        try:
                            telem = self.transporter.nixl_agent.get_xfer_telemetry(completed_task.xfer_handle)
                            total_us = telem.xferDuration
                            post_us = telem.postDuration
                            backend_us = telem.xferDuration - telem.postDuration
                            nixl_backend = self.transporter.nixl_agent.query_xfer_backend(completed_task.xfer_handle)
                            if self.args.pd_detail_log:
                                logger.debug(
                                    f"write trans task request_id={completed_task.request_id} "
                                    f"kv=[{completed_task.start_kv_index},{completed_task.end_kv_index}) "
                                    f"src_page={completed_task.src_page_index} "
                                    f"dst_page={completed_task.dst_page_index} "
                                    f"xfer time: {total_us:.3f} us, "
                                    f"post time: {post_us:.3f} us, backend time: {backend_us:.3f} us, "
                                    f"nixl_backend: {nixl_backend}, total_bytes: {telem.totalBytes}"
                                )
                        except BaseException:
                            logger.exception(f"get transfer telemetry failed: {completed_task.to_str()}")

                    try:
                        self.transporter.send_write_done_task_to_decode_node(completed_task)
                    except BaseException as e:
                        logger.exception(f"send WRITE done nixl notify failed: {completed_task.to_str()}")
                        completed_task.error_info = f"send WRITE done nixl notify failed: {str(e)}"
                        self.failed_queue.put(completed_task)
                        continue

                    if self.args.pd_detail_log:
                        logger.debug(
                            f"send WRITE done nixl notify "
                            f"request_id={completed_task.request_id} "
                            f"kv=[{completed_task.start_kv_index},{completed_task.end_kv_index}) "
                            f"src_page={completed_task.src_page_index} dst_page={completed_task.dst_page_index}"
                        )
                    self.success_queue.put(completed_task)
                elif ret == "ERR":
                    failed_task = self.waiting_dict.pop(trans_task.get_key(), None)
                    if failed_task is not None:
                        failed_task.error_info = "xfer error"
                        self.failed_queue.put(failed_task)
                elif trans_task.time_out():
                    failed_task = self.waiting_dict.pop(trans_task.get_key(), None)
                    if failed_task is not None:
                        failed_task.error_info = "time out in update_task_status_loop"
                        self.failed_queue.put(failed_task)

    @log_exception
    def success_loop(self):
        torch.cuda.set_device(self.device_id)
        while True:
            trans_task: PDChunckedTransTask = self.success_queue.get()
            # 写回后，回收页面
            if trans_task.src_page_index is not None:
                self.page_index_queue.put(trans_task.src_page_index)
            if trans_task.xfer_handle is not None:
                self.transporter.release_xfer_handle(trans_task.xfer_handle)

            ret = trans_task.createRetObj()
            ret.first_gen_token_id = None
            ret.first_gen_token_logprob = None
            self.task_out_queue.put(ret)

            if self.args.pd_detail_log:
                if trans_task.start_trans_time is not None:
                    logger.debug(f"trans task ret success:{ret} cost time: {trans_task.transfer_time()}s")
                else:
                    logger.debug(f"trans task ret success:{ret}")

    @log_exception
    def fail_loop(self):
        torch.cuda.set_device(self.device_id)
        while True:
            trans_task: PDChunckedTransTask = self.failed_queue.get()

            # 回收页面
            if trans_task.src_page_index is not None:
                self.page_index_queue.put(trans_task.src_page_index)
            if trans_task.xfer_handle is not None:
                self.transporter.release_xfer_handle(trans_task.xfer_handle)

            ret = trans_task.createRetObj()
            self.task_out_queue.put(ret)
            logger.warning(f"trans task ret fail:{ret}")

            if trans_task.error_info is not None:
                self._abort(request_id=trans_task.request_id, error_info=trans_task.error_info)
                self.transporter.send_error_info_to_decode_node(trans_task=trans_task)
