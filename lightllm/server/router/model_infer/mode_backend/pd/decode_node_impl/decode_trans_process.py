import torch
import time
import inspect
import threading
import setproctitle
import torch.multiprocessing as mp
import queue
import pickle
from typing import List, Dict, Union, Optional
from lightllm.utils.log_utils import init_logger
from lightllm.common.attention_state import AttentionStateManager
from lightllm.common.attention_state.copy_plan import CopyPlan
from lightllm.common.device_runtime import get_device_runtime
from lightllm.common.attention_state.lifecycle import RetiredTransfers, transfer_completion
from lightllm.server.router.model_infer.shared_att_resources_utils import load_att_resources
from lightllm.server.router.model_infer.mode_backend.pd.resource_io import allocate_transfer_buffer
from lightllm.server.pd_io_struct import (
    PDChunckedTransTask,
    PDChunckedTransTaskGroup,
    PDUpKVStatus,
    PDAbortReq,
)
from lightllm.server.pd_io_struct import PDDecodeNodeInfo
from lightllm.utils.graceful_utils import graceful_registry
from lightllm.server.core.objs import StartArgs
from ..kv_transporter import create_kv_transporter
from lightllm.utils.error_utils import log_exception
from lightllm.utils.envs_utils import get_unique_server_name
from lightllm.utils.process_check import install_fatal_thread_excepthook, start_parent_check_thread

logger = init_logger(__name__)


def start_decode_trans_process(
    args,
    device_id,
    task_in_queue: mp.Queue,
    task_out_queue: mp.Queue,
    up_status_in_queue: Optional[mp.SimpleQueue],
):
    proc = mp.Process(target=_init_env, args=(args, device_id, task_in_queue, task_out_queue, up_status_in_queue))
    proc.start()
    assert proc.is_alive()
    logger.info(f"prefill trans kv process for device: {device_id} started!")
    return proc


def _init_env(
    args: StartArgs,
    device_id: int,
    task_in_queue: mp.Queue,
    task_out_queue: mp.Queue,
    up_status_in_queue: Optional[mp.SimpleQueue],
):
    install_fatal_thread_excepthook()
    start_parent_check_thread()
    import lightllm.utils.rpyc_fix_utils as _

    import os

    # -------------------------------------------------------------------------
    # 问题背景（PD NIXL + 同卡多进程）：
    #   decode 物理 GPU 上至少有两个独立 CUDA 进程：model_infer（解码推理）与
    #   decode_trans（把 prefill 侧 KV page 拷入 decode KV cache）。
    #   lm_eval batch=64 时会在短时间内并发大量 read_page；拷贝在 copy_cuda_stream
    #   上排队，而推理在另一进程的 stream 上执行，彼此无法 cudaStreamWaitEvent
    #   协调。日志里的 read_page_gpu_time（event 差值）会把「等 GPU 时间片 /
    #   与推理争抢 SM」算进去，出现数十秒级毛刺，但并不代表单次 memcpy 真那么慢。
    #
    # 解决思路：依赖 NVIDIA MPS（Multi-Process Service）在同一 GPU 上多进程
    #   共享上下文并做客户端级调度；在子进程 import torch / 创建 CUDA 上下文
    #   **之前**设置下列环境变量（故必须放在本函数最前）。
    #
    # CUDA_MPS_CLIENT_PRIORITY="0"：
    #   MPS 下数值越小优先级越高。decode 侧 KV 拷贝处于 decode 关键路径（须先
    #   落盘 KV 才能出首 token），故给 trans 进程最高优先级，减轻被同卡推理
    #   饿死导致的排队放大。须集群已启动 nvidia-cuda-mps-control / mps-server，
    #   否则该变量不生效。 启动 mps 的命令为 nvidia-cuda-mps-control -d
    # -------------------------------------------------------------------------
    os.environ["CUDA_MPS_CLIENT_PRIORITY"] = "0"

    torch.backends.cudnn.enabled = False
    setproctitle.setproctitle(f"lightllm::{get_unique_server_name()}::decode_trans:Device{device_id}")

    try:
        torch.cuda.set_device(device_id)
        graceful_registry(inspect.currentframe().f_code.co_name)

        task_out_queue.put("proc_start")

        # 从共享内存读取所有 rank 的 attention 资源
        node_world_size = args.tp // args.nnodes
        resource_managers: List[AttentionStateManager] = [
            load_att_resources(rank, kind="pd")[0] for rank in range(node_world_size)
        ]

        task_out_queue.put("get_resources_ok")

        manager = _DecodeTransModule(
            args=args,
            device_id=device_id,
            task_in_queue=task_in_queue,
            task_out_queue=task_out_queue,
            resource_managers=resource_managers,
            up_status_in_queue=up_status_in_queue,
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


class _DecodeTransModule:
    def __init__(
        self,
        args: StartArgs,
        device_id: int,
        task_in_queue: mp.Queue,
        task_out_queue: mp.Queue,
        resource_managers: List[AttentionStateManager],
        up_status_in_queue: Optional[mp.SimpleQueue],
    ):
        self.args = args
        self.dp_world_size = self.args.tp // self.args.dp
        self.device_id = device_id
        self.task_in_queue = task_in_queue
        self.task_out_queue = task_out_queue
        self.resource_managers = resource_managers
        self.up_status_in_queue = up_status_in_queue
        kv_move_buffer = allocate_transfer_buffer(
            self.resource_managers[device_id], self.args.pd_kv_page_num, device=f"cuda:{device_id}"
        )
        self.kv_move_buffer = kv_move_buffer
        self.runtime = get_device_runtime(kv_move_buffer.device)
        self.copy_plans = tuple(
            CopyPlan.for_pd(resource_managers[start : start + self.dp_world_size], kv_move_buffer, write=False)
            for start in range(0, len(resource_managers), self.dp_world_size)
        )
        self.copy_cuda_stream = torch.cuda.Stream(priority=-1)
        from lightllm.common.attention_state.manifest import build_state_manifest

        self.transporter = create_kv_transporter(
            args=self.args,
            node_id=self.args.pd_node_id,
            tp_idx=device_id,
            kv_move_buffer=kv_move_buffer,
            state_manifest=build_state_manifest(self.resource_managers[device_id]),
        )
        self.recv_task_group_queue = queue.Queue()
        self.waiting_dict_lock = threading.Lock()
        self.waiting_dict: Dict[str, PDChunckedTransTask] = {}
        self.request_page_task_queue = queue.Queue()
        self.ready_page_task_queue = queue.Queue()
        self.success_queue = queue.Queue()
        self.failed_queue = queue.Queue()

        self.page_index_queue = queue.Queue()
        self.retired_transfers = RetiredTransfers(self.transporter, self.page_index_queue, "dst_page_index")
        for page_index in range(self.args.pd_kv_page_num):
            self.page_index_queue.put(page_index)

        # warmup 预先加载一次kv 数据到 mem manager，避免第一次拷贝时出现卡顿。
        self._warmup()

        for func in [
            self.recv_task_loop,
            self.dispatch_task_loop,
            self.accept_peer_task_loop,
            self.request_page_loop,
            self.read_page_to_mems_loop,
            self.success_loop,
            self.fail_loop,
        ]:
            threading.Thread(target=func, daemon=True).start()
        return

    def _warmup(self):
        for dp_index in range(self.args.dp // self.args.nnodes):
            with torch.cuda.stream(stream=self.copy_cuda_stream):
                cpu_token_indices = torch.tensor([0], dtype=torch.long, device="cpu")
                token_indices = cpu_token_indices.cuda(device=self.kv_move_buffer.device, non_blocking=True)
                self.copy_plans[dp_index].run(token_indices=token_indices, pd_page_index=0)
                torch.cuda.current_stream().synchronize()
        return

    @log_exception
    def recv_task_loop(self):
        while True:
            obj: Union[PDChunckedTransTaskGroup, PDAbortReq] = self.task_in_queue.get()
            if isinstance(obj, PDChunckedTransTaskGroup):
                self.recv_task_group_queue.put(obj)
            elif isinstance(obj, PDAbortReq):
                self._abort(request_id=obj.request_id)
            else:
                assert False, f"recv error obj {obj}"

    def _abort(self, request_id: int, error_info: str = "aborted req"):
        aborted_tasks = []
        with self.waiting_dict_lock:
            for key, trans_task in list(self.waiting_dict.items()):
                if trans_task.request_id == request_id:
                    transfer_completion(trans_task).cancelled = True
                    trans_task.error_info = error_info
                if trans_task.request_id == request_id and trans_task.dst_page_index is None:
                    # 对于 已经分配了page index 的任务，不能直接失败，需要两边走完正常流程再失败，不然可能
                    # 出现复杂的异步协同问题。
                    aborted_tasks.append(self.waiting_dict.pop(key))

        for trans_task in aborted_tasks:
            trans_task.error_info = error_info
            self.failed_queue.put(trans_task)
        return

    @log_exception
    def dispatch_task_loop(self):
        while True:
            trans_task_group: PDChunckedTransTaskGroup = self.recv_task_group_queue.get()

            with self.waiting_dict_lock:
                shared_time_out_mark = _SharedTimeOutMark()
                for task in trans_task_group.task_list:
                    task.shared_time_out_mark = shared_time_out_mark
                    if task.need_transfer_page():
                        self.waiting_dict[task.get_key()] = task
                    else:
                        task.start_trans_time = time.time()
                        self.success_queue.put((None, None, task))

            # up status
            task = trans_task_group.task_list[0]

            decode_node_info = PDDecodeNodeInfo(
                decode_node_id=self.args.pd_node_id,
                pd_master_node_id=task.pd_master_node_id,
                agent_name=self.transporter.agent_name,
                agent_metadata=self.transporter.agent_metadata,
                num_pages=self.transporter.num_pages,
                page_reg_desc=self.transporter.local_page_mem_desc,
                request_id=task.request_id,
                ready_kv_len=task.start_kv_index,
            )

            up_status = PDUpKVStatus(
                group_request_id=task.request_id,
                pd_master_node_id=task.pd_master_node_id,
                pd_kv_trans_params=pickle.dumps(decode_node_info),
            )

            self.up_status_in_queue.put(up_status)

    @log_exception
    def accept_peer_task_loop(
        self,
    ):
        torch.cuda.set_device(self.device_id)
        while True:
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

                        if not isinstance(notify_obj, PDChunckedTransTask):
                            continue

                        # 请求有错误
                        if notify_obj.error_info is not None:
                            # 直接清理掉所有的相关请求。
                            with self.waiting_dict_lock:
                                local_trans_task = self.waiting_dict.pop(notify_obj.get_key(), None)
                                if local_trans_task is not None:
                                    local_trans_task.error_info = notify_obj.error_info
                                    self.failed_queue.put(local_trans_task)

                            self._abort(
                                request_id=notify_obj.request_id,
                                error_info=notify_obj.error_info,
                            )
                            continue

                        # 到了请求页面的阶段
                        remote_trans_task = notify_obj
                        if remote_trans_task.write_stage == "request":
                            with self.waiting_dict_lock:
                                local_trans_task = self.waiting_dict.pop(remote_trans_task.get_key(), None)
                            if local_trans_task is not None:
                                local_trans_task.shared_time_out_mark.update()
                                local_trans_task.transfer_id = remote_trans_task.transfer_id
                                local_trans_task.prefill_agent_name = remote_trans_task.prefill_agent_name
                                local_trans_task.prefill_agent_metadata = remote_trans_task.prefill_agent_metadata
                                local_trans_task.prefill_num_pages = remote_trans_task.prefill_num_pages
                                local_trans_task.prefill_page_reg_desc = remote_trans_task.prefill_page_reg_desc
                                self.request_page_task_queue.put(local_trans_task)
                                if self.args.pd_detail_log:
                                    logger.debug(f"recv WRITE request from prefill: {remote_trans_task.to_str()}")
                            else:
                                # This does not necessarily mean the WRITE protocol state is corrupted.
                                # A common benign case is: decode has already received an abort for this
                                # request and removed its waiting task, while prefill's NIXL WRITE request
                                # notify arrives later. Keep the original cleanup/error path so true
                                # missing-task bugs are still visible, but make the log explicit enough
                                # to avoid misclassifying abort-after-cleanup as a transfer failure.
                                logger.warning(
                                    "can not find waiting WRITE task for request notify, "
                                    "possibly because request was already aborted and cleaned on decode side: "
                                    f"{remote_trans_task.to_str()}"
                                )
                                # 发一个error信息回去给 prefill 节点，让其可以知道这边有问题了，它可以选择其他清理掉请求。
                                remote_trans_task.error_info = "can not find waiting WRITE task for request notify"
                                self.transporter.send_error_info_to_prefill_node(trans_task=remote_trans_task)

                            continue

                        # prefill 写完数据到了 done 阶段
                        if remote_trans_task.write_stage == "done":
                            with self.waiting_dict_lock:
                                local_trans_task = self.waiting_dict.get(remote_trans_task.get_key())
                                if (
                                    local_trans_task is not None
                                    and local_trans_task.transfer_id == remote_trans_task.transfer_id
                                ):
                                    self.waiting_dict.pop(remote_trans_task.get_key())
                                else:
                                    local_trans_task = None
                            if local_trans_task is not None:
                                transfer_completion(local_trans_task).network_done = True
                                local_trans_task.shared_time_out_mark.update()
                                local_trans_task.first_gen_token_id = remote_trans_task.first_gen_token_id
                                local_trans_task.first_gen_token_logprob = remote_trans_task.first_gen_token_logprob
                                self.ready_page_task_queue.put(local_trans_task)
                                if self.args.pd_detail_log:
                                    logger.debug(f"recv WRITE done from prefill: {remote_trans_task.to_str()}")
                            elif hasattr(self, "retired_transfers") and self.retired_transfers.remote_done(
                                remote_trans_task.transfer_id
                            ):
                                continue
                            else:
                                # Same race as the WRITE request stage: decode may have cleaned the
                                # waiting task because the request was aborted, then a late done notify
                                # arrives from prefill. Preserve the original error path, but make the
                                # diagnostic tell future readers this can be abort-related noise.
                                logger.warning(
                                    "can not find waiting WRITE task for done notify, "
                                    "possibly because request was already aborted and cleaned on decode side: "
                                    f"{remote_trans_task.to_str()}"
                                )
                                # 发一个error信息回去给 prefill 节点，让其可以知道这边有问题了，它可以选择其他清理掉请求。
                                remote_trans_task.error_info = "can not find waiting WRITE task for done notify"
                                self.transporter.send_error_info_to_prefill_node(trans_task=remote_trans_task)
                            continue

            self._check_tasks_time_out()
            if not notifies_dict:
                time.sleep(0.001)

    def _check_tasks_time_out(self):
        if hasattr(self, "retired_transfers"):
            self.retired_transfers.poll()
        with self.waiting_dict_lock:
            timeout_tasks = []
            for key, trans_task in list(self.waiting_dict.items()):
                if trans_task.shared_time_out_mark.time_out(trans_task):
                    timeout_tasks.append(self.waiting_dict.pop(key))

        for trans_task in timeout_tasks:
            trans_task.error_info = "time out in accept_peer_task_loop"
            self.failed_queue.put(trans_task)
        return

    @log_exception
    def request_page_loop(self):
        torch.cuda.set_device(self.device_id)
        while True:
            dst_page_index = self.page_index_queue.get()
            trans_task: PDChunckedTransTask = self.request_page_task_queue.get()
            trans_task.dst_page_index = dst_page_index
            trans_task.start_trans_time = time.time()
            key = trans_task.get_key()
            try:
                with self.waiting_dict_lock:
                    self.waiting_dict[key] = trans_task
                transfer_completion(trans_task).network_submitted = True
                self.retired_transfers.watch(trans_task)
                self.transporter.send_write_ready_task_to_prefill_node(trans_task=trans_task)
            except BaseException as e:
                with self.waiting_dict_lock:
                    self.waiting_dict.pop(key, None)
                logger.error(f"send write ready task to prefill node failed: {trans_task.to_str()}")
                logger.exception(str(e))
                self.transporter.remove_remote_agent(peer_name=trans_task.prefill_agent_name)
                trans_task.error_info = f"send write ready task to prefill node failed: {str(e)}"
                self.failed_queue.put(trans_task)
                continue

        return

    @log_exception
    def read_page_to_mems_loop(self):
        torch.cuda.set_device(self.device_id)
        while True:
            trans_task: PDChunckedTransTask = self.ready_page_task_queue.get()
            if transfer_completion(trans_task).cancelled:
                self.failed_queue.put(trans_task)
                continue
            if trans_task.xfer_handle is not None:
                # 只有 NCCL 接收端会创建本地 Xfer 句柄；NIXL 的远端 WRITE 完成通知
                # 已代表其写入完成，不经过这里的本地 CUDA event 轮询。
                # P 的 done 不代表 D 本地接收已完成；统一轮询传输状态，
                # 直到接收 event 完成后，独立搬运流才读取 PD 页。
                while True:
                    status = self.transporter.check_task_status(trans_task=trans_task)
                    if status == "PROC":
                        time.sleep(0.001)
                        continue
                    if status != "DONE":
                        trans_task.error_info = trans_task.xfer_handle.error_info
                        self.failed_queue.put(trans_task)
                    break
                if status != "DONE":
                    continue
            copy_start_event = torch.cuda.Event(enable_timing=True)
            copy_end_event = torch.cuda.Event(enable_timing=True)
            with torch.cuda.stream(stream=self.copy_cuda_stream):
                copy_start_event.record(self.copy_cuda_stream)
                # 只有 KV 页使用 token 索引；att_state 直接按请求槽位搬运运行态。
                token_indices = None
                if trans_task.page_kind == "kv":
                    cpu_token_indices = torch.tensor(trans_task.mem_indexes, dtype=torch.long, device="cpu")
                    # 索引上传和后续页拷贝使用同一流，保证执行顺序。
                    token_indices = cpu_token_indices.cuda(device=self.kv_move_buffer.device, non_blocking=True)
                try:
                    self.copy_plans[trans_task.decode_dp_index].run(
                        token_indices=token_indices,
                        pd_page_index=trans_task.dst_page_index,
                        task_kind=trans_task.page_kind,
                        req_idx=trans_task.req_idx,
                        state_token_len=trans_task.end_kv_index if trans_task.page_kind == "att_state" else None,
                    )
                except Exception as error:
                    trans_task.error_info = f"unpacking state failed: {error}"
                copy_end_event.record(self.copy_cuda_stream)
                transfer_completion(trans_task).destination_ready = copy_end_event
            if getattr(trans_task, "error_info", None) is not None:
                self.failed_queue.put(trans_task)
            else:
                self.success_queue.put((copy_end_event, copy_start_event, trans_task))

    @log_exception
    def success_loop(self):
        torch.cuda.set_device(self.device_id)
        while True:
            copy_end_event, copy_start_event, trans_task = self.success_queue.get()
            trans_task: PDChunckedTransTask = trans_task
            copy_end_event: Optional[torch.cuda.Event] = copy_end_event
            copy_start_event: Optional[torch.cuda.Event] = copy_start_event
            read_page_gpu_time_ms = -1.0
            if copy_end_event is not None:
                copy_end_event.synchronize()
                read_page_gpu_time_ms = copy_start_event.elapsed_time(copy_end_event)

            self.retired_transfers.retire(trans_task)

            ret = trans_task.createRetObj()
            self.task_out_queue.put(ret)

            if self.args.pd_detail_log:
                if trans_task.start_trans_time is not None:
                    logger.debug(
                        f"trans task ret success:{ret} cost time: {trans_task.transfer_time()} s "
                        f"read_page_gpu_time: {read_page_gpu_time_ms:.3f} ms"
                    )
                else:
                    logger.debug(f"trans task ret success:{ret}")

    @log_exception
    def fail_loop(self):
        torch.cuda.set_device(self.device_id)
        while True:
            trans_task: PDChunckedTransTask = self.failed_queue.get()
            completion = transfer_completion(trans_task)
            completion.cancelled = True
            completion.wait_local_accesses()

            # 回收页面
            self.retired_transfers.retire(trans_task)

            ret = trans_task.createRetObj()
            self.task_out_queue.put(ret)
            logger.warning(f"trans task ret fail:{ret}")

            if trans_task.error_info is not None:
                # 提前终结所有有问题的属于同一个请求的任务。
                self._abort(
                    request_id=trans_task.request_id,
                    error_info=trans_task.error_info,
                )
                self.transporter.send_error_info_to_prefill_node(trans_task=trans_task)


class _SharedTimeOutMark:
    """Pending pages share request progress; active pages keep their own deadline."""

    def __init__(self):
        self.update()

    def update(self):
        self.time_mark = time.time()

    def time_out(self, task: PDChunckedTransTask) -> bool:
        if task.start_trans_time is not None:
            return task.time_out()
        return time.time() - self.time_mark > task.time_out_secs
