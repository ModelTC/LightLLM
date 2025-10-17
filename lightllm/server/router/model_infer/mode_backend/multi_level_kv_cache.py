import threading
import torch.distributed as dist
import torch
import dataclasses
from typing import Optional, List, Deque
from collections import deque
from lightllm.server.multi_level_kv_cache.cpu_cache_client import CpuKvCacheClient
from lightllm.utils.envs_utils import get_env_start_args
from ..infer_batch import InferReq
from lightllm.utils.dist_utils import create_new_group_for_current_dp
from lightllm.common.basemodel.triton_kernel.kv_cache_offload import offload_gpu_kv_to_cpu, load_cpu_kv_to_gpu
from lightllm.server.router.model_infer.infer_batch import g_infer_context
from lightllm.utils.log_utils import init_logger
from lightllm.utils.kv_cache_utils import compute_token_list_hash

logger = init_logger(__name__)


class MultiLevelKvCacheModule(object):
    def __init__(self, backend):
        self.args = get_env_start_args()
        from .base_backend import ModeBackend

        self.backend: ModeBackend = backend
        self.gloo_group = create_new_group_for_current_dp("gloo")
        self.filter_group = create_new_group_for_current_dp("gloo")
        self.init_sync_group = create_new_group_for_current_dp("nccl")
        dist.barrier(group=self.init_sync_group)

        self.cpu_cache_handle_queue: Deque[TransTask] = deque()
        self.cpu_cache_client = CpuKvCacheClient(only_create_meta_data=False, init_shm_data=False)

    def wait(self):
        """
        等待 cpu cache 相关页面注册完成
        """
        attach_shm_handle = self.cpu_cache_client.attach_shm_handle
        if attach_shm_handle is not None:
            attach_shm_handle.wait()

    def load_cpu_cache_to_reqs(self, reqs: List[InferReq]):
        idle_token_num = g_infer_context.get_can_alloc_token_num()
        token_page_size = self.args.cpu_cache_token_page_size
        all_page_list = []
        is_master_in_dp = self.backend.is_master_in_dp
        for req in reqs:
            page_list = req.shm_req.cpu_cache_match_page_indexes.get_all()
            match_tokens = len(page_list) * token_page_size
            # 更新命中的 cpu kv cache 长度.
            if is_master_in_dp:
                req.shm_req.cpu_prompt_cache_len = match_tokens

            need_token_num = match_tokens - req.cur_kv_len
            # 多匹配了一定数量的token 才进行复制操作，不然操作效率不高
            if need_token_num >= 64:
                if need_token_num <= idle_token_num:
                    if self.backend.radix_cache is not None:
                        g_infer_context.radix_cache.free_radix_cache_to_get_enough_token(need_token_num=need_token_num)

                    # 计算需要加载的页面（只加载未匹配的部分）
                    cur_kv_pages = req.cur_kv_len // token_page_size
                    need_pages = page_list[cur_kv_pages:]  # 只取需要的页面
                    actual_need_tokens = len(need_pages) * token_page_size

                    mem_indexes = g_infer_context.req_manager.mem_manager.alloc(need_size=actual_need_tokens)

                    # 将 cpu page 的内容拷贝到 gpu 页面中
                    load_cpu_kv_to_gpu(
                        mem_indexes=mem_indexes,
                        gpu_kv_cache=self.backend.model.mem_manager.kv_buffer,
                        cpu_kv_cache=self.cpu_cache_client.cpu_kv_cache_tensor,
                        page_indexes=torch.tensor(need_pages, dtype=torch.int32, device="cpu").cuda(non_blocking=True),
                    )

                torch.cuda.current_stream().synchronize()

                idle_token_num -= actual_need_tokens
                g_infer_context.req_manager.req_to_token_indexs[
                    req.req_idx, req.cur_kv_len : (req.cur_kv_len + actual_need_tokens)
                ] = mem_indexes
                req.cur_kv_len = req.cur_kv_len + actual_need_tokens
                if self.backend.is_master_in_dp:
                    req.shm_req.shm_cur_kv_len = req.cur_kv_len

            all_page_list.extend(page_list)

        dist.barrier(group=self.init_sync_group)

        if self.backend.is_master_in_dp:
            self.cpu_cache_client.lock.acquire_sleep1ms()
            self.cpu_cache_client.deref_pages(page_list=all_page_list)
            self.cpu_cache_client.lock.release()
        return

    def handle_finished_reqs(self, finished_reqs: List[InferReq]) -> List[InferReq]:
        """
        将满足cpu kv cache 卸载条件的请求进行处理，并返回需要真正退出的请求列表。
        """

        if self.args.enable_cpu_cache:
            # 如果开启了cpu cache，将达到finished状态的请求开启将gpu kv cache 卸载到 cpu cache中的操作。
            # 当 kv cache 卸载完成后，才会进行请求的真实退出操作。
            true_finished_reqs = []
            cpu_stream = g_infer_context.get_cpu_kv_cache_stream()
            for req in finished_reqs:
                # 只有 group_req_id 和 request_id 相同的请求才会被卸载到 cpu cache 中。
                # 这个限制是为了兼容 diverse 模式下的请求处理。
                if req.shm_req.group_req_id != req.shm_req.request_id:
                    true_finished_reqs.append(req)
                    continue

                # 过滤不适合进行 kv 卸载到 cpu cache 的请求。
                if req.cur_kv_len < self.args.cpu_cache_token_page_size:
                    true_finished_reqs.append(req)
                    continue

                # 如果请求已经完成了 cpu cache 的任务，则满足了退出条件
                if req.cpu_cache_task_status.is_finished():
                    true_finished_reqs.append(req)
                elif req.cpu_cache_task_status.is_running():
                    # 如果请求已经发起过卸载任务，则在当前轮不进行处理
                    continue
                else:
                    assert req.cpu_cache_task_status.is_not_started()
                    # 必须等待 overlap stream 上的计算任务完成，不然会崩溃
                    if g_infer_context.overlap_stream is not None:
                        cpu_stream.wait_stream(g_infer_context.overlap_stream)
                    else:
                        cpu_stream.wait_stream(torch.cuda.current_stream())

                    # 发起将请求的 kv cache 卸载到 cpu cache 中的任务
                    trans_task = self._start_kv_cache_offload_task(req=req, cpu_kv_cache_stream=cpu_stream)
                    if trans_task is not None:
                        self.cpu_cache_handle_queue.append(trans_task)
                    else:
                        true_finished_reqs.append(req)

            # 必须在这里同步，不然会崩溃
            cpu_stream.synchronize()
            return true_finished_reqs
        else:
            return finished_reqs

    def _start_kv_cache_offload_task(
        self, req: InferReq, cpu_kv_cache_stream: torch.cuda.Stream
    ) -> Optional["TransTask"]:
        with torch.cuda.stream(cpu_kv_cache_stream):
            if self.backend.is_master_in_dp:
                # 综合考虑后只对prompt做缓存管理，不包含decode内容，这里与radix cache不一致
                token_hash_list = req.shm_req.token_hash_list.get_all()
                block_size = req.cur_kv_len // self.args.cpu_cache_token_page_size
                move_block_size = min(block_size, len(token_hash_list))

                if move_block_size == 0:
                    dist.broadcast_object_list([0], group=self.gloo_group, group_src=0)
                    req.cpu_cache_task_status = InferReq._CpuCacheTaskStatus.FINISHED
                    return None

                try:
                    self.cpu_cache_client.lock.acquire_sleep1ms()
                    page_list, ready_list = self.cpu_cache_client.allocate_pages(
                        token_hash_list[:move_block_size],
                        disk_offload_enable=self.args.enable_disk_cache,
                    )
                finally:
                    self.cpu_cache_client.lock.release()

                item_size = len(page_list)
                if item_size == 0:
                    dist.broadcast_object_list([0], group=self.gloo_group, group_src=0)
                    req.cpu_cache_task_status = InferReq._CpuCacheTaskStatus.FINISHED
                    return None

                broadcast_data = {"item_size": item_size, "page_list": page_list, "ready_list": ready_list}
                dist.broadcast_object_list([broadcast_data], group=self.gloo_group, group_src=0)
            else:
                recv_list = [None]
                dist.broadcast_object_list(recv_list, group=self.gloo_group, group_src=0)
                if isinstance(recv_list[0], int) and recv_list[0] == 0:
                    req.cpu_cache_task_status = InferReq._CpuCacheTaskStatus.FINISHED
                    return None
                broadcast_data = recv_list[0]
                item_size = broadcast_data["item_size"]
                page_list = broadcast_data["page_list"]
                ready_list = broadcast_data["ready_list"]

            page_indexes = torch.tensor(page_list, dtype=torch.int32, device="cpu", pin_memory=True)
            page_readies = torch.tensor(ready_list, dtype=torch.bool, device="cpu", pin_memory=True)
            token_indexes = self.backend.model.req_manager.req_to_token_indexs[req.req_idx, 0 : req.cur_kv_len]
            offload_gpu_kv_to_cpu(
                token_indexes=token_indexes,
                gpu_kv_cache=self.backend.model.mem_manager.kv_buffer,
                cpu_kv_cache=self.cpu_cache_client.cpu_kv_cache_tensor,
                page_indexes=page_indexes,
                page_readies=page_readies,
            )

            sync_event = torch.cuda.Event()
            sync_event.record()
            req.cpu_cache_task_status = InferReq._CpuCacheTaskStatus.RUNNING
            trans_task = TransTask(
                page_indexes=page_indexes, page_readies=page_readies, req_obj=req, sync_event=sync_event
            )

        return trans_task

    def update_cpu_cache_task_states(self):
        if self.backend.is_master_in_dp:
            trans_ok_tasks = []
            while len(self.cpu_cache_handle_queue) != 0:
                task: TransTask = self.cpu_cache_handle_queue.popleft()
                if task.sync_event.query():
                    trans_ok_tasks.append(task)
                else:
                    self.cpu_cache_handle_queue.appendleft(task)
                    break
            item_size = len(trans_ok_tasks)
            dist.broadcast_object_list([item_size], group=self.filter_group, group_src=0)
        else:
            recv_list = [None]
            dist.broadcast_object_list(recv_list, group=self.filter_group, group_src=0)
            item_size = recv_list[0]
            trans_ok_tasks: List[TransTask] = [self.cpu_cache_handle_queue.popleft() for _ in range(item_size)]

        if item_size > 0:
            page_array_list = [task.page_indexes for task in trans_ok_tasks]
            page_list = torch.cat(page_array_list, dim=0).tolist()
            if self.backend.is_master_in_dp:
                self.cpu_cache_client.lock.acquire_sleep1ms()
                self.cpu_cache_client.update_pages_status_to_ready(
                    page_list=page_list, deref=True, disk_offload_enable=self.args.enable_disk_cache
                )
                self.cpu_cache_client.lock.release()
            for task in trans_ok_tasks:
                task.req_obj.cpu_cache_task_status = InferReq._CpuCacheTaskStatus.FINISHED
        return


@dataclasses.dataclass
class TransTask:
    page_indexes: torch.Tensor
    page_readies: torch.Tensor
    req_obj: InferReq
    sync_event: torch.cuda.Event
