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
from lightllm.utils.infer_utils import mark_start, mark_end

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

        self.sync_group = create_new_group_for_current_dp("nccl")
        dist.barrier(group=self.sync_group)
        self.sync_tensor = torch.zeros((1,), dtype=torch.int64, device="cuda")

        self.cpu_cache_handle_queue: Deque[TransTask] = deque()
        self.cpu_cache_client = CpuKvCacheClient(init_shm_data=False)

    def _compute_full_sequence_hash(self, req: InferReq):
        """
        计算基于完整序列（输入+输出）的hash值，而不是只基于输入
        """
        from lightllm.utils.kv_cache_utils import compute_token_list_hash

        # 获取完整的token序列：输入 + 已生成的输出
        input_tokens = req.shm_req.get_prompt_ids()

        # 获取已生成的输出token
        total_len = req.shm_req.input_len + req.shm_req.shm_cur_output_len
        if total_len > req.shm_req.input_len:
            # 从共享内存中获取完整序列
            full_sequence = req.shm_req.shm_prompt_ids.arr[:total_len].tolist()
        else:
            full_sequence = input_tokens

        # 基于完整序列计算hash
        hash_values = compute_token_list_hash(full_sequence, self.args.cpu_cache_token_page_size)
        return hash_values

    def handle_finished_reqs(self, finished_reqs: List[InferReq]) -> List[InferReq]:
        """
        将满足cpu kv cache 卸载条件的请求进行处理，并返回需要真正退出的请求列表。
        """

        if self.args.enable_cpu_cache:
            # 如果开启了cpu cache，将达到finished状态的请求开启将gpu kv cache 卸载到 cpu cache中的操作。
            # 当 kv cache 卸载完成后，才会进行请求的真实退出操作。
            true_finished_reqs = []
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
                    # 发起将请求的 kv cache 卸载到 cpu cache 中的任务
                    trans_task = self._start_kv_cache_offload_task(
                        req=req, cpu_kv_cache_stream=g_infer_context.get_cpu_kv_cache_stream()
                    )

                    if trans_task is not None:
                        self.cpu_cache_handle_queue.append(trans_task)
                    else:
                        true_finished_reqs.append(req)

            return true_finished_reqs
        else:
            return finished_reqs

    def _start_kv_cache_offload_task(
        self, req: InferReq, cpu_kv_cache_stream: torch.cuda.Stream
    ) -> Optional["TransTask"]:
        with torch.cuda.stream(cpu_kv_cache_stream):
            # 性能优化：只有 master 进程计算 hash，减少重复计算
            if self.backend.is_master_in_dp:
                all_token_hash_list = self._compute_full_sequence_hash(req)
                block_size = req.cur_kv_len // self.args.cpu_cache_token_page_size
                move_block_size = min(block_size, len(all_token_hash_list))
                
                if move_block_size == 0:
                    # 广播失败状态给其他进程
                    dist.broadcast_object_list([0], group=self.gloo_group, group_src=0)
                    req.cpu_cache_task_status = InferReq._CpuCacheTaskStatus.FINISHED
                    return None
                
                # 性能优化：减少锁持有时间，只在必要时获取锁
                try:
                    self.cpu_cache_client.lock.acquire_sleep1ms()
                    page_list, ready_list = self.cpu_cache_client.allocate_pages(
                        all_token_hash_list[:move_block_size],
                        disk_offload_enable=self.args.enable_disk_cache,
                    )
                finally:
                    self.cpu_cache_client.lock.release()
                
                item_size = len(page_list)
                if item_size == 0:
                    # 广播失败状态
                    dist.broadcast_object_list([0], group=self.gloo_group, group_src=0)
                    req.cpu_cache_task_status = InferReq._CpuCacheTaskStatus.FINISHED
                    return None
                
                # 性能优化：合并广播操作，减少通信次数
                broadcast_data = {
                    'item_size': item_size,
                    'page_list': page_list,
                    'ready_list': ready_list
                }
                dist.broadcast_object_list([broadcast_data], group=self.gloo_group, group_src=0)
            else:
                # 非 master 进程只接收广播结果
                recv_list = [None]
                dist.broadcast_object_list(recv_list, group=self.gloo_group, group_src=0)
                
                if isinstance(recv_list[0], int) and recv_list[0] == 0:
                    # 接收到失败状态
                    req.cpu_cache_task_status = InferReq._CpuCacheTaskStatus.FINISHED
                    return None
                
                broadcast_data = recv_list[0]
                item_size = broadcast_data['item_size']
                page_list = broadcast_data['page_list']
                ready_list = broadcast_data['ready_list']

            # 性能优化：预分配 tensor，避免重复创建
            page_indexes = torch.tensor(page_list, dtype=torch.int32, device="cpu", pin_memory=True)
            page_readies = torch.tensor(ready_list, dtype=torch.bool, device="cpu", pin_memory=True)

            token_indexes = self.backend.model.req_manager.req_to_token_indexs[req.req_idx, 0 : req.cur_kv_len]
            
            # 执行 GPU 到 CPU 的数据传输
            offload_gpu_kv_to_cpu(
                token_indexes=token_indexes,
                gpu_kv_cache=self.backend.model.mem_manager.kv_buffer,
                cpu_kv_cache=self.cpu_cache_client.cpu_kv_cache_tensor,
                page_indexes=page_indexes,
                page_readies=page_readies,
            )

            # 性能优化：使用异步 allreduce，减少阻塞时间
            async_work = dist.all_reduce(tensor=self.sync_tensor, group=self.sync_group, async_op=True)
            
            # 在等待同步的同时，先创建其他对象
            sync_event = torch.cuda.Event()
            sync_event.record()
            req.cpu_cache_task_status = InferReq._CpuCacheTaskStatus.RUNNING
            
            # # 等待异步操作完成
            async_work.wait()
            
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

    def fill_cpu_cache_to_reqs(self, reqs: List[InferReq]):
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
            if need_token_num > 128:
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


@dataclasses.dataclass
class TransTask:
    page_indexes: torch.Tensor
    page_readies: torch.Tensor
    req_obj: InferReq
    sync_event: torch.cuda.Event
