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

        # 一些算子模式需要同步计算和 cpu cache 的 load 和 offload 操作
        self.need_sync_compute_stream: bool = True

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
            # 更新命中的 cpu kv cache 长度, 减去radix cache和disk cache的部分.
            if is_master_in_dp:
                req.shm_req.cpu_prompt_cache_len = max(
                    0, match_tokens - req.cur_kv_len - req.shm_req.disk_prompt_cache_len
                )

            need_token_num = match_tokens - req.cur_kv_len
            # 多匹配了一定数量的token同时请求长度大于一定的长度，才进行复制操作，不然操作效率不高，代价过高
            if need_token_num >= 128 and req.shm_req.input_len >= 256:
                if need_token_num <= idle_token_num:
                    if self.backend.radix_cache is not None:
                        g_infer_context.radix_cache.free_radix_cache_to_get_enough_token(need_token_num=need_token_num)

                    # 计算需要加载的页面（只加载未匹配的部分）
                    cur_kv_pages = req.cur_kv_len // token_page_size
                    need_pages = page_list[cur_kv_pages:]  # 只取需要的页面

                    mem_indexes = g_infer_context.req_manager.mem_manager.alloc(need_size=need_token_num)

                    if self.need_sync_compute_stream:
                        # TODO fa3 现在必须使用同步模式, 未来需要移除
                        g_infer_context.get_overlap_stream().synchronize()

                    # TODO 更有效的分配策略。
                    grid_num = 16

                    mem_manager = self.backend.model.mem_manager
                    if hasattr(mem_manager, "scale_buffer") and mem_manager.scale_buffer is not None:
                        cpu_cache_meta = self.cpu_cache_client.kv_cache_tensor_meta
                        cpu_kv_cache = self.cpu_cache_client.cpu_kv_cache_tensor[
                            :, :, :, :, 0 : cpu_cache_meta.head_dim
                        ]
                        cpu_kv_cache_scale = self.cpu_cache_client.cpu_kv_cache_tensor[
                            :, :, :, :, cpu_cache_meta.head_dim :
                        ].view(mem_manager.scale_buffer.dtype)
                        gpu_kv_cache_scale = mem_manager.scale_buffer
                    else:
                        cpu_kv_cache = self.cpu_cache_client.cpu_kv_cache_tensor
                        cpu_kv_cache_scale = None
                        gpu_kv_cache_scale = None

                    # 将 cpu page 的内容拷贝到 gpu 页面中
                    load_cpu_kv_to_gpu(
                        gpu_mem_indexes=mem_indexes.cuda(non_blocking=True),
                        gpu_kv_cache=mem_manager.kv_buffer,
                        gpu_kv_cache_scale=gpu_kv_cache_scale,
                        cpu_kv_cache=cpu_kv_cache,
                        cpu_kv_cache_scale=cpu_kv_cache_scale,
                        page_indexes=torch.tensor(need_pages, dtype=torch.int32, device="cpu").cuda(non_blocking=True),
                        tp_index=self.backend.rank_in_dp,
                        tp_world_size=self.backend.dp_world_size,
                        grid_num=grid_num,
                    )

                torch.cuda.current_stream().synchronize()

                idle_token_num -= need_token_num
                g_infer_context.req_manager.req_to_token_indexs[
                    req.req_idx, req.cur_kv_len : (req.cur_kv_len + need_token_num)
                ] = mem_indexes
                req.cur_kv_len = req.cur_kv_len + need_token_num
                if self.backend.is_master_in_dp:
                    req.shm_req.shm_cur_kv_len = req.cur_kv_len

            all_page_list.extend(page_list)

        dist.barrier(group=self.init_sync_group)

        if self.backend.is_master_in_dp:
            self.cpu_cache_client.lock.acquire_sleep1ms()
            self.cpu_cache_client.deref_pages(page_list=all_page_list)
            self.cpu_cache_client.lock.release()
        return

    def offload_finished_reqs_to_cpu_cache(self, finished_reqs: List[InferReq]) -> List[InferReq]:
        """
        将满足cpu kv cache 卸载条件的请求进行处理, 并返回真的满足退出条件的请求list。
        """
        # 如果开启了cpu cache，将达到finished状态的请求开启将gpu kv cache 卸载到 cpu cache中的操作。
        # 当 kv cache 卸载完成后，才会进行请求的真实退出操作。
        true_finished_reqs = []
        cpu_stream = g_infer_context.get_cpu_kv_cache_stream()
        for req in finished_reqs:
            # 只有 group_req_id 和 request_id 相同的请求才会被卸载到 cpu cache 中。
            # 这个限制是为了兼容 diverse 模式下的请求处理, 只有主请求才 offload kv 到 cpu
            # cache 中
            if req.shm_req.group_req_id != req.shm_req.request_id:
                true_finished_reqs.append(req)
                continue

            # 过滤不适合进行 kv 卸载到 cpu cache 的请求。
            if (
                req.cur_kv_len < self.args.cpu_cache_token_page_size
                or req.shm_req.input_len <= self.args.cpu_cache_token_page_size
            ):
                true_finished_reqs.append(req)
                continue

            # 如果请求已经完成了 cpu cache 的任务，则满足了退出条件
            if req.cpu_cache_task_status.is_finished():
                true_finished_reqs.append(req)
                continue

            # 如果请求已经发起过卸载任务且正在卸载过程中，则在当前轮不进行处理
            if req.cpu_cache_task_status.is_running():
                continue

            assert req.cpu_cache_task_status.is_not_started()

            if self.need_sync_compute_stream:
                # TODO fa3 现在必须使用同步模式, 未来需要移除, 必须等待 overlap stream 上的计算任务完成，不然会崩溃
                g_infer_context.get_overlap_stream().synchronize()

            # 发起将请求的 kv cache 卸载到 cpu cache 中的任务
            trans_task = self._start_kv_cache_offload_task(req=req, cpu_kv_cache_stream=cpu_stream)

            # 根据是否成功创建了卸载任务，决定是否将请求加入到处理队列中
            if trans_task is not None:
                self.cpu_cache_handle_queue.append(trans_task)
            else:
                true_finished_reqs.append(req)

        if self.need_sync_compute_stream:
            # TODO fa3 现在必须使用同步模式, 未来需要移除
            cpu_stream.synchronize()

        return true_finished_reqs

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
            move_token_num = item_size * self.args.cpu_cache_token_page_size
            assert req.cur_kv_len >= item_size * self.args.cpu_cache_token_page_size
            token_indexes = self.backend.model.req_manager.req_to_token_indexs[req.req_idx, 0:move_token_num]

            # TODO 更有效的分配策略。
            grid_num = 16

            mem_manager = self.backend.model.mem_manager
            if hasattr(mem_manager, "scale_buffer") and mem_manager.scale_buffer is not None:
                cpu_cache_meta = self.cpu_cache_client.kv_cache_tensor_meta
                cpu_kv_cache = self.cpu_cache_client.cpu_kv_cache_tensor[:, :, :, :, 0 : cpu_cache_meta.head_dim]
                cpu_kv_cache_scale = self.cpu_cache_client.cpu_kv_cache_tensor[
                    :, :, :, :, cpu_cache_meta.head_dim :
                ].view(mem_manager.scale_buffer.dtype)
                gpu_kv_cache_scale = mem_manager.scale_buffer
            else:
                cpu_kv_cache = self.cpu_cache_client.cpu_kv_cache_tensor
                cpu_kv_cache_scale = None
                gpu_kv_cache_scale = None

            # assert max(page_list) < self.cpu_cache_client.cpu_kv_cache_tensor.shape[0]
            offload_gpu_kv_to_cpu(
                token_indexes=token_indexes,
                gpu_kv_cache=mem_manager.kv_buffer,
                gpu_kv_cache_scale=gpu_kv_cache_scale,
                cpu_kv_cache=cpu_kv_cache,
                cpu_kv_cache_scale=cpu_kv_cache_scale,
                page_indexes=page_indexes,
                page_readies=page_readies,
                tp_index=self.backend.rank_in_dp,
                tp_world_size=self.backend.dp_world_size,
                grid_num=grid_num,
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
            page_array_list = [task.page_indexes.tolist() for task in trans_ok_tasks]
            if self.backend.is_master_in_dp:
                self.cpu_cache_client.lock.acquire_sleep1ms()
                # 分组update，避免不同请求的page交叉，导致disk cache hash不一致
                for pages in page_array_list:
                    self.cpu_cache_client.update_pages_status_to_ready(
                        page_list=pages, deref=True, disk_offload_enable=self.args.enable_disk_cache
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
