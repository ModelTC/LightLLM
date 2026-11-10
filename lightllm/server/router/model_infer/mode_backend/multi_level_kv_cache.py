import torch.distributed as dist
import torch
import dataclasses
import bisect
from functools import lru_cache
from typing import Optional, List, Deque
from collections import deque
from lightllm.server.multi_level_kv_cache import CacheTier
from lightllm.server.multi_level_kv_cache.cpu_cache_client import CpuKvCacheClient
from lightllm.utils.envs_utils import get_env_start_args
from ..infer_batch import InferReq
from lightllm.utils.dist_utils import create_new_group_for_current_dp
from lightllm.server.router.model_infer.infer_batch import g_infer_context
from lightllm.utils.log_utils import init_logger
from lightllm.utils.multimodal_chunk_utils import clamp_mm_cache_len

logger = init_logger(__name__)


class MultiLevelKvCacheModule(object):
    def __init__(self, backend):
        self.args = get_env_start_args()
        assert self.args.cpu_cache_token_page_size % self.args.page_size == 0
        from .base_backend import ModeBackend

        self.backend: ModeBackend = backend
        self.gloo_group = create_new_group_for_current_dp("gloo")
        self.filter_group = create_new_group_for_current_dp("gloo")
        self.init_sync_group = create_new_group_for_current_dp("nccl")
        dist.barrier(group=self.init_sync_group)
        self.offload_sync_group = create_new_group_for_current_dp("nccl")
        dist.barrier(group=self.offload_sync_group)
        self.offload_sync_tensor = torch.empty((1,), dtype=torch.int32, device="cuda")
        self.load_sync_tensor = torch.empty((1,), dtype=torch.int32, device="cuda")

        self.offload_tasks: Deque[OffloadTask] = deque()
        self.load_tasks: Deque["LoadTask"] = deque()
        self.cpu_cache_client = CpuKvCacheClient(only_create_meta_data=False, init_shm_data=False)
        resources = self.backend.model.att_state_manager
        self.cpu_cache_operator = resources.get_cpu_cache_operator()
        meta = self.cpu_cache_client.kv_cache_tensor_meta
        if meta.page_bytes != self.cpu_cache_operator.get_page_bytes():
            raise ValueError("CPU cache service and model resource page sizes differ")

    @lru_cache()
    def need_sync_compute_stream(self) -> bool:
        """不支持异步 CPU cache 的 backend（目前是 FA3）仍要等计算流，否则搬运可能崩溃。"""

        model = self.backend.model
        att_backends = [
            model.prefill_att_backend,
            model.decode_att_backend,
            model.prefill_att_backend1,
            model.decode_att_backend1,
        ]
        for att_backend in att_backends:
            if att_backend is not None and att_backend.requires_cpu_cache_sync:
                logger.info(
                    "MultiLevelKvCacheModule: %s requires synchronous CPU cache IO.", type(att_backend).__name__
                )
                return True
        logger.info("MultiLevelKvCacheModule: CPU cache load waits on a recorded event.")
        return False

    def get_cpu_load_stream(self) -> torch.cuda.Stream:
        """把 CPU cache 拉回 GPU 的流，与计算流分开。"""
        stream = getattr(self, "_cpu_load_stream", None)
        if stream is None:
            stream = torch.cuda.Stream()
            setattr(self, "_cpu_load_stream", stream)
        return stream

    def get_cpu_offload_stream(self) -> torch.cuda.Stream:
        """把 GPU KV 写出到 CPU cache 的流。"""
        stream = getattr(self, "_cpu_offload_stream", None)
        if stream is None:
            stream = torch.cuda.Stream()
            setattr(self, "_cpu_offload_stream", stream)
        return stream

    def load_cpu_cache_to_reqs(self, reqs: List[InferReq]):
        idle_token_num = g_infer_context.get_can_alloc_token_num()
        direct_releases = []
        load_stream = self.get_cpu_load_stream()
        # 当前流上刚写好 token index 等元数据，拉取先等它。
        load_stream.wait_stream(torch.cuda.current_stream())
        if self.need_sync_compute_stream():
            # 计算在 overlap 流上，拷贝前先等已经提交的计算结束。
            load_stream.wait_stream(g_infer_context.get_overlap_stream())
        is_master_in_dp = self.backend.is_master_in_dp
        for req in reqs:
            page_list = req.shm_req.cpu_cache_match_page_indexes.get_all()
            # 需要返回 prompt logprobs 的请求不应加载 cpu cache：
            # 命中后会复用缓存 kv、跳过推理，拿不到对应 logprobs。
            # match 侧通常已跳过；这里仍要 deref 已 match 的 page，避免引用泄漏。
            if req.sampling_param.shm_param.prompt_logprobs >= 0:
                if is_master_in_dp:
                    req.shm_req.cpu_prompt_cache_len = 0
                direct_releases.append((req, page_list))
                continue

            page_len_list = req.shm_req.token_hash_page_len_list.get_all()
            assert len(page_list) <= len(page_len_list)

            matched_page_num = len(page_list)
            original_match_tokens = page_len_list[matched_page_num - 1] if matched_page_num else 0
            # CPU 和 disk 命中统一在恢复前裁剪；图片内部的 KV 可以保留，但不能作为恢复终点。
            while matched_page_num:
                match_tokens = page_len_list[matched_page_num - 1]
                cache_len = clamp_mm_cache_len(match_tokens, req.mm_input_spans)
                if cache_len == match_tokens:
                    break
                matched_page_num = bisect.bisect_right(page_len_list, cache_len, hi=matched_page_num)
            match_tokens = page_len_list[matched_page_num - 1] if matched_page_num else 0

            # 更新命中的 cpu kv cache 长度, 减去radix cache和disk cache的部分.
            if is_master_in_dp:
                req.shm_req.disk_prompt_cache_len = max(
                    0, req.shm_req.disk_prompt_cache_len - (original_match_tokens - match_tokens)
                )
                req.shm_req.cpu_prompt_cache_len = max(
                    0, match_tokens - req.cur_kv_len - req.shm_req.disk_prompt_cache_len
                )

            need_token_num = match_tokens - req.cur_kv_len
            # 命中够长、请求够长、剩余显存放得下，才进行复制，不然代价过高。
            if need_token_num >= 128 and req.shm_req.input_len >= 256 and need_token_num <= idle_token_num:
                assert req.cur_kv_len % self.args.page_size == 0
                assert match_tokens % self.args.page_size == 0
                assert need_token_num % self.args.page_size == 0
                assert req.hold_kv_len == req.cur_kv_len
                # 计算需要加载的页面（只加载未匹配的部分）
                ready_page_num = bisect.bisect_right(page_len_list, req.cur_kv_len)
                assert ready_page_num <= matched_page_num
                need_pages = page_list[ready_page_num:matched_page_num]

                start_token_len = page_len_list[ready_page_num - 1] if ready_page_num else 0
                with torch.cuda.stream(load_stream):
                    self.read_req_from_cpu(
                        req=req,
                        cpu_page_indices=need_pages,
                        page_lengths=page_len_list[ready_page_num:matched_page_num],
                        start_token_len=start_token_len,
                    )
                    # 等所有 DP rank 的这次拷贝都提交后，再记录完成事件。
                    if self.backend.dp_world_size > 1:
                        dist.all_reduce(self.load_sync_tensor, group=self.init_sync_group)
                    event = torch.cuda.Event()
                    event.record()
                self.load_tasks.append(LoadTask(req_obj=req, pages=page_list, event=event))
                req.cpu_cache_load_task_status = InferReq._CpuCacheTaskStatus.RUNNING
                idle_token_num -= need_token_num
                if is_master_in_dp:
                    req.shm_req.shm_cur_kv_len = req.cur_kv_len
            else:
                direct_releases.append((req, page_list))

        LoadTaskHelper(self).release_uncopied_pages(direct_releases)
        if self.need_sync_compute_stream():
            # 后续推理排在 overlap 流上，等到这次拉取结束再读 KV。CPU 不在这里阻塞。
            g_infer_context.get_overlap_stream().wait_stream(load_stream)
        return

    def update_cpu_cache_load_task_states(self) -> None:
        """把队头已经拷完的 load 出队，请求标为完成。多卡按 master 的数量一起出队，CPU 页由 master 释放。"""
        helper = LoadTaskHelper(self)
        finished_tasks = helper.pop_finished_load_tasks()
        for task in finished_tasks:
            task.req_obj.cpu_cache_load_task_status = InferReq._CpuCacheTaskStatus.FINISHED
        helper.deref_pages_on_master([task.pages for task in finished_tasks])

    def read_req_from_cpu(
        self,
        req: InferReq,
        cpu_page_indices: List[int],
        page_lengths: List[int],
        start_token_len: int,
    ) -> None:
        """恢复已选定的请求前缀页面，分配资源并登记完整大页引用。

        start_token_len 是首个 CPU 页的逻辑起点，可能早于已有 GPU 命中长度。
        调用方负责选择命中范围、确认容量以及搬运完成后的 CPU 页引用释放。
        """
        if not cpu_page_indices:
            return
        resources = self.backend.model.att_state_manager
        cpu_cache_operator = resources.get_cpu_cache_operator()
        req_manager = self.backend.model.req_manager
        need_token_num = page_lengths[-1] - req.cur_kv_len

        # 阶段一：分配新增 KV，并补齐首个 CPU 页在 GPU 上已经命中的 token 索引。
        mem_indexes = self.backend._alloc_req_kv_mem(req, need_token_num)
        assert mem_indexes is not None
        assert 0 <= start_token_len <= req.cur_kv_len, "invalid CPU page start"
        token_indices = torch.cat(
            [
                req_manager.req_to_token_indexs[req.req_idx, start_token_len : req.cur_kv_len],
                mem_indexes.cuda(non_blocking=True),
            ]
        )
        assert token_indices.numel() == page_lengths[-1] - start_token_len

        # 阶段二：只为完整 bounded 页分配普通大页，碎页 LOAD 槽位由资源层管理。
        big_page_slot_ids = None
        if self.backend.is_hybrid_att_model:
            page_tokens = self.args.cpu_cache_token_page_size
            complete_page_lengths = [length for length in page_lengths if length % page_tokens == 0]
            assert len(complete_page_lengths) == len(page_lengths) - int(
                page_lengths[-1] % page_tokens != 0
            ), "only the final CPU page may be partial"
            allocator = resources.bounded_att_big_page_allocator
            big_page_slot_ids = allocator.alloc(len(complete_page_lengths))
            assert big_page_slot_ids is not None, "not enough bounded state slots for CPU cache load"

        # 阶段三：统一恢复各类 State；异常直接向上传播，不在此处回收或继续执行。
        cpu_cache_operator.read_req(
            cpu_page_buffers=self.cpu_cache_client.cpu_kv_cache_tensor,
            cpu_page_indices=cpu_page_indices,
            token_indices=token_indices,
            layouts=cpu_cache_operator.get_page_layouts(),
            big_page_slot_ids=big_page_slot_ids,
            req_idx=req.req_idx if self.backend.is_hybrid_att_model else None,
            state_token_len=page_lengths[-1],
        )
        if big_page_slot_ids is not None:
            req.hybrid_len_to_big_page_id.update(zip(complete_page_lengths, big_page_slot_ids))
        req.cur_kv_len = page_lengths[-1]

    def offload_finished_reqs_to_cpu_cache(self, finished_reqs: List[InferReq]) -> List[InferReq]:
        """
        将满足cpu kv cache 卸载条件的请求进行处理, 并返回真的满足退出条件的请求list。
        """
        # 如果开启了cpu cache，将达到finished状态的请求开启将gpu kv cache 卸载到 cpu cache中的操作。
        # 当 kv cache 卸载完成后，才会进行请求的真实退出操作。
        if not finished_reqs:
            return []
        true_finished_reqs = []
        offload_stream = self.get_cpu_offload_stream()
        # 当前流上刚写好的 KV，写出先等它。
        offload_stream.wait_stream(torch.cuda.current_stream())
        if self.need_sync_compute_stream():
            # 计算在 overlap 流上，写出前先等已经提交的计算结束。
            offload_stream.wait_stream(g_infer_context.get_overlap_stream())
        for req in finished_reqs:
            # 只有 group_req_id 和 request_id 相同的请求才会被卸载到 cpu cache 中。
            # 这个限制是为了兼容 diverse 模式下的请求处理, 只有主请求才 offload kv 到 cpu
            # cache 中
            if req.shm_req.group_req_id != req.shm_req.request_id:
                true_finished_reqs.append(req)
                continue

            # 过滤不适合进行 kv 卸载到 cpu cache 的请求。
            if g_infer_context.is_hybrid_att_model:
                offload_limit_size = self.args.linear_att_hash_page_size
            else:
                offload_limit_size = self.args.cpu_cache_token_page_size

            if req.cur_kv_len < offload_limit_size or req.shm_req.input_len <= offload_limit_size:
                true_finished_reqs.append(req)
                continue

            # 如果请求已经完成了 cpu cache 的任务，则满足了退出条件
            if req.cpu_cache_offload_task_status.is_finished():
                true_finished_reqs.append(req)
                continue

            # 如果请求已经发起过卸载任务且正在卸载过程中，则在当前轮不进行处理
            if req.cpu_cache_offload_task_status.is_running():
                continue

            assert req.cpu_cache_offload_task_status.is_not_started()

            # 发起将请求的 kv cache 卸载到 cpu cache 中的任务
            trans_task = self.write_req_to_cpu(req)

            # 根据是否成功创建了卸载任务，决定是否将请求加入到处理队列中
            if trans_task is not None:
                self.offload_tasks.append(trans_task)
            else:
                true_finished_reqs.append(req)

        if self.need_sync_compute_stream():
            # 后续计算排在 overlap 流上，等到这次写出结束再读这块 KV。
            g_infer_context.get_overlap_stream().wait_stream(offload_stream)

        return true_finished_reqs

    def write_req_to_cpu(self, req: InferReq) -> Optional["OffloadTask"]:
        """选择请求可缓存的前缀，分配 CPU 页并提交写出，返回待确认完成的任务。"""
        assert CacheTier.CPU in req.cache_tiers
        disk_offload_enable = CacheTier.DISK in req.cache_tiers
        # KV 和 bounded 快照均在 offload 流上复制，由后续完成事件统一确认就绪。
        with torch.cuda.stream(self.get_cpu_offload_stream()):
            # 综合考虑后只对prompt做缓存管理，不包含decode内容，这里与radix cache不一致
            token_hash_list = req.shm_req.token_hash_list.get_all()
            page_len_list = req.shm_req.token_hash_page_len_list.get_all()
            assert len(token_hash_list) == len(page_len_list)

            # 阶段一：确定本次可写出范围，申请 CPU 页。
            if self.backend.is_master_in_dp:

                find_index = bisect.bisect_right(page_len_list, req.cur_kv_len)
                move_block_size = find_index

                # hybrid 模型的最后一个页面可能是碎页，需判断该碎页是否满足卸载条件。
                move_block_size = self._handle_hybrid_att_last_page(
                    req=req, move_block_size=move_block_size, page_len_list=page_len_list
                )

                if move_block_size == 0:
                    dist.broadcast_object_list([0], group=self.gloo_group, group_src=0)
                    req.cpu_cache_offload_task_status = InferReq._CpuCacheTaskStatus.FINISHED
                    return None

                try:
                    self.cpu_cache_client.lock.acquire_sleep1ms()
                    page_list, ready_list = self.cpu_cache_client.allocate_pages(
                        token_hash_list[:move_block_size],
                        disk_offload_enable=disk_offload_enable,
                    )
                finally:
                    self.cpu_cache_client.lock.release()

                # 分配不足时返回 -1 并补齐剩余位置，只保留成功分配的连续前缀。
                # 第一个 -1 及后续页面（包括可能的尾部碎页）均不参与本次写出。
                if -1 in page_list:
                    allocated_page_count = page_list.index(-1)
                    page_list = page_list[:allocated_page_count]
                    ready_list = ready_list[:allocated_page_count]
                item_size = len(page_list)
                if item_size == 0:
                    dist.broadcast_object_list([0], group=self.gloo_group, group_src=0)
                    req.cpu_cache_offload_task_status = InferReq._CpuCacheTaskStatus.FINISHED
                    return None

                broadcast_data = {"item_size": item_size, "page_list": page_list, "ready_list": ready_list}
                dist.broadcast_object_list([broadcast_data], group=self.gloo_group, group_src=0)
            else:
                recv_list = [None]
                dist.broadcast_object_list(recv_list, group=self.gloo_group, group_src=0)
                if isinstance(recv_list[0], int) and recv_list[0] == 0:
                    req.cpu_cache_offload_task_status = InferReq._CpuCacheTaskStatus.FINISHED
                    return None
                broadcast_data = recv_list[0]
                item_size = broadcast_data["item_size"]
                page_list = broadcast_data["page_list"]
                ready_list = broadcast_data["ready_list"]

            page_indexes = torch.tensor(page_list, dtype=torch.int32)
            page_readies = torch.tensor(ready_list, dtype=torch.bool)

            move_token_num = page_len_list[item_size - 1]
            assert req.cur_kv_len >= move_token_num
            token_indexes = self.backend.model.req_manager.req_to_token_indexs[req.req_idx, 0:move_token_num]

            # 阶段二：只查找实际分配的 CPU 页所需快照；完整大页必须存在，不做缺失回退。
            resources = self.backend.model.att_state_manager
            big_page_slot_ids, small_page_slot_ids = None, None
            if self.backend.is_hybrid_att_model:
                page_tokens = self.args.cpu_cache_token_page_size
                radix_cache = g_infer_context.radix_cache
                shared_slots = (
                    radix_cache.get_big_page_ids_by_node(req.shared_kv_node) if radix_cache is not None else []
                )
                big_slots = {(i + 1) * page_tokens: slot for i, slot in enumerate(shared_slots)}
                big_slots.update(req.hybrid_len_to_big_page_id)
                big_page_slot_ids = [
                    big_slots[length] for length in page_len_list[:item_size] if length % page_tokens == 0
                ]
                small_page_slot_ids = []
                # 只有实际分配的前缀包含原始尾页，才可能使用小页快照。
                # 若分配遇到 -1，item_size 已截短，尾页不会进入此分支。
                includes_tail_page = item_size == len(page_len_list)
                if includes_tail_page and page_len_list[-1] % page_tokens != 0:
                    small_page_slot_ids.append(req.tail_small_page_buffer_id)
            # 整批提交并保留 token/page 对齐，由资源层跳过 ready 页。
            resources.get_cpu_cache_operator().write_req(
                token_indices=token_indexes,
                cpu_page_buffers=self.cpu_cache_client.cpu_kv_cache_tensor,
                cpu_page_indices=page_list,
                layouts=resources.get_cpu_cache_operator().get_page_layouts(),
                cpu_page_ready_flags=ready_list,
                big_page_slot_ids=big_page_slot_ids,
                small_page_slot_ids=small_page_slot_ids,
            )

            # 这个操作只是为了在offload 对应的cuda stream中，同步标记下对应的kv cache offload 操作已经完成，
            if self.backend.dp_world_size > 1:
                dist.all_reduce(self.offload_sync_tensor, op=dist.ReduceOp.MAX, group=self.offload_sync_group)

            sync_event = torch.cuda.Event()
            sync_event.record()
            req.cpu_cache_offload_task_status = InferReq._CpuCacheTaskStatus.RUNNING
            trans_task = OffloadTask(
                move_token_num=move_token_num,
                page_indexes=page_indexes,
                page_readies=page_readies,
                req_obj=req,
                sync_event=sync_event,
            )

        return trans_task

    def _handle_hybrid_att_last_page(self, req: InferReq, move_block_size: int, page_len_list: List[int]) -> int:
        if not g_infer_context.is_hybrid_att_model:
            return move_block_size

        if move_block_size == 0:
            return 0

        if move_block_size == len(page_len_list):
            tail_len = page_len_list[move_block_size - 1]
            if tail_len % self.args.cpu_cache_token_page_size != 0:
                # 全局关闭了碎页的cpu cache 存储功能。
                if self.args.disable_linear_att_small_page_cpu_cache:
                    return move_block_size - 1
                # 说明是碎页，碎页需要判定是否满足cpu cache 的offload条件。
                if req.tail_small_page_buffer_id is None:
                    return move_block_size - 1
        return move_block_size

    def update_cpu_cache_offload_task_states(self):
        if self.backend.is_master_in_dp:
            trans_ok_tasks = []
            while len(self.offload_tasks) != 0:
                task: OffloadTask = self.offload_tasks.popleft()
                if task.sync_event.query():
                    trans_ok_tasks.append(task)
                else:
                    self.offload_tasks.appendleft(task)
                    break
            item_size = len(trans_ok_tasks)
            dist.broadcast_object_list([item_size], group=self.filter_group, group_src=0)
        else:
            recv_list = [None]
            dist.broadcast_object_list(recv_list, group=self.filter_group, group_src=0)
            item_size = recv_list[0]
            trans_ok_tasks: List[OffloadTask] = [self.offload_tasks.popleft() for _ in range(item_size)]

        if item_size > 0:
            page_array_list = [task.page_indexes.tolist() for task in trans_ok_tasks]
            move_token_nums = [task.move_token_num for task in trans_ok_tasks]
            if self.backend.is_master_in_dp:
                self.cpu_cache_client.lock.acquire_sleep1ms()
                # 分组update，避免不同请求的page交叉，导致disk cache hash不一致
                for task, pages, move_token_num in zip(trans_ok_tasks, page_array_list, move_token_nums):
                    self.cpu_cache_client.update_pages_status_to_ready(
                        page_list=pages,
                        deref=True,
                        disk_offload_enable=CacheTier.DISK in task.req_obj.cache_tiers,
                        token_num_in_page_list=move_token_num,
                    )
                self.cpu_cache_client.lock.release()
            for task in trans_ok_tasks:
                task.req_obj.cpu_cache_offload_task_status = InferReq._CpuCacheTaskStatus.FINISHED
        return


@dataclasses.dataclass
class OffloadTask:
    move_token_num: int
    page_indexes: torch.Tensor
    page_readies: torch.Tensor
    req_obj: InferReq
    sync_event: torch.cuda.Event


@dataclasses.dataclass
class LoadTask:
    req_obj: InferReq
    pages: List[int]
    event: torch.cuda.Event


class LoadTaskHelper:
    """load 任务的出队和 CPU 页释放。多卡按 master 查到的完成数量一起出队。"""

    def __init__(self, module: "MultiLevelKvCacheModule"):
        self.module = module

    def release_uncopied_pages(self, direct_releases: List[tuple]) -> None:
        """没有发起拷贝的命中页可以马上还引用，load 状态直接记为完成。"""
        if not direct_releases:
            return
        for req, _pages in direct_releases:
            req.cpu_cache_load_task_status = InferReq._CpuCacheTaskStatus.FINISHED
        self.deref_pages_on_master([pages for _req, pages in direct_releases if pages])

    def pop_finished_load_tasks(self) -> List[LoadTask]:
        if not self.module.load_tasks:
            return []

        finished_count = self._finished_load_count()
        finished_tasks = []
        for _ in range(finished_count):
            finished_tasks.append(self.module.load_tasks.popleft())
        return finished_tasks

    def deref_pages_on_master(self, page_lists: List[List[int]]) -> None:
        """CPU cache 引用只由 DP master 持有。"""
        if not page_lists or not self.module.backend.is_master_in_dp:
            return
        client = self.module.cpu_cache_client
        client.lock.acquire_sleep1ms()
        try:
            for pages in page_lists:
                client.deref_pages(pages)
        finally:
            client.lock.release()

    def _finished_load_count(self) -> int:
        """队头连续完成的任务数。多卡只由 master 查 event，再把数量广播给同组。"""
        backend = self.module.backend
        if backend.is_master_in_dp:
            finished_count = 0
            for task in self.module.load_tasks:
                if not task.event.query():
                    break
                finished_count += 1
            if backend.dp_world_size == 1:
                return finished_count
        else:
            finished_count = None

        payload = [finished_count]
        dist.broadcast_object_list(payload, group=self.module.filter_group, group_src=0)
        return payload[0]
