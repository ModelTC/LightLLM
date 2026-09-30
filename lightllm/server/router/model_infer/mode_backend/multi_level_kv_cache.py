import threading
import torch.distributed as dist
import torch
import dataclasses
import bisect
from functools import lru_cache
from typing import Optional, List, Deque, Dict
from collections import deque
from lightllm.server.multi_level_kv_cache import CacheTier
from lightllm.server.multi_level_kv_cache.cpu_cache_client import CpuKvCacheClient, CpuPageAllocState
from lightllm.utils.config_utils import is_hybrid_att_model
from lightllm.utils.envs_utils import get_env_start_args, get_dsv4_cpu_cache_max_pages_per_task
from ..infer_batch import InferReq
from lightllm.utils.dist_utils import create_new_group_for_current_dp
from lightllm.common.basemodel.triton_kernel.kv_cache_offload import offload_gpu_kv_to_cpu, load_cpu_kv_to_gpu
from lightllm.server.router.model_infer.infer_batch import g_infer_context
from lightllm.utils.log_utils import init_logger
from lightllm.common.kv_cache_mem_manager.operator.deepseek import DeepseekV4MemOperator

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

        self.page_index_buffer = torch.empty((1024 * 1024 * 4,), dtype=torch.int32, device="cuda")
        self.page_ready_buffer = torch.empty((1024 * 1024 * 4,), dtype=torch.bool, device="cuda")

        self.cpu_cache_handle_queue: Deque[TransTask] = deque()
        self.cpu_cache_client = CpuKvCacheClient(only_create_meta_data=False, init_shm_data=False)
        if isinstance(self.backend.model.mem_manager.operator, DeepseekV4MemOperator):
            self._dsv4_store_sessions: Dict[int, Dsv4CpuStoreSession] = {}
            self._dsv4_store_tasks: Deque[Dsv4StoreTask] = deque()
            self._dsv4_max_pages_per_store_task = get_dsv4_cpu_cache_max_pages_per_task()

    @lru_cache()
    def need_sync_compute_stream(self) -> bool:
        """
        fa3 在 offload 和 load kv cache 的时候，需要等待计算流完成，否则可能会概率崩溃。
        """

        model = self.backend.model
        att_backends = [
            model.prefill_att_backend,
            model.decode_att_backend,
            model.prefill_att_backend1,
            model.decode_att_backend1,
        ]
        for att_backend in att_backends:
            if att_backend is not None and "fa3" in att_backend.__class__.__name__.lower():
                logger.info("MultiLevelKvCacheModule: need sync compute stream for fa3 backend.")
                return True
        logger.info("MultiLevelKvCacheModule: no need sync compute stream.")
        return False

    def load_cpu_cache_to_reqs(self, reqs: List[InferReq]):
        cache_reqs = []
        pages_to_release = []
        is_master_in_dp = self.backend.is_master_in_dp
        is_deepseek_v4 = g_infer_context.is_deepseek_v4
        for req in reqs:
            # KV 命中会跳过计算，无法返回对应的 prompt logprobs。
            # match 侧通常已跳过；这里仍需释放之前匹配的页面引用。
            skip_cpu_cache = req.sampling_param.shm_param.prompt_logprobs >= 0
            if skip_cpu_cache:
                if is_master_in_dp:
                    req.shm_req.cpu_prompt_cache_len = 0
                    req.shm_req.disk_prompt_cache_len = 0
            else:
                cache_reqs.append(req)
            # DSV4 正常加载的页面由 session 持有，等待异步 load/store 完成。
            if is_master_in_dp and (skip_cpu_cache or not is_deepseek_v4):
                pages_to_release.extend(req.shm_req.cpu_cache_match_page_indexes.get_all())

        if is_deepseek_v4:
            self._load_dsv4_cpu_cache_to_reqs(cache_reqs)
        else:
            self._load_standard_cpu_cache_to_reqs(cache_reqs)

        if is_master_in_dp and pages_to_release:
            self.cpu_cache_client.lock.acquire_sleep1ms()
            try:
                self.cpu_cache_client.deref_pages(pages_to_release)
            finally:
                self.cpu_cache_client.lock.release()
        return

    def _load_standard_cpu_cache_to_reqs(self, reqs: List[InferReq]):
        idle_token_num = g_infer_context.get_can_alloc_token_num()
        is_master_in_dp = self.backend.is_master_in_dp
        for req in reqs:
            page_list = req.shm_req.cpu_cache_match_page_indexes.get_all()
            page_len_list = req.shm_req.token_hash_page_len_list.get_all()
            page_len_start_list = [0] + page_len_list
            assert len(page_list) <= len(page_len_list)

            if page_list:
                match_tokens = page_len_list[len(page_list) - 1]
            else:
                match_tokens = 0

            # 更新命中的 cpu kv cache 长度, 减去radix cache和disk cache的部分.
            if is_master_in_dp:
                req.shm_req.cpu_prompt_cache_len = max(
                    0, match_tokens - req.cur_kv_len - req.shm_req.disk_prompt_cache_len
                )

            need_token_num = match_tokens - req.cur_kv_len
            # 多匹配了一定数量的token同时请求长度大于一定的长度，才进行复制操作，不然操作效率不高，代价过高
            if need_token_num >= 128 and req.shm_req.input_len >= 256:
                assert req.cur_kv_len % self.args.page_size == 0
                assert match_tokens % self.args.page_size == 0
                assert need_token_num % self.args.page_size == 0
                assert req.hold_kv_len == req.cur_kv_len
                if need_token_num <= idle_token_num:
                    # 计算需要加载的页面（只加载未匹配的部分）
                    ready_page_num = bisect.bisect_right(page_len_list, req.cur_kv_len)
                    assert ready_page_num <= len(page_list)
                    need_pages = page_list[ready_page_num:]  # 只取需要的页面

                    mem_indexes = self.backend._alloc_req_kv_mem(req, need_token_num)
                    assert mem_indexes is not None

                    if self.need_sync_compute_stream():
                        # TODO fa3 现在必须使用同步模式, 未来需要移除
                        torch.cuda.current_stream().wait_stream(g_infer_context.get_overlap_stream())
                        # g_infer_context.get_overlap_stream().synchronize()

                    mem_manager = self.backend.model.mem_manager
                    req_manager = self.backend.model.req_manager

                    mem_indexes_cuda = mem_indexes.cuda(non_blocking=True)
                    page_indexes_cuda = torch.tensor(need_pages, dtype=torch.int32, device="cpu").cuda(
                        non_blocking=True
                    )
                    # hybrid 页面加载必须按完整 page 处理，否则可能缺失恢复运行态所需的 checkpoint，所以
                    # 这里需要进行pad操作，使操作的页面是完整的。
                    _start = page_len_start_list[ready_page_num]

                    _end = req.cur_kv_len
                    assert 0 <= _start <= _end, f"invalid pad range [{_start}, {_end}]"
                    mem_indexes_cuda = torch.cat(
                        [req_manager.req_to_token_indexs[req.req_idx, _start:_end], mem_indexes_cuda]
                    )

                    assert (
                        len(mem_indexes_cuda) == page_len_list[len(page_list) - 1] - page_len_start_list[ready_page_num]
                    )

                    # 更新 req 状态。
                    idle_token_num -= need_token_num
                    req.cur_kv_len = req.cur_kv_len + need_token_num

                    mem_manager.operator.load_cpu_cache_to_gpu(
                        mem_indexes=mem_indexes_cuda,
                        page_indexes=page_indexes_cuda,
                        cpu_cache_client=self.cpu_cache_client,
                        req=req,
                    )

                torch.cuda.current_stream().synchronize()

                if self.backend.is_master_in_dp:
                    req.shm_req.shm_cur_kv_len = req.cur_kv_len

        dist.barrier(group=self.init_sync_group)
        return

    def offload_finished_reqs_to_cpu_cache(self, finished_reqs: List[InferReq]) -> List[InferReq]:
        """
        将满足cpu kv cache 卸载条件的请求进行处理, 并返回真的满足退出条件的请求list。
        """
        # 如果开启了cpu cache，将达到finished状态的请求开启将gpu kv cache 卸载到 cpu cache中的操作。
        # 当 kv cache 卸载完成后，才会进行请求的真实退出操作。
        if g_infer_context.is_deepseek_v4:
            return self._finish_dsv4_cpu_cache_sessions(finished_reqs)
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
            if g_infer_context.is_hybrid_att_model:
                offload_limit_size = self.args.linear_att_hash_page_size
            else:
                offload_limit_size = self.args.cpu_cache_token_page_size

            if req.cur_kv_len < offload_limit_size or req.shm_req.input_len <= offload_limit_size:
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

            if self.need_sync_compute_stream():
                # TODO fa3 现在必须使用同步模式, 未来需要移除, 必须等待 overlap stream 上的计算任务完成，不然会崩溃
                g_infer_context.get_overlap_stream().synchronize()

            # 发起将请求的 kv cache 卸载到 cpu cache 中的任务
            trans_task = self._start_kv_cache_offload_task(req=req, cpu_kv_cache_stream=cpu_stream)

            # 根据是否成功创建了卸载任务，决定是否将请求加入到处理队列中
            if trans_task is not None:
                self.cpu_cache_handle_queue.append(trans_task)
            else:
                true_finished_reqs.append(req)

        if self.need_sync_compute_stream():
            # TODO fa3 现在必须使用同步模式, 未来需要移除
            cpu_stream.synchronize()

        return true_finished_reqs

    def _start_kv_cache_offload_task(
        self, req: InferReq, cpu_kv_cache_stream: torch.cuda.Stream
    ) -> Optional["TransTask"]:
        assert CacheTier.CPU in req.cache_tiers
        disk_offload_enable = CacheTier.DISK in req.cache_tiers
        with torch.cuda.stream(cpu_kv_cache_stream):
            # 综合考虑后只对prompt做缓存管理，不包含decode内容，这里与radix cache不一致
            token_hash_list = req.shm_req.token_hash_list.get_all()
            page_len_list = req.shm_req.token_hash_page_len_list.get_all()
            assert len(token_hash_list) == len(page_len_list)

            if self.backend.is_master_in_dp:

                find_index = bisect.bisect_right(page_len_list, req.cur_kv_len)
                move_block_size = find_index

                # hybrid 模型的最后一个页面可能是碎页，需判断该碎页是否满足卸载条件。
                move_block_size = self._handle_hybrid_att_last_page(
                    req=req, move_block_size=move_block_size, page_len_list=page_len_list
                )

                if move_block_size == 0:
                    dist.broadcast_object_list([0], group=self.gloo_group, group_src=0)
                    req.cpu_cache_task_status = InferReq._CpuCacheTaskStatus.FINISHED
                    return None

                try:
                    self.cpu_cache_client.lock.acquire_sleep1ms()
                    page_list, alloc_states = self.cpu_cache_client.allocate_pages(
                        token_hash_list[:move_block_size],
                        disk_offload_enable=disk_offload_enable,
                    )
                    ready_list = [state is CpuPageAllocState.READY_EXISTING for state in alloc_states]
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
            assert len(page_indexes) <= self.page_index_buffer.shape[0]
            cuda_page_indexes = self.page_index_buffer[: len(page_indexes)]
            cuda_page_readies = self.page_ready_buffer[: len(page_readies)]
            cuda_page_indexes.copy_(page_indexes, non_blocking=True)
            cuda_page_readies.copy_(page_readies, non_blocking=True)

            move_token_num = page_len_list[item_size - 1]
            assert req.cur_kv_len >= move_token_num
            token_indexes = self.backend.model.req_manager.req_to_token_indexs[req.req_idx, 0:move_token_num]

            mem_manager = self.backend.model.mem_manager

            mem_manager.operator.offload_gpu_kv_to_cpu_cache(
                mem_indexes=token_indexes,
                page_indexes=cuda_page_indexes,
                page_readies=cuda_page_readies,
                cpu_cache_client=self.cpu_cache_client,
                req=req,
            )

            # 这个操作只是为了在offload 对应的cuda stream中，同步标记下对应的kv cache offload 操作已经完成，
            if self.backend.dp_world_size > 1:
                dist.all_reduce(self.offload_sync_tensor, op=dist.ReduceOp.MAX, group=self.offload_sync_group)

            sync_event = torch.cuda.Event()
            sync_event.record()
            req.cpu_cache_task_status = InferReq._CpuCacheTaskStatus.RUNNING
            trans_task = TransTask(
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

    def update_cpu_cache_task_states(self):
        if g_infer_context.is_deepseek_v4:
            if self.backend.is_master_in_dp:
                self._poll_dsv4_store_tasks()
            return
        if not g_infer_context.infer_req_ids:
            return
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
                task.req_obj.cpu_cache_task_status = InferReq._CpuCacheTaskStatus.FINISHED
        return

    def _try_release_dsv4_session(self, session: "Dsv4CpuStoreSession") -> None:
        if not session.closing or not session.load_submitted or session.pending_task_num != 0:
            return
        if session.load_event is not None and not session.load_event.query():
            return

        if session.leased_pages:
            self.cpu_cache_client.lock.acquire_sleep1ms()
            try:
                if self.args.enable_disk_cache:
                    # A disk-cache group must contain one complete request prefix in
                    # root-to-tail order.  Incremental store batches may complete in
                    # a different order, so do not publish or release any page until
                    # every page leased by this session is ready.
                    if not self.cpu_cache_client.check_allpages_ready(session.leased_pages):
                        return
                    self.cpu_cache_client.update_pages_status_to_ready(
                        page_list=session.leased_pages,
                        deref=True,
                        disk_offload_enable=True,
                        token_num_in_page_list=(len(session.leased_pages) * self.args.cpu_cache_token_page_size),
                    )
                else:
                    # Cumulative hashes make the root page the most valuable entry.
                    # Releasing tail-to-root makes the tail oldest in the LRU.
                    self.cpu_cache_client.deref_pages(list(reversed(session.leased_pages)))
            finally:
                self.cpu_cache_client.lock.release()
        del self._dsv4_store_sessions[session.request_id]

    def _poll_dsv4_store_tasks(self, wait_for_one: bool = False) -> None:
        if not self._dsv4_store_tasks:
            for session in list(self._dsv4_store_sessions.values()):
                self._try_release_dsv4_session(session)
            return

        completed = []
        if wait_for_one:
            self._dsv4_store_tasks[0].store_event.synchronize()
        while self._dsv4_store_tasks and self._dsv4_store_tasks[0].store_event.query():
            completed.append(self._dsv4_store_tasks.popleft())
        if completed:
            self.cpu_cache_client.lock.acquire_sleep1ms()
            try:
                for task in completed:
                    self.cpu_cache_client.update_pages_status_to_ready(task.owner_pages, deref=False)
            finally:
                self.cpu_cache_client.lock.release()

            touched_sessions = {}
            for task in completed:
                slot = self.backend.model.mem_manager.operator.cpu_cache_staging_slots[task.staging_slot]
                slot.in_use = False
                for session in task.sessions:
                    session.pending_task_num -= 1
                    assert session.pending_task_num >= 0
                    touched_sessions[session.request_id] = session
            for session in touched_sessions.values():
                self._try_release_dsv4_session(session)
        for session in list(self._dsv4_store_sessions.values()):
            self._try_release_dsv4_session(session)

    def _submit_dsv4_store_batch(
        self,
        store_pages: List["Dsv4StorePage"],
        producer_stream: torch.cuda.Stream,
    ) -> None:
        assert 0 < len(store_pages) <= self._dsv4_max_pages_per_store_task
        sessions = {item.session.request_id: item.session for item in store_pages}
        owner_pages = [item.cpu_page_index for item in store_pages]
        operator: DeepseekV4MemOperator = self.backend.model.mem_manager.operator
        cpu_stream = g_infer_context.get_cpu_kv_cache_stream()

        self._poll_dsv4_store_tasks()
        slot_index = None
        while slot_index is None:
            for candidate, slot in enumerate(operator.cpu_cache_staging_slots):
                if not slot.in_use:
                    slot_index = candidate
                    break
            if slot_index is None:
                self._poll_dsv4_store_tasks(wait_for_one=True)

        pack_event, store_event = operator.store_cpu_cache_pages(
            staging_slot=slot_index,
            source_mem_indexes=[item.source_mem_indexes for item in store_pages],
            source_req_meta=[[item.req_idx, item.checkpoint_len] for item in store_pages],
            page_indexes=owner_pages,
            cpu_cache_client=self.cpu_cache_client,
            producer_stream=producer_stream,
            cpu_stream=cpu_stream,
        )

        for session in sessions.values():
            session.pending_task_num += 1
        self._dsv4_store_tasks.append(
            Dsv4StoreTask(
                owner_pages=owner_pages,
                sessions=list(sessions.values()),
                staging_slot=slot_index,
                pack_event=pack_event,
                store_event=store_event,
            )
        )

    def store_completed_prefill_pages(
        self,
        reqs: List[InferReq],
        producer_stream: torch.cuda.Stream,
    ) -> None:
        """Incrementally snapshot newly completed DS4 checkpoints before source reuse."""
        if not g_infer_context.is_deepseek_v4 or not self.backend.is_master_in_dp:
            return
        layout = self.backend.model.mem_manager.cpu_cache_layout
        token_page_size = layout.token_page_size
        store_pages: List[Dsv4StorePage] = []
        closing_sessions = {}
        self.cpu_cache_client.lock.acquire_sleep1ms()
        try:
            for req in reqs:
                session = self._dsv4_store_sessions.get(req.req_id)
                if session is None or session.closing:
                    continue
                token_hashes = req.shm_req.token_hash_list.get_all()
                if session.disabled:
                    closing_sessions[session.request_id] = session
                    continue
                if session.next_page_index >= len(token_hashes):
                    closing_sessions[session.request_id] = session
                    continue
                page_lens = req.shm_req.token_hash_page_len_list.get_all()
                target_page_index = bisect.bisect_right(page_lens, req.cur_kv_len)
                if target_page_index <= session.next_page_index:
                    continue

                start_page_index = session.next_page_index
                page_indexes, alloc_states = self.cpu_cache_client.allocate_pages(
                    token_hashes[start_page_index:target_page_index],
                    disk_offload_enable=False,
                )
                for offset, (cpu_page_index, alloc_state) in enumerate(zip(page_indexes, alloc_states)):
                    if cpu_page_index == -1:
                        session.disabled = True
                        break
                    checkpoint_index = start_page_index + offset
                    session.leased_pages.append(cpu_page_index)
                    session.next_page_index += 1
                    if alloc_state is CpuPageAllocState.NEW_STORE_OWNER:
                        token_start = checkpoint_index * token_page_size
                        source_mem_indexes = self.backend.model.req_manager.req_to_token_indexs[
                            req.req_idx, token_start : token_start + token_page_size
                        ]
                        store_pages.append(
                            Dsv4StorePage(
                                session=session,
                                cpu_page_index=cpu_page_index,
                                source_mem_indexes=source_mem_indexes,
                                req_idx=req.req_idx,
                                checkpoint_len=token_start + token_page_size,
                            )
                        )
                if session.disabled or session.next_page_index >= len(token_hashes):
                    closing_sessions[session.request_id] = session
        finally:
            self.cpu_cache_client.lock.release()

        for offset in range(0, len(store_pages), self._dsv4_max_pages_per_store_task):
            self._submit_dsv4_store_batch(
                store_pages[offset : offset + self._dsv4_max_pages_per_store_task],
                producer_stream=producer_stream,
            )
        for session in closing_sessions.values():
            session.closing = True
            self._try_release_dsv4_session(session)

    @staticmethod
    def _get_image_safe_load_end(req: InferReq, loaded_start: int, load_end: int, page_size: int) -> int:
        """Move an image-internal CPU resume point before that image."""
        for image_start, image_end in reversed(req.image_block_spans):
            if image_start < load_end < image_end:
                load_end = image_start // page_size * page_size
        return load_end if load_end > loaded_start else 0

    def _load_dsv4_cpu_cache_to_reqs(self, reqs: List[InferReq]):
        idle_token_num = g_infer_context.get_can_alloc_token_num()
        is_master_in_dp = self.backend.is_master_in_dp
        for req in reqs:
            page_list = req.shm_req.cpu_cache_match_page_indexes.get_all()
            page_len_list = req.shm_req.token_hash_page_len_list.get_all()
            assert len(page_list) <= len(page_len_list)

            gpu_kv_len = int(req.cur_kv_len)
            requested_end = gpu_kv_len
            matched_disk_len = int(req.shm_req.disk_prompt_cache_len)
            if is_master_in_dp:
                session = Dsv4CpuStoreSession(
                    request_id=req.req_id,
                    next_page_index=len(page_list),
                    leased_pages=list(page_list),
                )
                page_size = self.backend.model.mem_manager.cpu_cache_layout.token_page_size
                # A radix-owned checkpoint without its CPU prefix creates an unreachable hash-chain hole.
                if gpu_kv_len // page_size > session.next_page_index:
                    session.disabled = True
                    session.closing = True
                self._dsv4_store_sessions[req.req_id] = session

            loaded_end = gpu_kv_len
            if page_list:
                mem_manager = self.backend.model.mem_manager
                layout = mem_manager.cpu_cache_layout
                requested_end = int(page_len_list[len(page_list) - 1])
                if requested_end > gpu_kv_len:
                    swa_capacity = g_infer_context.get_can_alloc_dsv4_swa_page_num()
                    loadable_end = mem_manager.get_loadable_cpu_cache_end(
                        gpu_kv_len,
                        requested_end,
                        idle_token_num,
                        swa_capacity,
                    )
                    loadable_end = self._get_image_safe_load_end(req, gpu_kv_len, loadable_end, layout.token_page_size)
                    if loadable_end != 0:
                        token_num = loadable_end - gpu_kv_len
                        full_need = token_num
                        if self.backend.radix_cache is not None:
                            radix_cache = self.backend.radix_cache
                            radix_cache.free_radix_cache_to_get_enough_token(full_need)

                        loadable_end = mem_manager.get_loadable_cpu_cache_end(
                            gpu_kv_len,
                            loadable_end,
                            int(mem_manager.allocator.can_use_mem_size),
                            int(mem_manager.swa_page_allocator.can_use_mem_size),
                        )
                        loadable_end = self._get_image_safe_load_end(
                            req, gpu_kv_len, loadable_end, layout.token_page_size
                        )
                        if loadable_end != 0:
                            loaded_end = loadable_end
                            token_num = loaded_end - gpu_kv_len
                            first_page_index = gpu_kv_len // layout.token_page_size
                            cpu_pages = page_list[first_page_index : loaded_end // layout.token_page_size]
                            page_indexes_cuda = torch.tensor(cpu_pages, dtype=torch.int32, device="cuda")
                            mem_indexes = mem_manager.alloc(token_num).cuda(non_blocking=True)
                            try:
                                mem_manager.operator.load_cpu_cache_to_gpu(
                                    mem_indexes=mem_indexes,
                                    page_indexes=page_indexes_cuda,
                                    cpu_cache_client=self.cpu_cache_client,
                                    req=req,
                                )
                            except Exception:
                                mem_manager.free(mem_indexes)
                                raise
                            self.backend.model.req_manager.req_to_token_indexs[
                                req.req_idx, gpu_kv_len:loaded_end
                            ] = mem_indexes
                            req.cur_kv_len = loaded_end
                            req.hold_kv_len = loaded_end
                            idle_token_num -= token_num

            if is_master_in_dp:
                cpu_prompt_cache_len, disk_prompt_cache_len = _split_dsv4_loaded_cache_lengths(
                    original_gpu_kv_len=gpu_kv_len,
                    loaded_end=loaded_end,
                    requested_end=requested_end,
                    disk_prompt_cache_len=matched_disk_len,
                )
                req.shm_req.cpu_prompt_cache_len = cpu_prompt_cache_len
                req.shm_req.disk_prompt_cache_len = disk_prompt_cache_len
                req.shm_req.shm_cur_kv_len = loaded_end
                session.load_submitted = True
                if loaded_end > gpu_kv_len:
                    session.load_event = torch.cuda.Event()
                    session.load_event.record()

        dist.barrier(group=self.init_sync_group)
        if is_master_in_dp:
            for session in list(self._dsv4_store_sessions.values()):
                self._try_release_dsv4_session(session)
        return

    def _finish_dsv4_cpu_cache_sessions(self, finished_reqs: List[InferReq]) -> List[InferReq]:
        if self.backend.is_master_in_dp:
            for req in finished_reqs:
                session = self._dsv4_store_sessions.get(req.req_id)
                if session is not None:
                    session.closing = True
                    self._try_release_dsv4_session(session)
            self._poll_dsv4_store_tasks()
        # Source pages are fenced by the pack event.  Request teardown does
        # not wait for the independent staging-to-host transfer.
        return finished_reqs


@dataclasses.dataclass
class TransTask:
    move_token_num: int
    page_indexes: torch.Tensor
    page_readies: torch.Tensor
    req_obj: InferReq
    sync_event: torch.cuda.Event


def _split_dsv4_loaded_cache_lengths(
    original_gpu_kv_len: int,
    loaded_end: int,
    requested_end: int,
    disk_prompt_cache_len: int,
) -> tuple[int, int]:
    """Split an actual CPU-cache load into CPU and disk matched token counts."""
    load_start = max(0, int(original_gpu_kv_len))
    load_end = max(load_start, int(loaded_end))
    actual_loaded_len = load_end - load_start

    matched_end = max(0, int(requested_end))
    matched_disk_len = min(max(0, int(disk_prompt_cache_len)), matched_end)
    disk_start = matched_end - matched_disk_len
    actual_disk_len = max(0, min(load_end, matched_end) - max(load_start, disk_start))
    actual_disk_len = min(actual_disk_len, actual_loaded_len)
    actual_cpu_len = actual_loaded_len - actual_disk_len
    return actual_cpu_len, actual_disk_len


@dataclasses.dataclass
class Dsv4CpuStoreSession:
    """跟踪单个 DS4 请求持有的 CPU pages 及其异步 load/store 生命周期。"""

    request_id: int
    # [0, next_page_index) 的 checkpoint pages 已处理；该值指向下一个待处理 page。
    next_page_index: int = 0
    # 本 session 持有引用的 CPU page 编号，session 释放时统一 deref。
    leased_pages: List[int] = dataclasses.field(default_factory=list)
    # 已提交但尚未完成的 GPU -> CPU store batch 数量。
    pending_task_num: int = 0
    # 当前 hash 链无法继续存储，不再预留新的 CPU page。
    disabled: bool = False
    # 不再接收新的 store page，等待 load/store 完成后释放 session。
    closing: bool = False
    # 本请求的初始 CPU -> GPU load 流程已经提交。
    load_submitted: bool = False
    # 初始 CPU -> GPU load 的完成事件；没有实际 load 时为 None。
    load_event: Optional[torch.cuda.Event] = None


@dataclasses.dataclass
class Dsv4StorePage:
    """描述一个由当前请求负责写入的GPU page -> CPU page"""

    # 持有该 CPU page 引用并跟踪异步任务的请求 session。
    session: Dsv4CpuStoreSession
    # 已预留、等待写入的目标 CPU page 编号。
    cpu_page_index: int
    # 该 checkpoint page 对应的 GPU KV slot 编号。
    source_mem_indexes: torch.Tensor
    req_idx: int
    checkpoint_len: int


@dataclasses.dataclass
class Dsv4StoreTask:
    """跟踪一个已经提交的异步 GPU -> CPU store batch。"""

    # 本 batch 负责写入的 CPU pages，完成后统一发布为 READY。
    owner_pages: List[int]
    # 本 batch 涉及的请求 session，完成后分别减少 pending_task_num。
    sessions: List[Dsv4CpuStoreSession]
    # 本 batch 占用的 staging slot 编号。
    staging_slot: int
    # GPU KV 已打包完成；此事件完成后原始 KV slot 可以被回收。
    pack_event: torch.cuda.Event
    # staging 数据已写入 CPU；轮询此事件判断整个 batch 是否完成。
    store_event: torch.cuda.Event
