import bisect
import dataclasses
from collections import deque
from typing import Deque, Dict, List, Optional

import torch
import torch.distributed as dist

from lightllm.common.kv_cache_mem_manager import DeepseekV4MemoryManager
from lightllm.common.kv_cache_mem_manager.operator.deepseek import DeepseekV4MemOperator
from lightllm.server.multi_level_kv_cache.cpu_cache_client import CpuPageAllocState
from lightllm.server.router.model_infer.infer_batch import InferReq, g_infer_context
from lightllm.utils.envs_utils import get_dsv4_cpu_cache_max_pages_per_task

from .multi_level_kv_cache import MultiLevelKvCacheModule


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


class Dsv4MultiLevelKvCacheModule(MultiLevelKvCacheModule):
    def __init__(self, backend):
        super().__init__(backend)
        self._dsv4_store_sessions: Dict[int, Dsv4CpuStoreSession] = {}
        self._dsv4_store_tasks: Deque[Dsv4StoreTask] = deque()
        self._dsv4_staging_slots = [Dsv4StagingSlot(), Dsv4StagingSlot()]
        self._dsv4_max_pages_per_store_task = get_dsv4_cpu_cache_max_pages_per_task()

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
                slot = self._dsv4_staging_slots[task.staging_slot]
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
            for candidate, slot in enumerate(self._dsv4_staging_slots):
                if not slot.in_use:
                    slot_index = candidate
                    break
            if slot_index is None:
                self._poll_dsv4_store_tasks(wait_for_one=True)

        slot = self._dsv4_staging_slots[slot_index]
        page_num = len(store_pages)
        page_nbytes = self.backend.model.mem_manager.cpu_cache_layout.page_nbytes
        if slot.buffer is None or slot.buffer.shape[0] < page_num:
            with torch.cuda.stream(producer_stream):
                buffer = torch.empty((page_num, page_nbytes), dtype=torch.uint8, device="cuda")
                source_mem_indexes = torch.empty(
                    (page_num, self.backend.model.mem_manager.cpu_cache_layout.token_page_size),
                    dtype=torch.int32,
                    device="cuda",
                )
                page_indexes_cuda = torch.empty((page_num,), dtype=torch.int32, device="cuda")
            page_indexes_cpu = torch.empty((page_num,), dtype=torch.int32, device="cpu", pin_memory=True)
            slot.buffer = buffer
            slot.source_mem_indexes = source_mem_indexes
            slot.page_indexes_cuda = page_indexes_cuda
            slot.page_indexes_cpu = page_indexes_cpu
        slot.in_use = True

        slot.page_indexes_cpu[:page_num].numpy()[:] = owner_pages
        with torch.cuda.stream(producer_stream):
            source_mem_indexes = slot.source_mem_indexes[:page_num]
            torch.stack([item.source_mem_indexes for item in store_pages], out=source_mem_indexes)
            staging = slot.buffer[:page_num]
            operator.pack_cpu_cache_pages(source_mem_indexes, staging)
            pack_event = torch.cuda.Event()
            pack_event.record()

        # Request free/pause runs on the current stream.  It may recycle the
        # original DS4 slabs after packing, but never before it.
        torch.cuda.current_stream().wait_event(pack_event)
        with torch.cuda.stream(cpu_stream):
            cpu_stream.wait_event(pack_event)
            page_indexes_cuda = slot.page_indexes_cuda[:page_num]
            page_indexes_cuda.copy_(slot.page_indexes_cpu[:page_num], non_blocking=True)
            operator.scatter_packed_cpu_cache_pages(staging, page_indexes_cuda, self.cpu_cache_client)
            store_event = torch.cuda.Event()
            store_event.record()

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

    def load_cpu_cache_to_reqs(self, reqs: List[InferReq]):
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
                mem_manager: DeepseekV4MemoryManager = self.backend.model.mem_manager
                layout = mem_manager.cpu_cache_layout
                requested_end = int(page_len_list[len(page_list) - 1])
                if requested_end > gpu_kv_len:
                    swa_capacity, c4_capacity, c128_capacity = g_infer_context.get_can_alloc_dsv4_page_and_slot_num()
                    loadable_end = mem_manager.get_loadable_cpu_cache_end(
                        gpu_kv_len,
                        requested_end,
                        idle_token_num,
                        swa_capacity,
                        c4_capacity,
                        c128_capacity,
                    )
                    if loadable_end != 0:
                        token_num = loadable_end - gpu_kv_len
                        full_need = token_num
                        swa_need = 2
                        c4_need = token_num // 256 if mem_manager.n_c4 else 0
                        c128_need = token_num // 128 if mem_manager.n_c128 else 0
                        if self.backend.radix_cache is not None:
                            radix_cache = self.backend.radix_cache
                            radix_cache.free_radix_cache_to_get_enough_token(full_need)
                            radix_cache.free_radix_cache_to_get_enough_c4_pages(c4_need)
                            radix_cache.free_radix_cache_to_get_enough_c128_slots(c128_need)
                            swa_shortage = swa_need - int(mem_manager.swa_page_allocator.can_use_mem_size)
                            if swa_shortage > 0:
                                radix_cache.free_unreferenced_swa_pages(swa_shortage)

                        loadable_end = mem_manager.get_loadable_cpu_cache_end(
                            gpu_kv_len,
                            loadable_end,
                            int(mem_manager.allocator.can_use_mem_size),
                            int(mem_manager.swa_page_allocator.can_use_mem_size),
                            int(mem_manager.c4_page_allocator.can_use_mem_size) if mem_manager.n_c4 else 0,
                            int(mem_manager.c128_allocator.can_use_mem_size) if mem_manager.n_c128 else 0,
                        )
                        if loadable_end != 0:
                            loaded_end = loadable_end
                            token_num = loaded_end - gpu_kv_len
                            first_page_index = gpu_kv_len // layout.token_page_size
                            cpu_pages = page_list[first_page_index : loaded_end // layout.token_page_size]
                            page_indexes_cuda = torch.tensor(cpu_pages, dtype=torch.int32, device="cuda")
                            plan = mem_manager.prepare_cpu_cache_load(token_num=token_num, loaded_end=loaded_end)
                            mem_manager.operator.load_cpu_cache_pages(
                                plan=plan,
                                page_indexes=page_indexes_cuda,
                                cpu_cache_client=self.cpu_cache_client,
                                first_page_history_offset_tokens=gpu_kv_len % layout.token_page_size,
                            )
                            mem_manager.commit_cpu_cache_load_plan(plan)
                            self.backend.model.req_manager.req_to_token_indexs[
                                req.req_idx, gpu_kv_len:loaded_end
                            ] = plan.mem_indexes
                            self.backend.model.req_manager.finish_cpu_cache_load(req.req_idx, loaded_end)
                            req.cur_kv_len = loaded_end
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

    def offload_finished_reqs_to_cpu_cache(self, finished_reqs: List[InferReq]) -> List[InferReq]:
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

    def update_cpu_cache_task_states(self):
        self._poll_dsv4_store_tasks()
        return


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


@dataclasses.dataclass
class Dsv4StagingSlot:
    """可复用的 GPU staging buffer 及其 CPU page 索引缓冲区。"""

    # 打包后的连续 GPU 字节缓冲区，形状为 [page_capacity, page_nbytes]。
    buffer: Optional[torch.Tensor] = None
    # batch 内每个 checkpoint page 对应的 GPU KV slot 编号。
    source_mem_indexes: Optional[torch.Tensor] = None
    # Python 写入的 pinned CPU page 编号，用于异步拷贝到 GPU。
    page_indexes_cpu: Optional[torch.Tensor] = None
    # scatter kernel 使用的目标 CPU page 编号。
    page_indexes_cuda: Optional[torch.Tensor] = None
    # True 表示该 slot 仍被一个未完成的 store task 占用。
    in_use: bool = False
