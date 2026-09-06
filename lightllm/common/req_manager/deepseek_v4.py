from dataclasses import dataclass
from typing import List, Optional

import torch

from lightllm.common.kv_cache_mem_manager.deepseek4_mem_manager import (
    DSV4_C4_PAGE_SIZE,
    DSV4_PROMPT_CACHE_PAGE_SIZE,
    DeepseekV4MemoryManager,
)
from lightllm.server.router.model_infer.pin_mem_manager import g_pin_mem_manager

from .base import ReqManager


@dataclass
class DeepseekV4PromptCachePayload:
    """prompt cache 载荷: swa 按页有效性 bitmap 和最后有效页。

    槽位与 compressor 状态都不进载荷: full_to_swa/full_to_c4/full_to_c128 以 full token 槽位
    为键(radix 持有 full 槽 ⇒ 映射行存活,free 级联回收);c4 compressor 状态随 swa 页
    生灭。c128 状态按 request ring 寻址；prompt cache 的 256-token 边界同时是 c128
    分组边界，命中后新分组会在首次读取前覆写完整 128-token 窗口，因而无需保存状态。

    * ``swa_page_valid``: cpu bool [cache_len // page]，插入时按当下 full_to_swa 映射写定
      (页内 token 映射全有效才为 True)。匹配层据此把命中裁剪到"结尾页有效"的 page 边界,
      swa 压力阀回收节点页时清零。"""

    cache_len: int
    swa_page_valid: Optional[torch.Tensor] = None
    swa_last_valid_page: int = -1

    def refresh_swa_last_valid_page(self) -> None:
        if self.swa_page_valid is None:
            self.swa_last_valid_page = -1
            return
        valid_idx = torch.nonzero(self.swa_page_valid).flatten()
        self.swa_last_valid_page = -1 if valid_idx.numel() == 0 else int(valid_idx[-1].item())
        return

    def valid_match_length(self, natural_len: int, page: int) -> int:
        if self.swa_last_valid_page < 0:
            return 0
        return (int(self.swa_last_valid_page) + 1) * page


class DeepseekV4PromptCacheValueOps:
    def __init__(self, req_manager: "DeepseekV4ReqManager"):
        self.req_manager = req_manager

    def slice(self, payload: DeepseekV4PromptCachePayload, start: int, end: int):
        return self.req_manager.slice_prompt_cache_payload(payload, start, end)

    def concat(self, payloads: List[DeepseekV4PromptCachePayload]):
        return self.req_manager.concat_prompt_cache_payloads(payloads)

    def free(self, payload: DeepseekV4PromptCachePayload):
        # 槽位资源全部由 mem_manager.free(full_slots) 级联回收，载荷本身没有需要释放的资源。
        return

    def valid_match_length(self, payload: Optional[DeepseekV4PromptCachePayload], natural_len: int) -> int:
        """radix 匹配裁剪: 返回 <= natural_len 的最大 prompt-cache 边界 L'，使结尾页有效。

        有效性可能非单调(owner 生前从左驱逐、后续阀从尾回收)，中段 invalid 页不挡更
        靠后的有效命中(注意力只回看最后一个窗口)。"""
        if payload is None:
            return 0
        return payload.valid_match_length(natural_len, self.req_manager.get_prompt_cache_page_size())


class DeepseekV4ReqManager(ReqManager):
    """DeepSeek-V4 的请求级管理。

    负责 req/seq/MTP 布局、SWA 回收水位线和派生槽位准备；具体池结构、映射和分配器
    由 ``DeepseekV4MemoryManager`` 持有。对象先于 mem manager 创建，模型初始化后再接入。
    """

    def __init__(
        self,
        max_request_num,
        max_sequence_length,
        mem_manager: Optional[DeepseekV4MemoryManager] = None,
        sliding_window: Optional[int] = None,
    ):
        super().__init__(max_request_num, max_sequence_length, mem_manager)

        self.sliding_window = sliding_window
        # 出窗回收水位线: -1 表示该 req 尚未见过 prefill chunk(首个 chunk 的 ready_cache_len
        # 即共享前缀边界，作为永不下探的回收下界)。
        self._swa_evict_marks = [-1 for _ in range(max_request_num + 1)]
        return

    # ------------------------------------------------------------------ swa slot prep (per step)
    def _swa_retain_len(self) -> int:
        """出窗回收的保留长度 = window + 一个 radix 页。

        多留一页使「最近一个完成的 prompt-cache 边界」的结尾页恒驻留: 若回收只留 window，
        则任何非对齐时刻该边界的结尾页都已被部分回收，插入门会把所有插入裁到 0。
        V4 prompt-cache 页取 256 token，正好覆盖一个 c4 物理页对应的 token 范围。"""
        return int(self.sliding_window) + self.get_prompt_cache_page_size()

    def _align_swa_evict_frontier(self, raw_frontier: int) -> int:
        """SWA 回收水位线按 prompt-cache 页向下对齐。

        bitmap 的有效性是 prompt-cache page 粒度；若水位线切进页面中间，该页会被判为
        invalid，即使靠近命中边界的窗口实际仍完整驻留。"""
        page = self.get_prompt_cache_page_size()
        raw_frontier = max(0, int(raw_frontier))
        return raw_frontier // page * page

    def prepare_prefill_swa(
        self,
        req_list: List[int],
        ready_list: List[int],
        seq_list: List[int],
        mem_indexes: torch.Tensor,
    ) -> None:
        """prefill prep: 为本 chunk 全部新 token(位置 [ready, seq))分配位置对齐的 swa 槽，
        并回收已出窗位置的槽。

        本 chunk 起点 L = ready_cache_len，首个新 token(位置 L)的窗口是 [L-W+1, L]；回收
        边界再额外保留一个 radix 页(_swa_retain_len)，即位置 < L-retain+1。先回收再分配。
        当前 chunk 的 full slots 直接使用 generic preprocess 分配的 mem_indexes，因而可以
        在通用 req_to_token scatter 之前执行。"""
        self.mem_manager: DeepseekV4MemoryManager
        if self.sliding_window is not None:
            retain = self._swa_retain_len()
            evict_slots = []
            for req_idx, ready_len in zip(req_list, ready_list):
                if req_idx == self.HOLD_REQUEST_ID:
                    continue
                mark = self._swa_evict_marks[req_idx]
                if mark < 0:
                    # 首个 chunk: [0, ready_len) 是 radix 共享前缀，其 swa 槽归 radix 所有，不可回收。
                    self._swa_evict_marks[req_idx] = self._align_swa_evict_frontier(ready_len)
                    continue
                evict_end = self._align_swa_evict_frontier(ready_len - retain + 1)
                if evict_end > mark:
                    evict_slots.append(self.req_to_token_indexs[req_idx, mark:evict_end])
                    self._swa_evict_marks[req_idx] = evict_end
            if evict_slots:
                self.mem_manager.evict_swa(torch.cat(evict_slots))
        self.mem_manager.alloc_swa_prefill(
            mem_indexes,
            self.req_to_token_indexs,
            req_list=req_list,
            ready_list=ready_list,
            seq_list=seq_list,
        )
        return

    def prepare_decode(
        self,
        b_req_idx_cpu,
        b_seq_len_cpu,
        b_mtp_index_cpu,
        mem_indexes,
        mtp_decode_slot_prepare_indices,
        prepare_compress_slots=True,
    ):
        """decode 每步槽位 prep。在 BaseModel 的通用 req scatter 与 attention metadata
        构建前调用；DeepSeek-V4 MTP draft layer 只需要 SWA 槽位。"""
        max_mtp_index = int(b_mtp_index_cpu.max().item())
        if mtp_decode_slot_prepare_indices is None:
            steps = range(max_mtp_index + 1)
        else:
            steps = mtp_decode_slot_prepare_indices

        batch_size = b_mtp_index_cpu.shape[0]
        slots_per_req = max_mtp_index + 1
        assert batch_size % slots_per_req == 0
        req_list = b_req_idx_cpu.tolist()
        seq_list = b_seq_len_cpu.tolist()
        mem_indexes_by_req = mem_indexes.reshape(-1, slots_per_req)
        for step in steps:
            step_req_list = req_list[step::slots_per_req]
            step_seq_list = seq_list[step::slots_per_req]
            self.prepare_decode_swa(
                step_req_list,
                step_seq_list,
                mem_indexes_by_req[:, step],
                prev_mem_indexes=mem_indexes_by_req[:, step - 1] if step > 0 else None,
            )
            if prepare_compress_slots:
                self.prepare_decode_compress_slots(
                    step_req_list,
                    step_seq_list,
                    mem_indexes_by_req[:, step],
                    prev_group_end_mem_indexes=mem_indexes_by_req[:, step - 4] if step >= 4 else None,
                )
        return

    def prepare_prefill(
        self,
        b_req_idx_cpu: torch.Tensor,
        b_ready_cache_len_cpu: torch.Tensor,
        b_seq_len_cpu: torch.Tensor,
        mem_indexes: torch.Tensor,
    ) -> None:
        """prefill 槽位 prep: 直接消费 generic preprocess 分配的 full slots，在
        BaseModel 的通用 req scatter 与 attention metadata 构建之前完成。"""
        req_list = b_req_idx_cpu.tolist()
        ready_list = b_ready_cache_len_cpu.tolist()
        seq_list = b_seq_len_cpu.tolist()
        mem_indexes = mem_indexes.reshape(-1)
        self.prepare_prefill_swa(
            req_list=req_list,
            ready_list=ready_list,
            seq_list=seq_list,
            mem_indexes=mem_indexes,
        )
        self.prepare_prefill_compress_slots(
            req_list=req_list,
            ready_list=ready_list,
            seq_list=seq_list,
            mem_indexes=mem_indexes,
        )
        return

    def prepare_pd_decode_cache(
        self,
        req_list: List[int],
        ready_list: List[int],
        seq_list: List[int],
        new_full_slots: torch.Tensor,
    ) -> None:
        """Allocate DSV4 derived slots for request-major suffixes received from peers."""
        page = self.get_prompt_cache_page_size()
        assert len(req_list) == len(ready_list) == len(seq_list) and len(req_list) > 0
        assert all(ready % page == 0 and seq_len > ready for ready, seq_len in zip(ready_list, seq_list))
        assert new_full_slots.numel() == sum(seq_len - ready for ready, seq_len in zip(ready_list, seq_list))

        new_full_slots = new_full_slots.reshape(-1).to(self.req_to_token_indexs.device, non_blocking=True)
        self.prepare_prefill_compress_slots(
            req_list=req_list,
            ready_list=ready_list,
            seq_list=seq_list,
            mem_indexes=new_full_slots,
        )

        # swa 只保存最后一部分，前面的不需要
        swa_start_list = []
        swa_parts = []
        offset = 0
        # swa 不一样
        for ready, seq_len in zip(ready_list, seq_list):
            swa_start = max(ready, max(0, seq_len // page * page - page))
            swa_start_list.append(swa_start)
            suffix_len = seq_len - ready
            swa_parts.append(new_full_slots[offset + swa_start - ready : offset + suffix_len])
            offset += suffix_len
        swa_full_slots = swa_parts[0] if len(swa_parts) == 1 else torch.cat(swa_parts)

        self.mem_manager.alloc_swa_prefill(
            swa_full_slots,
            self.req_to_token_indexs,
            req_list=req_list,
            ready_list=swa_start_list,
            seq_list=seq_list,
        )
        for req_idx, swa_start in zip(req_list, swa_start_list):
            self._swa_evict_marks[req_idx] = swa_start
        return

    def prepare_decode_swa(
        self,
        req_list: List[int],
        seq_list: List[int],
        mem_indexes: torch.Tensor,
        prev_mem_indexes: Optional[torch.Tensor] = None,
    ) -> None:
        """decode prep: 回收出窗槽并为本步新 token 分配位置对齐的 swa 槽。当前 query 位置
        seq_len-1 的窗口是 [seq_len-W, seq_len-1]；回收边界额外保留一个 radix 页
        (_swa_retain_len)，即位置 < seq_len-retain。先回收再分配。
        seq_len/req_idx 从 CPU 镜像读(host 算术,无 D2H);水位线 _swa_evict_marks 仍是 host 状态。"""
        assert self.mem_manager is not None
        if self.sliding_window is not None:
            retain = self._swa_retain_len()
            evict_slots = []
            for req_idx, seq_len in zip(req_list, seq_list):
                if req_idx == self.HOLD_REQUEST_ID:
                    continue
                mark = self._swa_evict_marks[req_idx]
                if mark < 0:
                    # direct-decode 中 [0, seq_len-1) 是已有 KV；exact hit 时这段前缀归 radix 所有，
                    # 水位必须从前缀末端开始，不能由请求回收。
                    self._swa_evict_marks[req_idx] = self._align_swa_evict_frontier(seq_len - 1)
                    continue
                evict_end = self._align_swa_evict_frontier(seq_len - retain)
                if evict_end > mark:
                    evict_slots.append(self.req_to_token_indexs[req_idx, mark:evict_end])
                    self._swa_evict_marks[req_idx] = evict_end
            if evict_slots:
                self.mem_manager.evict_swa(torch.cat(evict_slots))
        if prev_mem_indexes is None:
            prev_meta = g_pin_mem_manager.gen_from_list(
                key="dsv4_swa_decode_prev",
                data=[x for req_idx, seq_len in zip(req_list, seq_list) for x in (req_idx, seq_len - 2)],
                dtype=torch.int64,
            ).to(self.req_to_token_indexs.device, non_blocking=True)
            prev_meta = prev_meta.view(-1, 2)
            prev_mem_indexes = self.req_to_token_indexs[prev_meta[:, 0], prev_meta[:, 1]]
        self.mem_manager.alloc_swa_decode(
            req_list,
            seq_list,
            mem_indexes,
            prev_mem_indexes,
        )
        return

    def init_compress_state(self, req_idx: int):
        """新请求开始时重置 runtime 水位线(对应 mamba 的 init_linear_att_state 调用点)。

        c4 状态随 swa 页寻址；c128 request ring 依靠 overwrite-before-read，不做大块清零。"""
        self.clear_runtime_state(req_idx)
        return

    def finish_cpu_cache_load(self, req_idx: int, loaded_len: int) -> None:
        """Keep only the final restored 256-token SWA page eligible for radix reuse."""
        self._swa_evict_marks[req_idx] = loaded_len - self.get_prompt_cache_page_size()
        return

    # ------------------------------------------------------------------ compress slot prep (per step)
    def _register_c4_slots(self, full_slots: torch.Tensor, slots: torch.Tensor) -> None:
        """写入 full->c4 槽映射并按页累加存活计数。"""
        self.mem_manager.full_to_c4_indexs[full_slots] = slots
        self.mem_manager.count_c4_slots(slots, 1)

    def _scatter_c4_prefill_slots_batched(self, req_list, ready_list, seq_list, mem_indexes) -> None:
        """Batch c4 prefill scatter from the generic preprocess full-slot layout.

        Each group's end token is in the current chunk, so its full slot is addressed directly in
        mem_indexes. Only a mid-page continuation reads the previous group's old req-table entry.
        New full slots guarantee a fresh mapping; no GPU-to-CPU idempotency check is needed."""
        page = DSV4_C4_PAGE_SIZE
        mapping = self.mem_manager.full_to_c4_indexs
        device = mapping.device

        plan = []
        mem_offset = 0
        for req_idx, ready_len, seq_len in zip(req_list, ready_list, seq_list):
            q_len = seq_len - ready_len
            if req_idx == self.HOLD_REQUEST_ID:
                mem_offset += q_len
                continue
            first, last = ready_len // 4, seq_len // 4
            if last <= first:
                mem_offset += q_len
                continue
            plan.append((req_idx, ready_len, mem_offset, first, last))
            mem_offset += q_len
        if not plan:
            return

        def to_cuda_long(key, data):
            return g_pin_mem_manager.gen_from_list(key=key, data=data, dtype=torch.int64).to(device, non_blocking=True)

        reqs, readies, mem_offsets, firsts, lasts = zip(*plan)
        counts = [last - first for first, last in zip(firsts, lasts)]
        first_pages = [first // page for first in firsts]
        page_counts = [((last - 1) // page) - fp + 1 for last, fp in zip(lasts, first_pages)]
        page_offsets, total_pages = [], 0
        for n_pages in page_counts:
            page_offsets.append(total_pages)
            total_pages += n_pages
        total_entries = sum(counts)
        cont = [(off, req, first) for off, req, first in zip(page_offsets, reqs, firsts) if first % page != 0]
        self._realize_c4_pages(total_pages - len(cont))

        # One pinned H2D copy for all per-request metadata, then per-entry ragged expansion.
        meta = to_cuda_long(
            "dsv4_c4_prefill_meta",
            [x for row in zip(readies, mem_offsets, firsts, first_pages, counts, page_offsets) for x in row],
        ).view(-1, 6)
        readies_t, mem_offsets_t, firsts_t, first_pages_t, counts_t, page_offsets_t = meta.unbind(1)
        seg = torch.repeat_interleave(torch.arange(len(plan), device=device), counts_t, output_size=total_entries)
        seg_starts = counts_t.cumsum(0) - counts_t
        entries = firsts_t[seg] + torch.arange(total_entries, device=device) - seg_starts[seg]
        full_offsets = mem_offsets_t[seg] + entries * 4 + 3 - readies_t[seg]
        full_slots = mem_indexes.reshape(-1)[full_offsets]

        # physical base per logical page: fresh pages from one alloc; mid-page continuations read prev
        if not cont:
            page_bases = self.mem_manager.alloc_c4_pages(total_pages).to(device, non_blocking=True) * page
        else:
            page_bases = torch.empty(total_pages, dtype=torch.int32, device=device)
            new_pos = [
                pos
                for off, n_pages, first in zip(page_offsets, page_counts, firsts)
                for pos in range(off + (first % page != 0), off + n_pages)
            ]
            if new_pos:
                new_pos_t = to_cuda_long("dsv4_c4_prefill_new_pos", new_pos)
                page_bases[new_pos_t] = (
                    self.mem_manager.alloc_c4_pages(len(new_pos)).to(device, non_blocking=True) * page
                )
            cont_t = to_cuda_long("dsv4_c4_prefill_cont", [x for row in cont for x in row]).view(-1, 3)
            prev_slot = mapping[self.req_to_token_indexs[cont_t[:, 1], cont_t[:, 2] * 4 - 1]]
            cont_off = ((cont_t[:, 2] - 1) % page).to(torch.int32)
            page_bases[cont_t[:, 0]] = prev_slot - cont_off

        page_idx = page_offsets_t[seg] + torch.div(entries, page, rounding_mode="floor") - first_pages_t[seg]
        slots = page_bases[page_idx] + (entries % page).to(torch.int32)
        self._register_c4_slots(full_slots, slots)
        return

    def _scatter_c4_decode_slots(
        self,
        req_list,
        seq_list,
        mem_indexes: torch.Tensor,
        prev_group_end_mem_indexes: Optional[torch.Tensor] = None,
    ) -> None:
        page = DSV4_C4_PAGE_SIZE
        mapping = self.mem_manager.full_to_c4_indexs
        mem_indexes = mem_indexes.reshape(-1)

        cont_rows, cont_prev_pos = [], []
        new_rows = []
        for i, (req_idx, seq_len) in enumerate(zip(req_list, seq_list)):
            if req_idx == self.HOLD_REQUEST_ID or seq_len <= 0 or seq_len % 4 != 0:
                continue
            entry = seq_len // 4 - 1
            offset = entry % page
            if offset == 0:
                new_rows.append(i)
            else:
                cont_rows.append(i)
                cont_prev_pos.append(entry * 4 - 1)

        if cont_rows:
            if prev_group_end_mem_indexes is None:
                prev_meta = g_pin_mem_manager.gen_from_list(
                    key="dsv4_c4_decode_prev",
                    data=[x for row in zip([req_list[i] for i in cont_rows], cont_prev_pos) for x in row],
                    dtype=torch.int64,
                ).to(mapping.device, non_blocking=True)
                prev_meta = prev_meta.view(-1, 2)
                prev_full = self.req_to_token_indexs[prev_meta[:, 0], prev_meta[:, 1]]
            else:
                prev_full = (
                    prev_group_end_mem_indexes.reshape(-1)
                    if len(cont_rows) == len(req_list)
                    else prev_group_end_mem_indexes.reshape(-1)[cont_rows]
                )
            prev_slots = mapping[prev_full]
            dst_indexes = mem_indexes if len(cont_rows) == len(req_list) else mem_indexes[cont_rows]
            self._register_c4_slots(dst_indexes, prev_slots + 1)

        if new_rows:
            self._realize_c4_pages(len(new_rows))  # 兑现: 精确需求, 复用已算的 new_rows
            pages = self.mem_manager.alloc_c4_pages(len(new_rows)).to(mapping.device, non_blocking=True)
            dst_indexes = mem_indexes if len(new_rows) == len(req_list) else mem_indexes[new_rows]
            self._register_c4_slots(dst_indexes, pages * page)
        return

    def _scatter_c128_slots(self, full_slots: torch.Tensor) -> None:
        """为本批新组末 full 槽分配 c128 槽并写入映射。"""
        if full_slots.numel() == 0:
            return
        full_slots = full_slots.reshape(-1)
        self._realize_c128_slots(full_slots.numel())
        new_slots = self.mem_manager.alloc_c128(full_slots.numel()).cuda(non_blocking=True)
        self.mem_manager.full_to_c128_indexs[full_slots] = new_slots
        return

    def _realize_c4_pages(self, need_pages: int) -> None:
        """压缩池兑现 —— 和主池在 prep 里调 free_radix_cache_to_get_enough_token 同一套路:
        base_backend admission 已按"空闲+可回收"放行本步请求,这里在真分配前(scatter 已算好 need)
        把可回收的无引用 radix 节点驱逐出来腾出 c4 页,避免 alloc_c4_pages 触底 assert。
        可回收仍不足时由 admission 的 wait_pause 兜底。"""
        if self.mem_manager.n_c4 == 0 or need_pages <= 0:
            return
        # 延迟 import: infer_batch 在模块顶 import 了 req_manager,顶层 import 会循环引用
        from lightllm.server.router.model_infer.infer_batch import g_infer_context

        if g_infer_context.radix_cache is not None:
            g_infer_context.radix_cache.free_radix_cache_to_get_enough_c4_pages(need_pages)
        return

    def _realize_c128_slots(self, need_slots: int) -> None:
        if self.mem_manager.n_c128 == 0 or need_slots <= 0:
            return
        from lightllm.server.router.model_infer.infer_batch import g_infer_context

        if g_infer_context.radix_cache is not None:
            g_infer_context.radix_cache.free_radix_cache_to_get_enough_c128_slots(need_slots)
        return

    def prepare_prefill_compress_slots(
        self,
        req_list: List[int],
        ready_list: List[int],
        seq_list: List[int],
        mem_indexes: torch.Tensor,
    ) -> None:
        """prefill prep: 为本 chunk 内的组末 token(位置 (g+1)*ratio-1 ∈ [ready, seq))分配压缩槽，
        组末 full 槽直接从 generic preprocess 的 mem_indexes 取。"""
        if self.mem_manager.n_c4 == 0 and self.mem_manager.n_c128 == 0:
            return
        if self.mem_manager.n_c4 > 0:
            self._scatter_c4_prefill_slots_batched(req_list, ready_list, seq_list, mem_indexes)

        if self.mem_manager.n_c128 > 0:
            ratio = 128
            full_offsets = []
            mem_offset = 0
            for req_idx, ready_len, seq_len in zip(req_list, ready_list, seq_list):
                q_len = seq_len - ready_len
                if req_idx == self.HOLD_REQUEST_ID:
                    mem_offset += q_len
                    continue
                first, last = ready_len // ratio, seq_len // ratio
                if last > first:
                    full_offsets.extend(
                        mem_offset + (entry + 1) * ratio - 1 - ready_len for entry in range(first, last)
                    )
                mem_offset += q_len
            if full_offsets:
                offsets = g_pin_mem_manager.gen_from_list(
                    key="dsv4_c128_prefill_offsets", data=full_offsets, dtype=torch.int64
                ).to(mem_indexes.device, non_blocking=True)
                self._scatter_c128_slots(mem_indexes.reshape(-1)[offsets])
        return

    def prepare_decode_compress_slots(
        self,
        req_list: List[int],
        seq_list: List[int],
        mem_indexes: torch.Tensor,
        prev_group_end_mem_indexes: Optional[torch.Tensor] = None,
    ) -> None:
        """decode prep: 本步 token 关闭一个组(seq_len % ratio == 0)时为其分配压缩槽并 scatter。
        组末 full 槽即本步的 mem_index。
        从 CPU 镜像读 seq_len/req_idx(host 算术,无 D2H);非关组步 rows 为空 => 不调 _scatter,零同步。"""
        if self.mem_manager.n_c4 == 0 and self.mem_manager.n_c128 == 0:
            return
        if self.mem_manager.n_c4 > 0:
            self._scatter_c4_decode_slots(
                req_list,
                seq_list,
                mem_indexes,
                prev_group_end_mem_indexes=prev_group_end_mem_indexes,
            )

        if self.mem_manager.n_c128 > 0:
            ratio = 128
            rows = [
                i
                for i, (req_idx, seq_len) in enumerate(zip(req_list, seq_list))
                if req_idx != self.HOLD_REQUEST_ID and seq_len > 0 and seq_len % ratio == 0
            ]
            if rows:
                full_slots = mem_indexes.reshape(-1)
                if len(rows) != len(req_list):
                    full_slots = full_slots[rows]
                self._scatter_c128_slots(full_slots)
        return

    def alloc(self):
        req_idx = super().alloc()
        if req_idx is not None:
            self.init_compress_state(req_idx)
        return req_idx

    def clear_runtime_state(self, req_idx: int):
        # swa 槽位本身由 mem_manager.free 级联回收(随 full 槽位)，这里只复位出窗水位线。
        self._swa_evict_marks[req_idx] = -1
        return

    def get_prompt_cache_value_ops(self):
        return DeepseekV4PromptCacheValueOps(self)

    def get_prompt_cache_page_size(self):
        return DSV4_PROMPT_CACHE_PAGE_SIZE

    def compute_swa_page_valid(self, full_slots: torch.Tensor) -> torch.Tensor:
        """按当下 full_to_swa 映射给出按页有效性: full_slots [L](L 为 page 整数倍) ->
        cpu bool [L/page]，页内全部映射有效才为 True。GPU gather + 同步,测试/校验用;
        插入热路径用 swa_page_valid_from_watermark(纯 CPU,免同步)。"""
        page = self.get_prompt_cache_page_size()
        assert full_slots.numel() % page == 0
        if full_slots.numel() == 0:
            return torch.zeros((0,), dtype=torch.bool)
        swa = self.mem_manager.full_to_swa_indexs[full_slots.cuda().long().reshape(-1)]
        return (swa.view(-1, page) >= 0).all(dim=1).cpu()

    def swa_page_valid_from_watermark(self, req_idx: int, cache_len: int) -> torch.Tensor:
        """插入时的按页有效性,纯 CPU: 请求自有 token 的 swa 映射只被出窗水位线回收
        (阀不触活跃请求,级联只在 free 时),页 p 全驻留 ⟺ 页起点 page*p >= 水位线。

        与 compute_swa_page_valid 在插入时刻对自有 token 等价,但不做 GPU gather/同步——
        router 关键路径上每次插入省一次对全部在途 kernel 的等待。bitmap 中借入前缀
        ([0, ready) 的页)的行在 radix insert 切片时被丢弃(既有节点保留自己的 bitmap),
        其取值无影响。"""
        page = self.get_prompt_cache_page_size()
        mark = max(0, self._swa_evict_marks[req_idx])
        n_pages = int(cache_len) // page
        return torch.arange(n_pages, dtype=torch.long) * page >= mark

    def slice_prompt_cache_payload(self, payload: DeepseekV4PromptCachePayload, start: int, end: int):
        start = int(start)
        end = int(end)
        page = self.get_prompt_cache_page_size()
        # radix page 保证分裂点页对齐，bitmap 可整页切分。
        ans = DeepseekV4PromptCachePayload(
            cache_len=end - start,
            swa_page_valid=payload.swa_page_valid[start // page : end // page].clone()
            if payload.swa_page_valid is not None
            else None,
        )
        ans.refresh_swa_last_valid_page()
        return ans

    def concat_prompt_cache_payloads(self, payloads: List[DeepseekV4PromptCachePayload]):
        if len(payloads) == 0:
            return None
        bitmaps = [p.swa_page_valid for p in payloads]
        ans = DeepseekV4PromptCachePayload(
            cache_len=sum(p.cache_len for p in payloads),
            swa_page_valid=torch.cat(bitmaps, dim=0) if all(b is not None for b in bitmaps) else None,
        )
        if ans.swa_page_valid is None:
            return ans

        page = self.get_prompt_cache_page_size()
        page_offset = 0
        last_valid_page = -1
        for item in payloads:
            item_last = int(getattr(item, "swa_last_valid_page", -1))
            if item_last >= 0:
                last_valid_page = page_offset + item_last
            page_offset += int(item.cache_len) // page
        ans.swa_last_valid_page = last_valid_page
        return ans

    def build_prompt_cache_payload(
        self,
        cache_len: int,
    ) -> DeepseekV4PromptCachePayload:
        """构造插入载荷。compressor 状态不进载荷(c4 随 swa 页生灭、c128 在 256 对齐
        恢复点依靠 overwrite-before-read),cache_len 不再受序列末端约束。
        swa_page_valid 不在此填: 它必须用插入时刻的映射(infer batch 在 insert 前补)。"""
        assert self.mem_manager is not None
        return DeepseekV4PromptCachePayload(cache_len=int(cache_len))

    def free(self, free_req_indexes, free_token_index):
        """dense/swa/压缩槽全部经 mem_manager.free(free_token_index) 级联回收。"""
        for req_index in free_req_indexes:
            self.clear_runtime_state(req_index)
        super().free(free_req_indexes, free_token_index)
        return

    def free_req(self, free_req_index: int):
        self.clear_runtime_state(free_req_index)
        return super().free_req(free_req_index)

    def free_all(self):
        super().free_all()
        self._swa_evict_marks = [-1 for _ in range(self.max_request_num + 1)]
        return
