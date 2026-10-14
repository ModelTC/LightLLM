import math
from typing import Optional

import torch

from .hybrid_base import HybridAttentionReqManager
from .base import RequestResourceBudget
from lightllm.common.kv_cache_mem_manager import DeepseekV4MemoryManager
from lightllm.common.kv_cache_mem_manager.deepseek4_cache_layout import (
    DSV4_SWA_PAGE_SIZE,
    DSV4_CACHE_BLOCK_SIZE,
)
from lightllm.common.state_cache_manager.deepseek4 import DeepseekV4StateCacheManager


class DeepseekV4ReqManager(HybridAttentionReqManager):
    """Own request-private SWA pages and restore aligned continuation checkpoints.

    The page table covers the whole sequence, so a prefill chunk can keep all of
    its new KV until attention completes. Only pages preceding the next chunk's
    retained window are recycled. Speculative retries reuse their held pages.
    """

    def __init__(
        self,
        max_request_num,
        max_sequence_length,
        mem_manager: Optional[DeepseekV4MemoryManager] = None,
        sliding_window=None,
    ):
        super().__init__(max_request_num, max_sequence_length, None)
        self.sliding_window = sliding_window
        self.req_to_swa_pages = torch.full(
            (max_request_num + 1, self.req_to_token_indexs.shape[1] // DSV4_SWA_PAGE_SIZE),
            -1,
            dtype=torch.int32,
            device="cuda",
        )
        self._swa_pages = [{} for _ in range(max_request_num)]
        if mem_manager is not None:
            self.bind_mem_manager(mem_manager)

    def bind_mem_manager(self, mem_manager):
        super().bind_mem_manager(mem_manager)
        self.req_to_swa_pages[self.HOLD_REQUEST_ID].fill_(mem_manager.swa_num_pages)
        # IPC pack/unpack workers need the GPU table, without the CPU owner lists.
        mem_manager.req_to_swa_pages = self.req_to_swa_pages

    def _swa_page_changes(self, req_idx, start, end):
        pages = self._swa_pages[req_idx]
        first_retained = max(0, start - self.sliding_window - DSV4_CACHE_BLOCK_SIZE + 1) // DSV4_SWA_PAGE_SIZE
        missing = [
            p
            for p in range(start // DSV4_SWA_PAGE_SIZE, (end + DSV4_SWA_PAGE_SIZE - 1) // DSV4_SWA_PAGE_SIZE)
            if p not in pages
        ]
        evicted = [p for p in pages if p < first_retained]
        return missing, evicted

    def get_swa_page_need(self, req_idx, start, end):
        if end <= start:
            return 0
        missing, evicted = self._swa_page_changes(req_idx, start, end)
        return max(0, len(missing) - len(evicted))

    def get_swa_pool_page_num(self, args, batch_max_tokens, mtp_step):
        """Capacity for retained windows, verification rows and the active prefill chunk."""
        retain = self.sliding_window + DSV4_CACHE_BLOCK_SIZE
        decode_width = max(mtp_step + 1, 2 * mtp_step)
        # A window can straddle one extra physical page.
        per_req_pages = math.ceil((retain + decode_width - 1) / DSV4_SWA_PAGE_SIZE) + 1
        page_num = self.max_request_num * per_req_pages
        if args.run_mode != "prefill" and args.mtp_mode == "dspark":
            page_num += self.max_request_num  # Separate DSpark proposal scratch.
        if args.run_mode != "decode":
            page_num += math.ceil(batch_max_tokens / DSV4_SWA_PAGE_SIZE)
        return page_num

    def get_request_resource_budget(self) -> RequestResourceBudget:
        return RequestResourceBudget(int(self.mem_manager.swa_page_allocator.can_use_mem_size))

    def get_request_resource_need(self, req, stage: str, *, is_chunked_prefill=False, target_kv_len=None) -> int:
        """SWA page demand using the same logical ranges as prepare_swa."""
        if stage == "restore":
            if target_kv_len is None:
                # Prefix matching restores a complete continuation checkpoint.
                return DSV4_CACHE_BLOCK_SIZE // DSV4_SWA_PAGE_SIZE
            return self.get_swa_page_need(req.req_idx, target_kv_len - DSV4_CACHE_BLOCK_SIZE, target_kv_len)
        if stage == "decode":
            seq_len = req.get_cur_total_len()
            if seq_len <= 0:
                return 0
            width = req.mtp_step + 1 if req.args.mtp_mode == "dspark" else max(req.mtp_step + 1, req.mtp_step * 2)
            return self.get_swa_page_need(req.req_idx, seq_len - 1, seq_len - 1 + width) + int(
                req.args.mtp_mode == "dspark"
            )
        if stage not in ("prefill", "recover"):
            raise ValueError(f"Unknown DeepSeek-V4 resource stage: {stage}")
        end = req.get_chuncked_input_token_len() if is_chunked_prefill else req.get_cur_total_len()
        need = self.get_swa_page_need(req.req_idx, req.cur_kv_len, end)
        if stage == "prefill" or need == 0 or req.args.disable_chunked_prefill:
            return need
        # Recovery reserves the peak across future chunks, including atomic image blocks.
        chunk_tokens = max(
            req.args.chunked_prefill_size,
            max((end - start for start, end in req.image_block_spans), default=0),
        )
        peak_tokens = min(end, chunk_tokens + self.sliding_window + 2 * DSV4_CACHE_BLOCK_SIZE)
        return math.ceil(peak_tokens / DSV4_SWA_PAGE_SIZE)

    def get_loadable_cpu_cache_end(
        self,
        req,
        requested_end: int,
        full_token_capacity: int,
    ) -> int:
        """Return the farthest loadable checkpoint boundary, or zero.

        History is independently addressable in 256-token blocks, but resume
        SWA/state exists only at the end of each CPU checkpoint page.  Therefore
        capacity cropping must never return an intermediate 256-token boundary.
        """
        loaded_start = int(req.cur_kv_len)
        requested_end = int(requested_end)
        page = self.mem_manager.cpu_cache_layout.token_page_size
        if loaded_start < 0 or loaded_start % DSV4_CACHE_BLOCK_SIZE != 0:
            raise ValueError(f"DeepSeek-V4 CPU cache loaded_start must be 256-token aligned, got {loaded_start}")
        if requested_end <= loaded_start or requested_end % page != 0:
            raise ValueError(
                f"DeepSeek-V4 CPU cache requested_end must be a checkpoint boundary after loaded_start, "
                f"got start={loaded_start}, end={requested_end}, page={page}"
            )
        token_capacity = int(full_token_capacity)
        token_capacity = token_capacity // DSV4_CACHE_BLOCK_SIZE * DSV4_CACHE_BLOCK_SIZE
        loadable_end = min(requested_end, (loaded_start + token_capacity) // page * page)
        if loadable_end <= loaded_start:
            return 0
        need = self.get_request_resource_need(req, "restore", target_kv_len=loadable_end)
        return loadable_end if self.get_request_resource_budget().reserve(need) else 0

    def prepare_cpu_cache_load(self, req, mem_indexes: torch.Tensor, loaded_end: int):
        """Allocate private continuation pages and describe the history copy explicitly."""
        start = loaded_end - DSV4_CACHE_BLOCK_SIZE
        self.prepare_swa(req.req_idx, start, loaded_end)
        resume_slots = self.get_swa_slots(req.req_idx, torch.arange(start, loaded_end, device=mem_indexes.device))
        return self.mem_manager.prepare_cpu_cache_load(
            token_num=mem_indexes.numel(),
            loaded_end=loaded_end,
            resume_swa_slots=resume_slots,
            mem_indexes=mem_indexes,
        )

    def prepare_swa(self, req_idx, start, end):
        if req_idx == self.HOLD_REQUEST_ID:
            return
        pages = self._swa_pages[req_idx]
        missing, evicted = self._swa_page_changes(req_idx, start, end)
        if evicted:
            self.mem_manager.swa_page_allocator.free([pages.pop(p) for p in evicted])
            self.req_to_swa_pages[req_idx, evicted] = -1
        if missing:
            allocated = self.mem_manager.swa_page_allocator.alloc(len(missing))
            pages.update(zip(missing, allocated.tolist()))
            self.req_to_swa_pages[req_idx, missing] = allocated.to(device="cuda", non_blocking=True)

    def get_prompt_cache_page_size(self):
        # Continuation needs one packed history block, independent of allocation/hash page sizes.
        return DSV4_CACHE_BLOCK_SIZE

    def get_swa_slots(self, req_idx, positions):
        return (
            self.req_to_swa_pages[req_idx, positions // DSV4_SWA_PAGE_SIZE] * DSV4_SWA_PAGE_SIZE
            + positions % DSV4_SWA_PAGE_SIZE
        )

    def prepare_prefill(self, b_req_idx_cpu, b_ready_cache_len_cpu, b_seq_len_cpu):
        for req_idx, start, end in zip(b_req_idx_cpu.tolist(), b_ready_cache_len_cpu.tolist(), b_seq_len_cpu.tolist()):
            self.prepare_swa(req_idx, start, end)

    def prepare_decode(self, b_req_idx_cpu, b_seq_len_cpu, b_mtp_index_cpu):
        # Verification rows are request-major and include consecutive MTP positions.
        width = int(b_mtp_index_cpu.max().item()) + 1
        reqs, seqs = b_req_idx_cpu.tolist(), b_seq_len_cpu.tolist()
        for i in range(0, len(reqs), width):
            self.prepare_swa(reqs[i], seqs[i] - 1, seqs[i + width - 1])

    def prepare_pd_decode_cache(self, req_list, seq_list):
        for req_idx, end in zip(req_list, seq_list):
            # PD may end inside a compression group; preserve the complete tail
            # required by its SWA and C4 continuation layout.
            start = max(0, (end - 1) // DSV4_CACHE_BLOCK_SIZE * DSV4_CACHE_BLOCK_SIZE - DSV4_CACHE_BLOCK_SIZE)
            self.prepare_swa(req_idx, start, end)

    def create_small_page_cache_manager(self, size):
        self.small_page_buffers = DeepseekV4StateCacheManager(size, **self.mem_manager.state_cache_shapes)
        return self.small_page_buffers

    def init_hybrid_attention_state(self, req):
        self.clear_runtime_state(req.req_idx)

    def save_state(self, req_idx, buffer_idx, state_cache_manager, checkpoint_len):
        manager = self.mem_manager
        end_page = checkpoint_len // DSV4_SWA_PAGE_SIZE
        swa, c4, indexer = state_cache_manager.get_state_cache(buffer_idx)
        pages = self.req_to_swa_pages[req_idx, end_page - swa.shape[1] : end_page].long()
        swa.copy_(manager.swa_pool.buffer.index_select(1, pages), non_blocking=True)
        if manager.n_c4:
            slots = pages[-1] * DSV4_SWA_PAGE_SIZE + torch.arange(
                DSV4_SWA_PAGE_SIZE - c4.shape[1], DSV4_SWA_PAGE_SIZE, device="cuda"
            )
            rows = (slots // DSV4_SWA_PAGE_SIZE) * manager.c4_state_ring + slots % manager.c4_state_ring
            c4.copy_(manager.c4_state_buffer.index_select(1, rows), non_blocking=True)
            indexer.copy_(manager.c4_indexer_state_buffer.index_select(1, rows), non_blocking=True)

    def restore_state(self, req, state_cache_manager, buffer_idx, checkpoint_len):
        self.clear_runtime_state(req.req_idx)
        self.prepare_swa(req.req_idx, checkpoint_len - DSV4_CACHE_BLOCK_SIZE, checkpoint_len)
        manager = self.mem_manager
        end_page = checkpoint_len // DSV4_SWA_PAGE_SIZE
        swa, c4, indexer = state_cache_manager.get_state_cache(buffer_idx)
        pages = self.req_to_swa_pages[req.req_idx, end_page - swa.shape[1] : end_page].long()
        manager.swa_pool.buffer.index_copy_(1, pages, swa.cuda(non_blocking=True))
        if manager.n_c4:
            slots = pages[-1] * DSV4_SWA_PAGE_SIZE + torch.arange(
                DSV4_SWA_PAGE_SIZE - c4.shape[1], DSV4_SWA_PAGE_SIZE, device="cuda"
            )
            rows = (slots // DSV4_SWA_PAGE_SIZE) * manager.c4_state_ring + slots % manager.c4_state_ring
            manager.c4_state_buffer.index_copy_(1, rows, c4.cuda(non_blocking=True))
            manager.c4_indexer_state_buffer.index_copy_(1, rows, indexer.cuda(non_blocking=True))
        # A 256-token checkpoint closes both compressor groups. The next C128
        # group overwrites all of its rows before reading them.

    def restore_cpu_cache_checkpoints(self, req, loaded_start, loaded_end, cpu_cache_client, radix_cache):
        """Retain loaded continuation checkpoints for subsequent GPU radix reuse."""
        from lightllm.utils.envs_utils import get_env_start_args

        args = get_env_start_args()
        layout = self.mem_manager.cpu_cache_layout
        page_list = req.shm_req.cpu_cache_match_page_indexes.get_all()
        big_page_tokens = args.linear_att_hash_page_size * args.linear_att_page_block_num
        first_boundary = (loaded_start // big_page_tokens + 1) * big_page_tokens
        boundaries = list(range(first_boundary, loaded_end + 1, big_page_tokens))
        buffer_indexes = self.big_page_buffers.alloc_state_cache(len(boundaries))
        if buffer_indexes is None:
            raise RuntimeError("No DeepSeek-V4 checkpoint slots available for CPU cache restore")
        tail_buffer = None
        try:
            for boundary, buffer_idx in zip(boundaries, buffer_indexes):
                page_idx = page_list[boundary // layout.token_page_size - 1]
                self.big_page_buffers.buffer[buffer_idx].copy_(
                    cpu_cache_client.cpu_kv_cache_tensor[page_idx, layout.swa_offset :]
                )
            if loaded_end == req.hybrid_cache_len and loaded_end % big_page_tokens:
                radix_cache.free_one_small_page_buffer()
                tail_buffer = self.small_page_buffers.alloc_one_state_cache()
                if tail_buffer is not None:
                    self.save_state(req.req_idx, tail_buffer, self.small_page_buffers, checkpoint_len=loaded_end)
        except Exception:
            self.big_page_buffers.free_state_cache(buffer_indexes)
            if tail_buffer is not None:
                self.small_page_buffers.free_state_cache([tail_buffer])
            raise
        req.hybrid_len_to_big_page_id.update(zip(boundaries, buffer_indexes))
        if tail_buffer is not None:
            req.tail_small_page_buffer_id = tail_buffer

    def clear_runtime_state(self, req_idx):
        pages = self._swa_pages[req_idx]
        if pages:
            self.mem_manager.swa_page_allocator.free(list(pages.values()))
            pages.clear()
            self.req_to_swa_pages[req_idx].fill_(-1)

    def free(self, free_req_indexes, free_token_index):
        for req_idx in free_req_indexes:
            self.clear_runtime_state(req_idx)
        super().free(free_req_indexes, free_token_index)

    def free_req(self, free_req_index):
        self.clear_runtime_state(free_req_index)
        super().free_req(free_req_index)

    def free_all(self):
        for req_idx in range(self.max_request_num):
            self.clear_runtime_state(req_idx)
        super().free_all()
