from typing import Optional

import torch

from .hybrid_base import HybridAttentionReqManager
from lightllm.common.kv_cache_mem_manager import DeepseekV4MemoryManager
from lightllm.common.kv_cache_mem_manager.deepseek4_mem_manager import (
    DSV4_SWA_PAGE_SIZE,
    DSV4_PROMPT_CACHE_PAGE_SIZE,
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

    def get_swa_page_need(self, req_idx, start, end):
        pages = self._swa_pages[req_idx]
        first_retained = max(0, start - self.sliding_window - DSV4_PROMPT_CACHE_PAGE_SIZE + 1) // DSV4_SWA_PAGE_SIZE
        missing = sum(
            p not in pages
            for p in range(start // DSV4_SWA_PAGE_SIZE, (end + DSV4_SWA_PAGE_SIZE - 1) // DSV4_SWA_PAGE_SIZE)
        )
        released = sum(p < first_retained for p in pages)
        return max(0, missing - released)

    def prepare_swa(self, req_idx, start, end):
        if req_idx == self.HOLD_REQUEST_ID:
            return
        page_size = DSV4_SWA_PAGE_SIZE
        pages = self._swa_pages[req_idx]
        retain = self.sliding_window + DSV4_PROMPT_CACHE_PAGE_SIZE
        first_retained_page = max(0, start - retain + 1) // page_size
        evicted = [position for position in pages if position < first_retained_page]
        if evicted:
            self.mem_manager.swa_page_allocator.free([pages.pop(p) for p in evicted])
            self.req_to_swa_pages[req_idx, :first_retained_page] = -1
        missing = [p for p in range(start // page_size, (end + page_size - 1) // page_size) if p not in pages]
        if missing:
            allocated = self.mem_manager.swa_page_allocator.alloc(len(missing))
            pages.update(zip(missing, allocated.tolist()))
            self.req_to_swa_pages[req_idx, missing] = allocated.to(device="cuda", non_blocking=True)

    def get_prompt_cache_page_size(self):
        return DSV4_PROMPT_CACHE_PAGE_SIZE

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
            start = max(
                0, (end - 1) // DSV4_PROMPT_CACHE_PAGE_SIZE * DSV4_PROMPT_CACHE_PAGE_SIZE - DSV4_PROMPT_CACHE_PAGE_SIZE
            )
            self.prepare_swa(req_idx, start, end)

    def create_small_page_cache_manager(self, size):
        self.small_page_buffers = DeepseekV4StateCacheManager(size, self.mem_manager.cpu_cache_layout)
        return self.small_page_buffers

    def init_hybrid_attention_state(self, req):
        self.clear_runtime_state(req.req_idx)

    def save_state(self, req_idx, buffer_idx, state_cache_manager, checkpoint_len):
        assert checkpoint_len > 0 and checkpoint_len % DSV4_PROMPT_CACHE_PAGE_SIZE == 0
        manager = self.mem_manager
        end_page = checkpoint_len // DSV4_SWA_PAGE_SIZE
        pages = self.req_to_swa_pages[req_idx, end_page - 2 : end_page].long()
        swa, c4, indexer = state_cache_manager.get_state_cache(buffer_idx)
        swa.copy_(manager.swa_pool.buffer.index_select(1, pages), non_blocking=True)
        if manager.n_c4:
            slots = pages[-1] * DSV4_SWA_PAGE_SIZE + torch.arange(124, 128, device="cuda")
            rows = (slots // DSV4_SWA_PAGE_SIZE) * manager.c4_state_ring + slots % manager.c4_state_ring
            c4.copy_(manager.c4_state_buffer.index_select(1, rows), non_blocking=True)
            indexer.copy_(manager.c4_indexer_state_buffer.index_select(1, rows), non_blocking=True)

    def restore_state(self, req, state_cache_manager, buffer_idx, checkpoint_len):
        assert checkpoint_len > 0 and checkpoint_len % DSV4_PROMPT_CACHE_PAGE_SIZE == 0
        self.clear_runtime_state(req.req_idx)
        self.prepare_swa(req.req_idx, checkpoint_len - DSV4_PROMPT_CACHE_PAGE_SIZE, checkpoint_len)
        manager = self.mem_manager
        end_page = checkpoint_len // DSV4_SWA_PAGE_SIZE
        pages = self.req_to_swa_pages[req.req_idx, end_page - 2 : end_page].long()
        swa, c4, indexer = state_cache_manager.get_state_cache(buffer_idx)
        manager.swa_pool.buffer.index_copy_(1, pages, swa.cuda(non_blocking=True))
        if manager.n_c4:
            slots = pages[-1] * DSV4_SWA_PAGE_SIZE + torch.arange(124, 128, device="cuda")
            rows = (slots // DSV4_SWA_PAGE_SIZE) * manager.c4_state_ring + slots % manager.c4_state_ring
            manager.c4_state_buffer.index_copy_(1, rows, c4.cuda(non_blocking=True))
            manager.c4_indexer_state_buffer.index_copy_(1, rows, indexer.cuda(non_blocking=True))
        # A 256-token checkpoint closes both compressor groups. The next C128
        # group overwrites all of its rows before reading them.

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
