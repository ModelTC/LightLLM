import torch
import triton

from .hybrid_base import HybridAttentionReqManager
from lightllm.common.state_cache_manager.deepseek4 import (
    DSV4_CACHE_BLOCK_SIZE,
    DSV4_SWA_PAGE_SIZE,
    DSV4_SWA_PAGE_BYTES,
)
from lightllm.common.state_cache_manager.deepseek4 import DeepseekV4StateCacheManager


class DeepseekV4ReqManager(HybridAttentionReqManager):
    """Finite request state, plus a bounded KV workspace for each prefill forward.

    The fixed SWA rings and the forward workspace share one allocation so FlashMLA
    can address both using ordinary packed-page indices. No SWA allocator is needed.
    """

    def __init__(
        self,
        max_request_num,
        max_sequence_length,
        mem_manager,
        cache_config,
        *,
        max_forward_tokens,
        mtp_step=0,
        prefill_regions=1,
    ):
        super().__init__(max_request_num, max_sequence_length, mem_manager)
        self.cache_config = cache_config
        self.sliding_window = cache_config.sliding_window
        self.swa_pages_per_req = triton.cdiv(self.sliding_window + DSV4_CACHE_BLOCK_SIZE + 2 * mtp_step, 128) + 1
        self.swa_runtime_pages = max_request_num * self.swa_pages_per_req
        # Only overflowing chunks need a workspace. The token budget bounds their count.
        min_overflow = max(1, (self.swa_pages_per_req - 1) * 128 - self.sliding_window)
        overflow_reqs = min(max_request_num, max_forward_tokens // min_overflow)
        self.swa_scratch_pages = (
            (triton.cdiv(max_forward_tokens + overflow_reqs * self.sliding_window, 128) + 2 * (overflow_reqs + 1))
            if overflow_reqs
            else 0
        )
        pages = self.swa_runtime_pages + prefill_regions * self.swa_scratch_pages + 1
        self.swa_buffer = torch.zeros(
            (cache_config.layer_num, pages, DSV4_SWA_PAGE_BYTES), dtype=torch.uint8, device="cuda"
        )
        self.HOLD_SWA_SLOT = (pages - 1) * DSV4_SWA_PAGE_SIZE
        # For the current forward only: first logical page, exclusive end, workspace base.
        self.prefill_pages = torch.zeros((max_request_num + 1, 3), dtype=torch.int32, device="cuda")
        self.c4_state_ring = 8 + mtp_step
        self.c128_state_ring = triton.cdiv(128 + mtp_step, 4) * 4
        self.c4_checkpoint_base = (max_request_num + 1) * self.c4_state_ring
        self.c4_state_buffer = self.c4_indexer_state_buffer = self.c128_state_buffer = None
        if cache_config.n_c4:
            rows = self.c4_checkpoint_base + pages * 4 + 1
            self.c4_state_buffer = self._state_buffer(cache_config.n_c4, rows, 4 * cache_config.head_dim)
            self.c4_indexer_state_buffer = self._state_buffer(
                cache_config.n_c4, rows, 4 * cache_config.indexer_head_dim
            )
        if cache_config.n_c128:
            self.c128_state_buffer = self._state_buffer(
                cache_config.n_c128, (max_request_num + 1) * self.c128_state_ring + 1, 2 * cache_config.head_dim
            )

    @staticmethod
    def _state_buffer(layers, rows, width):
        buffer = torch.zeros((layers, rows, width), dtype=torch.float32, device="cuda")
        buffer[:, -1, width // 2 :].fill_(float("-inf"))
        return buffer

    def clear_runtime_state(self, req_idx):
        # Positions determine validity; request reuse does not require clearing KV.
        self.prefill_pages[req_idx].zero_()

    def init_hybrid_attention_state(self, req):
        self.clear_runtime_state(req.req_idx)

    def init_forward_state(self, state):
        if state.is_draft_model:
            return
        if state.microbatch_index == 0:
            self.prefill_pages.zero_()
        if not state.is_prefill:
            return
        from lightllm.models.deepseek_v4.triton_kernel.swa_cache import prepare_prefill_workspace

        prepare_prefill_workspace(
            self, state.b_req_idx, state.b_ready_cache_len, state.b_seq_len, state.microbatch_index
        )

    def get_swa_pages(self, req_idx, logical_pages, *, prefill=True):
        ring = self.swa_pages_per_req
        physical = req_idx * ring + logical_pages % ring
        if prefill:
            first, end, start = self.prefill_pages[req_idx].unbind(-1)
            physical = torch.where(
                (logical_pages >= first) & (logical_pages < end), start + logical_pages - first, physical
            )
        return torch.where(
            torch.as_tensor(req_idx, device=logical_pages.device) == self.HOLD_REQUEST_ID,
            self.swa_buffer.shape[1] - 1,
            physical,
        )

    def get_swa_slots(self, req_idx, positions, *, prefill=True):
        return self.get_swa_pages(req_idx, positions // 128, prefill=prefill) * 128 + positions % 128

    def pack_mla_kv_to_cache_fused_norm_rope(
        self, layer_index, swa_slots, kv, kv_weight, eps, freqs_cis, positions, runtime_slots=None
    ):
        from lightllm.models.deepseek_v4.triton_kernel.norm_rope import fused_k_norm_rope_flashmla

        fused_k_norm_rope_flashmla(
            kv=kv,
            kv_weight=kv_weight,
            eps=eps,
            freqs_cis=freqs_cis,
            positions=positions,
            out_loc=swa_slots,
            out_loc2=runtime_slots,
            kvcache=self.swa_buffer[layer_index],
            page_size=128,
        )

    def get_att_input_params(self, layer_index):
        history, _ = self.mem_manager.get_att_input_params(layer_index)
        return self.swa_buffer[layer_index], history

    def get_prompt_cache_page_size(self):
        return DSV4_CACHE_BLOCK_SIZE

    def create_small_page_cache_manager(self, size):
        self.small_page_buffers = DeepseekV4StateCacheManager(size, self.cache_config)
        return self.small_page_buffers

    def save_state(self, req_idx, buffer_idx, state_cache_manager, checkpoint_len):
        end = checkpoint_len // 128
        swa, c4, indexer = state_cache_manager.get_state_cache(buffer_idx)
        pages = self.get_swa_pages(req_idx, torch.arange(end - swa.shape[1], end, device="cuda")).long()
        swa.copy_(self.swa_buffer.index_select(1, pages), non_blocking=True)
        if self.cache_config.n_c4:
            rows = self.c4_checkpoint_base + pages[-1] * 4 + torch.arange(4, device="cuda")
            c4.copy_(self.c4_state_buffer.index_select(1, rows), non_blocking=True)
            indexer.copy_(self.c4_indexer_state_buffer.index_select(1, rows), non_blocking=True)

    def restore_state(self, req, state_cache_manager, buffer_idx, checkpoint_len):
        self.restore_state_from_tensors(req, state_cache_manager.get_state_cache(buffer_idx), checkpoint_len)

    def restore_state_from_tensors(self, req, checkpoint, checkpoint_len):
        self.clear_runtime_state(req.req_idx)
        end = checkpoint_len // 128
        swa, c4, indexer = checkpoint
        pages = self.get_swa_pages(
            req.req_idx, torch.arange(end - swa.shape[1], end, device="cuda"), prefill=False
        ).long()
        self.swa_buffer.index_copy_(1, pages, swa.cuda(non_blocking=True))
        if self.cache_config.n_c4:
            c4_gpu, indexer_gpu = c4.cuda(non_blocking=True), indexer.cuda(non_blocking=True)
            rows = self.c4_checkpoint_base + pages[-1] * 4 + torch.arange(4, device="cuda")
            self.c4_state_buffer.index_copy_(1, rows, c4_gpu)
            self.c4_indexer_state_buffer.index_copy_(1, rows, indexer_gpu)
            rows = (
                req.req_idx * self.c4_state_ring
                + torch.arange(checkpoint_len - 4, checkpoint_len, device="cuda") % self.c4_state_ring
            )
            self.c4_state_buffer.index_copy_(1, rows, c4_gpu)
            self.c4_indexer_state_buffer.index_copy_(1, rows, indexer_gpu)
        # Aligned checkpoints close C128 groups; the next group overwrites its tail.
