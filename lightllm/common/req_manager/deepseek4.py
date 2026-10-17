import torch
import triton

from .hybrid_base import HybridAttentionReqManager
from lightllm.common.state_cache_manager.deepseek4 import DeepseekV4StateCacheManager


class DeepseekV4ReqManager(HybridAttentionReqManager):
    """Request-owned window and compressor tails, independent of history capacity.

    All capacities and addresses use token slots. Packed attention views and
    prefill workspace addressing are private to the model's cache operators.
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
        alignment = cache_config.swa_token_alignment
        self.swa_capacity_tokens = triton.cdiv(self.sliding_window + 2 * mtp_step, alignment) * alignment
        self.swa_runtime_tokens = (max_request_num + 1) * self.swa_capacity_tokens
        # Each request contributes its previous window and packing padding at both ends.
        self.prefill_capacity_tokens = (
            triton.cdiv(max_forward_tokens + max_request_num * (self.sliding_window + 2 * alignment), alignment)
            * alignment
            if prefill_regions
            else 0
        )
        total_tokens = self.swa_runtime_tokens + prefill_regions * self.prefill_capacity_tokens
        self.swa_buffer = torch.zeros(
            (cache_config.layer_num, total_tokens, cache_config.swa_storage_width), dtype=torch.uint8, device="cuda"
        )
        self.req_to_swa_state = self.swa_buffer[:, : self.swa_runtime_tokens].view(
            cache_config.layer_num, max_request_num + 1, self.swa_capacity_tokens, cache_config.swa_storage_width
        )
        self.HOLD_SWA_SLOT = self.HOLD_REQUEST_ID * self.swa_capacity_tokens
        # Absolute [ready, end) token positions and forward-local storage offsets.
        self.prefill_token_ranges = torch.zeros((max_request_num + 1, 2), dtype=torch.int32, device="cuda")
        self.prefill_workspace_offsets = torch.full((max_request_num + 1,), -1, dtype=torch.int32, device="cuda")
        self.prefill_checkpoint_offsets = torch.zeros((max_request_num + 1,), dtype=torch.int32, device="cuda")
        short_ratio, long_ratio = cache_config.short_compress_ratio, cache_config.long_compress_ratio
        self.c4_state_tokens = 2 * short_ratio + mtp_step
        self.c128_state_tokens = long_ratio + mtp_step
        self.req_to_c4_state = self.req_to_c4_indexer_state = self.req_to_c128_state = None
        if cache_config.n_c4:
            self.req_to_c4_state = self._state_buffer(
                cache_config.n_c4, self.c4_state_tokens, 4 * cache_config.head_dim
            )
            self.req_to_c4_indexer_state = self._state_buffer(
                cache_config.n_c4, self.c4_state_tokens, 4 * cache_config.indexer_head_dim
            )
        if cache_config.n_c128:
            self.req_to_c128_state = self._state_buffer(
                cache_config.n_c128, self.c128_state_tokens, 2 * cache_config.head_dim
            )
        # Intermediate prefill snapshots have their own storage, indexed by
        # checkpoint boundary, never by SWA physical location.
        self.prefill_checkpoint_capacity = max_forward_tokens // cache_config.history_token_alignment + max_request_num
        checkpoint_slots = prefill_regions * self.prefill_checkpoint_capacity
        self.prefill_c4_checkpoints = self.prefill_indexer_checkpoints = None
        if cache_config.n_c4 and checkpoint_slots:
            self.prefill_c4_checkpoints = torch.empty(
                (cache_config.n_c4, checkpoint_slots, short_ratio, 4 * cache_config.head_dim),
                dtype=torch.float32,
                device="cuda",
            )
            self.prefill_indexer_checkpoints = torch.empty(
                (cache_config.n_c4, checkpoint_slots, short_ratio, 4 * cache_config.indexer_head_dim),
                dtype=torch.float32,
                device="cuda",
            )

    def _state_buffer(self, layer_count, token_capacity, state_dim):
        buffer = torch.zeros(
            (layer_count, self.max_request_num + 1, token_capacity, state_dim), dtype=torch.float32, device="cuda"
        )
        buffer[:, self.HOLD_REQUEST_ID, :, state_dim // 2 :].fill_(float("-inf"))
        return buffer

    def clear_runtime_state(self, req_idx):
        # Absolute positions determine validity; stale KV never needs clearing.
        self.prefill_token_ranges[req_idx].zero_()
        self.prefill_workspace_offsets[req_idx] = -1

    def init_hybrid_attention_state(self, req):
        self.clear_runtime_state(req.req_idx)

    def init_forward_state(self, state):
        if state.is_draft_model:
            return
        if state.microbatch_index == 0:
            self.prefill_token_ranges.zero_()
            self.prefill_workspace_offsets.fill_(-1)
        if state.is_prefill:
            from lightllm.models.deepseek_v4.triton_kernel.swa_cache import prepare_prefill_workspace

            prepare_prefill_workspace(
                self, state.b_req_idx, state.b_ready_cache_len, state.b_seq_len, state.microbatch_index
            )

    def get_swa_token_indices(self, req_idx, positions, *, prefill=True):
        from lightllm.models.deepseek_v4.triton_kernel.swa_cache import get_swa_token_indices

        return get_swa_token_indices(self, req_idx, positions, prefill=prefill)

    def pack_mla_kv_to_cache_fused_norm_rope(
        self,
        layer_index,
        swa_slots,
        kv,
        kv_weight,
        eps,
        freqs_cis,
        positions,
        runtime_slots=None,
    ):
        from lightllm.models.deepseek_v4.triton_kernel.norm_rope import fused_k_norm_rope_flashmla
        from lightllm.models.deepseek_v4.triton_kernel.cache_io import view_swa_kv

        swa = view_swa_kv(self.swa_buffer[layer_index], self.cache_config)
        fused_k_norm_rope_flashmla(
            kv=kv,
            kv_weight=kv_weight,
            eps=eps,
            freqs_cis=freqs_cis,
            positions=positions,
            out_loc=swa_slots,
            out_loc2=runtime_slots,
            kvcache=swa,
            page_size=self.cache_config.swa_token_alignment,
        )

    def get_att_input_params(self, layer_index):
        from lightllm.models.deepseek_v4.triton_kernel.cache_io import get_compressed_kv_buffer, view_swa_kv

        return view_swa_kv(self.swa_buffer[layer_index], self.cache_config), get_compressed_kv_buffer(
            self.mem_manager, layer_index
        )

    def get_prompt_cache_page_size(self):
        return self.cache_config.history_token_alignment

    def create_small_page_cache_manager(self, size):
        self.small_page_buffers = DeepseekV4StateCacheManager(size, self.cache_config)
        return self.small_page_buffers

    def save_state(self, req_idx, buffer_idx, state_cache_manager, checkpoint_len):
        from lightllm.models.deepseek_v4.triton_kernel.swa_cache import gather_swa_tokens

        config = self.cache_config
        swa, c4, indexer = state_cache_manager.get_state_cache(buffer_idx)
        positions = torch.arange(checkpoint_len - config.sliding_window, checkpoint_len, device="cuda")
        swa.copy_(gather_swa_tokens(self, req_idx, positions), non_blocking=True)
        if not config.n_c4:
            return
        positions = torch.arange(checkpoint_len - config.short_compress_ratio, checkpoint_len, device="cuda")
        state_slots = positions % self.c4_state_tokens
        ready, end = self.prefill_token_ranges[req_idx].unbind()
        from_prefill = (checkpoint_len > ready) & (checkpoint_len <= end)
        checkpoint_slot = self.prefill_checkpoint_offsets[req_idx] + checkpoint_len // config.history_token_alignment
        for target, runtime, captured in (
            (c4, self.req_to_c4_state, self.prefill_c4_checkpoints),
            (indexer, self.req_to_c4_indexer_state, self.prefill_indexer_checkpoints),
        ):
            values = runtime[:, req_idx].index_select(1, state_slots)
            if captured is not None:
                saved = captured[:, checkpoint_slot.clamp(0, captured.shape[1] - 1)]
                values = torch.where(from_prefill, saved, values)
            target.copy_(values, non_blocking=True)

    def restore_state(self, req, state_cache_manager, buffer_idx):
        self.restore_state_from_tensors(req, state_cache_manager.get_state_cache(buffer_idx))

    def restore_state_from_tensors(self, req, checkpoint):
        from lightllm.models.deepseek_v4.triton_kernel.swa_cache import scatter_swa_tokens

        self.clear_runtime_state(req.req_idx)
        config = self.cache_config
        swa, c4, indexer = checkpoint
        positions = torch.arange(req.cur_kv_len - config.sliding_window, req.cur_kv_len, device="cuda")
        scatter_swa_tokens(self, req.req_idx, positions, swa.cuda(non_blocking=True))
        if config.n_c4:
            positions = torch.arange(req.cur_kv_len - config.short_compress_ratio, req.cur_kv_len, device="cuda")
            slots = positions % self.c4_state_tokens
            self.req_to_c4_state[:, req.req_idx].index_copy_(1, slots, c4.cuda(non_blocking=True))
            self.req_to_c4_indexer_state[:, req.req_idx].index_copy_(1, slots, indexer.cuda(non_blocking=True))
        # Aligned checkpoints close C128 groups; the next group overwrites its tail.
