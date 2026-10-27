import torch

from .hybrid_base import HybridAttentionReqManager
from lightllm.common.state_cache_manager.deepseek4 import DeepseekV4StateCacheManager


class DeepseekV4ReqManager(HybridAttentionReqManager):
    """持有跨 forward 的请求映射、SWA ring 和 compressor ring，并负责状态保存和恢复。

    当前 forward 的派生索引、页表和元数据由 infer_state 构建并持有，
    模型 workspace 提供投影与量化的计算 scratch。
    """

    def __init__(
        self,
        max_request_num,
        max_sequence_length,
        mem_manager,
        cache_config,
        *,
        mtp_step=0,
    ):
        super().__init__(max_request_num, max_sequence_length, mem_manager)
        self.cache_config = cache_config
        self.sliding_window = cache_config.sliding_window
        swa_shape, c4_shape, indexer_shape, c128_shape = cache_config.get_runtime_state_shapes(mtp_step)
        self.swa_capacity_tokens = swa_shape[1]
        self.HOLD_SWA_SLOT = self.HOLD_REQUEST_ID * self.swa_capacity_tokens
        self.c4_state_tokens = c4_shape[1]
        self.c128_state_tokens = c128_shape[1]
        self.req_to_swa = torch.zeros(
            (cache_config.layer_num, max_request_num + 1, self.swa_capacity_tokens, cache_config.swa_slot_bytes),
            dtype=torch.uint8,
            device="cuda",
        )
        self.req_to_c4_state = self._alloc_compressor_state(*c4_shape)
        self.req_to_c4_indexer_state = self._alloc_compressor_state(*indexer_shape)
        self.req_to_c128_state = self._alloc_compressor_state(*c128_shape)

    def _alloc_compressor_state(self, layer_count, token_capacity, state_dim):
        buffer = torch.zeros(
            (layer_count, self.max_request_num + 1, token_capacity, state_dim), dtype=torch.float32, device="cuda"
        )
        buffer[:, self.HOLD_REQUEST_ID, :, state_dim // 2 :].fill_(float("-inf"))
        return buffer

    def get_att_input_params(self, layer_index):
        from lightllm.models.deepseek_v4.triton_kernel.cache_io import get_compressed_kv_buffer, view_swa_kv

        return view_swa_kv(self.req_to_swa[layer_index], self.cache_config), get_compressed_kv_buffer(
            self.mem_manager, layer_index
        )

    def create_small_page_cache_manager(self, size):
        self.small_page_buffers = DeepseekV4StateCacheManager(size, self.cache_config)
        return self.small_page_buffers

    def save_state(self, req, buffer_idx, state_cache_manager):
        from lightllm.models.deepseek_v4.triton_kernel.swa_cache import gather_swa_state

        config = self.cache_config
        swa, c4, indexer = state_cache_manager.get_state_cache(buffer_idx)
        # 保存时 cur_kv_len 还是 chunk 起点；运行态对应本轮 prefill 终点。
        req_idx = req.req_idx
        checkpoint_len = req.get_chuncked_input_token_len()
        positions = torch.arange(checkpoint_len - config.sliding_window, checkpoint_len, device="cuda")
        swa.copy_(gather_swa_state(self, req_idx, positions), non_blocking=True)
        if config.short_compress_ratio != 4 or not config.n_c4:
            return
        positions = torch.arange(checkpoint_len - config.short_compress_ratio, checkpoint_len, device="cuda")
        state_slots = positions % self.c4_state_tokens
        # compressor 已将 chunk 尾部写回 ring；大小页只保存当前终点。
        for target, runtime in ((c4, self.req_to_c4_state), (indexer, self.req_to_c4_indexer_state)):
            target.copy_(runtime[:, req_idx].index_select(1, state_slots), non_blocking=True)

    def restore_state(self, req, state_cache_manager, buffer_idx):
        from lightllm.models.deepseek_v4.triton_kernel.swa_cache import scatter_swa_state

        config = self.cache_config
        swa, c4, indexer = state_cache_manager.get_state_cache(buffer_idx)
        positions = torch.arange(req.cur_kv_len - config.sliding_window, req.cur_kv_len, device="cuda")
        scatter_swa_state(self, req.req_idx, positions, swa.cuda(non_blocking=True))
        if config.short_compress_ratio == 4 and config.n_c4:
            positions = torch.arange(req.cur_kv_len - config.short_compress_ratio, req.cur_kv_len, device="cuda")
            slots = positions % self.c4_state_tokens
            self.req_to_c4_state[:, req.req_idx].index_copy_(1, slots, c4.cuda(non_blocking=True))
            self.req_to_c4_indexer_state[:, req.req_idx].index_copy_(1, slots, indexer.cuda(non_blocking=True))
        # Aligned checkpoints close C128 groups; the next group overwrites its tail.
