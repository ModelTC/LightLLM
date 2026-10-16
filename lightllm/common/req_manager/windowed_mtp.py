from lightllm.common.state_cache_manager import WindowStateCacheManager
from .hybrid_base import HybridAttentionReqManager


class ReqManagerForWindowedMTP(HybridAttentionReqManager):
    """Full-attention target with request-local draft state in hybrid checkpoints."""

    def create_small_page_cache_manager(self, size: int):
        self.small_page_buffers = WindowStateCacheManager(
            size=size, config=self.mem_manager.window_state_config, dtype=self.mem_manager.dtype
        )
        return self.small_page_buffers

    def init_hybrid_attention_state(self, req):
        self.mem_manager.windowed_draft_kv.reset_req(req.req_idx)

    def save_state(self, req_idx, buffer_idx, state_cache_manager):
        self.mem_manager.windowed_draft_kv.save_checkpoint(req_idx, state_cache_manager.draft_window, buffer_idx)

    def restore_state(self, req, state_cache_manager, buffer_idx):
        self.mem_manager.windowed_draft_kv.restore_checkpoint(req.req_idx, state_cache_manager.draft_window, buffer_idx)
