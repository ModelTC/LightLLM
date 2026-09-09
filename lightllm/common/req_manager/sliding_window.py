from typing import TYPE_CHECKING, Optional

from lightllm.common.basemodel.triton_kernel.sliding_window_cpu_cache_copy import copy_sliding_window_state
from lightllm.common.sliding_window_cache_manager import SlidingWindowStateCacheManager

from .hybrid_att import HybridAttentionReqManager


if TYPE_CHECKING:
    from lightllm.common.kv_cache_mem_manager.hybrid_sliding_mem_manager import HybridSlidingMemoryManager
    from lightllm.common.sliding_window_cache_manager import SlidingWindowCacheConfig
    from lightllm.server.router.model_infer.infer_batch import InferReq


class ReqManagerForSlidingWindow(HybridAttentionReqManager):
    """按请求保存窗口运行态，并在大小页边界保存和恢复 checkpoint。"""

    def __init__(
        self,
        max_request_num: int,
        max_sequence_length: int,
        mem_manager: Optional["HybridSlidingMemoryManager"],
        sliding_config: "SlidingWindowCacheConfig",
    ):
        super().__init__(max_request_num, max_sequence_length, mem_manager)
        self.sliding_config = sliding_config
        self.sliding_window = sliding_config.sliding_window

    @property
    def req_to_sliding_window(self):
        # A view, not a second allocation. req_idx owns the same W slots for its lifetime.
        pool = self.mem_manager.sliding_kv_buffer
        return pool[:, : self.mem_manager.sliding_prefill_start].unflatten(1, (-1, self.sliding_window))

    def create_state_cache_manager(self, size: int):
        return SlidingWindowStateCacheManager(size=size, sliding_config=self.sliding_config)

    def init_hybrid_attention_state(self, req: "InferReq"):
        self.req_to_sliding_window[:, req.req_idx].zero_()

    def restore_state(self, req: "InferReq", state_cache_manager, buffer_idx: int):
        copy_sliding_window_state(
            state_cache_manager.get_state_cache(buffer_idx),
            self.req_to_sliding_window[:, req.req_idx],
        )

    def save_state(self, req_idx: int, buffer_idx: int, state_cache_manager: SlidingWindowStateCacheManager):
        copy_sliding_window_state(
            self.req_to_sliding_window[:, req_idx],
            state_cache_manager.get_state_cache(buffer_idx),
        )
