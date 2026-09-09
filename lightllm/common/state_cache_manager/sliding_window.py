import torch

from .base import StateCacheManager
from .sliding_window_config import SlidingWindowCacheConfig


class SlidingWindowStateCacheManager(StateCacheManager):
    """CPU pinned 窗口 checkpoint，布局为 [slot, layer, window, 2 * heads, dim]。"""

    def __init__(self, size: int, sliding_config: SlidingWindowCacheConfig, keep_num: int = 0):
        super().__init__(size, keep_num)
        self.state_cache = torch.zeros(
            (size, *sliding_config.get_state_shape()),
            dtype=sliding_config.dtype,
            device="cpu",
            pin_memory=True,
        )

    def get_state_cache(self, buffer_idx: int):
        return self.state_cache[buffer_idx]

    def clear_to_init_state(self):
        self.state_cache.zero_()
        super().clear_to_init_state()
