import collections
from typing import List, Optional

import torch

from .config import SlidingWindowCacheConfig


class SlidingWindowStateCacheManager:
    """GPU storage for immutable request-level sliding-window checkpoints."""

    def __init__(self, size: int, sliding_config: SlidingWindowCacheConfig):
        self.size = size
        assert size >= 0
        self.state_cache = torch.empty(
            (size, *sliding_config.get_state_shape()), dtype=sliding_config.dtype, device="cuda"
        )
        self.clear_to_init_state()

    def get_state_cache(self, buffer_idx: int):
        return self.state_cache[buffer_idx]

    def alloc_one_state_cache(self) -> Optional[int]:
        return None if not self.free_list else self.free_list.popleft()

    def alloc_state_cache(self, need_size: int) -> Optional[List[int]]:
        if need_size > len(self.free_list):
            return None
        return [self.free_list.popleft() for _ in range(need_size)]

    def free_state_cache(self, free_indexes: List[int]):
        assert all(0 <= idx < self.size for idx in free_indexes)
        self.free_list.extend(free_indexes)
        assert len(self.free_list) <= self.size

    def get_free_cache_num(self):
        return len(self.free_list)

    def get_used_cache_num(self):
        return self.size - len(self.free_list)

    def clear_to_init_state(self):
        self.state_cache.zero_()
        self.free_list = collections.deque(range(self.size))
