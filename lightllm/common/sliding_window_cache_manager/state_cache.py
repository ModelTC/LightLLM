import collections
from typing import List, Optional

import torch

from .config import SlidingWindowCacheConfig


class SlidingWindowStateCacheManager:
    """大小页共用的 CPU pinned checkpoint 存储，两个池独立分配。

    布局为 size-first: [slot, layer, window, 2 * heads, dim]。
    本类只管理状态存储与空闲槽位，不判断页面大小或缓存边界，也不持有 GPU 运行态。
    """

    def __init__(self, size: int, sliding_config: SlidingWindowCacheConfig, keep_num: int = 0):
        self.size = size
        self.keep_num = keep_num
        assert 0 <= keep_num <= size
        self.state_cache = torch.empty(
            (size, *sliding_config.get_state_shape()),
            dtype=sliding_config.dtype,
            device="cpu",
            pin_memory=True,
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
        alloc_size = self.size - self.keep_num
        assert all(0 <= idx < alloc_size for idx in free_indexes)
        self.free_list.extend(free_indexes)
        assert len(self.free_list) <= alloc_size

    def get_free_cache_num(self):
        return len(self.free_list)

    def get_used_cache_num(self):
        return self.size - len(self.free_list)

    def clear_to_init_state(self):
        self.state_cache.zero_()
        self.free_list = collections.deque(range(self.size - self.keep_num))
