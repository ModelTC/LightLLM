import dataclasses
import math
from typing import Dict

import torch


@dataclasses.dataclass
class SlidingWindowCacheConfig:
    """Physical cache layout for a full + sliding-window transformer."""

    sliding_layer_to_cache_index: Dict[int, int]
    full_layer_to_cache_index: Dict[int, int]
    sliding_window: int
    sliding_head_num: int
    sliding_head_dim: int
    full_head_num: int
    full_head_dim: int
    dtype: torch.dtype

    def __post_init__(self):
        assert self.sliding_window > 0
        assert self.sliding_layer_to_cache_index and self.full_layer_to_cache_index
        assert not self.sliding_layer_to_cache_index.keys() & self.full_layer_to_cache_index.keys()
        self.sliding_layer_num = len(set(self.sliding_layer_to_cache_index.values()))
        self.full_layer_num = len(set(self.full_layer_to_cache_index.values()))
        assert set(self.sliding_layer_to_cache_index.values()) == set(range(self.sliding_layer_num))
        assert set(self.full_layer_to_cache_index.values()) == set(range(self.full_layer_num))

    def get_state_shape(self):
        return (
            self.sliding_layer_num,
            self.sliding_window,
            2 * self.sliding_head_num,
            self.sliding_head_dim,
        )

    def get_state_nbytes(self):
        return math.prod(self.get_state_shape()) * self.dtype.itemsize

    def get_cpu_cache_full_att_bytes(self, big_page_token_num: int, tp_world_size: int):
        return (
            big_page_token_num
            * self.full_layer_num
            * 2
            * self.full_head_num
            * self.full_head_dim
            * self.dtype.itemsize
            * tp_world_size
        )

    def get_cpu_cache_state_bytes(self, tp_world_size: int):
        return self.get_state_nbytes() * tp_world_size

    def get_cpu_cache_big_page_bytes(self, big_page_token_num: int, tp_world_size: int):
        # One CPU page contains all TP shards: full KV, window state, padding.
        payload_bytes = self.get_cpu_cache_full_att_bytes(big_page_token_num, tp_world_size)
        payload_bytes += self.get_cpu_cache_state_bytes(tp_world_size)
        return (payload_bytes + 15) // 16 * 16
