import dataclasses
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

    def get_sliding_layer_index(self, layer_index: int) -> int:
        return self.sliding_layer_to_cache_index[layer_index]

    def get_full_layer_index(self, layer_index: int) -> int:
        return self.full_layer_to_cache_index[layer_index]

    def get_state_shape(self):
        return (
            self.sliding_layer_num,
            self.sliding_window,
            2 * self.sliding_head_num,
            self.sliding_head_dim,
        )

    def get_state_nbytes(self):
        elements = 1
        for dim in self.get_state_shape():
            elements *= dim
        return elements * self.dtype.itemsize
