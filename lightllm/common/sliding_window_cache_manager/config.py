import dataclasses
from typing import Dict, List

import torch


@dataclasses.dataclass
class SlidingWindowCacheConfig:
    """Physical cache layout for a full + sliding-window transformer."""

    layer_types: List[str]
    num_kv_shared_layers: int
    sliding_window: int
    sliding_head_num: int
    sliding_head_dim: int
    full_head_num: int
    full_head_dim: int
    dtype: torch.dtype

    def __post_init__(self):
        assert self.sliding_window > 0
        assert "sliding_attention" in self.layer_types
        assert "full_attention" in self.layer_types
        cutoff = len(self.layer_types) - self.num_kv_shared_layers
        assert 0 < cutoff <= len(self.layer_types)

        self.sliding_layer_to_cache_index: Dict[int, int] = {}
        self.full_layer_to_cache_index: Dict[int, int] = {}
        owner_to_index = {"sliding_attention": {}, "full_attention": {}}
        next_index = {"sliding_attention": 0, "full_attention": 0}

        for layer_idx, layer_type in enumerate(self.layer_types[:cutoff]):
            assert layer_type in owner_to_index, f"unsupported attention layer type: {layer_type}"
            owner_to_index[layer_type][layer_idx] = next_index[layer_type]
            next_index[layer_type] += 1

        for layer_idx, layer_type in enumerate(self.layer_types):
            if layer_idx < cutoff:
                owner = layer_idx
            else:
                owner = next(idx for idx in range(cutoff - 1, -1, -1) if self.layer_types[idx] == layer_type)
            cache_index = owner_to_index[layer_type][owner]
            if layer_type == "sliding_attention":
                self.sliding_layer_to_cache_index[layer_idx] = cache_index
            else:
                self.full_layer_to_cache_index[layer_idx] = cache_index

        self.sliding_layer_num = next_index["sliding_attention"]
        self.full_layer_num = next_index["full_attention"]

    @property
    def all_layer_num(self):
        return len(self.layer_types)

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
