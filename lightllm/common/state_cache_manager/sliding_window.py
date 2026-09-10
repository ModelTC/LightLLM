import dataclasses
import math
from typing import Dict

import torch

from .base import StateCacheManager


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

    def get_cpu_cache_big_page_bytes(self, big_page_token_num: int = None, tp_world_size: int = None):
        if big_page_token_num is None or tp_world_size is None:
            from lightllm.utils.envs_utils import get_env_start_args

            args = get_env_start_args()
            if big_page_token_num is None:
                big_page_token_num = args.linear_att_hash_page_size * args.linear_att_page_block_num
                assert args.cpu_cache_token_page_size == big_page_token_num
            if tp_world_size is None:
                tp_world_size = args.tp // args.dp
        # One CPU page contains all TP shards: full KV, window state, padding.
        payload_bytes = self.get_cpu_cache_full_att_bytes(big_page_token_num, tp_world_size)
        payload_bytes += self.get_cpu_cache_state_bytes(tp_world_size)
        return (payload_bytes + 15) // 16 * 16

    @classmethod
    def load_from_args(cls):
        from lightllm.models.gemma4.kv_layout import build_sliding_cache_config
        from lightllm.utils.config_utils import get_config_json
        from lightllm.utils.envs_utils import get_env_start_args, get_llm_data_type

        args = get_env_start_args()
        model_config = get_config_json(args.model_dir)
        text_config = model_config.get("text_config", model_config)
        return build_sliding_cache_config(text_config, args.tp // args.dp, get_llm_data_type())


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
