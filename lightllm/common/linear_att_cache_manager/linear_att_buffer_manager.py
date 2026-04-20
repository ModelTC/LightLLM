import torch
import collections
from lightllm.utils.log_utils import init_logger
from .layer_cache import LayerCache
from typing import List, Optional, Tuple, Union
from .config_objs import LinearAttCacheConfig

logger = init_logger(__name__)


class LinearAttCacheManager:
    def __init__(
        self,
        size: int,
        linear_config: LinearAttCacheConfig,
    ):
        # init the mem state
        self.size = size
        self.linear_config = linear_config

        # init the layer cache
        self.conv_state_cache = LayerCache(
            size=self.size,
            dtype=self.linear_config.conv_state_dtype,
            shape=self.linear_config.get_conv_state_shape(),
            layer_num=self.linear_config.linear_layer_num,
            device="cpu",
        )
        self.ssm_state_cache = LayerCache(
            size=self.size,
            dtype=self.linear_config.ssm_state_dtype,
            shape=self.linear_config.get_ssm_state_shape(),
            layer_num=self.linear_config.linear_layer_num,
            device="cpu",
        )
        self.clear_to_init_state()
        return

    def get_state_cache(self, buffer_idx: int):
        return self.conv_state_cache.buffer[:, buffer_idx, ...], self.ssm_state_cache.buffer[:, buffer_idx, ...]

    def alloc_one_state_cache(self) -> Optional[int]:
        if len(self.free_list) == 0:
            return None

        alloc_index = self.free_list.popleft()
        return alloc_index

    def alloc_state_cache(self, need_size: int) -> Optional[List[int]]:
        if need_size > len(self.free_list):
            logger.error(f"warn no enough cache need_size {need_size} free_size {len(self.free_list)}")
            return None

        alloc_indexes = [self.free_list.popleft() for _ in range(need_size)]
        return alloc_indexes

    def free_state_cache(self, free_indexes: List[int]):
        self.free_list.extend(free_indexes)
        assert (
            len(self.free_list) <= self.size
        ), f"free cache num {len(self.free_list)} should not be larger than total size {self.size}"
        return

    def get_free_cache_num(self):
        return len(self.free_list)

    def get_used_cache_num(self):
        return self.size - len(self.free_list)

    def clear_to_init_state(self):
        self.conv_state_cache.buffer.zero_()
        self.ssm_state_cache.buffer.zero_()
        self.free_list.clear()
        for i in range(self.size):
            self.free_list.append(i)
        return
