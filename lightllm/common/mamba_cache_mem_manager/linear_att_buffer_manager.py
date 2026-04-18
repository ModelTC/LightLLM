import torch
import collections
from lightllm.utils.log_utils import init_logger
from .layer_cache import LayerCache
from typing import List, Optional, Tuple, Union

logger = init_logger(__name__)


class LinearAttCacheManager:
    def __init__(
        self,
        size: int,
        layer_num: int,
        conv_state_dtype: torch.dtype,
        ssm_state_dtype: torch.dtype,
        conv_kernel_size: int,
        num_linear_k_heads: int,
        num_linear_v_heads: int,
        head_linear_k_dim: int,
        head_linear_v_dim: int,
    ):
        # init the mem state
        self.size = size
        self.num_linear_k_heads = num_linear_k_heads
        self.num_linear_v_heads = num_linear_v_heads
        self.head_linear_k_dim = head_linear_k_dim
        self.head_linear_v_dim = head_linear_v_dim
        self.conv_dim = (
            self.head_linear_k_dim * self.num_linear_k_heads * 2 + self.head_linear_v_dim * self.num_linear_v_heads
        )
        self.layer_num = layer_num
        self.conv_kernel_size = conv_kernel_size
        conv_state_shape = (self.conv_dim, conv_kernel_size - 1)
        ssm_state_shape = (
            self.num_linear_v_heads,
            self.head_linear_k_dim,
            self.head_linear_v_dim,
        )
        self.ssm_state_dtype = ssm_state_dtype
        self.conv_state_dtype = conv_state_dtype

        # init the layer cache
        self.conv_state_cache = LayerCache(self.size, conv_state_dtype, conv_state_shape, layer_num, device="cpu")
        self.ssm_state_cache = LayerCache(self.size, ssm_state_dtype, ssm_state_shape, layer_num, device="cpu")
        self.free_list = collections.deque()
        for i in range(self.size):
            self.free_list.append(i)
        return

    def get_state_cache(self, layer_idx: int):
        return self.conv_state_cache.buffer[layer_idx], self.ssm_state_cache.buffer[layer_idx]

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
