import torch
import numpy as np
from typing import Dict, List, Protocol, Set, Union, Tuple, Optional
from typing_extensions import override
from lightllm.utils.log_utils import init_logger
from lightllm.common.kv_cache_mem_manager.mem_manager import BaseAllocator, MemoryManager
from lightllm.utils.envs_utils import get_env_start_args
from lightllm.server.router.model_infer.infer_batch import InferReq
from lightllm.utils.envs_utils import get_unique_server_name
from lightllm.utils.dist_utils import get_current_rank_in_node
from lightllm.server.router.dynamic_prompt.shared_arr import SharedInt
from lightllm.server.router.dynamic_prompt.hybrid_radix_cache import HybridMemManager

logger = init_logger(__name__)


class LayerCacheMemoryManager(BaseAllocator):
    def __init__(self, size: int, dtype: torch.dtype, shape: Tuple[int, ...], layer_num: int, mem_manager_nmae: str):
        super().__init__(size, mem_manager_nmae)

        self.dtype = dtype
        self.shape = shape
        self.layer_num = layer_num

        self._init_buffers(
            self.size,
            dtype,
            shape,
        )

    def _init_buffers(self, size, dtype, shape):
        self.buffer = torch.zeros((self.layer_num, size + 1, *shape), dtype=dtype, device="cuda")

    def get_cell_size(self):
        return np.prod(self.shape) * self.layer_num * torch._utils._element_size(self.dtype)


class Qwen3NextMemoryManager(HybridMemManager):
    def __init__(
        self,
        full_attn_cache_size,
        linear_attn_cache_size,
        dtype,
        num_kv_heads,
        head_dim,
        layer_num,
        mtp_layer_num,
        full_attention_interval: int,
        conv_state_dtype: torch.dtype,
        conv_state_shape: Tuple[int, ...],
        ssm_state_dtype: torch.dtype,
        ssm_state_shape: Tuple[int, ...],
        max_req_num: int,
        always_copy=False,
        mem_fraction=0.9,
    ):
        self.full_attention_interval = full_attention_interval

        assert layer_num % full_attention_interval == 0
        self.layer_num = layer_num
        self.mtp_layer_num = mtp_layer_num
        self.full_attn_layer_num = layer_num // full_attention_interval
        self.linear_attn_layer_num = layer_num - self.full_attn_layer_num

        self.conv_state_dtype = conv_state_dtype
        self.conv_state_shape = conv_state_shape
        self.ssm_state_dtype = ssm_state_dtype
        self.ssm_state_shape = ssm_state_shape

        assert linear_attn_cache_size is not None
        self.conv_state_mem_manager = LayerCacheMemoryManager(
            linear_attn_cache_size, conv_state_dtype, conv_state_shape, self.linear_attn_layer_num, "conv_state"
        )
        self.ssm_state_mem_manager = LayerCacheMemoryManager(
            linear_attn_cache_size, ssm_state_dtype, ssm_state_shape, self.linear_attn_layer_num, "ssm_state"
        )
        logger.info(
            f"Linear attention state cache size: {linear_attn_cache_size}\n"
            f"Conv state use : "
            f"{self.conv_state_mem_manager.get_cell_size() * linear_attn_cache_size / 1024 ** 3} GB Memory.\n"
            f"Ssm state use : "
            f"{self.ssm_state_mem_manager.get_cell_size() * linear_attn_cache_size / 1024 ** 3} GB Memory.\n"
        )
        super().__init__(full_attn_cache_size, dtype, num_kv_heads, head_dim, layer_num, always_copy, mem_fraction)

    @override
    def _init_buffers(self, size, dtype, head_num, head_dim, layer_num):
        # kv_buffer = [None, None, None, kv_cache, None, None, None, kv_cache, ...,
        #                None, kv_cache, mtp_kv_cache, mtp_kv_cache]
        self.kv_buffer = [None for _ in range(self.layer_num)]
        for layer_id in range(self.full_attn_layer_num):
            self.kv_buffer[(layer_id + 1) * self.full_attention_interval - 1] = torch.empty(
                (size + 1, 2 * head_num, head_dim), dtype=dtype, device="cuda"
            )

        for _ in range(self.mtp_layer_num):
            self.kv_buffer.append(torch.empty((size + 1, 2 * head_num, head_dim), dtype=dtype, device="cuda"))

    @override
    def free_all(self):
        super().free_all()
        self.conv_state_mem_manager.free_all()
        self.ssm_state_mem_manager.free_all()
        return

    @override
    def get_buffer(self, layer_index) -> Tuple[torch.Tensor, torch.Tensor]:
        assert layer_index < self.layer_num, "layer_index is out of range"
        assert (layer_index + 1) % self.full_attention_interval != 0, "layer_index is not linear attention layer"
        real_layer_index = layer_index - layer_index // self.full_attention_interval
        return self.conv_state_mem_manager.buffer[real_layer_index], self.ssm_state_mem_manager.buffer[real_layer_index]

    @override
    def free_buffer(self, free_buffer_indexes: List[int], reset=True):
        # conv_state 和 ssm_state 共享buffer_idx
        self.conv_state_mem_manager.free(free_buffer_indexes)
        if reset:
            self.conv_state_mem_manager.buffer[:, free_buffer_indexes] = 0
            self.ssm_state_mem_manager.buffer[:, free_buffer_indexes] = 0

    @override
    def alloc_buffer(self, need_size):
        # conv_state 和 ssm_state 共享buffer_idx
        buffer_indexes = self.conv_state_mem_manager.alloc(need_size)
        return buffer_indexes

    @override
    def get_buffer_can_use_size(self):
        return self.conv_state_mem_manager.can_use_mem_size

    @override
    def copy_buffer(self, src_idx, tgt_idx):
        assert src_idx is not None and tgt_idx is not None
        assert src_idx != tgt_idx
        # Use slice operation and in-place copy for better performance
        self.conv_state_mem_manager.buffer[:, tgt_idx].copy_(self.conv_state_mem_manager.buffer[:, src_idx])
        self.ssm_state_mem_manager.buffer[:, tgt_idx].copy_(self.ssm_state_mem_manager.buffer[:, src_idx])
        return
