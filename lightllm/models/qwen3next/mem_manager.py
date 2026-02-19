import torch
from typing import Tuple
from lightllm.utils.log_utils import init_logger
from lightllm.common.kv_cache_mem_manager.mem_manager import MemoryManager
from lightllm.common.mamba_cache_mem_manager.cache_manager import MambaCacheManager

logger = init_logger(__name__)


class Qwen3NextHybridMemManager(MemoryManager):
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

        self.mamba_cache_mem_manager = MambaCacheManager(
            linear_attn_cache_size,
            self.linear_attn_layer_num,
            conv_state_dtype,
            conv_state_shape,
            ssm_state_dtype,
            ssm_state_shape,
        )

        super().__init__(full_attn_cache_size, dtype, num_kv_heads, head_dim, layer_num, always_copy, mem_fraction)

    def _init_buffers(self, size, dtype, head_num, head_dim, layer_num):
        # KV buffer layout: [None, None, None, kv_cache, None, None, None, kv_cache, ...,
        #                    None, kv_cache, mtp_kv_cache, mtp_kv_cache]
        # Only full attention layers and MTP layers have KV cache.
        self.kv_buffer = [None for _ in range(self.layer_num)]
        for layer_id in range(self.full_attn_layer_num):
            self.kv_buffer[(layer_id + 1) * self.full_attention_interval - 1] = torch.empty(
                (size + 1, 2 * head_num, head_dim), dtype=dtype, device="cuda"
            )
        for _ in range(self.mtp_layer_num):
            self.kv_buffer.append(torch.empty((size + 1, 2 * head_num, head_dim), dtype=dtype, device="cuda"))

    def free_all(self):
        super().free_all()
        self.mamba_cache_mem_manager.free_all()
        return

    def get_cell_size(self):
        # Only full attention layers and MTP layers have KV cache
        kv_cache_layer_num = self.full_attn_layer_num + self.mtp_layer_num
        return 2 * self.head_num * self.head_dim * kv_cache_layer_num * torch._utils._element_size(self.dtype)

    def get_mamba_cache(self, layer_idx: int):
        layer_idx_in_linear = layer_idx - (layer_idx // self.full_attention_interval)
        return self.mamba_cache_mem_manager.get_mamba_cache(layer_idx_in_linear)
