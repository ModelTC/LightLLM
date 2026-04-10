from typing import Any

import torch

from lightllm.common.mamba_cache_mem_manager.cache_manager import MambaCacheManager

from .kv_buffer import KvBuffer


class HybridKvBuffer(KvBuffer):
    def __init__(
        self,
        buffer: torch.Tensor,
        head_num: int,
        transformer_layer_num: int,
        full_attention_interval: int,
        mamba_cache_size: int,
        linear_attn_layer_num: int,
        conv_state_dtype: torch.dtype,
        ssm_state_dtype: torch.dtype,
        conv_kernel_size: int,
        num_linear_k_heads: int,
        num_linear_v_heads: int,
        head_linear_k_dim: int,
        head_linear_v_dim: int,
    ):
        super().__init__(buffer, head_num)
        self._transformer_layer_num = transformer_layer_num
        self._full_attention_interval = full_attention_interval
        self.mamba_cache_manager = MambaCacheManager(
            size=mamba_cache_size,
            layer_num=linear_attn_layer_num,
            conv_state_dtype=conv_state_dtype,
            ssm_state_dtype=ssm_state_dtype,
            conv_kernel_size=conv_kernel_size,
            num_linear_k_heads=num_linear_k_heads,
            num_linear_v_heads=num_linear_v_heads,
            head_linear_k_dim=head_linear_k_dim,
            head_linear_v_dim=head_linear_v_dim,
        )

    def create_adapter(self):
        from .hybrid_kv_buffer_adapter import HybridKvBufferAdapter

        return HybridKvBufferAdapter(self)

    def _is_full_attention_layer(self, layer_idx: int) -> bool:
        return (layer_idx + 1) % self._full_attention_interval == 0

    def _get_full_attn_layer_idx(self, layer_idx: int) -> int:
        assert 0 <= layer_idx < self._transformer_layer_num, f"invalid transformer layer index {layer_idx}"
        assert self._is_full_attention_layer(layer_idx), f"layer {layer_idx} does not have kv cache storage"
        return layer_idx // self._full_attention_interval

    def get_mamba_cache(self, layer_idx: int):
        assert 0 <= layer_idx < self._transformer_layer_num, f"invalid transformer layer index {layer_idx}"
        assert not self._is_full_attention_layer(layer_idx), f"layer {layer_idx} is not a linear attention layer"
        layer_idx_in_linear = layer_idx - (layer_idx // self._full_attention_interval)
        return self.mamba_cache_manager.get_mamba_cache(layer_idx_in_linear)

    def copy_kv_to_mem_manager(self, layer_index: int, mem_index: torch.Tensor, kv: torch.Tensor) -> None:
        return super().copy_kv_to_mem_manager(self._get_full_attn_layer_idx(layer_index), mem_index, kv)

    def get_att_input_params(self, layer_index: int) -> Any:
        return super().get_att_input_params(self._get_full_attn_layer_idx(layer_index))

    def find_layer_index(self, k: torch.Tensor, v: torch.Tensor) -> int:
        kv_layer_index = super().find_layer_index(k, v)
        return (kv_layer_index + 1) * self._full_attention_interval - 1
