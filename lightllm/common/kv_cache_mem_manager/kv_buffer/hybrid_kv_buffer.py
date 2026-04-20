import torch
from typing import Any
from lightllm.common.linear_att_cache_manager.config_objs import LinearAttCacheConfig
from .kv_buffer import KvBuffer


class HybridKvBuffer(KvBuffer):
    def __init__(
        self,
        buffer: torch.Tensor,
        head_num: int,
        linear_config: LinearAttCacheConfig,
    ):
        super().__init__(buffer, head_num)
        self.linear_config = linear_config

    def create_adapter(self):
        from .hybrid_kv_buffer_adapter import HybridKvBufferAdapter

        return HybridKvBufferAdapter(self)

    def _is_full_attention_layer(self, layer_idx_in_all: int) -> bool:
        return (layer_idx_in_all + 1) % self.linear_config.full_attention_interval == 0

    def _get_full_attn_layer_idx(self, layer_idx_in_all: int) -> int:
        assert (
            0 <= layer_idx_in_all < self.linear_config.all_layer_num
        ), f"invalid transformer layer index {layer_idx_in_all}"
        assert self._is_full_attention_layer(
            layer_idx_in_all
        ), f"layer {layer_idx_in_all} does not have kv cache storage"
        return layer_idx_in_all // self.linear_config.full_attention_interval

    def copy_kv_to_mem_manager(self, layer_index_in_all: int, mem_index: torch.Tensor, kv: torch.Tensor) -> None:
        return super().copy_kv_to_mem_manager(self._get_full_attn_layer_idx(layer_index_in_all), mem_index, kv)

    def get_att_input_params(self, layer_index_in_all: int) -> Any:
        return super().get_att_input_params(self._get_full_attn_layer_idx(layer_index_in_all))

    def find_layer_index(self, k: torch.Tensor, v: torch.Tensor) -> int:
        kv_layer_index = super().find_layer_index(k, v)
        return (kv_layer_index + 1) * self.linear_config.full_attention_interval - 1
