import torch
from typing import Tuple, Any
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
        full_attention_interval: int,
        conv_state_dtype: torch.dtype,
        ssm_state_dtype: torch.dtype,
        conv_kernel_size: int,
        num_linear_k_heads: int,
        num_linear_v_heads: int,
        head_linear_k_dim: int,
        head_linear_v_dim: int,
        max_req_num: int,
        always_copy=False,
        mem_fraction=0.9,
        network_config: dict = None,
    ):

        self.full_attention_interval = full_attention_interval
        assert layer_num % full_attention_interval == 0
        self.transformer_layer_num = layer_num
        self.full_attn_layer_num = layer_num // full_attention_interval
        self.linear_attn_layer_num = layer_num - self.full_attn_layer_num

        self.mamba_cache_mem_manager = MambaCacheManager(
            size=linear_attn_cache_size,
            layer_num=self.linear_attn_layer_num,
            conv_state_dtype=conv_state_dtype,
            ssm_state_dtype=ssm_state_dtype,
            conv_kernel_size=conv_kernel_size,
            num_linear_k_heads=num_linear_k_heads,
            num_linear_v_heads=num_linear_v_heads,
            head_linear_k_dim=head_linear_k_dim,
            head_linear_v_dim=head_linear_v_dim,
        )

        super().__init__(
            full_attn_cache_size,
            dtype,
            num_kv_heads,
            head_dim,
            self.full_attn_layer_num,
            always_copy,
            mem_fraction,
        )

    def _init_buffers(self, size, dtype, head_num, head_dim, layer_num):
        self.kv_buffer = torch.empty((layer_num, size + 1, 2 * head_num, head_dim), dtype=dtype, device="cuda")

    def free_all(self):
        super().free_all()
        self.mamba_cache_mem_manager.free_all()
        return

    def _is_full_attention_layer(self, layer_idx: int) -> bool:
        return (layer_idx + 1) % self.full_attention_interval == 0

    def _get_full_attn_layer_idx(self, layer_idx: int) -> int:
        assert 0 <= layer_idx < self.transformer_layer_num, f"invalid transformer layer index {layer_idx}"
        assert self._is_full_attention_layer(layer_idx), f"layer {layer_idx} is not a full attention layer"
        return layer_idx // self.full_attention_interval

    def copy_kv_to_mem_manager(self, layer_index: int, mem_index: torch.Tensor, kv: torch.Tensor):
        kv_layer_index = self._get_full_attn_layer_idx(layer_index)
        return super().copy_kv_to_mem_manager(kv_layer_index, mem_index, kv)

    def get_att_input_params(self, layer_index: int) -> Tuple[Any, Any]:
        kv_layer_index = self._get_full_attn_layer_idx(layer_index)
        return super().get_att_input_params(kv_layer_index)

    def get_mamba_cache(self, layer_idx: int):
        assert 0 <= layer_idx < self.transformer_layer_num, f"invalid transformer layer index {layer_idx}"
        assert not self._is_full_attention_layer(layer_idx), f"layer {layer_idx} is not a linear attention layer"
        layer_idx_in_linear = layer_idx - (layer_idx // self.full_attention_interval)
        return self.mamba_cache_mem_manager.get_mamba_cache(layer_idx_in_linear)
