from typing import Any, List, Optional

import torch

from lightllm.common.mamba_cache_mem_manager.cache_manager import MambaCacheManager

from .kv_buffer import KvBuffer


class HybridKvBuffer(KvBuffer):
    def __init__(
        self,
        buffers: List[Optional[torch.Tensor]],
        head_num: int,
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
        self._buffers = buffers
        self._head_num = head_num
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

    def get_mamba_cache(self, layer_idx: int):
        layer_idx_in_linear = layer_idx - (layer_idx // self._full_attention_interval)
        return self.mamba_cache_manager.get_mamba_cache(layer_idx_in_linear)

    def __getitem__(self, item):
        return self._buffers[item]

    def copy_kv_to_mem_manager(self, layer_index: int, mem_index: torch.Tensor, kv: torch.Tensor) -> None:
        from lightllm.common.basemodel.triton_kernel.destindex_copy_kv import destindex_copy_kv

        layer_buffer = self._buffers[layer_index]
        if layer_buffer is None:
            raise RuntimeError(f"layer {layer_index} does not have kv cache storage")
        destindex_copy_kv(kv, mem_index, layer_buffer)

    def get_att_input_params(self, layer_index: int) -> Any:
        layer_buffer = self._buffers[layer_index]
        if layer_buffer is None:
            raise RuntimeError(f"layer {layer_index} does not have kv cache storage")
        k = layer_buffer[:, : self._head_num, :]
        v = layer_buffer[:, self._head_num :, :]
        return k, v

    def get_index_kv_buffer(self, index: Any) -> dict:
        return {"kv_buffer": [None if layer_buffer is None else layer_buffer[index] for layer_buffer in self._buffers]}

    def load_index_kv_buffer(self, index: Any, payload: dict) -> None:
        for layer_index, layer_payload in enumerate(payload["kv_buffer"]):
            if layer_payload is None:
                continue
            layer_buffer = self._buffers[layer_index]
            if layer_buffer is None:
                raise RuntimeError(f"layer {layer_index} does not have kv cache storage")
            layer_buffer[index].copy_(layer_payload)

    def get_device(self) -> int:
        for layer_buffer in self._buffers:
            if layer_buffer is not None:
                return layer_buffer.get_device()
        raise RuntimeError("HybridKvBuffer does not contain any kv cache tensor")

    def find_layer_index(self, k: torch.Tensor, v: torch.Tensor) -> int:
        key = min(k.data_ptr(), v.data_ptr())
        find_dict = {
            layer_buffer.data_ptr(): layer_index
            for layer_index, layer_buffer in enumerate(self._buffers)
            if layer_buffer is not None
        }
        assert key in find_dict
        return find_dict[key]
