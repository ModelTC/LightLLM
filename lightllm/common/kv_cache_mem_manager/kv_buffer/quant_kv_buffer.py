from typing import Any, Optional

import torch

from .kv_buffer import KvBuffer


class QuantKvBuffer(KvBuffer):
    def __init__(
        self,
        buffer: torch.Tensor,
        scale_buffer: torch.Tensor,
        head_num: int,
        quant_group_size: Optional[int] = None,
    ):
        super().__init__(buffer=buffer, head_num=head_num)
        self.scale_buffer = scale_buffer
        self.quant_group_size = quant_group_size

    def copy_kv_to_mem_manager(self, layer_index: int, mem_index: torch.Tensor, kv: torch.Tensor) -> None:
        raise NotImplementedError("QuantKvBuffer subclass must implement quantized token writes")

    def get_scale_buffer(self) -> torch.Tensor:
        return self.scale_buffer

    def get_att_input_params(self, layer_index: int) -> Any:
        layer_buffer = self._buffer[layer_index]
        layer_scale_buffer = self.scale_buffer[layer_index]
        k = layer_buffer[:, : self._head_num, :]
        k_scale = layer_scale_buffer[:, : self._head_num, :]
        v = layer_buffer[:, self._head_num :, :]
        v_scale = layer_scale_buffer[:, self._head_num :, :]
        return (k, k_scale), (v, v_scale)

    def get_index_kv_buffer(self, index: Any) -> dict:
        return {
            "kv_buffer": self._buffer[:, index],
            "scale_buffer": self.scale_buffer[:, index],
        }

    def load_index_kv_buffer(self, index: Any, payload: dict) -> None:
        self._buffer[:, index].copy_(payload["kv_buffer"])
        self.scale_buffer[:, index].copy_(payload["scale_buffer"])


class PPLInt4QuantKvBuffer(QuantKvBuffer):
    def copy_kv_to_mem_manager(self, layer_index: int, mem_index: torch.Tensor, kv: torch.Tensor) -> None:
        from lightllm.common.basemodel.triton_kernel.kv_copy.ppl_int4kv_copy_kv import destindex_copy_int4kv

        destindex_copy_int4kv(
            kv,
            mem_index,
            self[layer_index],
            self.scale_buffer[layer_index],
            quant_group_size=self.quant_group_size,
        )


class PPLInt8QuantKvBuffer(QuantKvBuffer):
    def copy_kv_to_mem_manager(self, layer_index: int, mem_index: torch.Tensor, kv: torch.Tensor) -> None:
        from lightllm.common.basemodel.triton_kernel.kv_copy.ppl_int8kv_copy_kv import destindex_copy_quantize_kv

        destindex_copy_quantize_kv(
            kv,
            mem_index,
            self[layer_index],
            self.scale_buffer[layer_index],
            quant_group_dim=self.quant_group_size,
        )
