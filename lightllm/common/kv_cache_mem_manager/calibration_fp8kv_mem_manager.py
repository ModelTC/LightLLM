import torch
from typing import Tuple, Any
from .offline_fp8_quant_mem_manager import OfflineFP8QuantMemManager


class CalibrationFP8KVMemoryManager(OfflineFP8QuantMemManager):
    def __init__(self, size, dtype, head_num, head_dim, layer_num, always_copy=False, mem_fraction=0.9):
        super().__init__(size, dtype, head_num, head_dim, layer_num, always_copy, mem_fraction, is_export_mode=False)

    def copy_kv_to_mem_manager(self, layer_index: int, mem_index: torch.Tensor, kv: torch.Tensor):
        """
        推理模式：使用预计算的FP8 scale将kv量化后拷贝到kv_buffer中
        """
        from lightllm.common.basemodel.triton_kernel.destindex_copy_kv_fp8 import destindex_copy_kv_fp8

        scales = self.scales
        destindex_copy_kv_fp8(
            kv,
            mem_index,
            scales[layer_index] if scales is not None else None,
            self.kv_buffer[layer_index].view(torch.float8_e4m3fn),
        )
        return

    def get_att_input_params(self, layer_index: int) -> Tuple[Any, Any]:
        k = self.kv_buffer[layer_index][:, : self.head_num, :]
        v = self.kv_buffer[layer_index][:, self.head_num :, :]
        return k, v
