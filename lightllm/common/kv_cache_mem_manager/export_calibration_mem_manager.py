import torch
from typing import Tuple, Any
from .offline_fp8_quant_mem_manager import OfflineFP8QuantMemManager


class ExportCalibrationMemoryManager(OfflineFP8QuantMemManager):
    def __init__(self, size, dtype, head_num, head_dim, layer_num, always_copy=False, mem_fraction=0.9):
        super().__init__(size, dtype, head_num, head_dim, layer_num, always_copy, mem_fraction, is_export_mode=True)

    def copy_kv_to_mem_manager(self, layer_index: int, mem_index: torch.Tensor, kv: torch.Tensor):
        """
        导出校准模式：先用普通方式拷贝原始kv（保持原始精度），然后收集校准统计数据
        """
        from lightllm.common.basemodel.triton_kernel.destindex_copy_kv import destindex_copy_kv

        destindex_copy_kv(kv, mem_index, self.kv_buffer[layer_index])
        self.update_calibration_data(kv, layer_index)
        return

    def get_att_input_params(self, layer_index: int) -> Tuple[Any, Any]:
        k = self.kv_buffer[layer_index][:, : self.head_num, :]
        v = self.kv_buffer[layer_index][:, self.head_num :, :]
        return k, v
