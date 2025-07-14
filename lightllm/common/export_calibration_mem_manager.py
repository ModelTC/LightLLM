from .offline_fp8_quant_mem_manager import OfflineFP8QuantMemManager


class ExportCalibrationMemoryManager(OfflineFP8QuantMemManager):
    def __init__(self, size, dtype, head_num, head_dim, layer_num, always_copy=False, mem_fraction=0.9):
        super().__init__(size, dtype, head_num, head_dim, layer_num, always_copy, mem_fraction, is_export_mode=True)
