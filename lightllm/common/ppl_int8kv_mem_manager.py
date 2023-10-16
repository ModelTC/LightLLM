import torch

from .mem_manager import MemoryManager


class PPLINT8KVMemoryManager(MemoryManager):
    def __init__(self, size, dtype, head_num, head_dim, layer_num, always_copy=True):
        super().__init__(size, dtype, head_num, head_dim, layer_num, always_copy=True)

    def _init_buffers(self, size, dtype, head_num, head_dim, layer_num):
        group_quant_size = 8
        self.key_buffer = [torch.empty((size, head_num, head_dim), dtype=torch.int8, device="cuda") for _ in range(layer_num)]
        self.value_buffer = [torch.empty((size, head_num, head_dim), dtype=torch.int8, device="cuda") for _ in range(layer_num)]
        self.key_scale_buffer = [torch.empty((size, head_num, head_dim // group_quant_size), dtype=dtype, device="cuda") for _ in range(layer_num)]
        self.value_scale_buffer = [torch.empty((size, head_num, head_dim // group_quant_size), dtype=dtype, device="cuda") for _ in range(layer_num)]
