import torch

from .mem_manager import MemoryManager


class INT8KVMemoryManager(MemoryManager):
    def __init__(self, size, dtype, head_num, head_dim, layer_num):
        super().__init__(size, dtype, head_num, head_dim, layer_num)
        
    def _init_buffers(self, size, dtype, head_num, head_dim, layer_num):
        self.key_buffer = [torch.empty((size, head_num, head_dim), dtype=torch.int8, device="cuda") for _ in range(layer_num)]
        self.value_buffer = [torch.empty((size, head_num, head_dim), dtype=torch.int8, device="cuda") for _ in range(layer_num)]
        self.key_scale_buffer = [torch.empty((size, head_num, 1), dtype=dtype, device="cuda") for _ in range(layer_num)]
        self.value_scale_buffer = [torch.empty((size, head_num, 1), dtype=dtype, device="cuda") for _ in range(layer_num)]

    @torch.no_grad()
    def alloc_contiguous(self, need_size):
        # INT8 KVCache can not alloc contiguous for quantized copy.
        return None
