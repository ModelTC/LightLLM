import torch
from .deepseek2_mem_manager import Deepseek2MemoryManager


class Deepseek2FP8KVMemoryManager(Deepseek2MemoryManager):
    def __init__(self, size, dtype, head_num, head_dim, layer_num, always_copy=False, mem_fraction=0.9):
        # scale被追加到kv_buffer末尾, 因此加2, dtype统一改成uint8
        super().__init__(size, torch.uint8, head_num, head_dim + 2, layer_num, always_copy, mem_fraction)
