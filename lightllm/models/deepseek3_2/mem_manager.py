from typing_extensions import override
import torch

from lightllm.common.kv_cache_mem_manager.mem_manager import MemoryManager
from lightllm.common.kv_cache_mem_manager.deepseek2_mem_manager import Deepseek2MemoryManager


class IndexerKSBuffer:
    def __init__(self, size: int, head_num: int, head_dim: int, layer_num: int, dtype=torch.uint8):
        self.kv_buffer = torch.empty((layer_num, size + 1, head_num, head_dim), dtype=dtype, device="cuda")


class Deepseek3_2MemoryManager(Deepseek2MemoryManager):
    def __init__(self, size, dtype, head_num, head_dim, layer_num, always_copy=False, mem_fraction=0.9):
        super().__init__(size, dtype, head_num, head_dim, layer_num, always_copy, mem_fraction)
        self.indexer_ks_buffer = IndexerKSBuffer(self.size, 1, 132, layer_num)

    @override
    def get_cell_size(self):
        return super().get_cell_size() + 132

    @override
    def _free_buffers(self):
        super()._free_buffers()
        self.indexer_ks_buffer = None

    @override
    def resize_mem(self, new_size):
        super().resize_mem(new_size)
        self.indexer_ks_buffer = IndexerKSBuffer(self.size, 1, 132, self.layer_num)


class Deepseek3_2FP8KVMemoryManager(Deepseek3_2MemoryManager):
    def __init__(self, size, dtype, head_num, head_dim, layer_num, always_copy=False, mem_fraction=0.9):
        super().__init__(size, torch.uint8, head_num, head_dim + 2, layer_num, always_copy, mem_fraction)
