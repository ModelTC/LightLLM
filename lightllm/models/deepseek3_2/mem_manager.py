from typing import List
from typing_extensions import override
import torch

from lightllm.common.mem_manager import MemoryManager
from lightllm.common.deepseek2_mem_manager import Deepseek2MemoryManager
from lightllm.server.pd_io_struct import KVMoveTask
from lightllm.distributed.pynccl import PyNcclCommunicator

class Deepseek3_2MemoryManager(Deepseek2MemoryManager):
    def __init__(self, size, dtype, head_num, head_dim, layer_num, always_copy=False, mem_fraction=0.9 ,is_sub_mem_manager=False):
        super().__init__(size, dtype, head_num, head_dim, layer_num, always_copy, mem_fraction, is_sub_mem_manager)
        self.indexer_ks_mem_manager = Deepseek2MemoryManager(self.size, torch.uint8, 1, 132, layer_num, is_sub_mem_manager=True)
        return

    @override
    def get_cell_size(self):
        return super().get_cell_size() + 132
    
class Deepseek3_2FP8KVMemoryManager(Deepseek3_2MemoryManager):
    def __init__(self, size, dtype, head_num, head_dim, layer_num, always_copy=False, mem_fraction=0.9, is_sub_mem_manager=False):
        super().__init__(size, torch.uint8, head_num, head_dim + 2, layer_num, always_copy, mem_fraction, is_sub_mem_manager)