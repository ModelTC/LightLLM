from .mem_manager import MemoryManager

class GQAMemoryManager(MemoryManager):
    def __init__(self, size, dtype, key_value_head_num, head_dim, layer_num):
        super().__init__(size, dtype, key_value_head_num, head_dim, layer_num)