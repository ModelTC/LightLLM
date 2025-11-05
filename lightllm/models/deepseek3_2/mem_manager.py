from typing_extensions import override
import torch

from lightllm.common.deepseek2_mem_manager import Deepseek2MemoryManager


class Deepseek3_2MemoryManager(Deepseek2MemoryManager):
    def __init__(
        self, 
        size, 
        dtype, 
        head_num, 
        head_dim, 
        layer_num, 
        index_head_dim, 
        index_quant_block_size,
        k_cache_dtype=torch.float8_e4m3fn,
        k_scale_dtype=torch.float32,
        always_copy=False, 
        mem_fraction=0.9
    ):
        super().__init__(size, dtype, head_num, head_dim, layer_num, always_copy, mem_fraction)
        assert index_head_dim % index_quant_block_size == 0, "index_head_dim must be divisible by index_quant_block_size"
        self.index_head_dim = index_head_dim
        self.index_quant_block_size = index_quant_block_size
        self.k_cache_dtype = k_cache_dtype
        self.k_scale_dtype = k_scale_dtype
        return

    @override
    def get_cell_size(self):
        index_k_cache_cell_size = self.index_head_dim * self.layer_num * torch._utils._element_size(self.k_cache_dtype)
        index_k_scale_cell_size = (self.index_head_dim // self.index_quant_block_size) * self.layer_num * torch._utils._element_size(self.k_scale_dtype)
        return super().get_cell_size() + index_k_cache_cell_size + index_k_scale_cell_size

    @override
    def _init_buffers(self, size, dtype, head_num, head_dim, layer_num):
        super()._init_buffers(size, dtype, head_num, head_dim, layer_num)
        self._init_indexer_k_cache_buffers()
        return

    def _init_indexer_k_cache_buffers(self):
        self.indexer_k_cache_buffers = torch.empty(
            (self.layer_num, self.size + 1, self.index_head_dim), dtype=self.k_cache_dtype, device="cuda")
        self.indexer_k_scale_buffers = torch.empty(
            (self.layer_num, self.size + 1, self.index_head_dim // self.index_quant_block_size), dtype=self.k_scale_dtype, device="cuda")
        return
