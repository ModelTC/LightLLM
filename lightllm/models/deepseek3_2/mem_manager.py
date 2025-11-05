from typing_extensions import override
import torch

from lightllm.common.basemodel.infer_struct import InferStateInfo
from lightllm.common.deepseek2_mem_manager import Deepseek2MemoryManager
from lightllm.utils.log_utils import init_logger

logger = init_logger(__name__)

class Deepseek3_2IndexerPagedMemoryManager:
    def __init__(self, page_size):
        self.page_size = page_size
        return
    
    def set_size(self, size):
        self.physics_size = size
        self.num_pages = size // self.page_size
        return

    def _init_buffers(self):
        self.k_cache_buffer = torch.empty(
            (self.page_size, 128), dtype=torch.float8_e4m3fn, device="cuda")
        self.k_scale_buffer = torch.empty(
            (self.page_size, 1), dtype=torch.float64, device="cuda")
        return
    
    def alloc_paged_index(self, last_index: int, need_size):
        pass

    def get_cell_size(self):
        # Use for deepseek v3.2 exp only, 128 for k_cache(128 torch.float8_e4m3fn), 4 for scale(1 torch.float64)
        return 128 + 4 

    
class Deepseek3_2MemoryManager(Deepseek2MemoryManager):
    def __init__(
        self, 
        size, 
        dtype, 
        head_num, 
        head_dim, 
        layer_num, 
        always_copy=False, 
        mem_fraction=0.9,
        page_size=64
    ):
        self.page_size = page_size
        self.indexer_paged_mem_manager = Deepseek3_2IndexerPagedMemoryManager(page_size)
        super().__init__(size, dtype, head_num, head_dim, layer_num, always_copy, mem_fraction)
        self.indexer_paged_mem_manager.set_size(self.size)
        return

    @override
    def get_cell_size(self):
        return super().get_cell_size() + self.indexer_paged_mem_manager.get_cell_size()

    @override
    def _init_buffers(self, size, dtype, head_num, head_dim, layer_num):
        super()._init_buffers(size, dtype, head_num, head_dim, layer_num)
        self.indexer_paged_mem_manager._init_buffers()
        return

    @override
    def profile_size(self, mem_fraction):
        super().profile_size(mem_fraction)
        if self.size % self.page_size != 0:
            size_paged = (self.size // self.page_size + 1) * self.page_size
            logger.warning(f"size {self.size} is not divisible by page_size {self.page_size}, will use paged_size {size_paged}")
            self.size = size_paged
        return