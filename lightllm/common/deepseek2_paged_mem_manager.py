import torch
import numpy as np
from .deepseek2_mem_manager import Deepseek2MemoryManager
from .paged_mem_manager import PagedMemoryManager
from lightllm.utils.log_utils import init_logger
from lightllm.utils.envs_utils import get_page_size


def cdiv(a, b):
    return (a + b - 1) // b


logger = init_logger(__name__)


class Deepseek2PagedMemoryManager(PagedMemoryManager, Deepseek2MemoryManager):
    def __init__(self, size, dtype, head_num, head_dim, layer_num, always_copy=False, mem_fraction=0.9):
        super().__init__(size, dtype, head_num, head_dim, layer_num, always_copy, mem_fraction)

    def _init_buffers(self, size, dtype, head_num, head_dim, layer_num):
        self.kv_buffer = torch.empty(
            (layer_num, cdiv(size, get_page_size()) * get_page_size(), head_num, head_dim),
            dtype=dtype,
            device="cuda",
        )
