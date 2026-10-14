from types import SimpleNamespace

import pytest
import torch

from lightllm.common.kv_cache_mem_manager.mem_manager import MemoryManager
from lightllm.common.kv_cache_mem_manager.deepseek2_mem_manager import Deepseek2MemoryManager
from lightllm.common.kv_cache_mem_manager.ppl_int4kv_mem_manager import PPLINT4KVMemoryManager
from lightllm.common.kv_cache_mem_manager.ppl_int8kv_mem_manager import PPLINT8KVMemoryManager


def test_kv_cache_allocation_requires_complete_pages():
    manager = MemoryManager.__new__(MemoryManager)
    manager.page_size = 4
    manager.allocator = SimpleNamespace(alloc=lambda size: torch.arange(size, dtype=torch.int32))

    with pytest.raises(AssertionError, match="must be a multiple of page_size 4"):
        manager.alloc(5)

    assert manager.alloc(8).tolist() == list(range(8))


@pytest.mark.parametrize(
    "manager_class",
    [MemoryManager, Deepseek2MemoryManager, PPLINT4KVMemoryManager, PPLINT8KVMemoryManager],
)
def test_kv_cache_buffer_size_requires_complete_pages(manager_class):
    manager = manager_class.__new__(manager_class)
    manager.page_size = 4

    with pytest.raises(AssertionError, match="KV cache size 5 must be a multiple of page_size 4"):
        manager._init_buffers(size=5, dtype=torch.float16, head_num=1, head_dim=8, layer_num=1)
