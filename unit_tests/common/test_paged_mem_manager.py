from types import SimpleNamespace

import pytest
import torch

from lightllm.common.kv_cache_mem_manager.mem_manager import MemoryManager


def test_kv_cache_allocation_requires_complete_pages():
    manager = MemoryManager.__new__(MemoryManager)
    manager.page_size = 4
    manager.allocator = SimpleNamespace(alloc=lambda size: torch.arange(size, dtype=torch.int32))

    with pytest.raises(AssertionError, match="must be a multiple of page_size 4"):
        manager.alloc(5)

    assert manager.alloc(8).tolist() == list(range(8))
