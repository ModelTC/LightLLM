from __future__ import annotations

import pytest

from lightllm.common.att_resources.allocators import FullAttKVAllocator


def test_kv_cache_allocation_requires_complete_pages():
    allocator = FullAttKVAllocator(16, page_size=4, pin_memory=False)
    with pytest.raises(AssertionError, match="complete KV pages"):
        allocator.alloc(5)
    allocated = allocator.alloc(8)
    assert len(allocated) == 8
    assert not set(allocator.hold_token_indices) & set(allocated.tolist())
    allocator.free(allocated)
    assert allocator.can_use_mem_size == 16


@pytest.mark.parametrize("size,page_size", [(5, 4), (-4, 4), (4, 0)])
def test_kv_allocator_capacity_requires_complete_pages(size: int, page_size: int):
    with pytest.raises(AssertionError, match="complete pages"):
        FullAttKVAllocator(size, page_size=page_size, pin_memory=False)
