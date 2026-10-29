from __future__ import annotations

import pytest
import torch

from lightllm.common.att_resources.allocators import StatePageAllocator, FullAttKVAllocator


def test_slot_allocation_domains_and_reserved_storage():
    big = StatePageAllocator(5, reserved_slots=2)
    small = StatePageAllocator(5)
    assert big.alloc(3) == [0, 1, 2]
    assert small.alloc(1) == [0]
    assert big.alloc(1) is None
    with pytest.raises(AssertionError):
        big.free([0, 3])
    assert big.free_count == 0
    big.free([0])
    assert big.free_count == 1 and small.free_count == 4
    with pytest.raises(AssertionError):
        big.free([0])
    with pytest.raises(AssertionError):
        big.free([1, 1])
    assert big.alloc(1) == [0]


def test_kv_returns_complete_pages_and_rejects_partial_free_atomically():
    allocator = FullAttKVAllocator(16, 4, pin_memory=False)
    tokens = allocator.alloc(8)
    snapshot = tokens.clone()
    assert tokens.tolist() == list(range(8))
    assert allocator.hold_token_indices == (16, 17, 18, 19)
    assert ((tokens.view(-1, 4)[:, 0] % 4) == 0).all()
    assert (tokens.view(-1, 4).diff(dim=1) == 1).all()
    with pytest.raises(AssertionError):
        allocator.free(tokens[:2])
    assert allocator.can_use_mem_size == 8
    more = allocator.alloc(8)
    assert set(tokens.tolist()).isdisjoint(more.tolist())
    assert torch.equal(tokens, snapshot)
    with pytest.raises(AssertionError, match="available"):
        allocator.alloc(4)
    # Pages may be returned in any order; tokens within each page remain ordered.
    returned = tokens.view(-1, 4).flip(0).reshape(-1)
    allocator.free(returned)
    assert allocator.can_use_mem_size == 8
    assert more.tolist() == list(range(8, 16))
    reused = allocator.alloc(8)
    assert torch.equal(reused, returned)
    allocator.free(more)
    allocator.free(reused)
    assert allocator.can_use_mem_size == 16
    assert set(allocator.alloc(16).tolist()) == set(range(16))


def test_kv_validation_and_empty_operations():
    with pytest.raises(AssertionError, match="HOLD"):
        FullAttKVAllocator((1 << 31) - 2, 3, pin_memory=False)
    allocator = FullAttKVAllocator(8, 4, pin_memory=False)
    for n in [-4, 3]:
        with pytest.raises(AssertionError):
            allocator.alloc(n)
    assert allocator.alloc(0).numel() == 0
    allocator.free([])
    assert allocator.can_use_mem_size == 8
    tokens = allocator.alloc(4)
    with pytest.raises(AssertionError, match="complete KV pages"):
        allocator.free(tokens[:3])
    with pytest.raises(AssertionError, match="exceeds allocated"):
        allocator.free(list(range(8)))
    assert allocator.can_use_mem_size == 4
    allocator.free_all()
    assert allocator.can_use_mem_size == 8
    assert set(allocator.alloc(8).tolist()) == set(range(8))


@pytest.mark.parametrize("page_size", [1, 4, 16])
def test_kv_repeated_allocation_and_return_buffer_wrap(page_size: int):
    allocator = FullAttKVAllocator(64, page_size, pin_memory=False)
    # Keep half the domain live while cycling the other half through the return buffer.
    retained = allocator.alloc(32).clone()
    for _ in range(16):
        current = allocator.alloc(32)
        snapshot = current.clone()
        assert set(current.tolist()) == set(range(32, 64))
        allocator.free(current.tolist())
        assert torch.equal(current, snapshot)
        assert allocator.can_use_mem_size == 32
    allocator.free(retained)
    assert set(allocator.alloc(64).tolist()) == set(range(64))
