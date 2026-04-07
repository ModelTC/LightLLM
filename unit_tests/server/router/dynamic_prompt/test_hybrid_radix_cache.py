import pytest
import torch
from unittest.mock import MagicMock


def _make_mock_kv_mem_manager(size=1000):
    mgr = MagicMock()
    mgr.can_use_mem_size = size
    mgr.free = MagicMock()
    return mgr


def _make_mock_buffer_mem_manager(size=50):
    mgr = MagicMock()
    mgr.size = size
    mgr.can_use_mem_size = size
    mgr.free = MagicMock()
    return mgr


def _make_hybrid_cache(unique_suffix, kv_size=1000, buf_size=50):
    from lightllm.server.router.dynamic_prompt.hybrid_radix_cache import HybridRadixCache

    kv_mgr = _make_mock_kv_mem_manager(kv_size)
    kv_mgr.mamba_cache_mem_manager = _make_mock_buffer_mem_manager(buf_size)
    cache = HybridRadixCache(f"test_{unique_suffix}", kv_size, 0, kv_mgr)
    return cache


def test_evict_buffer_set_hotspot_sort_order():
    cache = _make_hybrid_cache("hotspot_sort")

    # Insert two nodes with tokens
    key1 = torch.tensor([1, 2], dtype=torch.int64)
    val1 = torch.tensor([10, 11], dtype=torch.int64)
    _, node1 = cache.insert(key1, val1)
    cache.add_buffer_idx_to_node(node1, 100, is_hotspot=True)

    key2 = torch.tensor([3, 4], dtype=torch.int64)
    val2 = torch.tensor([20, 21], dtype=torch.int64)
    _, node2 = cache.insert(key2, val2)
    cache.add_buffer_idx_to_node(node2, 200, is_hotspot=False)

    # Non-hotspot should be evicted first (sorted before hotspot)
    first = cache.evict_buffer_set[0]
    assert first is node2, "Non-hotspot node should sort first for eviction"


def test_two_pass_eviction_skips_referenced():
    cache = _make_hybrid_cache("two_pass")

    # Create two nodes
    key1 = torch.tensor([1, 2], dtype=torch.int64)
    val1 = torch.tensor([10, 11], dtype=torch.int64)
    _, node1 = cache.insert(key1, val1)
    cache.add_buffer_idx_to_node(node1, 100)

    key2 = torch.tensor([3, 4], dtype=torch.int64)
    val2 = torch.tensor([20, 21], dtype=torch.int64)
    _, node2 = cache.insert(key2, val2)
    cache.add_buffer_idx_to_node(node2, 200)

    # Reference node1 (simulates an active request)
    cache.add_node_ref_counter(node1)

    evicted = []
    cache._evict_buffer(1, lambda buf: evicted.append(buf))

    # Should evict unreferenced node2 first, not referenced node1
    assert evicted == [200]
    assert node1.buffer_idx == 100, "Referenced node should keep its buffer"
    assert node2.buffer_idx is None


def test_two_pass_eviction_falls_back_to_referenced():
    cache = _make_hybrid_cache("fallback")

    key1 = torch.tensor([1, 2], dtype=torch.int64)
    val1 = torch.tensor([10, 11], dtype=torch.int64)
    _, node1 = cache.insert(key1, val1)
    cache.add_buffer_idx_to_node(node1, 100)
    cache.add_node_ref_counter(node1)

    # Only one node and it's referenced — must evict it as fallback
    evicted = []
    cache._evict_buffer(1, lambda buf: evicted.append(buf))
    assert evicted == [100]
    assert node1.buffer_idx is None


def test_node_stays_alive_after_buffer_eviction():
    cache = _make_hybrid_cache("alive")

    key1 = torch.tensor([1, 2], dtype=torch.int64)
    val1 = torch.tensor([10, 11], dtype=torch.int64)
    _, node1 = cache.insert(key1, val1)
    cache.add_buffer_idx_to_node(node1, 100)

    evicted = []
    cache._evict_buffer(1, lambda buf: evicted.append(buf))

    # Node should still exist in the tree (not destroyed)
    assert node1.parent is not None, "Node should stay in tree after buffer eviction"
    assert node1.buffer_idx is None
    # Node should be in evict_tree_set for later KV reclamation
    assert node1 in cache.evict_tree_set
