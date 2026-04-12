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


def test_evict_buffer_set_lru_sort_order():
    """Eviction should be purely LRU by buffer_time."""
    cache = _make_hybrid_cache("lru_sort")

    key1 = torch.tensor([1, 2], dtype=torch.int64)
    val1 = torch.tensor([10, 11], dtype=torch.int64)
    _, node1 = cache.insert(key1, val1)
    cache.add_buffer_idx_to_node(node1, 100)

    key2 = torch.tensor([3, 4], dtype=torch.int64)
    val2 = torch.tensor([20, 21], dtype=torch.int64)
    _, node2 = cache.insert(key2, val2)
    cache.add_buffer_idx_to_node(node2, 200)

    # node1 was inserted first, so it should be evicted first (lower buffer_time)
    first = cache.evict_buffer_set[0]
    assert first is node1, "Older buffer should sort first for LRU eviction"


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


def test_eviction_protects_newly_matched_node():
    cache = _make_hybrid_cache("protected")

    key1 = torch.tensor([1, 2], dtype=torch.int64)
    val1 = torch.tensor([10, 11], dtype=torch.int64)
    _, node1 = cache.insert(key1, val1)
    cache.add_buffer_idx_to_node(node1, 100)
    cache.add_node_ref_counter(node1)

    key2 = torch.tensor([3, 4], dtype=torch.int64)
    val2 = torch.tensor([20, 21], dtype=torch.int64)
    _, node2 = cache.insert(key2, val2)
    cache.add_buffer_idx_to_node(node2, 200)

    evicted = []
    cache._evict_buffer(1, lambda buf: evicted.append(buf), protected_nodes={node1})

    assert evicted == [200]
    assert node1.buffer_idx == 100
    assert node2.buffer_idx is None


def test_eviction_keeps_protected_node_even_when_only_candidate():
    cache = _make_hybrid_cache("protected_only")

    key1 = torch.tensor([1, 2], dtype=torch.int64)
    val1 = torch.tensor([10, 11], dtype=torch.int64)
    _, node1 = cache.insert(key1, val1)
    cache.add_buffer_idx_to_node(node1, 100)
    cache.add_node_ref_counter(node1)

    evicted = []
    cache._evict_buffer(1, lambda buf: evicted.append(buf), protected_nodes={node1})

    assert evicted == []
    assert node1.buffer_idx == 100


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


def test_match_prefix_walkback_does_not_destroy_nodes():
    """match_prefix walk-back must NOT destroy nodes — they must survive for prefix matching."""
    cache = _make_hybrid_cache("walkback")

    # Build a chain: root -> [1,2] -> [3,4]
    # Insert [1,2] first to create that intermediate node, then [1,2,3,4] to extend it
    cache.insert(torch.tensor([1, 2], dtype=torch.int64), torch.tensor([10, 11], dtype=torch.int64))
    key = torch.tensor([1, 2, 3, 4], dtype=torch.int64)
    val = torch.tensor([10, 11, 12, 13], dtype=torch.int64)
    cache.insert(key, val)

    # Only attach buffer to the [1,2] node (parent), not [3,4]
    node_12 = cache.root_node.children[1]
    cache.add_buffer_idx_to_node(node_12, 42)

    # match_prefix with [1,2,3,4] — should walk back from [3,4] to [1,2]
    share_node, kv_len, value_tensor, unbuffered_prefix_pos = cache.match_prefix(key, update_refs=True)

    assert share_node is node_12, "Should match at [1,2] which has buffer"
    assert kv_len == 2, "kv_len should be the matched node's prefix total len"
    assert unbuffered_prefix_pos == 2, "Missed 2 tokens in [3,4] node"

    # Critical: the [3,4] node must still exist in the tree
    assert 3 in node_12.children, "Child [3,4] must not be destroyed during walk-back"
    child = node_12.children[3]
    assert child.ref_counter == 0, "Walk-back should have decremented ref"

    cache.dec_node_ref_counter(share_node)


def test_insert_with_buffer_attaches_buffer_to_leaf():
    """insert_with_buffer should attach a buffer to the inserted leaf node."""
    cache = _make_hybrid_cache("iwb_leaf")

    buffer_calls = []

    def snapshot_fn(node):
        buffer_calls.append(node)
        cache.add_buffer_idx_to_node(node, 100 + len(buffer_calls))

    key = torch.tensor([1, 2, 3, 4], dtype=torch.int64)
    val = torch.tensor([10, 11, 12, 13], dtype=torch.int64)
    prefix_len, leaf_node = cache.insert_with_buffer(key, val, 0, snapshot_fn)

    assert leaf_node.buffer_idx is not None, "Leaf node should have a buffer"
    assert len(buffer_calls) == 1, "snapshot_fn should be called once (for leaf only)"


def test_insert_with_buffer_splits_at_boundary():
    """insert_with_buffer should ensure a split at unbuffered_prefix_pos and attach buffer there."""
    cache = _make_hybrid_cache("iwb_split")

    buffer_nodes = []

    def snapshot_fn(node):
        buffer_nodes.append(node)
        cache.add_buffer_idx_to_node(node, 200 + len(buffer_nodes))

    key = torch.tensor([1, 2, 3, 4, 5, 6], dtype=torch.int64)
    val = torch.tensor([10, 11, 12, 13, 14, 15], dtype=torch.int64)
    prefix_len, leaf_node = cache.insert_with_buffer(key, val, 3, snapshot_fn)

    # Should have called snapshot_fn twice: once for boundary node, once for leaf
    assert len(buffer_nodes) == 2

    # Check the boundary node at position 3
    boundary_node = buffer_nodes[0]
    assert boundary_node.node_prefix_total_len == 3
    assert boundary_node.buffer_idx is not None

    # Check the leaf node
    assert leaf_node.buffer_idx is not None
    assert leaf_node.node_prefix_total_len == 6


def test_insert_with_buffer_skips_existing_buffer():
    """insert_with_buffer should not call snapshot_fn for nodes that already have buffers."""
    cache = _make_hybrid_cache("iwb_skip")

    # Pre-insert the prefix with a buffer
    prefix_key = torch.tensor([1, 2, 3], dtype=torch.int64)
    prefix_val = torch.tensor([10, 11, 12], dtype=torch.int64)
    _, prefix_node = cache.insert(prefix_key, prefix_val)
    cache.add_buffer_idx_to_node(prefix_node, 42)

    buffer_calls = []

    def snapshot_fn(node):
        buffer_calls.append(node)
        cache.add_buffer_idx_to_node(node, 300 + len(buffer_calls))

    # Insert longer key with boundary at 3 (where buffer already exists)
    key = torch.tensor([1, 2, 3, 4, 5, 6], dtype=torch.int64)
    val = torch.tensor([10, 11, 12, 13, 14, 15], dtype=torch.int64)
    prefix_len, leaf_node = cache.insert_with_buffer(key, val, 3, snapshot_fn)

    # Only the leaf should get a snapshot call — boundary already has a buffer
    assert len(buffer_calls) == 1
    assert buffer_calls[0] is leaf_node


def test_no_adaptive_threshold_fields():
    """HybridRadixCache should not have adaptive threshold or hotspot tracking fields."""
    cache = _make_hybrid_cache("no_adaptive")
    removed_attrs = [
        "min_insert_threshold",
        "MIN_THRESHOLD",
        "MAX_THRESHOLD",
        "adjust_interval",
        "buffer_insert_count",
        "buffer_hit_count",
        "buffer_waste_count",
    ]
    for attr in removed_attrs:
        assert not hasattr(cache, attr), f"{attr} should be removed"


def test_match_prefix_returns_unbuffered_prefix_pos():
    """match_prefix should return (node, kv_len, value, unbuffered_prefix_pos)."""
    cache = _make_hybrid_cache("return_pos")

    # Build chain: root -> [1,2] -> [3,4]
    cache.insert(torch.tensor([1, 2], dtype=torch.int64), torch.tensor([10, 11], dtype=torch.int64))
    cache.insert(torch.tensor([1, 2, 3, 4], dtype=torch.int64), torch.tensor([10, 11, 12, 13], dtype=torch.int64))

    # Attach buffer only to [1,2] node
    node_12 = cache.root_node.children[1]
    cache.add_buffer_idx_to_node(node_12, 42)

    # Match [1,2,3,4] — walks back from [3,4] to [1,2]
    result = cache.match_prefix(torch.tensor([1, 2, 3, 4], dtype=torch.int64), update_refs=True)
    assert len(result) == 4, "match_prefix should return 4 values"
    share_node, kv_len, value_tensor, unbuffered_prefix_pos = result
    assert share_node is node_12
    assert kv_len == 2
    assert unbuffered_prefix_pos == 2

    cache.dec_node_ref_counter(share_node)


def test_match_prefix_no_walkback_returns_zero():
    """When matched node has a buffer, unbuffered_prefix_pos should be 0."""
    cache = _make_hybrid_cache("no_walkback")

    cache.insert(torch.tensor([1, 2, 3], dtype=torch.int64), torch.tensor([10, 11, 12], dtype=torch.int64))
    node = cache.root_node.children[1]
    cache.add_buffer_idx_to_node(node, 42)

    share_node, kv_len, value_tensor, unbuffered_prefix_pos = cache.match_prefix(
        torch.tensor([1, 2, 3], dtype=torch.int64), update_refs=True
    )
    assert share_node is node
    assert unbuffered_prefix_pos == 0

    cache.dec_node_ref_counter(share_node)


def test_match_prefix_no_match_returns_zero():
    """When nothing matches, unbuffered_prefix_pos should be 0."""
    cache = _make_hybrid_cache("no_match")

    result = cache.match_prefix(torch.tensor([1, 2, 3], dtype=torch.int64), update_refs=False)
    node, kv_len, value_tensor, unbuffered_prefix_pos = result
    assert node is None
    assert unbuffered_prefix_pos == 0
