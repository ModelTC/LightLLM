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


def test_adaptive_threshold_initial_state():
    cache = _make_hybrid_cache("init_state")
    assert cache.min_insert_threshold == 1024
    assert cache.MIN_THRESHOLD == 256
    assert cache.MAX_THRESHOLD == 16384
    assert cache.adjust_interval == 100
    assert cache.buffer_insert_count == 0
    assert cache.buffer_hit_count == 0
    assert cache.buffer_waste_count == 0


def test_evict_buffer_set_hotspot_sort_order():
    cache = _make_hybrid_cache("hotspot_sort")

    # Insert two nodes with tokens
    key1 = torch.tensor([10, 11, 12], dtype=torch.int64, device="cpu")
    _, node1 = cache.insert(key1)
    cache.add_buffer_idx_to_node(node1, 0, is_hotspot=True)

    key2 = torch.tensor([20, 21, 22], dtype=torch.int64, device="cpu")
    _, node2 = cache.insert(key2)
    cache.add_buffer_idx_to_node(node2, 1, is_hotspot=False)

    # Non-hotspot should sort first (lower sort key) => evicted first
    first = cache.evict_buffer_set[0]
    assert first.is_hotspot is False
    assert first is node2


def test_add_buffer_idx_to_node_tracks_insert_count():
    cache = _make_hybrid_cache("insert_count")

    key1 = torch.tensor([10, 11], dtype=torch.int64, device="cpu")
    _, node1 = cache.insert(key1)

    key2 = torch.tensor([20, 21], dtype=torch.int64, device="cpu")
    _, node2 = cache.insert(key2)

    cache.add_buffer_idx_to_node(node1, 0, is_hotspot=True)
    assert cache.buffer_insert_count == 1

    cache.add_buffer_idx_to_node(node2, 1, is_hotspot=False)
    assert cache.buffer_insert_count == 1  # not incremented for non-hotspot


def test_match_prefix_returns_miss_prefix_len():
    cache = _make_hybrid_cache("miss_len")

    # Insert 5 tokens as a chain
    key = torch.tensor([1, 2, 3, 4, 5], dtype=torch.int64, device="cpu")
    _, node = cache.insert(key)

    # Split at depth 3 by inserting a diverging path
    key2 = torch.tensor([1, 2, 3, 6, 7], dtype=torch.int64, device="cpu")
    _, node2 = cache.insert(key2)

    # Add buffer at depth 3 node (the shared prefix node [1,2,3])
    # Find the node at depth 3
    depth3_node = cache.root_node.children[1]
    # After split, depth3_node should have prefix_total_len == 3
    assert depth3_node.node_prefix_total_len == 3
    cache.add_buffer_idx_to_node(depth3_node, 42, is_hotspot=True)

    # Match with key [1,2,3,4,5] -- walks to depth 5, buffer at depth 3
    # miss_prefix_len should be 2 (tokens 4,5 walked back)
    result_node, miss_prefix_len, value = cache.match_prefix(
        torch.tensor([1, 2, 3, 4, 5], dtype=torch.int64, device="cpu"), update_refs=False
    )
    assert result_node is depth3_node
    assert miss_prefix_len == 2


def test_match_prefix_r3_destroys_orphaned_leaves():
    cache = _make_hybrid_cache("r3_orphan")

    # Insert tokens
    key = torch.tensor([1, 2, 3], dtype=torch.int64, device="cpu")
    _, node = cache.insert(key)
    assert cache.tree_total_tokens_num.arr[0] == 3

    # Match with update_refs=True, no buffer anywhere => walks back to root
    # All leaves with ref_counter==0 should be destroyed
    result_node, miss_prefix_len, value = cache.match_prefix(
        torch.tensor([1, 2, 3], dtype=torch.int64, device="cpu"), update_refs=True
    )
    assert result_node is None
    assert cache.tree_total_tokens_num.arr[0] == 0
    # mem_manager.free should have been called
    cache.mem_manager.free.assert_called_once()


def test_match_prefix_marks_buffer_as_hit():
    cache = _make_hybrid_cache("hit_mark")

    key = torch.tensor([1, 2, 3], dtype=torch.int64, device="cpu")
    _, node = cache.insert(key)
    cache.add_buffer_idx_to_node(node, 10, is_hotspot=True)
    assert node.was_hit is False

    # Match with update_refs=True => buffer node found, was_hit set to True
    result_node, _, _ = cache.match_prefix(torch.tensor([1, 2, 3], dtype=torch.int64, device="cpu"), update_refs=True)
    assert result_node is node
    assert node.was_hit is True


def test_evict_buffer_r4_destroys_orphaned_leaf():
    cache = _make_hybrid_cache("r4_orphan")

    key = torch.tensor([1, 2, 3], dtype=torch.int64, device="cpu")
    _, node = cache.insert(key)
    cache.add_buffer_idx_to_node(node, 5, is_hotspot=False)
    assert node.ref_counter == 0
    assert node.is_leaf()
    initial_tokens = cache.tree_total_tokens_num.arr[0]

    released_buffers = []
    released_kv = []
    cache._evict_buffer(1, lambda b: released_buffers.append(b), lambda t: released_kv.append(t))

    assert len(released_buffers) == 1
    assert released_buffers[0] == 5
    # R4: node should be destroyed, KV freed
    assert len(released_kv) == 1
    assert cache.tree_total_tokens_num.arr[0] == initial_tokens - 3


def test_evict_buffer_tracks_waste():
    cache = _make_hybrid_cache("waste_track")

    key = torch.tensor([1, 2], dtype=torch.int64, device="cpu")
    _, node = cache.insert(key)
    cache.add_buffer_idx_to_node(node, 5, is_hotspot=True)
    # was_hit is False by default => waste

    cache._evict_buffer(1, lambda b: None, lambda t: None)
    assert cache.buffer_waste_count == 1
    assert cache.buffer_hit_count == 0


def test_evict_buffer_tracks_hit():
    cache = _make_hybrid_cache("hit_track")

    key = torch.tensor([1, 2], dtype=torch.int64, device="cpu")
    _, node = cache.insert(key)
    cache.add_buffer_idx_to_node(node, 5, is_hotspot=True)
    node.was_hit = True

    cache._evict_buffer(1, lambda b: None, lambda t: None)
    assert cache.buffer_hit_count == 1
    assert cache.buffer_waste_count == 0


def test_adaptive_threshold_increases_on_high_waste():
    cache = _make_hybrid_cache("thresh_up")
    cache.adjust_interval = 5
    initial_threshold = cache.min_insert_threshold

    # Simulate high waste: 4 waste, 1 insert => waste_ratio = 4/4 = 1.0 > 0.5
    cache.buffer_insert_count = 1
    cache.buffer_waste_count = 4
    cache.buffer_hit_count = 0

    cache._maybe_adjust_threshold()
    assert cache.min_insert_threshold == initial_threshold * 2
    # Counters should be reset
    assert cache.buffer_insert_count == 0
    assert cache.buffer_waste_count == 0
    assert cache.buffer_hit_count == 0


def test_adaptive_threshold_decreases_on_low_waste():
    cache = _make_hybrid_cache("thresh_down")
    cache.adjust_interval = 5
    initial_threshold = cache.min_insert_threshold

    # Simulate low waste with free pool > 30%
    cache.buffer_insert_count = 1
    cache.buffer_waste_count = 0
    cache.buffer_hit_count = 4
    # Ensure can_use_mem_size > size * 0.3
    cache.buffer_mem_manager.can_use_mem_size = 40
    cache.buffer_mem_manager.size = 50

    cache._maybe_adjust_threshold()
    assert cache.min_insert_threshold == initial_threshold // 2
    # Counters should be reset
    assert cache.buffer_insert_count == 0


def test_full_hotspot_lifecycle():
    """End-to-end: insert -> match (miss) -> hotspot insert -> match (hit) -> evict."""
    cache = _make_hybrid_cache("lifecycle", kv_size=200, buf_size=10)

    # 1. Create tree structure (no buffer)
    key = torch.tensor([1, 2, 3, 4, 5, 6, 7, 8], dtype=torch.int64, device="cpu")
    val = torch.tensor([10, 11, 12, 13, 14, 15, 16, 17], dtype=torch.int64, device="cpu")
    cache.insert(key, val)

    # 2. Match walks back to root, R3 cleans up orphaned leaves
    node2, miss, val2 = cache.match_prefix(
        torch.tensor([1, 2, 3, 4, 5, 6, 7, 8], dtype=torch.int64, device="cpu"),
        update_refs=True,
    )
    assert node2 is None
    assert miss == 8
    assert cache.get_tree_total_tokens_num() == 0

    # 3. Hotspot insert: re-insert tree + add buffer
    key2 = torch.tensor([1, 2, 3, 4, 5, 6, 7, 8], dtype=torch.int64, device="cpu")
    val2 = torch.tensor([20, 21, 22, 23, 24, 25, 26, 27], dtype=torch.int64, device="cpu")
    _, hot_node = cache.insert(key2, val2)
    cache.add_buffer_idx_to_node(hot_node, 0, is_hotspot=True)
    assert cache.buffer_insert_count == 1

    # 4. Match finds buffer, no miss
    node3, miss3, val3 = cache.match_prefix(
        torch.tensor([1, 2, 3, 4, 5, 6, 7, 8], dtype=torch.int64, device="cpu"),
        update_refs=True,
    )
    assert node3 is hot_node
    assert miss3 == 0
    assert len(val3) == 8
    assert hot_node.was_hit is True

    # 5. Evict — tracked as hit
    release_bufs = []
    release_mems = []
    cache._evict_buffer(1, lambda x: release_bufs.append(x), lambda m: release_mems.append(m))
    assert cache.buffer_hit_count == 1
    assert cache.buffer_waste_count == 0
