# Hybrid Radix Cache Hit Rate Restoration — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Restore hybrid radix cache hit rate by reintegrating walk-back cleanup (R3), cascading eviction cleanup (R4), hotspot-only buffer insertion with adaptive threshold, and hotspot eviction resistance.

**Architecture:** Four layers of change: (1) `TreeNode` gets `is_hotspot`/`was_hit` fields, (2) `HybridRadixCache` gets rewritten `match_prefix`, `_evict_buffer`, adaptive threshold, and hotspot-aware eviction, (3) `InferReq`/`InferenceContext` get hotspot detection, buffer snapshot, and deferred-free logic, (4) `ChunkedPrefillBackend` integrates the snapshot call between Stage 3 and Stage 4.

**Tech Stack:** Python, PyTorch, Triton (via existing `copy_state_buffers`), sortedcontainers

---

### Task 1: Add `is_hotspot` and `was_hit` fields to `TreeNode`

**Files:**
- Modify: `lightllm/server/router/dynamic_prompt/radix_cache.py:22-38`
- Test: `unit_tests/server/router/dynamic_prompt/test_radix_cache.py`

- [ ] **Step 1: Write the failing test**

Add a new test to `unit_tests/server/router/dynamic_prompt/test_radix_cache.py`:

```python
def test_tree_node_hotspot_fields():
    """TreeNode should have is_hotspot and was_hit fields, defaulting to False."""
    from lightllm.server.router.dynamic_prompt.radix_cache import TreeNode
    node = TreeNode()
    assert node.is_hotspot is False
    assert node.was_hit is False
    node.is_hotspot = True
    node.was_hit = True
    assert node.is_hotspot is True
    assert node.was_hit is True
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd /workspace/LightLLM/worktree/qw35_oom && python -m pytest unit_tests/server/router/dynamic_prompt/test_radix_cache.py::test_tree_node_hotspot_fields -v`
Expected: FAIL with `AttributeError: 'TreeNode' object has no attribute 'is_hotspot'`

- [ ] **Step 3: Add fields to `TreeNode.__init__`**

In `lightllm/server/router/dynamic_prompt/radix_cache.py`, add two fields after `self.buffer_time` (line 38):

```python
        self.is_hotspot = False
        self.was_hit = False
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd /workspace/LightLLM/worktree/qw35_oom && python -m pytest unit_tests/server/router/dynamic_prompt/test_radix_cache.py::test_tree_node_hotspot_fields -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add lightllm/server/router/dynamic_prompt/radix_cache.py unit_tests/server/router/dynamic_prompt/test_radix_cache.py
git commit -m "feat: add is_hotspot and was_hit fields to TreeNode"
```

---

### Task 2: Rewrite `HybridRadixCache` — adaptive threshold, hotspot sort key, `add_buffer_idx_to_node`

**Files:**
- Modify: `lightllm/server/router/dynamic_prompt/hybrid_radix_cache.py:13-71`
- Test: `unit_tests/server/router/dynamic_prompt/test_hybrid_radix_cache.py` (create)

- [ ] **Step 1: Write the failing test**

Create `unit_tests/server/router/dynamic_prompt/test_hybrid_radix_cache.py`:

```python
import pytest
import torch
import numpy as np
from unittest.mock import MagicMock


def _make_mock_kv_mem_manager(size=1000):
    """Create a mock KV cache memory manager."""
    mgr = MagicMock()
    mgr.can_use_mem_size = size
    mgr.free = MagicMock()
    return mgr


def _make_mock_buffer_mem_manager(size=50):
    """Create a mock MambaCacheManager."""
    mgr = MagicMock()
    mgr.size = size
    mgr.can_use_mem_size = size
    mgr.free = MagicMock()
    return mgr


def _make_hybrid_cache(unique_suffix, kv_size=1000, buf_size=50):
    """Create a HybridRadixCache with mock mem managers."""
    from lightllm.server.router.dynamic_prompt.hybrid_radix_cache import HybridRadixCache

    kv_mgr = _make_mock_kv_mem_manager(kv_size)
    kv_mgr.mamba_cache_mem_manager = _make_mock_buffer_mem_manager(buf_size)
    cache = HybridRadixCache(f"test_{unique_suffix}", kv_size, 0, kv_mgr)
    return cache


def test_adaptive_threshold_initial_state():
    """HybridRadixCache should initialize with adaptive threshold state."""
    cache = _make_hybrid_cache("adaptive_init")
    assert cache.min_insert_threshold == 1024
    assert cache.MIN_THRESHOLD == 256
    assert cache.MAX_THRESHOLD == 16384
    assert cache.buffer_insert_count == 0
    assert cache.buffer_hit_count == 0
    assert cache.buffer_waste_count == 0


def test_evict_buffer_set_hotspot_sort_order():
    """Hotspot buffers should sort after non-hotspot buffers in evict_buffer_set."""
    cache = _make_hybrid_cache("sort_order")

    # Insert two nodes into the tree
    key1 = torch.tensor([1, 2, 3], dtype=torch.int64, device="cpu")
    val1 = torch.tensor([10, 11, 12], dtype=torch.int64, device="cpu")
    _, node1 = cache.insert(key1, val1)

    key2 = torch.tensor([4, 5, 6], dtype=torch.int64, device="cpu")
    val2 = torch.tensor([13, 14, 15], dtype=torch.int64, device="cpu")
    _, node2 = cache.insert(key2, val2)

    # Add buffers: node1 is hotspot, node2 is not
    cache.add_buffer_idx_to_node(node1, 0, is_hotspot=True)
    cache.add_buffer_idx_to_node(node2, 1, is_hotspot=False)

    # Non-hotspot (node2) should be evicted first (index 0 in sorted set)
    first = cache.evict_buffer_set[0]
    assert first.is_hotspot is False
    assert first is node2


def test_add_buffer_idx_to_node_tracks_insert_count():
    """add_buffer_idx_to_node with is_hotspot=True should increment buffer_insert_count."""
    cache = _make_hybrid_cache("insert_count")
    key = torch.tensor([1, 2], dtype=torch.int64, device="cpu")
    val = torch.tensor([10, 11], dtype=torch.int64, device="cpu")
    _, node = cache.insert(key, val)

    assert cache.buffer_insert_count == 0
    cache.add_buffer_idx_to_node(node, 0, is_hotspot=True)
    assert cache.buffer_insert_count == 1
    assert node.was_hit is False
    assert node.is_hotspot is True

    # Non-hotspot insertion should NOT increment counter
    cache.add_buffer_idx_to_node(node, 1, is_hotspot=False)
    assert cache.buffer_insert_count == 1
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd /workspace/LightLLM/worktree/qw35_oom && python -m pytest unit_tests/server/router/dynamic_prompt/test_hybrid_radix_cache.py -v`
Expected: FAIL (attributes don't exist yet)

- [ ] **Step 3: Implement adaptive threshold state, hotspot sort key, and `add_buffer_idx_to_node`**

In `lightllm/server/router/dynamic_prompt/hybrid_radix_cache.py`, replace the `__init__` and `add_buffer_idx_to_node` methods:

```python
class HybridRadixCache(RadixCache):
    def __init__(self, unique_name, total_token_num, rank_in_node, kv_cache_mem_manager):
        super().__init__(unique_name, total_token_num, rank_in_node, kv_cache_mem_manager)
        assert hasattr(kv_cache_mem_manager, "mamba_cache_mem_manager")
        self.buffer_mem_manager: MambaCacheManager = kv_cache_mem_manager.mamba_cache_mem_manager
        # Hotspot-aware eviction: non-hotspot (False) sorts before hotspot (True)
        self.evict_buffer_set: Set[TreeNode] = SortedSet(key=lambda x: (x.is_hotspot, x.buffer_time))

        # Adaptive insertion threshold
        self.min_insert_threshold = 1024
        self.MIN_THRESHOLD = 256
        self.MAX_THRESHOLD = 16384
        self.adjust_interval = 100

        # Sliding window counters for threshold adaptation
        self.buffer_insert_count = 0
        self.buffer_hit_count = 0
        self.buffer_waste_count = 0
```

Replace `add_buffer_idx_to_node`:

```python
    def add_buffer_idx_to_node(self, node: TreeNode, buffer_idx: int, is_hotspot: bool = False):
        """Set buffer_idx for a node and add it to evict_buffer_set."""
        self.evict_buffer_set.discard(node)
        if node.is_leaf():
            self.evict_tree_set.discard(node)
        if node.buffer_idx is not None:
            self.buffer_mem_manager.free([node.buffer_idx])
        node.buffer_idx = buffer_idx
        node.is_hotspot = is_hotspot
        node.was_hit = False
        node.update_buffer_time()
        self.evict_buffer_set.add(node)
        if node.is_leaf():
            self.evict_tree_set.add(node)
        if is_hotspot:
            self.buffer_insert_count += 1
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd /workspace/LightLLM/worktree/qw35_oom && python -m pytest unit_tests/server/router/dynamic_prompt/test_hybrid_radix_cache.py -v`
Expected: PASS

- [ ] **Step 5: Run existing tests to ensure no regressions**

Run: `cd /workspace/LightLLM/worktree/qw35_oom && python -m pytest unit_tests/server/router/dynamic_prompt/test_radix_cache.py -v`
Expected: PASS

- [ ] **Step 6: Commit**

```bash
git add lightllm/server/router/dynamic_prompt/hybrid_radix_cache.py unit_tests/server/router/dynamic_prompt/test_hybrid_radix_cache.py
git commit -m "feat: add adaptive threshold state and hotspot-aware eviction to HybridRadixCache"
```

---

### Task 3: Rewrite `match_prefix` — return `miss_prefix_len` + R3 walk-back cleanup + hit tracking

**Files:**
- Modify: `lightllm/server/router/dynamic_prompt/hybrid_radix_cache.py:20-57`
- Test: `unit_tests/server/router/dynamic_prompt/test_hybrid_radix_cache.py`

- [ ] **Step 1: Write the failing tests**

Append to `unit_tests/server/router/dynamic_prompt/test_hybrid_radix_cache.py`:

```python
def test_match_prefix_returns_miss_prefix_len():
    """match_prefix should return miss_prefix_len (gap between deepest match and buffer)."""
    cache = _make_hybrid_cache("miss_prefix")

    # Build tree: [1,2,3,4,5] with buffer at node for [1,2,3]
    key_full = torch.tensor([1, 2, 3, 4, 5], dtype=torch.int64, device="cpu")
    val_full = torch.tensor([10, 11, 12, 13, 14], dtype=torch.int64, device="cpu")
    _, leaf_node = cache.insert(key_full, val_full)

    # Split at [1,2,3] by inserting a shorter prefix, then add buffer to the split parent
    key_short = torch.tensor([1, 2, 3, 7], dtype=torch.int64, device="cpu")
    val_short = torch.tensor([10, 11, 12, 20], dtype=torch.int64, device="cpu")
    _, branch_node = cache.insert(key_short, val_short)

    # The split parent at [1,2,3] — find it by walking from branch_node
    parent_node = branch_node.parent
    assert parent_node.node_prefix_total_len == 3
    cache.add_buffer_idx_to_node(parent_node, 42, is_hotspot=True)

    # Match [1,2,3,4,5]: deepest match is at depth 5, buffer at depth 3, miss = 2
    node, miss_prefix_len, value = cache.match_prefix(
        torch.tensor([1, 2, 3, 4, 5], dtype=torch.int64, device="cpu"), update_refs=True
    )
    assert node is parent_node
    assert node.node_prefix_total_len == 3
    assert miss_prefix_len == 2
    assert len(value) == 3


def test_match_prefix_r3_destroys_orphaned_leaves():
    """Walk-back should destroy buffer-less unreferenced leaves (R3)."""
    cache = _make_hybrid_cache("r3_cleanup", kv_size=100)

    # Insert [1,2,3,4,5] — creates a single leaf node
    key = torch.tensor([1, 2, 3, 4, 5], dtype=torch.int64, device="cpu")
    val = torch.tensor([10, 11, 12, 13, 14], dtype=torch.int64, device="cpu")
    cache.insert(key, val)
    assert cache.get_tree_total_tokens_num() == 5

    # Match with update_refs=True: no buffer anywhere → walks back to root
    # R3 should destroy the leaf and free its 5 tokens
    node, miss_prefix_len, value = cache.match_prefix(
        torch.tensor([1, 2, 3, 4, 5], dtype=torch.int64, device="cpu"), update_refs=True
    )
    assert node is None
    assert miss_prefix_len == 5
    assert value is None
    # The orphaned leaf should have been destroyed
    assert cache.get_tree_total_tokens_num() == 0


def test_match_prefix_marks_buffer_as_hit():
    """match_prefix with update_refs=True should mark the found buffer as hit."""
    cache = _make_hybrid_cache("hit_mark")

    key = torch.tensor([1, 2, 3], dtype=torch.int64, device="cpu")
    val = torch.tensor([10, 11, 12], dtype=torch.int64, device="cpu")
    _, node = cache.insert(key, val)
    cache.add_buffer_idx_to_node(node, 42)
    assert node.was_hit is False

    cache.match_prefix(
        torch.tensor([1, 2, 3], dtype=torch.int64, device="cpu"), update_refs=True
    )
    assert node.was_hit is True
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd /workspace/LightLLM/worktree/qw35_oom && python -m pytest unit_tests/server/router/dynamic_prompt/test_hybrid_radix_cache.py::test_match_prefix_returns_miss_prefix_len unit_tests/server/router/dynamic_prompt/test_hybrid_radix_cache.py::test_match_prefix_r3_destroys_orphaned_leaves unit_tests/server/router/dynamic_prompt/test_hybrid_radix_cache.py::test_match_prefix_marks_buffer_as_hit -v`
Expected: FAIL

- [ ] **Step 3: Rewrite `match_prefix`**

Replace the entire `match_prefix` method in `lightllm/server/router/dynamic_prompt/hybrid_radix_cache.py` with the implementation from the design spec (Section 2). The full code is in the spec — copy it exactly.

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd /workspace/LightLLM/worktree/qw35_oom && python -m pytest unit_tests/server/router/dynamic_prompt/test_hybrid_radix_cache.py -v`
Expected: PASS

- [ ] **Step 5: Run existing radix cache tests**

Run: `cd /workspace/LightLLM/worktree/qw35_oom && python -m pytest unit_tests/server/router/dynamic_prompt/test_radix_cache.py -v`
Expected: PASS

- [ ] **Step 6: Commit**

```bash
git add lightllm/server/router/dynamic_prompt/hybrid_radix_cache.py unit_tests/server/router/dynamic_prompt/test_hybrid_radix_cache.py
git commit -m "feat: rewrite match_prefix with miss_prefix_len return, R3 cleanup, and hit tracking"
```

---

### Task 4: Rewrite `_evict_buffer` and `free_radix_cache_to_get_enough_buffer` — R4 cascading cleanup + adaptive threshold

**Files:**
- Modify: `lightllm/server/router/dynamic_prompt/hybrid_radix_cache.py:73-94`
- Test: `unit_tests/server/router/dynamic_prompt/test_hybrid_radix_cache.py`

- [ ] **Step 1: Write the failing tests**

Append to `unit_tests/server/router/dynamic_prompt/test_hybrid_radix_cache.py`:

```python
def test_evict_buffer_r4_destroys_orphaned_leaf():
    """Evicting a buffer from an unreferenced leaf should destroy the node and free KV tokens (R4)."""
    cache = _make_hybrid_cache("r4_cleanup", kv_size=100)

    key = torch.tensor([1, 2, 3], dtype=torch.int64, device="cpu")
    val = torch.tensor([10, 11, 12], dtype=torch.int64, device="cpu")
    _, node = cache.insert(key, val)
    cache.add_buffer_idx_to_node(node, 42)
    assert cache.get_tree_total_tokens_num() == 3

    # Evict the buffer — node is unreferenced leaf → R4 should destroy it
    release_buffers = []
    release_mems = []
    cache._evict_buffer(1, lambda idx: release_buffers.append(idx), lambda mem: release_mems.append(mem))
    assert release_buffers == [42]
    assert len(release_mems) == 1  # KV tokens freed
    assert cache.get_tree_total_tokens_num() == 0


def test_evict_buffer_tracks_waste():
    """Evicting a buffer that was never hit should increment buffer_waste_count."""
    cache = _make_hybrid_cache("waste_track")

    key = torch.tensor([1, 2], dtype=torch.int64, device="cpu")
    val = torch.tensor([10, 11], dtype=torch.int64, device="cpu")
    _, node = cache.insert(key, val)
    cache.add_buffer_idx_to_node(node, 42, is_hotspot=True)
    assert node.was_hit is False

    release_bufs = []
    release_mems = []
    cache._evict_buffer(1, lambda x: release_bufs.append(x), lambda m: release_mems.append(m))
    assert cache.buffer_waste_count == 1
    assert cache.buffer_hit_count == 0


def test_evict_buffer_tracks_hit():
    """Evicting a buffer that WAS hit should increment buffer_hit_count."""
    cache = _make_hybrid_cache("hit_track")

    key = torch.tensor([1, 2], dtype=torch.int64, device="cpu")
    val = torch.tensor([10, 11], dtype=torch.int64, device="cpu")
    _, node = cache.insert(key, val)
    cache.add_buffer_idx_to_node(node, 42, is_hotspot=True)
    node.was_hit = True

    release_bufs = []
    release_mems = []
    cache._evict_buffer(1, lambda x: release_bufs.append(x), lambda m: release_mems.append(m))
    assert cache.buffer_hit_count == 1
    assert cache.buffer_waste_count == 0


def test_adaptive_threshold_increases_on_high_waste():
    """Threshold should double when waste_ratio > 0.5."""
    cache = _make_hybrid_cache("threshold_up")
    cache.adjust_interval = 5
    cache.min_insert_threshold = 1024

    # Simulate: 4 wasted, 1 hit → waste_ratio = 0.8
    cache.buffer_waste_count = 4
    cache.buffer_hit_count = 1
    cache.buffer_insert_count = 0  # total_events = 5 >= adjust_interval
    cache._maybe_adjust_threshold()
    assert cache.min_insert_threshold == 2048


def test_adaptive_threshold_decreases_on_low_waste():
    """Threshold should halve when waste_ratio < 0.1 and pool has >30% free."""
    cache = _make_hybrid_cache("threshold_down", buf_size=100)
    cache.adjust_interval = 5
    cache.min_insert_threshold = 1024

    # Simulate: 0 wasted, 5 hit → waste_ratio = 0.0, pool is 100% free
    cache.buffer_waste_count = 0
    cache.buffer_hit_count = 5
    cache.buffer_insert_count = 0
    cache._maybe_adjust_threshold()
    assert cache.min_insert_threshold == 512
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd /workspace/LightLLM/worktree/qw35_oom && python -m pytest unit_tests/server/router/dynamic_prompt/test_hybrid_radix_cache.py::test_evict_buffer_r4_destroys_orphaned_leaf unit_tests/server/router/dynamic_prompt/test_hybrid_radix_cache.py::test_adaptive_threshold_increases_on_high_waste -v`
Expected: FAIL

- [ ] **Step 3: Implement `_evict_buffer`, `free_radix_cache_to_get_enough_buffer`, `_maybe_adjust_threshold`, `_reset_counters`**

Replace `_evict_buffer` and `free_radix_cache_to_get_enough_buffer` in `lightllm/server/router/dynamic_prompt/hybrid_radix_cache.py` with the code from the design spec (Sections 3). Add `_maybe_adjust_threshold` and `_reset_counters` as new methods.

The full `_evict_buffer`:
```python
    def _evict_buffer(self, need_evict_buffer_num, evict_buffer_callback, evict_token_callback):
        while need_evict_buffer_num > 0:
            node = self.evict_buffer_set.pop(0)
            assert node.buffer_idx is not None

            if node.was_hit:
                self.buffer_hit_count += 1
            else:
                self.buffer_waste_count += 1

            evict_buffer_callback(node.buffer_idx)
            node.buffer_idx = None
            node.is_hotspot = False
            node.was_hit = False
            need_evict_buffer_num -= 1

            if node.is_leaf() and node.ref_counter == 0:
                self.evict_tree_set.discard(node)
                evict_token_callback(node.token_mem_index_value)
                self.tree_total_tokens_num.arr[0] -= len(node.token_mem_index_value)
                parent_node = node.parent
                parent_node.remove_child(node)
                if parent_node.is_leaf():
                    self.evict_tree_set.add(parent_node)

        self._maybe_adjust_threshold()
```

The full `free_radix_cache_to_get_enough_buffer`:
```python
    def free_radix_cache_to_get_enough_buffer(self, need_buffer_num):
        if need_buffer_num > self.buffer_mem_manager.can_use_mem_size:
            need_evict_buffer_num = need_buffer_num - self.buffer_mem_manager.can_use_mem_size
            release_buffers = []
            release_mems = []

            self._evict_buffer(
                need_evict_buffer_num,
                lambda idx: release_buffers.append(idx),
                lambda mem: release_mems.append(mem),
            )
            if len(release_buffers) > 0:
                self.buffer_mem_manager.free(release_buffers)
            if len(release_mems) > 0:
                self.mem_manager.free(torch.concat(release_mems))
```

The full `_maybe_adjust_threshold` and `_reset_counters`:
```python
    def _maybe_adjust_threshold(self):
        total_events = self.buffer_insert_count + self.buffer_waste_count + self.buffer_hit_count
        if total_events < self.adjust_interval:
            return

        total_resolved = self.buffer_hit_count + self.buffer_waste_count
        if total_resolved == 0:
            self._reset_counters()
            return

        waste_ratio = self.buffer_waste_count / total_resolved

        if waste_ratio > 0.5:
            self.min_insert_threshold = min(self.min_insert_threshold * 2, self.MAX_THRESHOLD)
        elif waste_ratio < 0.1 and self.buffer_mem_manager.can_use_mem_size > self.buffer_mem_manager.size * 0.3:
            self.min_insert_threshold = max(self.min_insert_threshold // 2, self.MIN_THRESHOLD)

        self._reset_counters()

    def _reset_counters(self):
        self.buffer_insert_count = 0
        self.buffer_hit_count = 0
        self.buffer_waste_count = 0
```

- [ ] **Step 4: Run all hybrid cache tests**

Run: `cd /workspace/LightLLM/worktree/qw35_oom && python -m pytest unit_tests/server/router/dynamic_prompt/test_hybrid_radix_cache.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add lightllm/server/router/dynamic_prompt/hybrid_radix_cache.py unit_tests/server/router/dynamic_prompt/test_hybrid_radix_cache.py
git commit -m "feat: rewrite _evict_buffer with R4 cascading cleanup and adaptive threshold"
```

---

### Task 5: Update `evict` method for dual-callback consistency

**Files:**
- Modify: `lightllm/server/router/dynamic_prompt/hybrid_radix_cache.py:119-143`

The existing `evict` method (called by `free_radix_cache_to_get_enough_token`) already uses dual callbacks. Verify it still works correctly — no code changes expected, just validation.

- [ ] **Step 1: Run all tests**

Run: `cd /workspace/LightLLM/worktree/qw35_oom && python -m pytest unit_tests/server/router/dynamic_prompt/ -v`
Expected: PASS

- [ ] **Step 2: Commit (if any fix was needed)**

Only commit if changes were required.

---

### Task 6: Add `InferReq` fields and hotspot detection in `_match_radix_cache`

**Files:**
- Modify: `lightllm/server/router/model_infer/infer_batch.py:430-498` and `543-551`

- [ ] **Step 1: Add fields to `InferReq.__init__`**

In `lightllm/server/router/model_infer/infer_batch.py`, after `self.finish_status = FinishStatus()` (line 478), add:

```python
        # Hybrid radix cache hotspot detection
        self.mamba_buffer_insert_len = 0
        self.is_hotspot_prefill = False
        self.extra_need_to_free_token_index = []
```

- [ ] **Step 2: Update `_match_radix_cache` for hotspot detection**

Replace the `_match_radix_cache` method (lines 481-498) with:

```python
    def _match_radix_cache(self):
        if self.sampling_param.disable_prompt_cache:
            return
        if g_infer_context.radix_cache is not None and self.get_cur_total_len() > 1 and self.cur_kv_len == 0:
            input_token_ids = self.shm_req.shm_prompt_ids.arr[0 : self.get_cur_total_len()]
            key = torch.tensor(input_token_ids, dtype=torch.int64, device="cpu")
            key = key[0 : len(key) - 1]
            share_node, miss_prefix_len, value_tensor = g_infer_context.radix_cache.match_prefix(
                key, update_refs=True
            )
            if share_node is not None:
                self.shared_kv_node = share_node
                ready_cache_len = share_node.node_prefix_total_len
                g_infer_context.req_manager.req_to_token_indexs[self.req_idx, 0:ready_cache_len] = value_tensor
                self.cur_kv_len = int(ready_cache_len)
                self.shm_req.prompt_cache_len = self.cur_kv_len

                if g_infer_context.has_recurrent_state:
                    threshold = g_infer_context.radix_cache.min_insert_threshold
                    if miss_prefix_len > threshold:
                        self.mamba_buffer_insert_len = miss_prefix_len
                        self.is_hotspot_prefill = True

        self.shm_req.shm_cur_kv_len = self.cur_kv_len
        return
```

- [ ] **Step 3: Update `get_chuncked_input_token_len` for first-chunk enlargement**

Replace `get_chuncked_input_token_len` (lines 548-551) with:

```python
    def get_chuncked_input_token_len(self):
        chunked_start = self.cur_kv_len
        chunked_end = min(self.get_cur_total_len(), chunked_start + self.shm_req.chunked_prefill_size)
        if self.mamba_buffer_insert_len > 0:
            chunked_end = min(self.get_cur_total_len(), chunked_start + self.mamba_buffer_insert_len)
            self.mamba_buffer_insert_len = 0
        return chunked_end
```

- [ ] **Step 4: Update `get_chuncked_input_token_ids` to use `get_chuncked_input_token_len`**

Replace `get_chuncked_input_token_ids` (lines 543-546) with:

```python
    def get_chuncked_input_token_ids(self):
        chunked_end = self.get_chuncked_input_token_len()
        return self.shm_req.shm_prompt_ids.arr[0:chunked_end]
```

- [ ] **Step 5: Commit**

```bash
git add lightllm/server/router/model_infer/infer_batch.py
git commit -m "feat: add InferReq hotspot fields, detection in _match_radix_cache, first-chunk enlargement"
```

---

### Task 7: Remove non-hot buffer donation + add deferred free drain in `free_a_req_mem_for_mamba`

**Files:**
- Modify: `lightllm/server/router/model_infer/infer_batch.py:163-186`

- [ ] **Step 1: Replace `free_a_req_mem_for_mamba`**

Replace the method (lines 163-186) with:

```python
    def free_a_req_mem_for_mamba(self, free_token_index: List, req: "InferReq") -> bool:
        if self.radix_cache is None:
            free_token_index.append(self.req_manager.req_to_token_indexs[req.req_idx][0 : req.cur_kv_len])
        else:
            input_token_ids = req.get_input_token_ids()
            key = torch.tensor(input_token_ids[0 : req.cur_kv_len], dtype=torch.int64, device="cpu")
            value = self.req_manager.req_to_token_indexs[req.req_idx][: req.cur_kv_len].detach().cpu()

            prefix_len, node = self.radix_cache.insert(key, value)
            old_prefix_len = 0 if req.shared_kv_node is None else req.shared_kv_node.node_prefix_total_len
            free_token_index.append(self.req_manager.req_to_token_indexs[req.req_idx][old_prefix_len:prefix_len])
            if req.shared_kv_node is not None:
                assert req.shared_kv_node.node_prefix_total_len <= prefix_len
                self.radix_cache.dec_node_ref_counter(req.shared_kv_node)
                req.shared_kv_node = None

            # Drain deferred frees from hotspot buffer insertion
            if len(req.extra_need_to_free_token_index) > 0:
                free_token_index.extend(req.extra_need_to_free_token_index)
                req.extra_need_to_free_token_index = []

        return True  # always free request's buffer back to pool
```

- [ ] **Step 2: Update `_free_req_mem_and_buffers` to remove conditional buffer handling**

Since `free_a_req_mem_for_mamba` now always returns `True`, simplify `_free_req_mem_and_buffers` (lines 188-196):

```python
    def _free_req_mem_and_buffers(self, free_token_index: List, free_buffer_index: List, req: "InferReq"):
        """释放请求的 KV cache 和 buffer 内存"""
        if self.has_recurrent_state:
            self.free_a_req_mem_for_mamba(free_token_index, req)
            req_to_buffer_index = self.req_manager.req_to_buffer_index
            free_buffer_index.extend(req_to_buffer_index[req.req_idx, :].tolist())
        else:
            self.free_a_req_mem(free_token_index, req)
```

- [ ] **Step 3: Commit**

```bash
git add lightllm/server/router/model_infer/infer_batch.py
git commit -m "feat: remove non-hot buffer donation, add deferred free drain"
```

---

### Task 8: Add `snapshot_hybrid_buffers` to `InferenceContext`

**Files:**
- Modify: `lightllm/server/router/model_infer/infer_batch.py` (add method after `_free_req_mem_and_buffers`)

- [ ] **Step 1: Add import for `HybridRadixCache`**

At the top of `lightllm/server/router/model_infer/infer_batch.py`, after the existing `HybridRadixCache` import (line 14), verify it exists:

```python
from lightllm.server.router.dynamic_prompt.hybrid_radix_cache import HybridRadixCache
```

- [ ] **Step 2: Add `snapshot_hybrid_buffers` method to `InferenceContext`**

Add after `_free_req_mem_and_buffers` (around line 197):

```python
    def snapshot_hybrid_buffers(self, run_reqs: List["InferReq"]):
        """Snapshot Mamba states for hotspot requests after their enlarged first chunk."""
        reqs_to_insert = [r for r in run_reqs
                           if r.is_hotspot_prefill and r.cur_kv_len < r.get_cur_total_len()]
        if not reqs_to_insert:
            return

        radix_cache: HybridRadixCache = self.radix_cache
        radix_cache.free_radix_cache_to_get_enough_buffer(len(reqs_to_insert))

        new_buffer_indexes = radix_cache.buffer_mem_manager.alloc(len(reqs_to_insert))
        req_idxes = torch.tensor([r.req_idx for r in reqs_to_insert], dtype=torch.int64, device="cuda")
        cur_buffers = self.req_manager.req_to_buffer_index[req_idxes, 0].contiguous()
        new_buffers_cuda = new_buffer_indexes.to(device="cuda", dtype=torch.int64).contiguous()
        radix_cache.buffer_mem_manager.copy_state_buffers(cur_buffers, new_buffers_cuda)

        for i, req in enumerate(reqs_to_insert):
            key = torch.tensor(req.get_input_token_ids()[:req.cur_kv_len], dtype=torch.int64, device="cpu")
            value = self.req_manager.req_to_token_indexs[req.req_idx][:req.cur_kv_len].cpu()

            prefix_len, new_node = radix_cache.insert(key, value)
            old_prefix_len = 0 if req.shared_kv_node is None else req.shared_kv_node.node_prefix_total_len

            radix_cache.dec_node_ref_counter(req.shared_kv_node)
            radix_cache.add_node_ref_counter(new_node)
            radix_cache.add_buffer_idx_to_node(new_node, new_buffer_indexes[i].item(), is_hotspot=True)

            req.extra_need_to_free_token_index.append(
                self.req_manager.req_to_token_indexs[req.req_idx][old_prefix_len:prefix_len]
            )
            req.shared_kv_node = new_node
            req.is_hotspot_prefill = False
```

- [ ] **Step 3: Commit**

```bash
git add lightllm/server/router/model_infer/infer_batch.py
git commit -m "feat: add snapshot_hybrid_buffers to InferenceContext"
```

---

### Task 9: Integrate `_maybe_snapshot_hybrid_buffers` in chunked prefill backend

**Files:**
- Modify: `lightllm/server/router/model_infer/mode_backend/chunked_prefill/impl.py:103-224`

- [ ] **Step 1: Add `_maybe_snapshot_hybrid_buffers` method**

Add this method to `ChunkedPrefillBackend`, after the `__init__` method (around line 53):

```python
    def _maybe_snapshot_hybrid_buffers(self, run_reqs: List[InferReq]):
        """Snapshot Mamba states for hotspot requests. Called after prefill, before next iteration."""
        if g_infer_context.has_recurrent_state and self.radix_cache is not None:
            torch.cuda.synchronize()
            g_infer_state_lock.acquire()
            g_infer_context.snapshot_hybrid_buffers(run_reqs)
            g_infer_state_lock.release()
```

- [ ] **Step 2: Insert call in `prefill_normal` between Stage 3 and Stage 4**

In `prefill_normal` (line 138), add the call after `_post_handle` and before `notify_pre_post_handle`:

```python
        # Buffer snapshot for hotspot requests (between Stage 3 and Stage 4)
        self._maybe_snapshot_hybrid_buffers(run_reqs)

        # 第四阶段
        event_pack.notify_pre_post_handle()
```

- [ ] **Step 3: Insert call in `prefill_mtp` between Stage 3 and Stage 4**

In `prefill_mtp` (line 220), add the call after `_post_handle` and before `notify_pre_post_handle`:

```python
        # Buffer snapshot for hotspot requests (between Stage 3 and Stage 4)
        self._maybe_snapshot_hybrid_buffers(run_reqs)

        # 第四阶段
        event_pack.notify_pre_post_handle()
```

- [ ] **Step 4: Commit**

```bash
git add lightllm/server/router/model_infer/mode_backend/chunked_prefill/impl.py
git commit -m "feat: integrate hotspot buffer snapshot in chunked prefill backend"
```

---

### Task 10: Final integration test and cleanup

**Files:**
- Test: `unit_tests/server/router/dynamic_prompt/test_hybrid_radix_cache.py`

- [ ] **Step 1: Add an end-to-end test for the full hotspot lifecycle**

Append to `unit_tests/server/router/dynamic_prompt/test_hybrid_radix_cache.py`:

```python
def test_full_hotspot_lifecycle():
    """End-to-end test: insert → match (miss) → hotspot insert → match (hit) → evict."""
    cache = _make_hybrid_cache("lifecycle", kv_size=200, buf_size=10)

    # 1. First request creates tree structure (simulating request completion)
    key = torch.tensor([1, 2, 3, 4, 5, 6, 7, 8], dtype=torch.int64, device="cpu")
    val = torch.tensor([10, 11, 12, 13, 14, 15, 16, 17], dtype=torch.int64, device="cpu")
    _, node = cache.insert(key, val)
    # No buffer donated (only hot insertions)

    # 2. Second request matches — walks back to root, R3 cleans up
    node2, miss, val2 = cache.match_prefix(
        torch.tensor([1, 2, 3, 4, 5, 6, 7, 8], dtype=torch.int64, device="cpu"),
        update_refs=True,
    )
    assert node2 is None  # no buffer anywhere
    assert miss == 8
    assert cache.get_tree_total_tokens_num() == 0  # R3 cleaned up

    # 3. Simulate hotspot insert: re-insert tree + add buffer
    key2 = torch.tensor([1, 2, 3, 4, 5, 6, 7, 8], dtype=torch.int64, device="cpu")
    val2 = torch.tensor([20, 21, 22, 23, 24, 25, 26, 27], dtype=torch.int64, device="cpu")
    _, hot_node = cache.insert(key2, val2)
    cache.add_buffer_idx_to_node(hot_node, 0, is_hotspot=True)
    assert cache.buffer_insert_count == 1

    # 4. Third request matches — finds buffer, no miss
    node3, miss3, val3 = cache.match_prefix(
        torch.tensor([1, 2, 3, 4, 5, 6, 7, 8], dtype=torch.int64, device="cpu"),
        update_refs=True,
    )
    assert node3 is hot_node
    assert miss3 == 0
    assert len(val3) == 8
    assert hot_node.was_hit is True

    # 5. Evict — should track as hit
    release_bufs = []
    release_mems = []
    cache._evict_buffer(1, lambda x: release_bufs.append(x), lambda m: release_mems.append(m))
    assert cache.buffer_hit_count == 1
    assert cache.buffer_waste_count == 0
```

- [ ] **Step 2: Run all tests**

Run: `cd /workspace/LightLLM/worktree/qw35_oom && python -m pytest unit_tests/server/router/dynamic_prompt/ -v`
Expected: ALL PASS

- [ ] **Step 3: Commit**

```bash
git add unit_tests/server/router/dynamic_prompt/test_hybrid_radix_cache.py
git commit -m "test: add full hotspot lifecycle integration test"
```
