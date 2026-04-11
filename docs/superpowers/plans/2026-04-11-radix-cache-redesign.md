# Radix Cache Insertion Redesign Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace adaptive threshold/hotspot heuristics in HybridRadixCache with deterministic two-point insertion (mid-prefill unbuffered prefix snapshot + post-prefill full input snapshot).

**Architecture:** Remove all heuristic fields (`is_hotspot`, `was_hit`, adaptive thresholds, waste ratios) from `TreeNode` and `HybridRadixCache`. Add `insert_with_buffer()` method to `HybridRadixCache` that guarantees buffer attachment at an explicit boundary position. Modify `match_prefix()` to return `unbuffered_prefix_pos` directly instead of stashing it. Replace `snapshot_hybrid_buffers()` and `snapshot_prefill_complete_buffers()` in `InferBatchContext` with `_maybe_snapshot_unbuffered_prefix()` and `_maybe_snapshot_full_input()`.

**Tech Stack:** Python, PyTorch, sortedcontainers, pytest

---

## File Map

| File | Action | Responsibility |
|------|--------|---------------|
| `lightllm/server/router/dynamic_prompt/radix_cache.py` | Modify | Remove `is_hotspot`, `was_hit` from `TreeNode` |
| `lightllm/server/router/dynamic_prompt/hybrid_radix_cache.py` | Modify | Remove heuristic state, add `insert_with_buffer()`, simplify `match_prefix()`, simplify `add_buffer_idx_to_node()`, simplify eviction |
| `lightllm/server/router/model_infer/infer_batch.py` | Modify | Replace snapshot methods, simplify `_match_radix_cache()`, replace request state fields |
| `lightllm/server/router/model_infer/mode_backend/chunked_prefill/impl.py` | Modify | Rename `_maybe_snapshot_hybrid_buffers` |
| `lightllm/server/router/model_infer/mode_backend/dp_backend/impl.py` | Modify | Rename `_maybe_snapshot_hybrid_buffers` |
| `unit_tests/server/router/dynamic_prompt/test_hybrid_radix_cache.py` | Modify | Update tests for new API, add `insert_with_buffer` tests |

---

### Task 1: Remove heuristic fields from TreeNode

**Files:**
- Modify: `lightllm/server/router/dynamic_prompt/radix_cache.py:22-41`

- [ ] **Step 1: Write failing test — TreeNode has no is_hotspot or was_hit**

```python
# In unit_tests/server/router/dynamic_prompt/test_radix_cache.py
# Add at end of file, before `if __name__`:

def test_tree_node_has_no_hotspot_fields():
    """TreeNode should not have is_hotspot or was_hit after redesign."""
    from lightllm.server.router.dynamic_prompt.radix_cache import TreeNode

    node = TreeNode()
    assert not hasattr(node, "is_hotspot"), "is_hotspot should be removed"
    assert not hasattr(node, "was_hit"), "was_hit should be removed"
    # buffer_idx and buffer_time should still exist
    assert hasattr(node, "buffer_idx")
    assert hasattr(node, "buffer_time")
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest unit_tests/server/router/dynamic_prompt/test_radix_cache.py::test_tree_node_has_no_hotspot_fields -v`
Expected: FAIL — `is_hotspot` still exists on TreeNode

- [ ] **Step 3: Remove is_hotspot and was_hit from TreeNode**

In `lightllm/server/router/dynamic_prompt/radix_cache.py`, remove lines 39-40 from `TreeNode.__init__`:

```python
        self.is_hotspot = False
        self.was_hit = False
```

These two lines are removed entirely. The rest of `__init__` stays unchanged.

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest unit_tests/server/router/dynamic_prompt/test_radix_cache.py::test_tree_node_has_no_hotspot_fields -v`
Expected: PASS

- [ ] **Step 5: Run all existing radix cache tests to check nothing breaks**

Run: `python -m pytest unit_tests/server/router/dynamic_prompt/test_radix_cache.py -v`
Expected: All 9 existing tests PASS (they don't use is_hotspot or was_hit)

- [ ] **Step 6: Commit**

```bash
git add lightllm/server/router/dynamic_prompt/radix_cache.py unit_tests/server/router/dynamic_prompt/test_radix_cache.py
git commit -m "refactor: remove is_hotspot and was_hit from TreeNode"
```

---

### Task 2: Remove heuristic state from HybridRadixCache

**Files:**
- Modify: `lightllm/server/router/dynamic_prompt/hybrid_radix_cache.py:13-30`
- Modify: `unit_tests/server/router/dynamic_prompt/test_hybrid_radix_cache.py`

- [ ] **Step 1: Write failing test — HybridRadixCache has no adaptive threshold fields**

```python
# In unit_tests/server/router/dynamic_prompt/test_hybrid_radix_cache.py
# Add at end of file:

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
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest unit_tests/server/router/dynamic_prompt/test_hybrid_radix_cache.py::test_no_adaptive_threshold_fields -v`
Expected: FAIL — `min_insert_threshold` still exists

- [ ] **Step 3: Remove adaptive threshold state from __init__**

In `lightllm/server/router/dynamic_prompt/hybrid_radix_cache.py`, replace lines 22-30:

```python
        self.evict_buffer_set: Set[TreeNode] = SortedSet(key=lambda x: (x.is_hotspot, x.buffer_time))
        # Adaptive threshold state
        self.min_insert_threshold = 1024
        self.MIN_THRESHOLD = 256
        self.MAX_THRESHOLD = 16384
        self.adjust_interval = 100
        self.buffer_insert_count = 0
        self.buffer_hit_count = 0
        self.buffer_waste_count = 0
```

With:

```python
        self.evict_buffer_set: Set[TreeNode] = SortedSet(key=lambda x: x.buffer_time)
```

- [ ] **Step 4: Remove _maybe_adjust_threshold, _reset_counters, and available_opportunistic_buffer_count methods**

Delete the following methods entirely from `hybrid_radix_cache.py`:

`available_opportunistic_buffer_count` (lines 216-220):
```python
    def available_opportunistic_buffer_count(self) -> int:
        """Count how many buffers can be obtained without evicting hotspot buffers."""
        free = self.buffer_mem_manager.can_use_mem_size
        non_hotspot = sum(1 for n in self.evict_buffer_set if not n.is_hotspot)
        return free + non_hotspot
```

`_maybe_adjust_threshold` (lines 222-235):
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
```

`_reset_counters` (lines 237-240):
```python
    def _reset_counters(self):
        self.buffer_insert_count = 0
        self.buffer_hit_count = 0
        self.buffer_waste_count = 0
```

- [ ] **Step 5: Remove hotspot tracking from add_buffer_idx_to_node**

In `hybrid_radix_cache.py`, replace `add_buffer_idx_to_node` (lines 83-99):

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
        if is_hotspot:
            self.buffer_insert_count += 1
        if node.is_leaf():
            self.evict_tree_set.add(node)
        return
```

With:

```python
    def add_buffer_idx_to_node(self, node: TreeNode, buffer_idx: int):
        """Set buffer_idx for a node and add it to evict_buffer_set."""
        self.evict_buffer_set.discard(node)
        if node.is_leaf():
            self.evict_tree_set.discard(node)
        if node.buffer_idx is not None:
            self.buffer_mem_manager.free([node.buffer_idx])
        node.buffer_idx = buffer_idx
        node.update_buffer_time()
        self.evict_buffer_set.add(node)
        if node.is_leaf():
            self.evict_tree_set.add(node)
        return
```

- [ ] **Step 6: Remove hotspot tracking from _evict_buffer**

In `hybrid_radix_cache.py`, replace `_evict_buffer` (lines 115-165):

```python
    def _evict_buffer(self, need_evict_buffer_num, evict_buffer_callback):
        # Two-pass eviction: first evict buffers from unreferenced nodes,
        # then from referenced nodes only if necessary.  Evicting a buffer
        # from a referenced node makes its KV cache unreachable for hybrid
        # model reuse (no Mamba state to resume from).
        deferred_referenced = []
        while need_evict_buffer_num > 0:
            if not self.evict_buffer_set:
                break
            node = self.evict_buffer_set.pop(0)
            assert node.buffer_idx is not None

            if node.ref_counter > 0:
                deferred_referenced.append(node)
                continue

            if node.was_hit:
                self.buffer_hit_count += 1
            else:
                self.buffer_waste_count += 1
            evict_buffer_callback(node.buffer_idx)
            node.buffer_idx = None
            node.is_hotspot = False
            node.was_hit = False
            need_evict_buffer_num -= 1

            # Keep tree node alive after buffer eviction so prefix matching
            # still works in multi-turn dialogues.  The node's tokens are cheap;
            # only the buffer was the expensive resource.  Tree-level eviction
            # (evict_tree_set) will reclaim the node later if memory is needed.
            if node.is_leaf() and node.ref_counter == 0:
                self.evict_tree_set.add(node)

        # Fallback: evict referenced buffers if unreferenced ones were not enough
        for node in deferred_referenced:
            if need_evict_buffer_num <= 0:
                self.evict_buffer_set.add(node)
                continue

            if node.was_hit:
                self.buffer_hit_count += 1
            else:
                self.buffer_waste_count += 1
            evict_buffer_callback(node.buffer_idx)
            node.buffer_idx = None
            node.is_hotspot = False
            node.was_hit = False
            need_evict_buffer_num -= 1
            # Node is referenced, so it stays out of evict_tree_set.
        self._maybe_adjust_threshold()
        return
```

With:

```python
    def _evict_buffer(self, need_evict_buffer_num, evict_buffer_callback):
        # Two-pass eviction: first evict buffers from unreferenced nodes,
        # then from referenced nodes only if necessary.
        deferred_referenced = []
        while need_evict_buffer_num > 0:
            if not self.evict_buffer_set:
                break
            node = self.evict_buffer_set.pop(0)
            assert node.buffer_idx is not None

            if node.ref_counter > 0:
                deferred_referenced.append(node)
                continue

            evict_buffer_callback(node.buffer_idx)
            node.buffer_idx = None
            need_evict_buffer_num -= 1

            if node.is_leaf() and node.ref_counter == 0:
                self.evict_tree_set.add(node)

        for node in deferred_referenced:
            if need_evict_buffer_num <= 0:
                self.evict_buffer_set.add(node)
                continue

            evict_buffer_callback(node.buffer_idx)
            node.buffer_idx = None
            need_evict_buffer_num -= 1
        return
```

- [ ] **Step 7: Remove was_hit tracking from match_prefix**

In `hybrid_radix_cache.py`, in `match_prefix` (around line 68-69), remove:

```python
        # Mark buffer node as hit when update_refs is True
        if update_refs:
            tree_node.was_hit = True
```

- [ ] **Step 8: Update existing tests — remove is_hotspot parameter from add_buffer_idx_to_node calls**

In `unit_tests/server/router/dynamic_prompt/test_hybrid_radix_cache.py`:

In `test_evict_buffer_set_hotspot_sort_order`, replace the entire test function:

```python
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
```

In `test_two_pass_eviction_skips_referenced`, `test_two_pass_eviction_falls_back_to_referenced`, and `test_node_stays_alive_after_buffer_eviction`: remove `is_hotspot` parameter from `add_buffer_idx_to_node` calls. Specifically, all calls like `cache.add_buffer_idx_to_node(node, 100, is_hotspot=True)` or `cache.add_buffer_idx_to_node(node, 100)` stay as `cache.add_buffer_idx_to_node(node, 100)` (which is already correct for those tests).

In `test_match_prefix_walkback_does_not_destroy_nodes`: remove the assertion on `_last_miss_prefix_len` (line 129):

```python
    assert cache._last_miss_prefix_len == 2, "Missed 2 tokens in [3,4] node"
```

Replace with (the new return value):

```python
    # unbuffered_prefix_pos is now returned directly, not stashed
    # It was already captured in share_node, kv_len, value_tensor above
```

Actually, we need to update the unpack to 4 values. Replace line 125:

```python
    share_node, kv_len, value_tensor = cache.match_prefix(key, update_refs=True)
```

With:

```python
    share_node, kv_len, value_tensor, unbuffered_prefix_pos = cache.match_prefix(key, update_refs=True)
```

And replace the `_last_miss_prefix_len` assertion with:

```python
    assert unbuffered_prefix_pos == 2, "Missed 2 tokens in [3,4] node"
```

- [ ] **Step 9: Run all hybrid cache tests**

Run: `python -m pytest unit_tests/server/router/dynamic_prompt/test_hybrid_radix_cache.py -v`
Expected: All tests PASS

- [ ] **Step 10: Commit**

```bash
git add lightllm/server/router/dynamic_prompt/hybrid_radix_cache.py unit_tests/server/router/dynamic_prompt/test_hybrid_radix_cache.py
git commit -m "refactor: remove adaptive threshold and hotspot heuristics from HybridRadixCache"
```

---

### Task 3: Modify match_prefix to return unbuffered_prefix_pos

**Files:**
- Modify: `lightllm/server/router/dynamic_prompt/hybrid_radix_cache.py:32-81`

- [ ] **Step 1: Write failing test — match_prefix returns 4-tuple**

```python
# In unit_tests/server/router/dynamic_prompt/test_hybrid_radix_cache.py
# Add at end:

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
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest unit_tests/server/router/dynamic_prompt/test_hybrid_radix_cache.py::test_match_prefix_returns_unbuffered_prefix_pos -v`
Expected: FAIL — match_prefix returns 3 values, not 4

- [ ] **Step 3: Modify match_prefix to return 4-tuple**

In `hybrid_radix_cache.py`, replace the `match_prefix` method with:

```python
    def match_prefix(self, key, update_refs=False):
        assert len(key) != 0
        ans_value_list = []
        tree_node = self._match_prefix_helper(self.root_node, key, ans_value_list, update_refs=update_refs)
        unbuffered_prefix_pos = 0
        while tree_node != self.root_node and tree_node.buffer_idx is None:
            unbuffered_prefix_pos += len(ans_value_list[-1]) if ans_value_list else 0

            next_node = tree_node.parent

            if update_refs:
                if tree_node.is_leaf():
                    self.evict_tree_set.discard(tree_node)
                if tree_node.ref_counter == 1:
                    self.refed_tokens_num.arr[0] -= len(tree_node.token_mem_index_value)
                tree_node.ref_counter -= 1
                if tree_node.is_leaf():
                    self.evict_tree_set.add(tree_node)

            ans_value_list.pop()
            tree_node = next_node

        if tree_node == self.root_node:
            return None, 0, None, unbuffered_prefix_pos

        update_node = tree_node
        while update_node != self.root_node:
            if update_node.buffer_idx is not None:
                self.evict_buffer_set.discard(update_node)
                update_node.update_buffer_time()
                self.evict_buffer_set.add(update_node)
            update_node = update_node.parent

        kv_len = tree_node.node_prefix_total_len
        value = torch.concat(ans_value_list)
        return tree_node, kv_len, value, unbuffered_prefix_pos
```

Key changes from current:
- Returns `unbuffered_prefix_pos` as 4th tuple element instead of stashing `_last_miss_prefix_len`
- Removed `tree_node.was_hit = True` line (already removed in Task 2 Step 7)
- Removed `self._last_miss_prefix_len = miss_prefix_len` stash line

- [ ] **Step 4: Run the new tests**

Run: `python -m pytest unit_tests/server/router/dynamic_prompt/test_hybrid_radix_cache.py -v`
Expected: All tests PASS

- [ ] **Step 5: Commit**

```bash
git add lightllm/server/router/dynamic_prompt/hybrid_radix_cache.py unit_tests/server/router/dynamic_prompt/test_hybrid_radix_cache.py
git commit -m "refactor: match_prefix returns unbuffered_prefix_pos instead of stashing it"
```

---

### Task 4: Add insert_with_buffer method

**Files:**
- Modify: `lightllm/server/router/dynamic_prompt/hybrid_radix_cache.py`
- Modify: `unit_tests/server/router/dynamic_prompt/test_hybrid_radix_cache.py`

- [ ] **Step 1: Write failing test — insert_with_buffer basic behavior**

```python
# In unit_tests/server/router/dynamic_prompt/test_hybrid_radix_cache.py
# Add at end:

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
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest unit_tests/server/router/dynamic_prompt/test_hybrid_radix_cache.py::test_insert_with_buffer_attaches_buffer_to_leaf -v`
Expected: FAIL — `insert_with_buffer` does not exist

- [ ] **Step 3: Implement insert_with_buffer**

In `hybrid_radix_cache.py`, add this method after `match_prefix`:

```python
    def insert_with_buffer(self, key, value, unbuffered_prefix_pos, buffer_snapshot_fn):
        """Insert tokens and ensure buffers at the boundary and leaf.

        Args:
            key: full input token ids (int64 tensor)
            value: corresponding KV cache memory indices (int64 tensor)
            unbuffered_prefix_pos: token position of prefix needing a buffer.
                                   0 means no boundary snapshot needed.
            buffer_snapshot_fn: callable(node) -> None, called for nodes needing
                                a buffer. Must check node.buffer_idx before acting.

        Returns:
            (prefix_len, leaf_node)
        """
        if unbuffered_prefix_pos > 0:
            # First insert the prefix portion to ensure a node exists at the boundary
            boundary_key = key[:unbuffered_prefix_pos]
            boundary_value = value[:unbuffered_prefix_pos]
            _, boundary_node = self.insert(boundary_key, boundary_value)
            if boundary_node is not None and boundary_node.buffer_idx is None:
                buffer_snapshot_fn(boundary_node)

        # Insert the full key
        prefix_len, leaf_node = self.insert(key, value)
        if leaf_node is not None and leaf_node.buffer_idx is None:
            buffer_snapshot_fn(leaf_node)

        return prefix_len, leaf_node
```

- [ ] **Step 4: Run all tests**

Run: `python -m pytest unit_tests/server/router/dynamic_prompt/test_hybrid_radix_cache.py -v`
Expected: All tests PASS

- [ ] **Step 5: Commit**

```bash
git add lightllm/server/router/dynamic_prompt/hybrid_radix_cache.py unit_tests/server/router/dynamic_prompt/test_hybrid_radix_cache.py
git commit -m "feat: add insert_with_buffer to HybridRadixCache"
```

---

### Task 5: Simplify InferReq state and _match_radix_cache

**Files:**
- Modify: `lightllm/server/router/model_infer/infer_batch.py:710-773`

- [ ] **Step 1: Replace request state fields**

In `lightllm/server/router/model_infer/infer_batch.py`, in `_init_all_state` (around line 736-739), replace:

```python
        # Hybrid radix cache hotspot detection
        self.mamba_buffer_target_len = 0  # absolute token position for buffer snapshot
        self.is_hotspot_prefill = False
        self.extra_need_to_free_token_index = []
```

With:

```python
        # Position of unbuffered prefix node (from match_prefix walk-back).
        # 0 means no mid-prefill snapshot needed.
        self.unbuffered_prefix_pos = 0
```

- [ ] **Step 2: Simplify _match_radix_cache**

In `lightllm/server/router/model_infer/infer_batch.py`, replace `_match_radix_cache` (lines 742-773):

```python
    def _match_radix_cache(self):
        if self.sampling_param.disable_prompt_cache:
            return
        if g_infer_context.radix_cache is not None and self.get_cur_total_len() > 1 and self.cur_kv_len == 0:
            input_token_ids = self.shm_req.shm_prompt_ids.arr[0 : self.get_cur_total_len()]
            key = torch.tensor(input_token_ids, dtype=torch.int64, device="cpu")
            key = key[0 : len(key) - 1]  # 最后一个不需要，因为需要一个额外的token，让其在prefill的时候输出下一个token的值
            share_node, kv_len, value_tensor = g_infer_context.radix_cache.match_prefix(key, update_refs=True)
            # For hybrid models, retrieve the walk-back miss length stashed by match_prefix
            miss_prefix_len = getattr(g_infer_context.radix_cache, "_last_miss_prefix_len", 0)
            if share_node is not None:
                self.shared_kv_node = share_node
                ready_cache_len = share_node.node_prefix_total_len
                # 从 cpu 到 gpu 是流内阻塞操作
                g_infer_context.req_manager.req_to_token_indexs[self.req_idx, 0:ready_cache_len] = value_tensor
                self.cur_kv_len = int(ready_cache_len)  # 序列化问题, 该对象可能为numpy.int64，用 int(*)转换
                self.shm_req.prompt_cache_len = self.cur_kv_len  # 记录 prompt cache 的命中长度

                if g_infer_context.has_recurrent_state:
                    threshold = g_infer_context.radix_cache.min_insert_threshold
                    if miss_prefix_len > threshold:
                        self.mamba_buffer_target_len = self.cur_kv_len + miss_prefix_len
                        self.is_hotspot_prefill = True
            elif g_infer_context.has_recurrent_state and miss_prefix_len > 0:
                # Bootstrap case: tree had KV structure but no buffer anywhere.
                threshold = g_infer_context.radix_cache.min_insert_threshold
                if miss_prefix_len > threshold:
                    self.mamba_buffer_target_len = miss_prefix_len
                    self.is_hotspot_prefill = True

        self.shm_req.shm_cur_kv_len = self.cur_kv_len
        return
```

With:

```python
    def _match_radix_cache(self):
        if self.sampling_param.disable_prompt_cache:
            return
        if g_infer_context.radix_cache is not None and self.get_cur_total_len() > 1 and self.cur_kv_len == 0:
            input_token_ids = self.shm_req.shm_prompt_ids.arr[0 : self.get_cur_total_len()]
            key = torch.tensor(input_token_ids, dtype=torch.int64, device="cpu")
            key = key[0 : len(key) - 1]  # 最后一个不需要，因为需要一个额外的token，让其在prefill的时候输出下一个token的值

            if g_infer_context.has_recurrent_state:
                share_node, kv_len, value_tensor, unbuffered_prefix_pos = (
                    g_infer_context.radix_cache.match_prefix(key, update_refs=True)
                )
                if share_node is not None:
                    self.shared_kv_node = share_node
                    ready_cache_len = share_node.node_prefix_total_len
                    g_infer_context.req_manager.req_to_token_indexs[self.req_idx, 0:ready_cache_len] = value_tensor
                    self.cur_kv_len = int(ready_cache_len)
                    self.shm_req.prompt_cache_len = self.cur_kv_len
                    self.unbuffered_prefix_pos = self.cur_kv_len + unbuffered_prefix_pos
                elif unbuffered_prefix_pos > 0:
                    # Bootstrap: tree had KV but no buffer anywhere
                    self.unbuffered_prefix_pos = unbuffered_prefix_pos
            else:
                share_node, kv_len, value_tensor = g_infer_context.radix_cache.match_prefix(key, update_refs=True)
                if share_node is not None:
                    self.shared_kv_node = share_node
                    ready_cache_len = share_node.node_prefix_total_len
                    g_infer_context.req_manager.req_to_token_indexs[self.req_idx, 0:ready_cache_len] = value_tensor
                    self.cur_kv_len = int(ready_cache_len)
                    self.shm_req.prompt_cache_len = self.cur_kv_len

        self.shm_req.shm_cur_kv_len = self.cur_kv_len
        return
```

- [ ] **Step 3: Commit**

```bash
git add lightllm/server/router/model_infer/infer_batch.py
git commit -m "refactor: simplify InferReq state and _match_radix_cache"
```

---

### Task 6: Replace snapshot methods in InferBatchContext

**Files:**
- Modify: `lightllm/server/router/model_infer/infer_batch.py:303-451`

- [ ] **Step 1: Remove snapshot_hybrid_buffers and snapshot_prefill_complete_buffers**

Delete both methods (`snapshot_hybrid_buffers` at lines 303-370 and `snapshot_prefill_complete_buffers` at lines 372-451).

- [ ] **Step 2: Add _make_buffer_snapshot_fn helper**

Add this method to the `InferBatchContext` class (after `_insert_kv_for_mamba`):

```python
    def _make_buffer_snapshot_fn(self, req: "InferReq"):
        """Create a buffer_snapshot_fn callback for insert_with_buffer.

        The callback handles CPU-offload or GPU-to-GPU copy depending on
        whether a CPU buffer manager is available.
        """
        radix_cache = self.radix_cache
        req_to_buffer_index = self.req_manager.req_to_buffer_index
        cpu_mgr = getattr(self.req_manager, "cpu_buffer_mem_manager", None)

        def snapshot(node):
            if node.buffer_idx is not None:
                return
            primary_buffer_idx = req_to_buffer_index[req.req_idx, 0].item()
            if cpu_mgr is not None:
                radix_cache.free_radix_cache_to_get_enough_buffer(1)
                if cpu_mgr.can_use_mem_size < 1:
                    return  # no buffer available
                cpu_slot = cpu_mgr.alloc(1)
                gpu_idx = torch.tensor([primary_buffer_idx], dtype=torch.int64, device="cuda")
                cpu_mgr.offload_to_cpu(
                    self.req_manager.buffer_mem_manager.conv_state_cache.buffer,
                    self.req_manager.buffer_mem_manager.ssm_state_cache.buffer,
                    gpu_idx,
                    cpu_slot,
                )
                radix_cache.add_buffer_idx_to_node(node, cpu_slot[0].item())
            else:
                radix_cache.free_radix_cache_to_get_enough_buffer(1)
                if radix_cache.buffer_mem_manager.can_use_mem_size < 1:
                    return  # no buffer available
                new_buf = radix_cache.buffer_mem_manager.alloc(1)
                cur_buf = torch.tensor([primary_buffer_idx], dtype=torch.int64, device="cuda")
                new_buf_cuda = new_buf.to(device="cuda", dtype=torch.int64).contiguous()
                radix_cache.buffer_mem_manager.copy_state_buffers(cur_buf, new_buf_cuda)
                radix_cache.add_buffer_idx_to_node(node, new_buf[0].item())

        return snapshot
```

- [ ] **Step 3: Add _maybe_snapshot_unbuffered_prefix**

Add this method to `InferBatchContext`:

```python
    def _maybe_snapshot_unbuffered_prefix(self, run_reqs: List["InferReq"]):
        """Snapshot Mamba state at the unbuffered prefix boundary (system prompt)."""
        reqs = [r for r in run_reqs if r.unbuffered_prefix_pos > 0 and r.cur_kv_len >= r.unbuffered_prefix_pos]
        if not reqs:
            return
        radix_cache = self.radix_cache
        for req in reqs:
            key = torch.tensor(req.get_input_token_ids()[: req.unbuffered_prefix_pos], dtype=torch.int64, device="cpu")
            value = self.req_manager.req_to_token_indexs[req.req_idx][: req.unbuffered_prefix_pos].cpu()
            snapshot_fn = self._make_buffer_snapshot_fn(req)
            prefix_len, new_node = radix_cache.insert(key, value)
            if new_node is not None and new_node.buffer_idx is None:
                snapshot_fn(new_node)
            # Update shared_kv_node ref counting
            old_prefix_len = 0 if req.shared_kv_node is None else req.shared_kv_node.node_prefix_total_len
            if req.shared_kv_node is not None:
                radix_cache.dec_node_ref_counter(req.shared_kv_node)
            radix_cache.add_node_ref_counter(new_node)
            req.shared_kv_node = new_node
            # Free overlapping tokens
            if old_prefix_len < prefix_len:
                self.req_manager.free_token(
                    self.req_manager.req_to_token_indexs[req.req_idx][old_prefix_len:prefix_len]
                )
            req.unbuffered_prefix_pos = 0
```

- [ ] **Step 4: Add _maybe_snapshot_full_input**

Add this method to `InferBatchContext`:

```python
    def _maybe_snapshot_full_input(self, run_reqs: List["InferReq"]):
        """Snapshot Mamba state for the full input after prefill completes."""
        reqs = [
            r
            for r in run_reqs
            if r.cur_kv_len + 1 >= r.get_cur_total_len() and r.cur_output_len == 0
        ]
        if not reqs:
            return
        radix_cache = self.radix_cache
        for req in reqs:
            # Skip if shared node already has a buffer (from unbuffered prefix snapshot or prior match)
            if req.shared_kv_node is not None and req.shared_kv_node.buffer_idx is not None:
                # Check if shared_kv_node covers the full input already
                if req.shared_kv_node.node_prefix_total_len >= req.cur_kv_len:
                    continue
            key = torch.tensor(req.get_input_token_ids()[: req.cur_kv_len], dtype=torch.int64, device="cpu")
            value = self.req_manager.req_to_token_indexs[req.req_idx][: req.cur_kv_len].cpu()
            snapshot_fn = self._make_buffer_snapshot_fn(req)
            prefix_len, new_node = radix_cache.insert(key, value)
            if new_node is not None and new_node.buffer_idx is None:
                snapshot_fn(new_node)
            old_prefix_len = 0 if req.shared_kv_node is None else req.shared_kv_node.node_prefix_total_len
            if req.shared_kv_node is not None:
                radix_cache.dec_node_ref_counter(req.shared_kv_node)
            radix_cache.add_node_ref_counter(new_node)
            req.shared_kv_node = new_node
            if old_prefix_len < prefix_len:
                self.req_manager.free_token(
                    self.req_manager.req_to_token_indexs[req.req_idx][old_prefix_len:prefix_len]
                )
```

- [ ] **Step 5: Commit**

```bash
git add lightllm/server/router/model_infer/infer_batch.py
git commit -m "feat: replace snapshot methods with deterministic _maybe_snapshot_unbuffered_prefix and _maybe_snapshot_full_input"
```

---

### Task 7: Simplify request completion and pause paths

**Files:**
- Modify: `lightllm/server/router/model_infer/infer_batch.py:193-301`

- [ ] **Step 1: Simplify _insert_kv_for_mamba — remove extra_need_to_free_token_index drain**

In `_insert_kv_for_mamba` (lines 193-218), remove lines 213-216:

```python
        # Drain deferred frees from hotspot buffer insertion
        if len(req.extra_need_to_free_token_index) > 0:
            free_token_index.extend(req.extra_need_to_free_token_index)
            req.extra_need_to_free_token_index = []
```

The `extra_need_to_free_token_index` field no longer exists.

- [ ] **Step 2: Simplify _free_req_mem_and_buffers — remove buffer attachment at completion**

In `_free_req_mem_and_buffers` (lines 224-258), replace:

```python
    def _free_req_mem_and_buffers(self, free_token_index: List, free_buffer_index: List, req: "InferReq"):
        """释放请求的 KV cache 和 buffer 内存"""
        if self.has_recurrent_state:
            node = self._insert_kv_for_mamba(free_token_index, req)
            req_to_buffer_index = self.req_manager.req_to_buffer_index
            cpu_mgr = getattr(self.req_manager, "cpu_buffer_mem_manager", None)

            if node is not None and node.buffer_idx is None:
                if cpu_mgr is not None:
                    # Evict existing CPU buffers if needed, then allocate
                    self.radix_cache.free_radix_cache_to_get_enough_buffer(1)
                    if cpu_mgr.can_use_mem_size >= 1:
                        cpu_slot = cpu_mgr.alloc(1)
                        primary_buffer_idx = req_to_buffer_index[req.req_idx, 0].item()
                        gpu_idx = torch.tensor([primary_buffer_idx], dtype=torch.int64, device="cuda")
                        cpu_mgr.offload_to_cpu(
                            self.req_manager.buffer_mem_manager.conv_state_cache.buffer,
                            self.req_manager.buffer_mem_manager.ssm_state_cache.buffer,
                            gpu_idx,
                            cpu_slot,
                        )
                        # No sync_transfer() needed: load_to_gpu flushes pending scatter
                        # and runs on the same transfer stream, so CUDA stream ordering
                        # guarantees the DMA completes before any subsequent read.
                        self.radix_cache.add_buffer_idx_to_node(node, cpu_slot[0].item(), is_hotspot=False)
                    # GPU buffer is freed back to working pool regardless
                    free_buffer_index.extend(req_to_buffer_index[req.req_idx, :].tolist())
                else:
                    # Original path: transfer GPU buffer ownership to tree
                    primary_buffer_idx = req_to_buffer_index[req.req_idx, 0].item()
                    self.radix_cache.add_buffer_idx_to_node(node, primary_buffer_idx, is_hotspot=False)
            else:
                free_buffer_index.extend(req_to_buffer_index[req.req_idx, :].tolist())
        else:
            self.free_a_req_mem(free_token_index, req)
```

With:

```python
    def _free_req_mem_and_buffers(self, free_token_index: List, free_buffer_index: List, req: "InferReq"):
        """释放请求的 KV cache 和 buffer 内存"""
        if self.has_recurrent_state:
            self._insert_kv_for_mamba(free_token_index, req)
            req_to_buffer_index = self.req_manager.req_to_buffer_index
            # Buffer was already attached at snapshot moments — just free GPU buffer
            free_buffer_index.extend(req_to_buffer_index[req.req_idx, :].tolist())
        else:
            self.free_a_req_mem(free_token_index, req)
```

- [ ] **Step 3: Simplify _pause_req_mem_and_buffers**

Replace `_pause_req_mem_and_buffers` (lines 260-301):

```python
    def _pause_req_mem_and_buffers(self, free_token_index: List, free_buffer_index: List, req: "InferReq"):
        """Free request memory during pause, transferring the Mamba buffer to the
        tree node so that recovery can find it via match_prefix.

        Without this, pause inserts KV tokens into the radix tree but attaches
        no buffer.  On recovery, match_prefix walks back and finds no buffer,
        returns None, and the request must fully re-prefill.
        """
        if not self.has_recurrent_state or self.radix_cache is None:
            self._free_req_mem_and_buffers(free_token_index, free_buffer_index, req)
            return

        node = self._insert_kv_for_mamba(free_token_index, req)

        # Transfer primary buffer to the tree node as hotspot
        req_to_buffer_index = self.req_manager.req_to_buffer_index
        cpu_mgr = getattr(self.req_manager, "cpu_buffer_mem_manager", None)

        if node is not None and req.cur_kv_len > 0:
            primary_buffer_idx = req_to_buffer_index[req.req_idx, 0].item()
            if cpu_mgr is not None:
                # Evict existing CPU buffers if needed, then allocate
                self.radix_cache.free_radix_cache_to_get_enough_buffer(1)
                if cpu_mgr.can_use_mem_size >= 1:
                    cpu_slot = cpu_mgr.alloc(1)
                    gpu_idx = torch.tensor([primary_buffer_idx], dtype=torch.int64, device="cuda")
                    cpu_mgr.offload_to_cpu(
                        self.req_manager.buffer_mem_manager.conv_state_cache.buffer,
                        self.req_manager.buffer_mem_manager.ssm_state_cache.buffer,
                        gpu_idx,
                        cpu_slot,
                    )
                    # No sync_transfer() needed: same reasoning as _free_req_mem_and_buffers.
                    self.radix_cache.add_buffer_idx_to_node(node, cpu_slot[0].item(), is_hotspot=True)
                free_buffer_index.extend(req_to_buffer_index[req.req_idx, :].tolist())
            else:
                self.radix_cache.add_buffer_idx_to_node(node, primary_buffer_idx, is_hotspot=True)
                # Free only the non-primary buffers; primary is now owned by the tree
                free_buffer_index.extend(req_to_buffer_index[req.req_idx, 1:].tolist())
        else:
            # No valid node — free all buffers normally
            free_buffer_index.extend(req_to_buffer_index[req.req_idx, :].tolist())
```

With:

```python
    def _pause_req_mem_and_buffers(self, free_token_index: List, free_buffer_index: List, req: "InferReq"):
        """Free request memory during pause. Buffer is already on the tree node
        from the snapshot moments, so no transfer needed."""
        if not self.has_recurrent_state or self.radix_cache is None:
            self._free_req_mem_and_buffers(free_token_index, free_buffer_index, req)
            return

        self._insert_kv_for_mamba(free_token_index, req)
        req_to_buffer_index = self.req_manager.req_to_buffer_index
        free_buffer_index.extend(req_to_buffer_index[req.req_idx, :].tolist())
```

- [ ] **Step 4: Commit**

```bash
git add lightllm/server/router/model_infer/infer_batch.py
git commit -m "refactor: simplify request completion and pause paths — buffers attached at snapshot moments"
```

---

### Task 8: Update backend call sites

**Files:**
- Modify: `lightllm/server/router/model_infer/mode_backend/chunked_prefill/impl.py:53-61`
- Modify: `lightllm/server/router/model_infer/mode_backend/dp_backend/impl.py:65-73`

- [ ] **Step 1: Update chunked_prefill backend**

In `lightllm/server/router/model_infer/mode_backend/chunked_prefill/impl.py`, replace `_maybe_snapshot_hybrid_buffers` (lines 53-61):

```python
    def _maybe_snapshot_hybrid_buffers(self, run_reqs: List[InferReq]):
        """Snapshot Mamba states after prefill. Handles both hotspot and opportunistic inserts."""
        if g_infer_context.has_recurrent_state and self.radix_cache is not None:
            g_infer_state_lock.acquire()
            try:
                g_infer_context.snapshot_hybrid_buffers(run_reqs)
                g_infer_context.snapshot_prefill_complete_buffers(run_reqs)
            finally:
                g_infer_state_lock.release()
```

With:

```python
    def _maybe_snapshot_hybrid_buffers(self, run_reqs: List[InferReq]):
        """Snapshot Mamba states after prefill — unbuffered prefix and full input."""
        if g_infer_context.has_recurrent_state and self.radix_cache is not None:
            g_infer_state_lock.acquire()
            try:
                g_infer_context._maybe_snapshot_unbuffered_prefix(run_reqs)
                g_infer_context._maybe_snapshot_full_input(run_reqs)
            finally:
                g_infer_state_lock.release()
```

- [ ] **Step 2: Update dp_backend**

In `lightllm/server/router/model_infer/mode_backend/dp_backend/impl.py`, apply the same replacement to `_maybe_snapshot_hybrid_buffers` (lines 65-73):

```python
    def _maybe_snapshot_hybrid_buffers(self, run_reqs: List[InferReq]):
        """Snapshot Mamba states after prefill — unbuffered prefix and full input."""
        if g_infer_context.has_recurrent_state and self.radix_cache is not None:
            g_infer_state_lock.acquire()
            try:
                g_infer_context._maybe_snapshot_unbuffered_prefix(run_reqs)
                g_infer_context._maybe_snapshot_full_input(run_reqs)
            finally:
                g_infer_state_lock.release()
```

- [ ] **Step 3: Commit**

```bash
git add lightllm/server/router/model_infer/mode_backend/chunked_prefill/impl.py lightllm/server/router/model_infer/mode_backend/dp_backend/impl.py
git commit -m "refactor: update backend call sites to use new snapshot methods"
```

---

### Task 9: Final validation

- [ ] **Step 1: Run all radix cache unit tests**

Run: `python -m pytest unit_tests/server/router/dynamic_prompt/ -v`
Expected: All tests PASS

- [ ] **Step 2: Run full unit test suite**

Run: `python -m pytest unit_tests/ -v`
Expected: All tests PASS (or only pre-existing failures unrelated to this change)

- [ ] **Step 3: Verify no stale references to removed fields**

Search for any remaining references to removed fields:

```bash
grep -rn "is_hotspot\|was_hit\|min_insert_threshold\|buffer_insert_count\|buffer_hit_count\|buffer_waste_count\|_last_miss_prefix_len\|mamba_buffer_target_len\|is_hotspot_prefill\|extra_need_to_free_token_index\|available_opportunistic_buffer_count\|snapshot_hybrid_buffers\|snapshot_prefill_complete_buffers" lightllm/ --include="*.py"
```

Expected: No matches (all references removed)

- [ ] **Step 4: Commit any fixes from validation**

If any stale references found, fix them and commit:

```bash
git add -A
git commit -m "fix: remove stale references to removed heuristic fields"
```
