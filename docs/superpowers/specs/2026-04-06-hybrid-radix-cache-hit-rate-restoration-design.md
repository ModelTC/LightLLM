# Design: Hybrid Radix Cache Hit Rate Restoration

## Problem

The current `qw35_oom` branch simplified the hybrid radix cache from `origin/qwen3next_last`, causing a noticeable hit rate drop. Four optimizations were removed:

1. **R1**: Mid-prefill buffer insertion (`insert_for_hybrid_radix_cache`)
2. **R2**: Hotspot-aware chunked prefill (`miss_prefix_len` + `mamba_buffer_insert_len`)
3. **R3**: Aggressive cleanup of buffer-less leaves during `match_prefix` walk-back
4. **R4**: Cascading cleanup when buffers are evicted via `_evict_buffer`

## Critical Insight: KV Cache Without Buffer Is Useless

In a hybrid model, when `match_prefix` walks back to the nearest buffer, the KV cache in the miss region (between the buffer and the deepest tree match) provides **zero computational benefit**. The request must recompute ALL layers (attention + Mamba) from the buffer position. Those orphaned KV tokens are 100% wasted GPU memory.

This makes R3 (walk-back cleanup) essential for memory correctness, not just optimization.

## Design

### 1. Startup-Computed Threshold

The threshold for buffer insertion is the cost-equivalence point: how many KV tokens' worth of memory does one buffer slot cost?

```
buffer_cell_size = conv_cell_size + ssm_cell_size
kv_per_token_size = 2 * num_kv_heads * head_dim * num_full_attn_layers * dtype_size

min_insert_threshold = buffer_cell_size // kv_per_token_size
```

A buffer is worth caching when `miss_prefix_len > min_insert_threshold`. Below that, the memory is better spent on KV cache capacity.

**Computed at startup** from model architecture parameters already available in `Qwen3NextHybridMemManager` and `MambaCacheManager`:

- `Qwen3NextHybridMemManager.get_cell_size()` returns `kv_per_token_size`
- `MambaCacheManager` has `conv_state_cache.get_cell_size()` and `ssm_state_cache.get_cell_size()` for `buffer_cell_size`

**Where to store**: On `HybridRadixCache` as `self.min_insert_threshold`, computed during `__init__` from the mem managers.

### 2. `match_prefix` — Return `miss_prefix_len` + Walk-Back Cleanup (R3)

**Return value change**: Replace the redundant `kv_len` (equals `node.node_prefix_total_len`) with `miss_prefix_len` (the gap size). Callers already use `share_node.node_prefix_total_len` for the usable prefix length.

**Walk-back cleanup (R3)**: When `update_refs=True` and a walked-back node becomes an unreferenced leaf, destroy it and free its KV tokens immediately. These tokens are proven useless (no buffer = no computational benefit).

```python
def match_prefix(self, key, update_refs=False):
    assert len(key) != 0
    ans_value_list = []
    tree_node = self._match_prefix_helper(self.root_node, key, ans_value_list, update_refs=update_refs)
    miss_prefix_len = 0
    evict_token_list = []
    while tree_node != self.root_node and tree_node.buffer_idx is None:
        if tree_node.is_leaf():
            self.evict_tree_set.discard(tree_node)

        if update_refs:
            if tree_node.ref_counter == 1:
                self.refed_tokens_num.arr[0] -= len(tree_node.token_mem_index_value)
            tree_node.ref_counter -= 1

        # R3: destroy orphaned buffer-less leaves — their KV tokens are 100% waste
        if update_refs and tree_node.is_leaf() and tree_node.ref_counter == 0:
            evict_token_list.append(tree_node.token_mem_index_value)
            self.tree_total_tokens_num.arr[0] -= len(tree_node.token_mem_index_value)
            parent_node = tree_node.parent
            parent_node.remove_child(tree_node)
            if parent_node.is_leaf():
                self.evict_tree_set.add(parent_node)
            tree_node = parent_node
        else:
            if tree_node.is_leaf():
                self.evict_tree_set.add(tree_node)
            tree_node = tree_node.parent

        miss_prefix_len += len(ans_value_list.pop())

    if len(evict_token_list) > 0:
        evict_token_value = torch.concat(evict_token_list)
        self.mem_manager.free(evict_token_value)

    if tree_node == self.root_node:
        return None, miss_prefix_len, None

    # Refresh buffer LRU times along the matched path
    update_node = tree_node
    while update_node != self.root_node:
        if update_node.buffer_idx is not None:
            self.evict_buffer_set.discard(update_node)
            update_node.update_buffer_time()
            self.evict_buffer_set.add(update_node)
        update_node = update_node.parent

    value = torch.concat(ans_value_list)
    return tree_node, miss_prefix_len, value
```

### 3. `_evict_buffer` — Cascading Cleanup (R4)

When a buffer is evicted from a node, and the node is an unreferenced leaf, the node and its KV tokens are useless (same reasoning as R3). Destroy them.

**Signature change**: two callbacks (`evict_buffer_callback` + `evict_token_callback`).

```python
def _evict_buffer(self, need_evict_buffer_num, evict_buffer_callback, evict_token_callback):
    while need_evict_buffer_num > 0:
        node = self.evict_buffer_set.pop(0)
        assert node.buffer_idx is not None
        evict_buffer_callback(node.buffer_idx)
        node.buffer_idx = None
        node.is_hotspot = False
        need_evict_buffer_num -= 1
        # R4: buffer-less unreferenced leaf is dead weight
        if node.is_leaf() and node.ref_counter == 0:
            self.evict_tree_set.discard(node)
            evict_token_callback(node.token_mem_index_value)
            self.tree_total_tokens_num.arr[0] -= len(node.token_mem_index_value)
            parent_node = node.parent
            parent_node.remove_child(node)
            if parent_node.is_leaf():
                self.evict_tree_set.add(parent_node)
```

`free_radix_cache_to_get_enough_buffer` passes both callbacks:

```python
def free_radix_cache_to_get_enough_buffer(self, need_buffer_num):
    if need_buffer_num > self.buffer_mem_manager.can_use_mem_size:
        need_evict_buffer_num = need_buffer_num - self.buffer_mem_manager.can_use_mem_size
        release_buffers = []
        release_mems = []

        self._evict_buffer(need_evict_buffer_num,
                           lambda idx: release_buffers.append(idx),
                           lambda mem: release_mems.append(mem))
        self.buffer_mem_manager.free(release_buffers)
        if len(release_mems) > 0:
            self.mem_manager.free(torch.concat(release_mems))
```

### 4. Hotspot Eviction Resistance

Buffers inserted at natural reuse boundaries (multi-turn conversation history, system prompts) should be harder to evict than buffers planted at request completion.

**TreeNode**: Add `is_hotspot: bool = False` field.

**`evict_buffer_set` sort key**: Change from `(buffer_time,)` to `(is_hotspot, buffer_time)`.

- Non-hotspot `(False, time)` — evicted first
- Hotspot `(True, time)` — evicted last, LRU within tier

```python
self.evict_buffer_set: Set[TreeNode] = SortedSet(key=lambda x: (x.is_hotspot, x.buffer_time))
```

**`add_buffer_idx_to_node`**: Accepts `is_hotspot` parameter.

```python
def add_buffer_idx_to_node(self, node: TreeNode, buffer_idx: int, is_hotspot: bool = False):
    self.evict_buffer_set.discard(node)
    if node.is_leaf():
        self.evict_tree_set.discard(node)
    if node.buffer_idx is not None:
        self.buffer_mem_manager.free([node.buffer_idx])
    node.buffer_idx = buffer_idx
    node.is_hotspot = is_hotspot
    node.update_buffer_time()
    self.evict_buffer_set.add(node)
    if node.is_leaf():
        self.evict_tree_set.add(node)
```

### 5. Targeted Buffer Insertion (Replaces R1)

**No mid-chunk insertion.** Buffers are only inserted in two high-ROI scenarios:

1. **System prompt miss**: system prompt has tree structure but no buffer (every request benefits)
2. **Multi-turn conversation return**: previous conversation history has tree structure but no buffer

Both manifest identically: `match_prefix` returns `share_node is not None` with `miss_prefix_len > min_insert_threshold`.

**Method on `InferenceContext`**: `snapshot_hybrid_buffers(run_reqs)`.

Only processes requests with `is_hotspot_prefill=True`. Called once after the enlarged first chunk, not after every chunk.

```python
def snapshot_hybrid_buffers(self, run_reqs: List[InferReq]):
    """Snapshot Mamba states for hotspot requests after their enlarged first chunk."""
    reqs_to_insert = [r for r in run_reqs
                       if r.is_hotspot_prefill and r.cur_kv_len < r.get_cur_total_len()]
    if not reqs_to_insert:
        return

    radix_cache: HybridRadixCache = self.radix_cache
    radix_cache.free_radix_cache_to_get_enough_buffer(len(reqs_to_insert))

    # Allocate fresh buffer slots and snapshot current Mamba states
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

        # Deferred free for overlapping KV token indices
        req.extra_need_to_free_token_index.append(
            self.req_manager.req_to_token_indexs[req.req_idx][old_prefix_len:prefix_len]
        )
        req.shared_kv_node = new_node
        req.is_hotspot_prefill = False  # one-shot: don't re-insert on subsequent chunks
```

### 6. Hotspot Detection in `_match_radix_cache` (R2)

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

### 7. First-Chunk Enlargement in `get_chuncked_input_token_len`

```python
def get_chuncked_input_token_len(self):
    chunked_start = self.cur_kv_len
    chunked_end = min(self.get_cur_total_len(), chunked_start + self.shm_req.chunked_prefill_size)
    if self.mamba_buffer_insert_len > 0:
        chunked_end = min(self.get_cur_total_len(), chunked_start + self.mamba_buffer_insert_len)
        self.mamba_buffer_insert_len = 0  # one-shot
    return chunked_end
```

Also update `get_chuncked_input_token_ids` to stay consistent:

```python
def get_chuncked_input_token_ids(self):
    chunked_end = self.get_chuncked_input_token_len()
    return self.shm_req.shm_prompt_ids.arr[0:chunked_end]
```

### 8. `InferReq` Field Additions

```python
self.mamba_buffer_insert_len = 0
self.is_hotspot_prefill = False
self.extra_need_to_free_token_index = []
```

### 9. Deferred Free in `free_a_req_mem_for_mamba`

Drain the deferred free list accumulated by `snapshot_hybrid_buffers`:

```python
def free_a_req_mem_for_mamba(self, free_token_index, req):
    # ... existing code ...
    if self.radix_cache is not None:
        # ... existing insert + buffer attach logic ...

        # Drain deferred frees from mid-prefill buffer insertion
        if len(req.extra_need_to_free_token_index) > 0:
            free_token_index.extend(req.extra_need_to_free_token_index)
            req.extra_need_to_free_token_index = []

        # ... rest of existing logic ...
```

### 10. Integration in Chunked Prefill Backend

**Synchronization**: follows the original proven pattern. The snapshot runs between Stage 3 and Stage 4, after `sync_event.synchronize()` + `_post_handle()`, before `notify_pre_post_handle()`.

```python
def _maybe_snapshot_hybrid_buffers(self, run_reqs: List[InferReq]):
    """Snapshot Mamba states for hotspot requests. Called after prefill, before next iteration."""
    if g_infer_context.has_recurrent_state and self.radix_cache is not None:
        torch.cuda.synchronize()
        g_infer_state_lock.acquire()
        g_infer_context.snapshot_hybrid_buffers(run_reqs)
        g_infer_state_lock.release()
```

**Placement in `prefill_normal`**:

```python
def prefill_normal(self, event_pack, prefill_reqs):
    # Stage 1: GPU forward
    model_input, run_reqs = prepare_prefill_inputs(prefill_reqs, is_chuncked_mode=...)
    with torch.cuda.stream(g_infer_context.get_overlap_stream()):
        model_output = self.model.forward(model_input)
        ...
        sync_event = torch.cuda.Event()
        sync_event.record()

    # Stage 2: pre-post handle (updates cur_kv_len)
    event_pack.notify_post_handle_and_wait_pre_post_handle()
    update_packs = self._pre_post_handle(run_reqs, is_chuncked_mode=...)

    # Stage 3: post handle (token finalization)
    event_pack.notify_forward_and_wait_post_handle()
    sync_event.synchronize()
    self._post_handle(run_reqs=run_reqs, ...)

    # Buffer snapshot for hotspot requests (between Stage 3 and Stage 4)
    self._maybe_snapshot_hybrid_buffers(run_reqs)

    # Stage 4: signal next iteration
    event_pack.notify_pre_post_handle()
```

Applied to both `prefill_normal` and `prefill_mtp`.

## Files Changed

| File | Changes |
|---|---|
| `radix_cache.py` | Add `is_hotspot` field to `TreeNode` |
| `hybrid_radix_cache.py` | `match_prefix` returns `miss_prefix_len` + R3 cleanup; `_evict_buffer` R4 cascading cleanup; hotspot-aware `evict_buffer_set` sort key; `add_buffer_idx_to_node` with `is_hotspot`; `free_radix_cache_to_get_enough_buffer` dual callbacks; `min_insert_threshold` computed at init |
| `infer_batch.py` | `InferReq` fields (`mamba_buffer_insert_len`, `is_hotspot_prefill`, `extra_need_to_free_token_index`); `_match_radix_cache` hotspot detection; `get_chuncked_input_token_len` + `get_chuncked_input_token_ids` enlargement; `snapshot_hybrid_buffers` on `InferenceContext`; `free_a_req_mem_for_mamba` deferred free |
| `chunked_prefill/impl.py` | `_maybe_snapshot_hybrid_buffers` method; calls in `prefill_normal` and `prefill_mtp` |

## Key Differences from Original `origin/qwen3next_last`

| Aspect | Original | This Design |
|---|---|---|
| Buffer insertion timing | After every prefill chunk | Only for hotspot requests, one-shot after enlarged first chunk |
| Threshold | Magic numbers (128, 1024) | `buffer_cell_size // kv_per_token_size`, computed at startup |
| Buffer copy API | `copy_buffer_p2p` (PyTorch indexing) | `copy_state_buffers` (Triton kernels) |
| `match_prefix` ref counting | Unconditional decrement (bug) | Conditional on `update_refs` (correct) |
| Walk-back cleanup | Always destroys orphans | Destroys orphans only when `update_refs=True` |
| Eviction priority | Uniform LRU | Hotspot-aware: non-hotspot evicted first |
| `insert_for_hybrid_radix_cache` location | On `HybridRadixCache` (circular import) | On `InferenceContext` (clean dependency) |
| Cache architecture awareness | None | `HybridRadixCache` stays pure; request logic in `InferenceContext` |
