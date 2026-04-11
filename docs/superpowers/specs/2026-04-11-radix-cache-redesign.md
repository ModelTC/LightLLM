# Radix Cache Insertion Redesign

## Problem

The current `HybridRadixCache` insertion logic is overly complex. It uses adaptive
thresholds, waste ratio tracking, hotspot heuristics, and opportunistic buffering to
decide when and where to snapshot Mamba state into the radix tree. This complexity led
to multiple crash-fix-revert cycles and makes the code hard to reason about.

In practice, for hybrid models like Qwen 3.5 with thinking mode, only **input tokens**
are cacheable. There are exactly two caching scenarios:

1. **System prompt sharing** — multiple requests reuse the same system prompt prefix.
2. **Full input sharing** — a single user's complete input is cached for multi-turn
   conversation reuse.

Both require KV cache *and* Mamba state buffer to be useful. KV cache without a buffer
is not recoverable for hybrid models.

## Design: Deterministic Two-Point Insertion

Replace all heuristic insertion logic with two deterministic insertion moments, both
producing KV + buffer snapshots.

### Approach

Single radix tree with an explicit insertion contract. The tree keeps its prefix-sharing
capability, but callers interact through a simplified API that guarantees buffer
attachment at the right positions.

### New API: `insert_with_buffer`

```python
def insert_with_buffer(self, key, value, unbuffered_prefix_pos, buffer_snapshot_fn):
    """
    Insert tokens into the radix tree, ensuring a buffer exists at
    unbuffered_prefix_pos.

    Args:
        key: full input token ids (int64 tensor)
        value: corresponding KV cache memory indices (int64 tensor)
        unbuffered_prefix_pos: token position of a prefix node that has KV but
                               no Mamba buffer (from match_prefix walk-back).
                               0 means no mid-prefill snapshot needed.
        buffer_snapshot_fn: callable(node) -> None
                            Called to snapshot Mamba state for a node that
                            needs a buffer. The cache calls this for any node
                            that lacks a buffer after insertion.

    Returns:
        (prefix_len, leaf_node)

    Guarantees:
        - If unbuffered_prefix_pos > 0, a node split exists at that position
          and buffer_snapshot_fn is called for it (if it has no buffer).
        - buffer_snapshot_fn is called for the leaf node (full input).
    """
```

The `buffer_snapshot_fn` callback separates cache logic from memory management. The
cache never touches GPU/CPU memory managers directly.

### Modified API: `match_prefix`

```python
def match_prefix(self, key, update_refs=False):
    """
    Returns:
        (node, kv_len, value_tensor, unbuffered_prefix_pos)

        - node: deepest matched node that has a buffer (or None)
        - kv_len: token count shared from cache (= node.node_prefix_total_len)
        - value_tensor: KV cache indices for matched tokens
        - unbuffered_prefix_pos: if walk-back occurred, the token position of
                                 the deepest bufferless node we walked past.
                                 0 if no walk-back was needed.
    """
```

Changes from current:
- `_last_miss_prefix_len` stashed on the cache object is replaced by
  `unbuffered_prefix_pos` returned directly (no hidden mutable state).

### Insertion Moments

#### Moment 1: Mid-prefill unbuffered prefix snapshot

**When:** During prefill, when `cur_kv_len >= unbuffered_prefix_pos` and
`unbuffered_prefix_pos > 0`.

**What:** Insert `key[0:unbuffered_prefix_pos]` into the tree with a buffer snapshot
at the system prompt boundary.

Replaces `snapshot_hybrid_buffers`. Backend check:

```python
def _maybe_snapshot_unbuffered_prefix(self, run_reqs):
    reqs = [r for r in run_reqs
            if r.unbuffered_prefix_pos > 0
            and r.cur_kv_len >= r.unbuffered_prefix_pos]
    if not reqs:
        return
    for req in reqs:
        # insert key[0:unbuffered_prefix_pos] with buffer
        # update req.shared_kv_node, clear unbuffered_prefix_pos
        req.unbuffered_prefix_pos = 0
```

No threshold gating, no buffer budget checks.

#### Moment 2: Post-prefill full input snapshot

**When:** After prefill completes (before first decode token), always.

**What:** Insert `key[0:cur_kv_len]` into the tree with a buffer.

Replaces `snapshot_prefill_complete_buffers`. Check:

```python
def _maybe_snapshot_full_input(self, run_reqs):
    reqs = [r for r in run_reqs
            if r.cur_kv_len + 1 >= r.get_cur_total_len()
            and r.cur_output_len == 0]
    if not reqs:
        return
    for req in reqs:
        # insert key[0:cur_kv_len] with buffer via insert_with_buffer
```

#### Other insertion paths

| Current path | Redesign |
|---|---|
| `snapshot_hybrid_buffers` (mid-prefill hotspot) | Replaced by `_maybe_snapshot_unbuffered_prefix` |
| `snapshot_prefill_complete_buffers` (post-prefill opportunistic) | Replaced by `_maybe_snapshot_full_input` |
| `_free_req_mem_and_buffers` (request completion) | Keeps KV insertion, no new buffer snapshot |
| `_pause_req_mem_and_buffers` (request pause) | Keeps KV insertion, no new buffer snapshot. The buffer is already on the tree node from moment 1 or 2, so pause no longer needs to transfer the request's GPU buffer to the tree. |

### Thinking Mode

With thinking mode enabled, output/thinking tokens are not cacheable. The redesign
handles this implicitly:

- Moment 1 (system prompt boundary) fires during prefill of input tokens only.
- Moment 2 (full input) fires right after input prefill, before thinking begins.
- At request completion, no insertion of output/thinking tokens (same as current).

`disable_radix_cache_insert` remains respected at completion time but is no longer
relevant for the two snapshot moments.

### Buffer Allocation via Callback

The `buffer_snapshot_fn` callback encapsulates GPU/CPU memory management:

```python
def make_buffer_snapshot_fn(req, cpu_mgr, gpu_buffer_mgr, cache):
    def snapshot(node):
        if node.buffer_idx is not None:
            return  # already has a buffer

        primary_buffer_idx = req_to_buffer_index[req.req_idx, 0].item()

        if cpu_mgr is not None:
            cpu_slot = cpu_mgr.alloc(1)
            gpu_idx = torch.tensor([primary_buffer_idx], ...)
            cpu_mgr.offload_to_cpu(..., gpu_idx, cpu_slot)
            cache.add_buffer_idx_to_node(node, cpu_slot[0].item())
        else:
            new_buf = gpu_buffer_mgr.alloc(1)
            gpu_buffer_mgr.copy_state_buffers(cur, new_buf)
            cache.add_buffer_idx_to_node(node, new_buf[0].item())

    return snapshot
```

### Eviction Simplification

**Unchanged:**
- `evict_tree_set` (SortedSet for KV token eviction)
- Two-pass buffer eviction (unreferenced first, then referenced)

**Simplified:**
- `evict_buffer_set` sort key: `(is_hotspot, buffer_time)` becomes `(buffer_time)` only.
  All buffers are equal; eviction is purely LRU.

**Node lifecycle:**

```
Node created (insert) -> has KV, no buffer
    |
buffer_snapshot_fn called -> has KV + buffer
    |
Request releases ref -> ref_counter drops, enters eviction set
    |
Buffer eviction -> KV stays, buffer freed
    |
Tree eviction -> KV freed, node removed
```

### Request State Changes

On `InferReq`:
- `is_hotspot_prefill` -> removed, replaced by `unbuffered_prefix_pos > 0`
- `mamba_buffer_target_len` -> removed, replaced by `unbuffered_prefix_pos`

In `_match_radix_cache`:
```python
node, kv_len, value_tensor, unbuffered_prefix_pos = cache.match_prefix(key, update_refs=True)
if node is not None:
    req.shared_kv_node = node
    req.cur_kv_len = kv_len
    req.unbuffered_prefix_pos = unbuffered_prefix_pos
```

## Removal Inventory

### From `TreeNode`:
- `is_hotspot: bool`
- `was_hit: bool`

### From `HybridRadixCache`:
- `min_insert_threshold`
- `buffer_insert_count`
- `buffer_hit_count`
- `buffer_waste_count`
- `available_opportunistic_buffer_count()`
- Adaptive threshold tuning logic
- `_last_miss_prefix_len` stashed state

### From `infer_batch.py`:
- `snapshot_hybrid_buffers()`
- `snapshot_prefill_complete_buffers()`
- `is_hotspot_prefill` on `InferReq`
- `mamba_buffer_target_len` on `InferReq`
- Hotspot threshold checks in `_match_radix_cache()`
- `extra_need_to_free_token_index` deferred free list

### From eviction sort key:
- `is_hotspot` field removed from `evict_buffer_set` ordering

## Unchanged Components

- `RadixCache` base class (tree structure, node splitting, merging, ref counting, KV eviction)
- `match_prefix` walk-back logic in `HybridRadixCache` (returns `unbuffered_prefix_pos` instead of stashing it)
- Two-pass buffer eviction (unreferenced first, then referenced)
- Backend call sites in `chunked_prefill/impl.py` and `dp_backend/impl.py` (renamed methods, same pipeline positions)
- Request completion / pause KV insertion paths
- `RadixCacheReadOnlyClient` shared memory interface
