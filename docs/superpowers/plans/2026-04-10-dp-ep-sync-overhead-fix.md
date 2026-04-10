# DP+EP Synchronization Overhead Fix

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Eliminate redundant GPU synchronization and CPU-blocking transfers in the DP backend's mamba cache snapshot path, which cause periodic GPU utilization drops to near-zero.

**Architecture:** The fix targets five independent bottlenecks: (1) remove redundant `torch.cuda.synchronize()` in `_maybe_snapshot_hybrid_buffers`, (2) defer `sync_transfer()` in snapshot paths so GPU->CPU PCIe transfers overlap with subsequent decode work, (3) batch per-element Python loops in CPU cache transfers into single advanced-indexing operations, (4) merge two blocking all_gather calls into one, (5) remove dead code.

**Tech Stack:** PyTorch CUDA streams, NCCL collectives, Python

---

### Task 1: Remove redundant `torch.cuda.synchronize()` from `_maybe_snapshot_hybrid_buffers`

**Files:**
- Modify: `lightllm/server/router/model_infer/mode_backend/dp_backend/impl.py:65-74`
- Modify: `lightllm/server/router/model_infer/mode_backend/chunked_prefill/impl.py:53-62`

**Context:** `_maybe_snapshot_hybrid_buffers` is called after every prefill at 6 call sites (dp_backend lines 189, 300, 424, 726; chunked_prefill lines 151, 235). At every call site, `sync_event.synchronize()` has already completed, meaning all model forward GPU work is finished. The `torch.cuda.synchronize()` inside `_maybe_snapshot_hybrid_buffers` is a full GPU pipeline drain that does nothing useful — the GPU is already idle. Removing it eliminates a blocking syscall on every prefill completion.

- [ ] **Step 1: Edit dp_backend/impl.py to remove the redundant sync**

In `lightllm/server/router/model_infer/mode_backend/dp_backend/impl.py`, change `_maybe_snapshot_hybrid_buffers` from:

```python
def _maybe_snapshot_hybrid_buffers(self, run_reqs: List[InferReq]):
    """Snapshot Mamba states after prefill. Handles both hotspot and opportunistic inserts."""
    if g_infer_context.has_recurrent_state and self.radix_cache is not None:
        torch.cuda.synchronize()
        g_infer_state_lock.acquire()
        try:
            g_infer_context.snapshot_hybrid_buffers(run_reqs)
            g_infer_context.snapshot_prefill_complete_buffers(run_reqs)
        finally:
            g_infer_state_lock.release()
```

to:

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

- [ ] **Step 2: Edit chunked_prefill/impl.py to remove the redundant sync**

In `lightllm/server/router/model_infer/mode_backend/chunked_prefill/impl.py`, change `_maybe_snapshot_hybrid_buffers` from:

```python
def _maybe_snapshot_hybrid_buffers(self, run_reqs: List[InferReq]):
    """Snapshot Mamba states after prefill. Handles both hotspot and opportunistic inserts."""
    if g_infer_context.has_recurrent_state and self.radix_cache is not None:
        torch.cuda.synchronize()
        g_infer_state_lock.acquire()
        try:
            g_infer_context.snapshot_hybrid_buffers(run_reqs)
            g_infer_context.snapshot_prefill_complete_buffers(run_reqs)
        finally:
            g_infer_state_lock.release()
```

to:

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

- [ ] **Step 3: Verify no other code path calls `_maybe_snapshot_hybrid_buffers` without prior sync**

Run: `grep -rn "_maybe_snapshot_hybrid_buffers" lightllm/`

Verify that every call site has a `sync_event.synchronize()` before it. The expected call sites are:
- `dp_backend/impl.py:189` (after sync at line 179)
- `dp_backend/impl.py:300` (after sync at line 289)
- `dp_backend/impl.py:424` (after sync at line 413)
- `dp_backend/impl.py:726` (after sync at line 715)
- `chunked_prefill/impl.py:151` (after sync at line 141)
- `chunked_prefill/impl.py:235` (after sync at line 224)

- [ ] **Step 4: Commit**

```bash
git add lightllm/server/router/model_infer/mode_backend/dp_backend/impl.py \
        lightllm/server/router/model_infer/mode_backend/chunked_prefill/impl.py
git commit -m "perf: remove redundant torch.cuda.synchronize() from _maybe_snapshot_hybrid_buffers

All 6 call sites already have sync_event.synchronize() completed before
calling this method, meaning all GPU forward work is finished. The extra
torch.cuda.synchronize() was a full GPU pipeline drain that blocked the
CPU thread for no benefit."
```

---

### Task 2: Batch per-element CPU cache transfers with advanced indexing

**Files:**
- Modify: `lightllm/common/mamba_cache_mem_manager/cpu_cache_manager.py:193-222`
- Test: `unit_tests/common/mamba_cache_mem_manager/test_cpu_cache_manager.py`

**Context:** `offload_to_cpu` and `load_to_gpu` use Python for-loops iterating per-slot, issuing 2 `copy_()` calls per iteration. With advanced indexing, this can be a single operation per buffer type, reducing Python overhead and CUDA kernel launch count. The CPU buffers are pinned memory, and the GPU buffers use standard CUDA memory, so advanced indexing with `non_blocking=True` works correctly.

- [ ] **Step 1: Write a test for batched transfer correctness**

Add the following test to `unit_tests/common/mamba_cache_mem_manager/test_cpu_cache_manager.py`:

```python
def test_batched_offload_and_load_multiple_slots(self):
    """Verify batch transfer works with multiple non-contiguous GPU indices."""
    mgr = _make_manager()
    gpu_conv, gpu_ssm = self._make_gpu_buffers(mgr)

    gpu_indices = [1, 3, 5, 7]
    cpu_slots = mgr.alloc(4)
    cpu_slot_list = cpu_slots.tolist()

    expected_conv = gpu_conv[:, gpu_indices, ...].clone()
    expected_ssm = gpu_ssm[:, gpu_indices, ...].clone()

    mgr.offload_to_cpu(gpu_conv, gpu_ssm, gpu_indices, cpu_slot_list)
    mgr.sync_transfer()

    # Verify CPU buffers received correct data
    for i, c in enumerate(cpu_slot_list):
        torch.testing.assert_close(
            mgr.conv_state_buffer[:, c, ...],
            expected_conv[:, i, ...].cpu(),
        )
        torch.testing.assert_close(
            mgr.ssm_state_buffer[:, c, ...],
            expected_ssm[:, i, ...].cpu(),
        )

    # Zero GPU slots and restore
    for g in gpu_indices:
        gpu_conv[:, g, ...] = 0
        gpu_ssm[:, g, ...] = 0

    mgr.load_to_gpu(gpu_conv, gpu_ssm, cpu_slot_list, gpu_indices)
    mgr.sync_transfer()

    for i, g in enumerate(gpu_indices):
        torch.testing.assert_close(gpu_conv[:, g, ...], expected_conv[:, i, ...])
        torch.testing.assert_close(gpu_ssm[:, g, ...], expected_ssm[:, i, ...])
```

- [ ] **Step 2: Run test to verify it passes with current implementation**

Run: `cd /workspace/lightllm-worktree/qw35_dpep_fix && python -m pytest unit_tests/common/mamba_cache_mem_manager/test_cpu_cache_manager.py -v -k "test_batched_offload"`
Expected: PASS (the test validates behavior, not implementation details)

- [ ] **Step 3: Replace per-element loops with batched advanced indexing**

In `lightllm/common/mamba_cache_mem_manager/cpu_cache_manager.py`, replace `offload_to_cpu`:

```python
def offload_to_cpu(
    self,
    gpu_conv_state: torch.Tensor,
    gpu_ssm_state: torch.Tensor,
    gpu_buffer_indexes: Union[List[int], torch.Tensor],
    cpu_slot_indexes: Union[List[int], torch.Tensor],
):
    """Async GPU -> CPU copy on a dedicated CUDA stream.

    gpu_conv_state / gpu_ssm_state have shape (layer_num, buffer_size, *state_shape).
    """
    stream = self._get_transfer_stream()
    with torch.cuda.stream(stream):
        self.conv_state_buffer[:, cpu_slot_indexes].copy_(gpu_conv_state[:, gpu_buffer_indexes], non_blocking=True)
        self.ssm_state_buffer[:, cpu_slot_indexes].copy_(gpu_ssm_state[:, gpu_buffer_indexes], non_blocking=True)
```

And replace `load_to_gpu`:

```python
def load_to_gpu(
    self,
    gpu_conv_state: torch.Tensor,
    gpu_ssm_state: torch.Tensor,
    cpu_slot_indexes: Union[List[int], torch.Tensor],
    gpu_buffer_indexes: Union[List[int], torch.Tensor],
):
    """Async CPU -> GPU copy on a dedicated CUDA stream."""
    stream = self._get_transfer_stream()
    with torch.cuda.stream(stream):
        gpu_conv_state[:, gpu_buffer_indexes].copy_(self.conv_state_buffer[:, cpu_slot_indexes], non_blocking=True)
        gpu_ssm_state[:, gpu_buffer_indexes].copy_(self.ssm_state_buffer[:, cpu_slot_indexes], non_blocking=True)
```

- [ ] **Step 4: Run all CPU cache manager tests**

Run: `cd /workspace/lightllm-worktree/qw35_dpep_fix && python -m pytest unit_tests/common/mamba_cache_mem_manager/test_cpu_cache_manager.py -v`
Expected: All tests PASS

- [ ] **Step 5: Commit**

```bash
git add lightllm/common/mamba_cache_mem_manager/cpu_cache_manager.py \
        unit_tests/common/mamba_cache_mem_manager/test_cpu_cache_manager.py
git commit -m "perf: batch CPU mamba cache transfers with advanced indexing

Replace per-element Python for-loops in offload_to_cpu and load_to_gpu
with single advanced-indexing copy operations. Reduces Python overhead
and CUDA kernel launch count from 2*N to 2 per transfer batch."
```

---

### Task 3: Defer `sync_transfer()` in snapshot paths

**Files:**
- Modify: `lightllm/server/router/model_infer/infer_batch.py:316-344` (snapshot_hybrid_buffers CPU path)
- Modify: `lightllm/server/router/model_infer/infer_batch.py:401-426` (snapshot_prefill_complete_buffers CPU path)
- Modify: `lightllm/common/mamba_cache_mem_manager/cpu_cache_manager.py` (add `record_transfer_event` method)

**Context:** Currently `offload_to_cpu()` + `sync_transfer()` are called back-to-back, blocking the CPU thread waiting for PCIe transfers. The snapshot paths run between stage 3 and stage 4 of the inference loop — the GPU→CPU transfer could overlap with stage 4 work and subsequent decode iterations. However, the radix cache tree insertions immediately after depend on the CPU data being fully landed (they call `add_buffer_idx_to_node` which associates a CPU slot index with a tree node — the slot must contain valid data before any future `load_to_gpu` reads it).

**Key safety constraint:** The transfer MUST be complete before the CPU buffer slot is used in a future `load_to_gpu` call. Since `load_to_gpu` uses the same `_transfer_stream`, operations are serialized on that stream — a load issued after an offload on the same stream will see the completed offload data. The `sync_transfer()` before tree insertion is only needed if the tree node metadata must be consistent with the CPU buffer contents at insertion time.

**Analysis:** Looking at the code flow, `add_buffer_idx_to_node` only stores the slot index integer — it doesn't read the CPU buffer data. The actual CPU data is only read when `load_to_gpu` is later called during cache hit recovery (in `_handle_fork_requests`). Since `load_to_gpu` and `offload_to_cpu` both use the same `_transfer_stream`, CUDA stream ordering guarantees the offload completes before any subsequent load on that stream.

**Therefore:** We can safely remove `sync_transfer()` from the snapshot paths. The only callers that truly need `sync_transfer()` are `_free_req_mem_and_buffers` and `_pause_req_mem_and_buffers` (lines 245, 290) because they free the GPU buffer immediately after offload — the GPU buffer data must be fully copied before it's overwritten. Actually, even those are safe: the GPU buffer free only returns the slot to the pool — the actual memory isn't zeroed. The next user of that GPU slot will write to it during model.forward(), which runs on the overlap stream, not the transfer stream. So the transfer stream's offload will have completed long before the GPU slot is reused.

**However, to be conservative:** We'll keep `sync_transfer()` in `_free_req_mem_and_buffers` and `_pause_req_mem_and_buffers` (they handle single requests, low overhead), and only remove it from the batch snapshot paths.

- [ ] **Step 1: Add a `record_transfer_event` method to CpuMambaCacheManager**

In `lightllm/common/mamba_cache_mem_manager/cpu_cache_manager.py`, add after `sync_transfer`:

```python
def record_transfer_event(self) -> Optional[torch.cuda.Event]:
    """Record a CUDA event on the transfer stream for deferred synchronization.

    Returns None if no transfer stream exists (no transfers have been issued).
    """
    if self._transfer_stream is None:
        return None
    event = torch.cuda.Event()
    event.record(self._transfer_stream)
    return event
```

- [ ] **Step 2: Remove `sync_transfer()` from `snapshot_hybrid_buffers` CPU path**

In `lightllm/server/router/model_infer/infer_batch.py`, in `snapshot_hybrid_buffers` (around line 329), remove the `cpu_mgr.sync_transfer()` call. Change:

```python
            cpu_mgr.offload_to_cpu(
                self.req_manager.buffer_mem_manager.conv_state_cache.buffer,
                self.req_manager.buffer_mem_manager.ssm_state_cache.buffer,
                cur_buffers,
                cpu_slots,
            )
            cpu_mgr.sync_transfer()
```

to:

```python
            cpu_mgr.offload_to_cpu(
                self.req_manager.buffer_mem_manager.conv_state_cache.buffer,
                self.req_manager.buffer_mem_manager.ssm_state_cache.buffer,
                cur_buffers,
                cpu_slots,
            )
```

- [ ] **Step 3: Remove `sync_transfer()` from `snapshot_prefill_complete_buffers` CPU path**

In `lightllm/server/router/model_infer/infer_batch.py`, in `snapshot_prefill_complete_buffers` (around line 413), remove the `cpu_mgr.sync_transfer()` call. Change:

```python
            cpu_mgr.offload_to_cpu(
                self.req_manager.buffer_mem_manager.conv_state_cache.buffer,
                self.req_manager.buffer_mem_manager.ssm_state_cache.buffer,
                cur_buffers,
                cpu_slots,
            )
            cpu_mgr.sync_transfer()
```

to:

```python
            cpu_mgr.offload_to_cpu(
                self.req_manager.buffer_mem_manager.conv_state_cache.buffer,
                self.req_manager.buffer_mem_manager.ssm_state_cache.buffer,
                cur_buffers,
                cpu_slots,
            )
```

- [ ] **Step 4: Commit**

```bash
git add lightllm/server/router/model_infer/infer_batch.py \
        lightllm/common/mamba_cache_mem_manager/cpu_cache_manager.py
git commit -m "perf: defer sync_transfer in snapshot paths for async GPU->CPU overlap

Remove blocking sync_transfer() from snapshot_hybrid_buffers and
snapshot_prefill_complete_buffers. The CPU slot index is stored in the
tree node but the actual CPU data is only read during load_to_gpu,
which runs on the same transfer stream — CUDA stream ordering guarantees
the offload completes first. This allows GPU->CPU PCIe transfers to
overlap with subsequent decode iterations."
```

---

### Task 4: Merge two all_gather calls into one

**Files:**
- Modify: `lightllm/server/router/model_infer/mode_backend/base_backend.py:200-207,808-824`

**Context:** `_dp_all_gather_prefill_and_decode_req_num` issues two separate blocking `all_gather_into_tensor` calls (one for prefill count, one for decode count) on every inference loop iteration. Both values are always consumed together by `DPControlState.select_run_way`. By packing both counts into a single 2-element tensor per rank, we halve the number of NCCL collective operations per loop iteration.

- [ ] **Step 1: Update tensor initialization**

In `lightllm/server/router/model_infer/mode_backend/base_backend.py`, around lines 200-207, change:

```python
        if self.dp_size > 1:
            self.dp_reduce_tensor = torch.tensor([0], dtype=torch.int32, device="cuda", requires_grad=False)
            self.dp_gather_item_tensor = torch.tensor([0], dtype=torch.int32, device="cuda", requires_grad=False)
            self.dp_all_gather_tensor = torch.tensor(
                [0 for _ in range(self.global_world_size)], dtype=torch.int32, device="cuda", requires_grad=False
            )
```

to:

```python
        if self.dp_size > 1:
            self.dp_reduce_tensor = torch.tensor([0], dtype=torch.int32, device="cuda", requires_grad=False)
            self.dp_gather_item_tensor = torch.tensor([0, 0], dtype=torch.int32, device="cuda", requires_grad=False)
            self.dp_all_gather_tensor = torch.zeros(
                self.global_world_size * 2, dtype=torch.int32, device="cuda", requires_grad=False
            )
```

- [ ] **Step 2: Merge the two all_gather calls**

In `lightllm/server/router/model_infer/mode_backend/base_backend.py`, replace `_dp_all_gather_prefill_and_decode_req_num` (lines 808-824):

```python
    def _dp_all_gather_prefill_and_decode_req_num(
        self, prefill_reqs: List[InferReq], decode_reqs: List[InferReq]
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Gather the number of prefill requests across all DP ranks.
        """
        current_dp_prefill_num = len(prefill_reqs)
        self.dp_gather_item_tensor.fill_(current_dp_prefill_num)
        all_gather_into_tensor(self.dp_all_gather_tensor, self.dp_gather_item_tensor, group=None, async_op=False)
        dp_prefill_req_nums = self.dp_all_gather_tensor.cpu().numpy()

        current_dp_decode_num = len(decode_reqs)
        self.dp_gather_item_tensor.fill_(current_dp_decode_num)
        all_gather_into_tensor(self.dp_all_gather_tensor, self.dp_gather_item_tensor, group=None, async_op=False)
        dp_decode_req_nums = self.dp_all_gather_tensor.cpu().numpy()

        return dp_prefill_req_nums, dp_decode_req_nums
```

with:

```python
    def _dp_all_gather_prefill_and_decode_req_num(
        self, prefill_reqs: List[InferReq], decode_reqs: List[InferReq]
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Gather the number of prefill and decode requests across all DP ranks in a single collective.
        """
        self.dp_gather_item_tensor[0] = len(prefill_reqs)
        self.dp_gather_item_tensor[1] = len(decode_reqs)
        all_gather_into_tensor(self.dp_all_gather_tensor, self.dp_gather_item_tensor, group=None, async_op=False)
        gathered = self.dp_all_gather_tensor.cpu().numpy()
        dp_prefill_req_nums = gathered[0::2]
        dp_decode_req_nums = gathered[1::2]
        return dp_prefill_req_nums, dp_decode_req_nums
```

- [ ] **Step 3: Verify `select_run_way` still works correctly**

The callers in `DPControlState.select_run_way` use `np.max(dp_prefill_req_nums)` and `np.max(dp_decode_req_nums)`. The new sliced arrays have shape `(world_size,)` — same as before. Verify by reading `lightllm/server/router/model_infer/mode_backend/dp_backend/control_state.py` to confirm `np.max()` calls on both arrays.

- [ ] **Step 4: Commit**

```bash
git add lightllm/server/router/model_infer/mode_backend/base_backend.py
git commit -m "perf: merge two all_gather calls into one in DP infer loop

Pack prefill and decode request counts into a single 2-element tensor
and issue one all_gather_into_tensor instead of two. Halves the NCCL
collective overhead per inference loop iteration with DP>1."
```

---

### Task 5: Remove dead code `_dp_all_reduce_decode_req_num`

**Files:**
- Modify: `lightllm/server/router/model_infer/mode_backend/base_backend.py:826-834`

**Context:** `_dp_all_reduce_decode_req_num` is defined but never called anywhere in the codebase. It performs a MAX reduction on decode counts — functionality that's already covered by the all_gather path.

- [ ] **Step 1: Verify the method is unused**

Run: `grep -rn "_dp_all_reduce_decode_req_num" lightllm/`

Expected output should show only the definition site (base_backend.py), no callers.

- [ ] **Step 2: Remove the dead method**

In `lightllm/server/router/model_infer/mode_backend/base_backend.py`, delete:

```python
    def _dp_all_reduce_decode_req_num(self, decode_reqs: List[InferReq]) -> int:
        """
        Reduce the number of decode requests across all DP ranks.
        """
        current_dp_decode_num = len(decode_reqs)
        self.dp_reduce_tensor.fill_(current_dp_decode_num)
        all_reduce(self.dp_reduce_tensor, op=dist.ReduceOp.MAX, group=None, async_op=False)
        max_decode_num = self.dp_reduce_tensor.item()
        return max_decode_num
```

Also check if `dp_reduce_tensor` is used elsewhere. If not, remove its initialization too. However, keep it if other code references it.

- [ ] **Step 3: Verify `dp_reduce_tensor` has no other users**

Run: `grep -rn "dp_reduce_tensor" lightllm/`

If only used in the deleted method and its initialization, remove the initialization line (`self.dp_reduce_tensor = ...`) from `__init__` as well.

- [ ] **Step 4: Commit**

```bash
git add lightllm/server/router/model_infer/mode_backend/base_backend.py
git commit -m "chore: remove unused _dp_all_reduce_decode_req_num method

This method was never called. Also removes dp_reduce_tensor if unused."
```
