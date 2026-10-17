"""Token addressing and packed I/O for request windows and prefill scratch."""

import torch
import triton
import triton.language as tl


@triton.jit
def swa_token_index(
    req,
    position,
    ranges,
    workspace_offsets,
    CAPACITY: tl.constexpr,
    WINDOW: tl.constexpr,
    ALIGNMENT: tl.constexpr,
    HOLD_REQ: tl.constexpr,
    HOLD_SLOT: tl.constexpr,
    PREFILL: tl.constexpr,
):
    slot = req * CAPACITY + position % CAPACITY
    if PREFILL:
        ready = tl.load(ranges + req * 2)
        end = tl.load(ranges + req * 2 + 1)
        first = tl.maximum(0, ready - WINDOW + 1) // ALIGNMENT * ALIGNMENT
        workspace = tl.load(workspace_offsets + req)
        slot = tl.where((workspace >= 0) & (position >= first) & (position < end), workspace + position - first, slot)
    return tl.where(req == HOLD_REQ, HOLD_SLOT + position % ALIGNMENT, slot)


def get_swa_token_indices(requests, req_idx, positions, *, prefill=True):
    config = requests.cache_config
    slots = req_idx * requests.swa_capacity_tokens + positions % requests.swa_capacity_tokens
    if prefill:
        ready, end = requests.prefill_token_ranges[req_idx].unbind(-1)
        first = (
            (ready - config.sliding_window + 1).clamp_min(0) // config.swa_token_alignment * config.swa_token_alignment
        )
        workspace = requests.prefill_workspace_offsets[req_idx]
        slots = torch.where(
            (workspace >= 0) & (positions >= first) & (positions < end), workspace + positions - first, slots
        )
    return torch.where(
        torch.as_tensor(req_idx, device=positions.device) == requests.HOLD_REQUEST_ID,
        requests.HOLD_SWA_SLOT + positions % config.swa_token_alignment,
        slots,
    )


@triton.jit
def _prepare_prefill(
    ranges,
    workspace_offsets,
    checkpoint_offsets,
    req_ids,
    ready_lens,
    seq_lens,
    BATCH: tl.constexpr,
    WINDOW: tl.constexpr,
    CAPACITY: tl.constexpr,
    ALIGNMENT: tl.constexpr,
    CHECKPOINT_TOKENS: tl.constexpr,
    HOLD_REQ: tl.constexpr,
    WORKSPACE_BASE: tl.constexpr,
    CHECKPOINT_BASE: tl.constexpr,
    BLOCK: tl.constexpr,
):
    batch = tl.arange(0, BLOCK)
    valid = batch < BATCH
    req = tl.load(req_ids + batch, valid, HOLD_REQ)
    ready = tl.load(ready_lens + batch, valid, 0)
    end = tl.load(seq_lens + batch, valid, 0)
    first = tl.maximum(0, ready - WINDOW + 1) // ALIGNMENT * ALIGNMENT
    length = tl.cdiv(end - first, ALIGNMENT) * ALIGNMENT
    overflow = (length > CAPACITY) & (req != HOLD_REQ) & valid
    tokens = tl.where(overflow, length, 0)
    workspace = WORKSPACE_BASE + tl.cumsum(tokens) - tokens
    checkpoint_count = tl.where(valid & (req != HOLD_REQ), end // CHECKPOINT_TOKENS - ready // CHECKPOINT_TOKENS, 0)
    checkpoint = CHECKPOINT_BASE + tl.cumsum(checkpoint_count) - checkpoint_count - ready // CHECKPOINT_TOKENS - 1
    tl.store(ranges + req * 2, ready, valid)
    tl.store(ranges + req * 2 + 1, end, valid)
    tl.store(workspace_offsets + req, tl.where(overflow, workspace, -1), valid)
    tl.store(checkpoint_offsets + req, checkpoint, valid)


@triton.jit
def _copy_prefix(
    storage,
    ranges,
    workspace_offsets,
    req_ids,
    ready_lens,
    CAPACITY: tl.constexpr,
    WINDOW: tl.constexpr,
    ALIGNMENT: tl.constexpr,
    HOLD_REQ: tl.constexpr,
    LAYER_STRIDE: tl.constexpr,
    PACKED_BYTES: tl.constexpr,
    BLOCK: tl.constexpr,
):
    batch, block, layer = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    req = tl.load(req_ids + batch)
    target_slot = tl.load(workspace_offsets + req)
    if req == HOLD_REQ or target_slot < 0:
        return
    ready = tl.load(ready_lens + batch)
    first = tl.maximum(0, ready - WINDOW + 1) // ALIGNMENT * ALIGNMENT
    position = first + block * ALIGNMENT
    if position >= ready:
        return
    source_slot = req * CAPACITY + position % CAPACITY
    source = storage + layer * LAYER_STRIDE + source_slot // ALIGNMENT * PACKED_BYTES
    target = storage + layer * LAYER_STRIDE + (target_slot // ALIGNMENT + block) * PACKED_BYTES
    lanes = tl.arange(0, BLOCK)
    for chunk in range(0, PACKED_BYTES, BLOCK):
        offset = chunk + lanes
        tl.store(target + offset, tl.load(source + offset, offset < PACKED_BYTES, 0), offset < PACKED_BYTES)


def prepare_prefill_workspace(requests, req_ids, ready_lens, seq_lens, microbatch_index):
    if not req_ids.numel():
        return
    config = requests.cache_config
    _prepare_prefill[(1,)](
        requests.prefill_token_ranges,
        requests.prefill_workspace_offsets,
        requests.prefill_checkpoint_offsets,
        req_ids,
        ready_lens,
        seq_lens,
        BATCH=req_ids.numel(),
        WINDOW=config.sliding_window,
        CAPACITY=requests.swa_capacity_tokens,
        ALIGNMENT=config.swa_token_alignment,
        CHECKPOINT_TOKENS=config.history_token_alignment,
        HOLD_REQ=requests.HOLD_REQUEST_ID,
        WORKSPACE_BASE=requests.swa_runtime_tokens + microbatch_index * requests.prefill_capacity_tokens,
        CHECKPOINT_BASE=microbatch_index * requests.prefill_checkpoint_capacity,
        BLOCK=triton.next_power_of_2(req_ids.numel()),
        num_warps=4,
    )
    storage = requests.swa_buffer
    _copy_prefix[
        (req_ids.numel(), triton.cdiv(config.sliding_window - 1, config.swa_token_alignment) + 1, storage.shape[0])
    ](
        storage,
        requests.prefill_token_ranges,
        requests.prefill_workspace_offsets,
        req_ids,
        ready_lens,
        CAPACITY=requests.swa_capacity_tokens,
        WINDOW=config.sliding_window,
        ALIGNMENT=config.swa_token_alignment,
        HOLD_REQ=requests.HOLD_REQUEST_ID,
        LAYER_STRIDE=storage.stride(0),
        PACKED_BYTES=config.packed_kv_bytes(config.swa_token_alignment),
        BLOCK=4096,
        num_warps=4,
    )


@triton.jit
def _copy_swa_tokens(
    storage,
    canonical,
    slots,
    positions,
    TOKEN_COUNT: tl.constexpr,
    STORAGE_STRIDE: tl.constexpr,
    TOKEN_BYTES: tl.constexpr,
    PACK_TOKENS: tl.constexpr,
    PACK_BYTES: tl.constexpr,
    DATA_BYTES: tl.constexpr,
    GATHER: tl.constexpr,
    BLOCK: tl.constexpr,
):
    token, layer = tl.program_id(0), tl.program_id(1)
    slot = tl.load(slots + token).to(tl.int64)
    position = tl.load(positions + token)
    byte = tl.arange(0, BLOCK)
    offset = slot // PACK_TOKENS * PACK_BYTES
    packed_offset = tl.where(
        byte < DATA_BYTES,
        slot % PACK_TOKENS * DATA_BYTES + byte,
        PACK_TOKENS * DATA_BYTES + slot % PACK_TOKENS * (TOKEN_BYTES - DATA_BYTES) + byte - DATA_BYTES,
    )
    packed = storage + layer * STORAGE_STRIDE + offset + packed_offset
    dense = canonical + (layer * TOKEN_COUNT + token) * TOKEN_BYTES + byte
    valid = (byte < TOKEN_BYTES) & (position >= 0)
    if GATHER:
        tl.store(dense, tl.load(packed, valid, 0), byte < TOKEN_BYTES)
    else:
        tl.store(packed, tl.load(dense, valid, 0), valid)


def _copy_window(requests, req_idx, positions, canonical, *, gather, prefill):
    config = requests.cache_config
    # Triton launches on the current device. PD may call from another GPU;
    # address generation and packed I/O must share the owner's CUDA stream.
    with torch.cuda.device(requests.swa_buffer.device):
        slots = get_swa_token_indices(requests, req_idx, positions, prefill=prefill)
        _copy_swa_tokens[(positions.numel(), config.layer_num)](
            requests.swa_buffer,
            canonical,
            slots,
            positions,
            TOKEN_COUNT=positions.numel(),
            STORAGE_STRIDE=requests.swa_buffer.stride(0),
            TOKEN_BYTES=config.kv_bytes,
            PACK_TOKENS=config.swa_token_alignment,
            PACK_BYTES=config.packed_kv_bytes(config.swa_token_alignment),
            DATA_BYTES=config.kv_data_bytes,
            GATHER=gather,
            BLOCK=triton.next_power_of_2(config.kv_bytes),
            num_warps=4,
        )


def gather_swa_tokens(requests, req_idx, positions, *, prefill=True):
    canonical = torch.empty(
        (requests.cache_config.layer_num, positions.numel(), requests.cache_config.kv_bytes),
        dtype=torch.uint8,
        device=requests.swa_buffer.device,
    )
    _copy_window(requests, req_idx, positions, canonical, gather=True, prefill=prefill)
    return canonical


def scatter_swa_tokens(requests, req_idx, positions, canonical):
    _copy_window(requests, req_idx, positions, canonical, gather=False, prefill=False)
