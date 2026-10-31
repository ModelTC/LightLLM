"""SWA 物理索引与 prefill workspace 的布局准备和前缀搬运。"""

import torch
import triton
import triton.language as tl


@triton.jit
def get_swa_token_index(
    req_idx,
    position,
    prefill_token_ranges,
    swa_prefill_workspace_offsets,
    SWA_CAPACITY_TOKENS: tl.constexpr,
    SLIDING_WINDOW: tl.constexpr,
    SWA_PAGE_TOKEN_NUM: tl.constexpr,
    HOLD_REQUEST_ID: tl.constexpr,
    HOLD_SWA_SLOT: tl.constexpr,
    IS_PREFILL: tl.constexpr,
):
    slot = req_idx * SWA_CAPACITY_TOKENS + position % SWA_CAPACITY_TOKENS
    if IS_PREFILL:
        ready = tl.load(prefill_token_ranges + req_idx * 2)
        first = tl.maximum(0, ready - SLIDING_WINDOW + 1) // SWA_PAGE_TOKEN_NUM * SWA_PAGE_TOKEN_NUM
        workspace = tl.load(swa_prefill_workspace_offsets + req_idx)
        slot = workspace + position - first
    return tl.where(req_idx == HOLD_REQUEST_ID, HOLD_SWA_SLOT + position % SWA_PAGE_TOKEN_NUM, slot)


def get_swa_token_indices(swa_state, req_idx, position_ids, *, prefill_state=None):
    config = swa_state.config
    swa_token_indices = req_idx * swa_state.swa_capacity_tokens + position_ids % swa_state.swa_capacity_tokens
    hold_slot = swa_state.HOLD_SWA_SLOT
    if prefill_state is not None:
        ready = prefill_state.prefill_token_ranges[req_idx, 0]
        first = (
            (ready - config.window_size + 1).clamp_min(0) // config.page_token_size * config.page_token_size
        )
        workspace = prefill_state.swa_prefill_workspace_offsets[req_idx]
        swa_token_indices = workspace + position_ids - first
        hold_slot = prefill_state.HOLD_SWA_SLOT
    return torch.where(
        torch.as_tensor(req_idx, device=position_ids.device) == swa_state.HOLD_REQUEST_ID,
        hold_slot + position_ids % config.page_token_size,
        swa_token_indices,
    )


@triton.jit
def _prepare_swa_prefill_workspace_kernel(
    prefill_token_ranges,
    swa_prefill_workspace_offsets,
    b_req_idx,
    b_ready_cache_len,
    b_seq_len,
    BATCH: tl.constexpr,
    SLIDING_WINDOW: tl.constexpr,
    SWA_PAGE_TOKEN_NUM: tl.constexpr,
    HOLD_REQUEST_ID: tl.constexpr,
    SWA_WORKSPACE_BASE: tl.constexpr,
    BLOCK: tl.constexpr,
):
    batch = tl.arange(0, BLOCK)
    valid = batch < BATCH
    req_idx = tl.load(b_req_idx + batch, valid, HOLD_REQUEST_ID)
    ready = tl.load(b_ready_cache_len + batch, valid, 0)
    end = tl.load(b_seq_len + batch, valid, 0)
    first = tl.maximum(0, ready - SLIDING_WINDOW + 1) // SWA_PAGE_TOKEN_NUM * SWA_PAGE_TOKEN_NUM
    length = tl.cdiv(end - first, SWA_PAGE_TOKEN_NUM) * SWA_PAGE_TOKEN_NUM
    tokens = tl.where((req_idx != HOLD_REQUEST_ID) & valid, length, 0)
    workspace = SWA_WORKSPACE_BASE + tl.cumsum(tokens) - tokens
    tl.store(prefill_token_ranges + req_idx * 2, ready, valid)
    tl.store(prefill_token_ranges + req_idx * 2 + 1, end, valid)
    tl.store(swa_prefill_workspace_offsets + req_idx, workspace, valid)


@triton.jit
def _copy_swa_prefix_to_workspace_kernel(
    runtime_swa_buffer,
    workspace_swa_buffer,
    swa_prefill_workspace_offsets,
    b_req_idx,
    b_ready_cache_len,
    SWA_CAPACITY_TOKENS: tl.constexpr,
    SLIDING_WINDOW: tl.constexpr,
    SWA_PAGE_TOKEN_NUM: tl.constexpr,
    HOLD_REQUEST_ID: tl.constexpr,
    RUNTIME_LAYER_STRIDE: tl.constexpr,
    WORKSPACE_LAYER_STRIDE: tl.constexpr,
    PACKED_BYTES: tl.constexpr,
    BLOCK: tl.constexpr,
):
    batch, block, layer = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    req_idx = tl.load(b_req_idx + batch)
    target_slot = tl.load(swa_prefill_workspace_offsets + req_idx)
    if req_idx == HOLD_REQUEST_ID:
        return
    ready = tl.load(b_ready_cache_len + batch)
    first = tl.maximum(0, ready - SLIDING_WINDOW + 1) // SWA_PAGE_TOKEN_NUM * SWA_PAGE_TOKEN_NUM
    position = first + block * SWA_PAGE_TOKEN_NUM
    if position >= ready:
        return
    source_slot = req_idx * SWA_CAPACITY_TOKENS + position % SWA_CAPACITY_TOKENS
    source = runtime_swa_buffer + layer * RUNTIME_LAYER_STRIDE + source_slot // SWA_PAGE_TOKEN_NUM * PACKED_BYTES
    target = workspace_swa_buffer + layer * WORKSPACE_LAYER_STRIDE + (target_slot // SWA_PAGE_TOKEN_NUM + block) * PACKED_BYTES
    lanes = tl.arange(0, BLOCK)
    for chunk in range(0, PACKED_BYTES, BLOCK):
        offset = chunk + lanes
        tl.store(target + offset, tl.load(source + offset, offset < PACKED_BYTES, 0), offset < PACKED_BYTES)


def prepare_swa_prefill_workspace(swa_state, prefill_state, b_req_idx, b_ready_cache_len, b_seq_len, workspace_base):
    if not b_req_idx.numel():
        return
    config = swa_state.config
    _prepare_swa_prefill_workspace_kernel[(1,)](
        prefill_state.prefill_token_ranges,
        prefill_state.swa_prefill_workspace_offsets,
        b_req_idx,
        b_ready_cache_len,
        b_seq_len,
        BATCH=b_req_idx.numel(),
        SLIDING_WINDOW=config.window_size,
        SWA_PAGE_TOKEN_NUM=config.page_token_size,
        HOLD_REQUEST_ID=swa_state.HOLD_REQUEST_ID,
        SWA_WORKSPACE_BASE=workspace_base,
        BLOCK=triton.next_power_of_2(b_req_idx.numel()),
        num_warps=4,
    )
    workspace_swa_buffer = prefill_state.swa_buffer
    _copy_swa_prefix_to_workspace_kernel[
        (
            b_req_idx.numel(),
            triton.cdiv(config.window_size - 1, config.page_token_size) + 1,
            workspace_swa_buffer.shape[0],
        )
    ](
        runtime_swa_buffer=swa_state.swa_buffer,
        workspace_swa_buffer=workspace_swa_buffer,
        swa_prefill_workspace_offsets=prefill_state.swa_prefill_workspace_offsets,
        b_req_idx=b_req_idx,
        b_ready_cache_len=b_ready_cache_len,
        SWA_CAPACITY_TOKENS=swa_state.swa_capacity_tokens,
        SLIDING_WINDOW=config.window_size,
        SWA_PAGE_TOKEN_NUM=config.page_token_size,
        HOLD_REQUEST_ID=swa_state.HOLD_REQUEST_ID,
        RUNTIME_LAYER_STRIDE=swa_state.swa_buffer.stride(0),
        WORKSPACE_LAYER_STRIDE=workspace_swa_buffer.stride(0),
        PACKED_BYTES=config.get_packed_kv_bytes(config.page_token_size),
        BLOCK=4096,
        num_warps=4,
    )
