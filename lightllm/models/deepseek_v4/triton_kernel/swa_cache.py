"""SWA 物理索引、prefill 临时区，以及大小页 SWA 状态的保存和恢复。"""

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


def get_swa_token_indices(req_manager, req_idx, position_ids, *, prefill_state=None):
    config = req_manager.cache_config
    swa_token_indices = req_idx * req_manager.swa_capacity_tokens + position_ids % req_manager.swa_capacity_tokens
    hold_slot = req_manager.HOLD_SWA_SLOT
    if prefill_state is not None:
        ready = prefill_state.prefill_token_ranges[req_idx, 0]
        first = (
            (ready - config.sliding_window + 1).clamp_min(0) // config.swa_page_token_num * config.swa_page_token_num
        )
        workspace = prefill_state.swa_prefill_workspace_offsets[req_idx]
        swa_token_indices = workspace + position_ids - first
        hold_slot = prefill_state.HOLD_SWA_SLOT
    return torch.where(
        torch.as_tensor(req_idx, device=position_ids.device) == req_manager.HOLD_REQUEST_ID,
        hold_slot + position_ids % config.swa_page_token_num,
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
    req_to_swa,
    swa_buffer,
    swa_prefill_workspace_offsets,
    b_req_idx,
    b_ready_cache_len,
    SWA_CAPACITY_TOKENS: tl.constexpr,
    SLIDING_WINDOW: tl.constexpr,
    SWA_PAGE_TOKEN_NUM: tl.constexpr,
    HOLD_REQUEST_ID: tl.constexpr,
    REQ_LAYER_STRIDE: tl.constexpr,
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
    source = req_to_swa + layer * REQ_LAYER_STRIDE + source_slot // SWA_PAGE_TOKEN_NUM * PACKED_BYTES
    target = swa_buffer + layer * WORKSPACE_LAYER_STRIDE + (target_slot // SWA_PAGE_TOKEN_NUM + block) * PACKED_BYTES
    lanes = tl.arange(0, BLOCK)
    for chunk in range(0, PACKED_BYTES, BLOCK):
        offset = chunk + lanes
        tl.store(target + offset, tl.load(source + offset, offset < PACKED_BYTES, 0), offset < PACKED_BYTES)


def prepare_swa_prefill_workspace(req_manager, prefill_state, b_req_idx, b_ready_cache_len, b_seq_len, workspace_base):
    if not b_req_idx.numel():
        return
    config = req_manager.cache_config
    _prepare_swa_prefill_workspace_kernel[(1,)](
        prefill_state.prefill_token_ranges,
        prefill_state.swa_prefill_workspace_offsets,
        b_req_idx,
        b_ready_cache_len,
        b_seq_len,
        BATCH=b_req_idx.numel(),
        SLIDING_WINDOW=config.sliding_window,
        SWA_PAGE_TOKEN_NUM=config.swa_page_token_num,
        HOLD_REQUEST_ID=req_manager.HOLD_REQUEST_ID,
        SWA_WORKSPACE_BASE=workspace_base,
        BLOCK=triton.next_power_of_2(b_req_idx.numel()),
        num_warps=4,
    )
    swa_buffer = prefill_state.swa_buffer
    _copy_swa_prefix_to_workspace_kernel[
        (
            b_req_idx.numel(),
            triton.cdiv(config.sliding_window - 1, config.swa_page_token_num) + 1,
            swa_buffer.shape[0],
        )
    ](
        req_manager.req_to_swa,
        swa_buffer,
        prefill_state.swa_prefill_workspace_offsets,
        b_req_idx,
        b_ready_cache_len,
        SWA_CAPACITY_TOKENS=req_manager.swa_capacity_tokens,
        SLIDING_WINDOW=config.sliding_window,
        SWA_PAGE_TOKEN_NUM=config.swa_page_token_num,
        HOLD_REQUEST_ID=req_manager.HOLD_REQUEST_ID,
        REQ_LAYER_STRIDE=req_manager.req_to_swa.stride(0),
        WORKSPACE_LAYER_STRIDE=swa_buffer.stride(0),
        PACKED_BYTES=config.get_packed_kv_bytes(config.swa_page_token_num),
        BLOCK=4096,
        num_warps=4,
    )


@triton.jit
def _copy_swa_state_kernel(
    swa_buffer,
    swa_state,
    swa_token_indices,
    position_ids,
    TOKEN_COUNT: tl.constexpr,
    SWA_LAYER_STRIDE: tl.constexpr,
    KV_BYTES: tl.constexpr,
    SWA_PAGE_TOKEN_NUM: tl.constexpr,
    SWA_PAGE_BYTES: tl.constexpr,
    KV_DATA_BYTES: tl.constexpr,
    GATHER: tl.constexpr,
    BLOCK: tl.constexpr,
):
    token, layer = tl.program_id(0), tl.program_id(1)
    slot = tl.load(swa_token_indices + token).to(tl.int64)
    position = tl.load(position_ids + token)
    byte = tl.arange(0, BLOCK)
    offset = slot // SWA_PAGE_TOKEN_NUM * SWA_PAGE_BYTES
    packed_offset = tl.where(
        byte < KV_DATA_BYTES,
        slot % SWA_PAGE_TOKEN_NUM * KV_DATA_BYTES + byte,
        SWA_PAGE_TOKEN_NUM * KV_DATA_BYTES
        + slot % SWA_PAGE_TOKEN_NUM * (KV_BYTES - KV_DATA_BYTES)
        + byte
        - KV_DATA_BYTES,
    )
    packed = swa_buffer + layer * SWA_LAYER_STRIDE + offset + packed_offset
    dense = swa_state + (layer * TOKEN_COUNT + token) * KV_BYTES + byte
    valid = (byte < KV_BYTES) & (position >= 0)
    if GATHER:
        tl.store(dense, tl.load(packed, valid, 0), byte < KV_BYTES)
    else:
        tl.store(packed, tl.load(dense, valid, 0), valid)


def _copy_swa_state(req_manager, req_idx, position_ids, swa_state, *, gather):
    config = req_manager.cache_config
    with torch.cuda.device(req_manager.req_to_swa.device):
        swa_token_indices = get_swa_token_indices(req_manager, req_idx, position_ids)
        _copy_swa_state_kernel[(position_ids.numel(), config.layer_num)](
            req_manager.req_to_swa,
            swa_state,
            swa_token_indices,
            position_ids,
            TOKEN_COUNT=position_ids.numel(),
            SWA_LAYER_STRIDE=req_manager.req_to_swa.stride(0),
            KV_BYTES=config.kv_bytes,
            SWA_PAGE_TOKEN_NUM=config.swa_page_token_num,
            SWA_PAGE_BYTES=config.get_packed_kv_bytes(config.swa_page_token_num),
            KV_DATA_BYTES=config.kv_data_bytes,
            GATHER=gather,
            BLOCK=triton.next_power_of_2(config.kv_bytes),
            num_warps=4,
        )


def gather_swa_state(req_manager, req_idx, position_ids):
    """将 SWA packed 存储中的窗口读为 [层, token, kv_bytes]，供大小页保存。"""
    swa_state = torch.empty(
        (req_manager.cache_config.layer_num, position_ids.numel(), req_manager.cache_config.kv_bytes),
        dtype=torch.uint8,
        device=req_manager.req_to_swa.device,
    )
    _copy_swa_state(req_manager, req_idx, position_ids, swa_state, gather=True)
    return swa_state


def scatter_swa_state(req_manager, req_idx, position_ids, swa_state):
    _copy_swa_state(req_manager, req_idx, position_ids, swa_state, gather=False)
