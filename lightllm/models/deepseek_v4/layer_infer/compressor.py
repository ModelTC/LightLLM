from dataclasses import dataclass
from typing import Optional

import torch
import triton
import triton.language as tl
from triton.language.extra import libdevice

from lightllm.common.kv_cache_mem_manager.deepseek4_mem_manager import (
    DSV4_C4_STATE_RING,
    DSV4_C128_STATE_RING,
    DSV4_SWA_PAGE_SIZE,
)


_SGLANG_COMPRESS_ERR = None
_SGLANG_COMPRESS_MOD = None
_SGLANG_LINEAR_BF16_FP32 = None
_FREQ_CIS_CACHE = {}


@dataclass
class CoreCompressorMetadata:
    layer_idx: int
    compress_ratio: int
    out_slots: torch.Tensor
    mem_index: torch.Tensor
    state_buffer: torch.Tensor
    out_buffer: torch.Tensor
    out_page_size: int
    position_ids: torch.Tensor
    b_req_idx: torch.Tensor
    b_seq_len: torch.Tensor
    b_ready_cache_len: Optional[torch.Tensor]
    b_q_start_loc: Optional[torch.Tensor]
    req_to_token_indexs: torch.Tensor
    full_to_swa_indexs: torch.Tensor
    token_to_batch_idx: Optional[torch.Tensor]
    kv_score: Optional[torch.Tensor]
    is_prefill: bool


@triton.jit
def _add_ape_to_kv_score_kernel(
    kv_score,
    kv_score_stride0,
    kv_score_stride1,
    ape,
    ape_stride0,
    positions,
    STATE_WIDTH: tl.constexpr,
    COMPRESS_RATIO: tl.constexpr,
    BLOCK: tl.constexpr,
):
    token_idx = tl.program_id(0)
    offs = tl.arange(0, BLOCK)
    mask = offs < STATE_WIDTH

    position = tl.load(positions + token_idx)
    ape_row = position % COMPRESS_RATIO
    score = tl.load(kv_score + token_idx * kv_score_stride0 + (STATE_WIDTH + offs) * kv_score_stride1, mask=mask)
    ape_value = tl.load(ape + ape_row * ape_stride0 + offs, mask=mask)
    tl.store(
        kv_score + token_idx * kv_score_stride0 + (STATE_WIDTH + offs) * kv_score_stride1,
        score + ape_value,
        mask=mask,
    )
    return


@triton.jit
def _save_partial_states_kernel(
    kv_score,
    kv_score_stride0,
    kv_score_stride1,
    positions,
    token_to_batch_idx,
    b_req_idx,
    b_seq_len,
    mem_index,
    full_to_swa,
    state_buffer,
    STATE_WIDTH: tl.constexpr,
    STATE_LAST_DIM: tl.constexpr,
    COMPRESS_RATIO: tl.constexpr,
    IS_C4: tl.constexpr,
    IS_PREFILL: tl.constexpr,
    SWA_PAGE_SIZE: tl.constexpr,
    STATE_RING: tl.constexpr,
    BLOCK: tl.constexpr,
):
    token_idx = tl.program_id(0)
    batch_idx = tl.load(token_to_batch_idx + token_idx) if IS_PREFILL else token_idx
    position = tl.load(positions + token_idx)
    seq_len = tl.load(b_seq_len + batch_idx)

    if IS_C4:
        same_page_next = (position % SWA_PAGE_SIZE) + STATE_RING < SWA_PAGE_SIZE
        if same_page_next and position + STATE_RING < seq_len:
            return
    else:
        if position + COMPRESS_RATIO < seq_len:
            return

    full_slot = tl.load(mem_index + token_idx).to(tl.int64)
    swa_slot = tl.load(full_to_swa + full_slot).to(tl.int64)
    if swa_slot < 0:
        return
    state_row = (swa_slot // SWA_PAGE_SIZE) * STATE_RING + (swa_slot % STATE_RING)

    offs = tl.arange(0, BLOCK)
    mask = offs < STATE_WIDTH
    kv = tl.load(kv_score + token_idx * kv_score_stride0 + offs * kv_score_stride1, mask=mask)
    score = tl.load(kv_score + token_idx * kv_score_stride0 + (STATE_WIDTH + offs) * kv_score_stride1, mask=mask)
    state_base = state_buffer + state_row * STATE_LAST_DIM
    tl.store(state_base + offs, kv, mask=mask)
    tl.store(state_base + STATE_WIDTH + offs, score, mask=mask)
    return


@triton.jit
def _fused_compress_norm_rope_insert_kernel(
    kv_score,
    kv_score_stride0,
    kv_score_stride1,
    state_buffer,
    positions,
    token_to_batch_idx,
    b_req_idx,
    b_seq_len,
    b_ready_cache_len,
    b_q_start_loc,
    req_to_token,
    req_to_token_stride0,
    full_to_swa,
    out_slots,
    norm_weight,
    rms_eps,
    cos_table,
    cos_stride0,
    sin_table,
    sin_stride0,
    out_buffer,
    HEAD_DIM: tl.constexpr,
    STATE_WIDTH: tl.constexpr,
    STATE_LAST_DIM: tl.constexpr,
    COMPRESS_RATIO: tl.constexpr,
    WINDOW_SIZE: tl.constexpr,
    IS_C4: tl.constexpr,
    IS_PREFILL: tl.constexpr,
    SWA_PAGE_SIZE: tl.constexpr,
    STATE_RING: tl.constexpr,
    ROPE_HEAD_DIM: tl.constexpr,
    FP8_MAX: tl.constexpr,
    SCALE_MIN: tl.constexpr,
    NOPE_DIM: tl.constexpr,
    QUANT_BLOCK: tl.constexpr,
    SCALE_BYTES: tl.constexpr,
    PAGE_SIZE: tl.constexpr,
    BYTES_PER_PAGE: tl.constexpr,
    BLOCK: tl.constexpr,
):
    token_idx = tl.program_id(0)
    out_slot = tl.load(out_slots + token_idx).to(tl.int64)
    if out_slot < 0:
        return

    position = tl.load(positions + token_idx)
    if (position + 1) % COMPRESS_RATIO != 0:
        return

    batch_idx = tl.load(token_to_batch_idx + token_idx) if IS_PREFILL else token_idx
    req_idx = tl.load(b_req_idx + batch_idx).to(tl.int64)
    seq_len = tl.load(b_seq_len + batch_idx)
    if IS_PREFILL:
        ready_len = tl.load(b_ready_cache_len + batch_idx)
        q_start = tl.load(b_q_start_loc + batch_idx)
    else:
        ready_len = position
        q_start = token_idx

    token_offsets = tl.arange(0, WINDOW_SIZE)
    start = position - WINDOW_SIZE + 1
    gather_pos = start + token_offsets
    valid_pos = (gather_pos >= 0) & (gather_pos < seq_len)
    use_current = (gather_pos >= ready_len) & valid_pos if IS_PREFILL else gather_pos == position
    current_idx = q_start + (gather_pos - ready_len) if IS_PREFILL else token_idx + token_offsets * 0

    if IS_C4:
        full_slot = tl.load(
            req_to_token + req_idx * req_to_token_stride0 + gather_pos,
            mask=valid_pos & (~use_current),
            other=0,
        ).to(tl.int64)
        swa_slot = tl.load(full_to_swa + full_slot, mask=valid_pos & (~use_current), other=-1).to(tl.int64)
        state_row = (swa_slot // SWA_PAGE_SIZE) * STATE_RING + (swa_slot % STATE_RING)
        state_valid = valid_pos & (~use_current) & (swa_slot >= 0)
        head_offset = tl.where(token_offsets >= COMPRESS_RATIO, HEAD_DIM, 0)
    else:
        full_slot = tl.load(
            req_to_token + req_idx * req_to_token_stride0 + gather_pos,
            mask=valid_pos & (~use_current),
            other=0,
        ).to(tl.int64)
        swa_slot = tl.load(full_to_swa + full_slot, mask=valid_pos & (~use_current), other=-1).to(tl.int64)
        state_row = (swa_slot // SWA_PAGE_SIZE) * STATE_RING + (swa_slot % STATE_RING)
        state_valid = valid_pos & (~use_current) & (swa_slot >= 0)
        head_offset = token_offsets * 0

    offs = tl.arange(0, BLOCK)
    dim_mask = offs < HEAD_DIM
    current_mask = use_current[:, None] & dim_mask[None, :]
    state_mask = state_valid[:, None] & dim_mask[None, :]

    cur_kv = tl.load(
        kv_score + current_idx[:, None] * kv_score_stride0 + (head_offset[:, None] + offs[None, :]) * kv_score_stride1,
        mask=current_mask,
        other=0.0,
    )
    cur_score = tl.load(
        kv_score
        + current_idx[:, None] * kv_score_stride0
        + (STATE_WIDTH + head_offset[:, None] + offs[None, :]) * kv_score_stride1,
        mask=current_mask,
        other=float("-inf"),
    )
    state_kv = tl.load(
        state_buffer + state_row[:, None] * STATE_LAST_DIM + head_offset[:, None] + offs[None, :],
        mask=state_mask,
        other=0.0,
    )
    state_score = tl.load(
        state_buffer + state_row[:, None] * STATE_LAST_DIM + STATE_WIDTH + head_offset[:, None] + offs[None, :],
        mask=state_mask,
        other=float("-inf"),
    )

    kv = tl.where(current_mask, cur_kv, state_kv)
    score = tl.where(current_mask, cur_score, state_score)
    score = tl.softmax(score, dim=0)
    compressed_kv = tl.sum(kv * score, axis=0)

    rms_w = tl.load(norm_weight + offs, mask=dim_mask, other=0.0)
    variance = tl.sum(compressed_kv * compressed_kv, axis=0) / HEAD_DIM
    rrms = tl.rsqrt(variance + rms_eps)
    normed = compressed_kv * rrms * rms_w

    num_pairs: tl.constexpr = BLOCK // 2
    nope_pairs: tl.constexpr = NOPE_DIM // 2
    pair_2d = tl.reshape(normed, (num_pairs, 2))
    even, odd = tl.split(pair_2d)
    pair_idx = tl.arange(0, num_pairs)
    rope_pair_local = pair_idx - nope_pairs
    is_rope_pair = rope_pair_local >= 0
    cs_idx = tl.maximum(rope_pair_local, 0)
    compressed_pos = (position // COMPRESS_RATIO) * COMPRESS_RATIO
    cos_v = tl.load(cos_table + compressed_pos * cos_stride0 + cs_idx, mask=is_rope_pair, other=1.0)
    sin_v = tl.load(sin_table + compressed_pos * sin_stride0 + cs_idx, mask=is_rope_pair, other=0.0)
    new_even = even * cos_v - odd * sin_v
    new_odd = odd * cos_v + even * sin_v
    rotated = tl.interleave(new_even, new_odd)

    page = out_slot // PAGE_SIZE
    token_in_page = out_slot % PAGE_SIZE
    data_base = page * BYTES_PER_PAGE + token_in_page * (NOPE_DIM + ROPE_HEAD_DIM * 2)
    scale_base = page * BYTES_PER_PAGE + PAGE_SIZE * (NOPE_DIM + ROPE_HEAD_DIM * 2) + token_in_page * SCALE_BYTES

    n_quant_blocks: tl.constexpr = BLOCK // QUANT_BLOCK
    n_nope_blocks: tl.constexpr = NOPE_DIM // QUANT_BLOCK
    quant_input = normed.to(tl.bfloat16).to(tl.float32)
    quant_2d = tl.reshape(quant_input, (n_quant_blocks, QUANT_BLOCK))
    abs_2d = tl.abs(quant_2d)
    block_absmax = tl.max(abs_2d, axis=1)
    scale_exp = tl.ceil(libdevice.log2(tl.maximum(block_absmax / FP8_MAX, SCALE_MIN))).to(tl.int32)
    scale = ((scale_exp + 127) << 23).to(tl.float32, bitcast=True)
    kv_fp8 = tl.clamp(quant_2d / scale[:, None], -FP8_MAX, FP8_MAX).to(tl.float8e4nv)
    kv_u8 = tl.reshape(kv_fp8.to(tl.uint8, bitcast=True), (BLOCK,))
    tl.store(out_buffer + data_base + offs, kv_u8, mask=offs < NOPE_DIM)

    scale_idx = tl.arange(0, SCALE_BYTES)
    scale_bytes = tl.where(scale_idx < n_nope_blocks, scale_exp + 127, 0).to(tl.uint8)
    tl.store(out_buffer + scale_base + scale_idx, scale_bytes)

    rope_local = offs - NOPE_DIM
    rope_mask = (offs >= NOPE_DIM) & dim_mask
    rope_ptr = (out_buffer + data_base + NOPE_DIM).to(tl.pointer_type(tl.bfloat16))
    tl.store(rope_ptr + rope_local, rotated.to(tl.bfloat16), mask=rope_mask)
    return


def prepare_compress_states(*, infer_state, layer_idx: int, compress_ratio: int):
    if compress_ratio == 0:
        return None

    mem_manager = infer_state.mem_manager
    if compress_ratio == 4:
        out_slots = mem_manager.full_to_c4_indexs[infer_state.mem_index.long().reshape(-1)]
        state_buffer = mem_manager.get_c4_state_buffer(layer_idx)
        out_pool = mem_manager.c4_pool
    elif compress_ratio == 128:
        out_slots = mem_manager.full_to_c128_indexs[infer_state.mem_index.long().reshape(-1)]
        state_buffer = mem_manager.get_c128_state_buffer(layer_idx)
        out_pool = mem_manager.c128_pool
    else:
        raise AssertionError(f"invalid DeepSeek-V4 compress ratio {compress_ratio}")

    token_to_batch_idx = infer_state.b_req_idx
    if infer_state.is_prefill:
        token_to_batch_idx = getattr(infer_state, "_dsv4_token_to_batch_idx", None)
        if token_to_batch_idx is None or token_to_batch_idx.numel() != infer_state.position_ids.numel():
            q_lens = (infer_state.b_seq_len - infer_state.b_ready_cache_len).to(torch.long)
            batch_idx = torch.arange(infer_state.b_req_idx.shape[0], device=infer_state.b_req_idx.device)
            token_to_batch_idx = torch.repeat_interleave(batch_idx, q_lens).to(torch.int32)
            infer_state._dsv4_token_to_batch_idx = token_to_batch_idx

    return CoreCompressorMetadata(
        layer_idx=layer_idx,
        compress_ratio=compress_ratio,
        out_slots=out_slots,
        mem_index=infer_state.mem_index,
        state_buffer=state_buffer,
        out_buffer=mem_manager.get_compressed_kv_buffer(layer_idx),
        out_page_size=out_pool.page_size,
        position_ids=infer_state.position_ids,
        b_req_idx=infer_state.b_req_idx,
        b_seq_len=infer_state.b_seq_len,
        b_ready_cache_len=infer_state.b_ready_cache_len,
        b_q_start_loc=infer_state.b_q_start_loc,
        req_to_token_indexs=infer_state.req_manager.req_to_token_indexs,
        full_to_swa_indexs=mem_manager.full_to_swa_indexs,
        token_to_batch_idx=token_to_batch_idx,
        kv_score=None,
        is_prefill=infer_state.is_prefill,
    )


def prepare_partial_states(
    *,
    kv_score: torch.Tensor,
    metadata: Optional[CoreCompressorMetadata],
    ape: torch.Tensor,
    compress_ratio: int,
):
    if metadata is None or kv_score.shape[0] == 0:
        return
    state_width = kv_score.shape[-1] // 2
    _add_ape_to_kv_score_kernel[(kv_score.shape[0],)](
        kv_score,
        kv_score.stride(0),
        kv_score.stride(1),
        ape,
        ape.stride(0),
        metadata.position_ids,
        STATE_WIDTH=state_width,
        COMPRESS_RATIO=compress_ratio,
        BLOCK=triton.next_power_of_2(state_width),
        num_warps=4,
    )
    return


def fused_compress(
    *,
    kv_score: torch.Tensor,
    metadata: Optional[CoreCompressorMetadata],
    norm_weight: torch.Tensor,
    ape: torch.Tensor,
    eps: float,
    head_dim: int,
    qk_rope_head_dim: int,
    compress_ratio: int,
    cos_table: torch.Tensor,
    sin_table: torch.Tensor,
):
    if metadata is None or kv_score.shape[0] == 0:
        return

    state_width = kv_score.shape[-1] // 2
    state_last_dim = metadata.state_buffer.shape[-1]
    is_c4 = compress_ratio == 4
    state_ring = DSV4_C4_STATE_RING if is_c4 else DSV4_C128_STATE_RING
    block_state = triton.next_power_of_2(state_width)
    block_head = triton.next_power_of_2(head_dim)

    _fused_compress_norm_rope_insert_kernel[(kv_score.shape[0],)](
        kv_score,
        kv_score.stride(0),
        kv_score.stride(1),
        metadata.state_buffer,
        metadata.position_ids,
        metadata.token_to_batch_idx,
        metadata.b_req_idx,
        metadata.b_seq_len,
        metadata.b_ready_cache_len if metadata.b_ready_cache_len is not None else metadata.b_seq_len,
        metadata.b_q_start_loc if metadata.b_q_start_loc is not None else metadata.b_seq_len,
        metadata.req_to_token_indexs,
        metadata.req_to_token_indexs.stride(0),
        metadata.full_to_swa_indexs,
        metadata.out_slots,
        norm_weight,
        eps,
        cos_table,
        cos_table.stride(0),
        sin_table,
        sin_table.stride(0),
        metadata.out_buffer,
        HEAD_DIM=head_dim,
        STATE_WIDTH=state_width,
        STATE_LAST_DIM=state_last_dim,
        COMPRESS_RATIO=compress_ratio,
        WINDOW_SIZE=compress_ratio * (2 if is_c4 else 1),
        IS_C4=is_c4,
        IS_PREFILL=metadata.is_prefill,
        SWA_PAGE_SIZE=DSV4_SWA_PAGE_SIZE,
        STATE_RING=state_ring,
        ROPE_HEAD_DIM=qk_rope_head_dim,
        FP8_MAX=torch.finfo(torch.float8_e4m3fn).max,
        SCALE_MIN=1e-4,
        NOPE_DIM=head_dim - qk_rope_head_dim,
        QUANT_BLOCK=64,
        SCALE_BYTES=(head_dim - qk_rope_head_dim) // 64 + 1,
        PAGE_SIZE=metadata.out_page_size,
        BYTES_PER_PAGE=metadata.out_buffer.shape[-1],
        BLOCK=block_head,
        num_warps=4,
    )

    _save_partial_states_kernel[(kv_score.shape[0],)](
        kv_score,
        kv_score.stride(0),
        kv_score.stride(1),
        metadata.position_ids,
        metadata.token_to_batch_idx,
        metadata.b_req_idx,
        metadata.b_seq_len,
        metadata.mem_index,
        metadata.full_to_swa_indexs,
        metadata.state_buffer,
        STATE_WIDTH=state_width,
        STATE_LAST_DIM=state_last_dim,
        COMPRESS_RATIO=compress_ratio,
        IS_C4=is_c4,
        IS_PREFILL=metadata.is_prefill,
        SWA_PAGE_SIZE=DSV4_SWA_PAGE_SIZE,
        STATE_RING=state_ring,
        BLOCK=block_state,
        num_warps=4,
    )
    return


def _load_sglang_compressor():
    global _SGLANG_COMPRESS_ERR, _SGLANG_COMPRESS_MOD, _SGLANG_LINEAR_BF16_FP32
    if _SGLANG_COMPRESS_MOD is not None:
        return _SGLANG_COMPRESS_MOD, _SGLANG_LINEAR_BF16_FP32
    if _SGLANG_COMPRESS_ERR is not None:
        raise _SGLANG_COMPRESS_ERR
    try:
        from sglang.jit_kernel.dsv4 import linear_bf16_fp32
        from sglang.jit_kernel.dsv4 import compress_old as compress_mod
    except Exception as exc:
        _SGLANG_COMPRESS_ERR = RuntimeError(
            "DeepSeek-V4 fused compressor requires sglang.jit_kernel.dsv4 "
            "(linear_bf16_fp32 + compress_old). Install/export the SGLang package "
            "or vendor the DSv4 compressor JIT into LightLLM."
        )
        raise _SGLANG_COMPRESS_ERR from exc
    _SGLANG_COMPRESS_MOD = compress_mod
    _SGLANG_LINEAR_BF16_FP32 = linear_bf16_fp32
    return compress_mod, linear_bf16_fp32


def _load_paged_compress_data_fn():
    from sglang.jit_kernel.dsv4 import triton_create_paged_compress_data

    return triton_create_paged_compress_data


def _freq_cis(cos_table, sin_table):
    key = (
        cos_table.data_ptr(),
        sin_table.data_ptr(),
        cos_table.device,
        tuple(cos_table.shape),
        tuple(sin_table.shape),
    )
    cached = _FREQ_CIS_CACHE.get(key)
    if cached is None:
        cached = torch.complex(cos_table.float(), sin_table.float())
        _FREQ_CIS_CACHE[key] = cached
    return cached


def _sglang_ape(ape, ratio, head_dim):
    if ratio == 4:
        return torch.cat([ape[:, :head_dim], ape[:, head_dim:]], dim=0).contiguous()
    return ape.contiguous()


def _compressor_weight(wkv_w, wgate_w):
    return torch.cat([wkv_w, wgate_w], dim=0).contiguous()


def _project_kv_score(x, wkv_w, wgate_w):
    _, linear_bf16_fp32 = _load_sglang_compressor()
    return linear_bf16_fp32(x, _compressor_weight(wkv_w, wgate_w))


def _state_pool_view(state_pool):
    if state_pool is None:
        raise RuntimeError("DeepSeek-V4 fused compressor requires a persistent state_pool")
    if state_pool.dim() == 4 and state_pool.shape[1] == 1:
        return state_pool.squeeze(1)
    return state_pool


def compressor_prefill_state(
    x,
    wkv_w,
    wgate_w,
    norm_w,
    ape,
    ratio,
    head_dim,
    cos_table,
    sin_table,
    eps,
    state_pool,
):
    """start_pos==0 prefill for ONE request: x [s, dim] -> compressed entries [s//ratio, head_dim]
    (rope applied). state_pool is the request's persistent jit state slice [1, slots, coff*2*head_dim];
    it is rebuilt in place so the decode path can continue from the trailing partial window."""
    mod, _ = _load_sglang_compressor()
    kv_score = _project_kv_score(x, wkv_w, wgate_w)
    pool = _state_pool_view(state_pool)
    pool.zero_()
    seq_len = x.shape[0]
    plan = mod.CompressorPrefillPlan.generate(
        ratio,
        seq_len,
        torch.tensor([seq_len], dtype=torch.int64),
        torch.tensor([seq_len], dtype=torch.int64),
        x.device,
    )
    indices = torch.zeros((1,), device=x.device, dtype=torch.int32)
    out = mod.compress_forward(
        pool,
        kv_score,
        _sglang_ape(ape.float(), ratio, head_dim),
        indices,
        plan,
        head_dim=head_dim,
        compress_ratio=ratio,
    )
    ncomp = seq_len // ratio
    if ncomp == 0:
        return x.new_zeros(0, head_dim)
    mod.compress_fused_norm_rope_inplace(out, norm_w.float(), eps, _freq_cis(cos_table, sin_table), plan)
    ragged_ids = plan.compress_plan.view(torch.int32)[:ncomp, 0].long()
    return out.index_select(0, ragged_ids).to(x.dtype)


def compressor_decode_step_single(
    x_new,
    wkv_w,
    wgate_w,
    norm_w,
    ape,
    ratio,
    head_dim,
    cos_table,
    sin_table,
    eps,
    state_pool,
    start_pos,
):
    """One token for ONE request (chunked-prefill extend path). Returns the finished compressed
    entry [head_dim] when (start_pos+1) % ratio == 0, else None. Mutates state_pool in place."""
    mod, _ = _load_sglang_compressor()
    kv_score = _project_kv_score(x_new.view(1, -1), wkv_w, wgate_w)
    pool = _state_pool_view(state_pool)
    seq_len = start_pos + 1
    plan = mod.CompressorDecodePlan(
        ratio,
        torch.tensor([seq_len], device=x_new.device, dtype=torch.int32),
    )
    indices = torch.zeros((1,), device=x_new.device, dtype=torch.int32)
    out = mod.compress_forward(
        pool,
        kv_score,
        _sglang_ape(ape.float(), ratio, head_dim),
        indices,
        plan,
        head_dim=head_dim,
        compress_ratio=ratio,
    )
    if seq_len % ratio != 0:
        return None
    mod.compress_fused_norm_rope_inplace(out, norm_w.float(), eps, _freq_cis(cos_table, sin_table), plan)
    return out[0].to(x_new.dtype)


def compressor_decode_step_batch(
    x_new,
    wkv_w,
    wgate_w,
    norm_w,
    ape,
    ratio,
    head_dim,
    rope_dim,
    cos_table,
    sin_table,
    eps,
    state_pool,
    b_req_idx,
    start_pos,
):
    mod, _ = _load_sglang_compressor()
    kv_score = _project_kv_score(x_new, wkv_w, wgate_w)
    pool = _state_pool_view(state_pool)
    seq_lens = (start_pos + 1).to(torch.int32).contiguous()
    plan = mod.CompressorDecodePlan(ratio, seq_lens)
    out = mod.compress_forward(
        pool,
        kv_score,
        _sglang_ape(ape.float(), ratio, head_dim),
        b_req_idx.to(torch.int32).contiguous(),
        plan,
        head_dim=head_dim,
        compress_ratio=ratio,
    )
    should_compress = (seq_lens % ratio) == 0
    mod.compress_fused_norm_rope_inplace(out, norm_w.float(), eps, _freq_cis(cos_table, sin_table), plan)
    return out.to(x_new.dtype), should_compress


# ---------------------------------------------------------------------------- paged state
# 与 sglang srt compressor 的 paged 路径同构(compress_old 内核 + 分组槽 indices + overlap
# extra_data): state 槽位由 swa 槽位算术派生(翻译③ state_loc = page*ring + swa_loc%ring,
# 分组槽 = state_loc//ratio),state 随 swa 页生灭,radix 命中零拷贝续算。


def paged_state_rows(num_swa_pages: int, ring: int, ratio: int) -> int:
    """state 池行数 = 页数*ring + ring(HOLD 页) + 1(哨兵行),向上取整到 ratio 整除
    (分组视图 [-1, ratio, last_dim] 需要)。与 sglang CompressStatePool 的 _size 公式一致。"""
    rows = num_swa_pages * ring + ring + 1
    return (rows + ratio - 1) // ratio * ratio


def init_paged_state_pool(buffer: torch.Tensor) -> None:
    """末行为哨兵: kv 半边置 0、score 半边置 -inf(KVAndScore.clear 语义)。其余行无需初始化
    (内核在组起点覆写)。buffer: [rows, 2*coff*head_dim] fp32。"""
    half = buffer.shape[-1] // 2
    buffer[-1, :half].zero_()
    buffer[-1, half:].fill_(float("-inf"))
    return


def _paged_state_group_slot(req_to_token, full_to_swa, b_req_idx, positions, page_size, ring, ratio):
    """位置 -> state 分组槽(= sglang create_paged_compressor_data.get_raw_loc):
    state_loc = (swa_loc//page)*ring + swa_loc%ring; 分组槽 = state_loc//ratio。
    负位置按 sglang 语义 mask 到 0;已出窗(swa_loc<0)的位置落到 -1(哨兵行,score=-inf)。"""
    positions = positions.masked_fill(positions < 0, 0)
    full = req_to_token[b_req_idx.long(), positions]
    swa_loc = full_to_swa[full.long()].long()
    state_loc = torch.div(swa_loc, page_size, rounding_mode="floor") * ring + swa_loc % ring
    state_loc = torch.where(swa_loc < 0, torch.full_like(state_loc, -1), state_loc)
    return torch.div(state_loc, ratio, rounding_mode="floor").to(torch.int32)


def paged_decode_state_slots(
    req_to_token,
    full_to_swa,
    b_req_idx,
    b_seq_len,
    page_size: int,
    ring: int,
    ratio: int,
    hold_req_id: int,
    num_swa_pages: int,
    overlap: bool = True,
):
    """decode 步的 state 分组槽(写槽 = 当前组 clip_down(seq-1) 的槽,可选 overlap 前一组)。
    纯张量算术(prep 已写本步 req_to_token),图安全。padding(HOLD)行重定向到 HOLD 页的
    state 槽,隔离其垃圾累加。"""
    seq = b_seq_len.long()
    write_positions = torch.div(seq - 1, ratio, rounding_mode="floor") * ratio
    write_slot = _paged_state_group_slot(req_to_token, full_to_swa, b_req_idx, write_positions, page_size, ring, ratio)
    overlap_slot = None
    if overlap:
        overlap_slot = _paged_state_group_slot(
            req_to_token, full_to_swa, b_req_idx, write_positions - ratio, page_size, ring, ratio
        )
    hold_slot = num_swa_pages * ring // ratio  # HOLD 页区域([pages*ring, pages*ring+ring))的首个分组槽
    is_hold = b_req_idx.long() == hold_req_id
    write_slot = torch.where(is_hold, torch.full_like(write_slot, hold_slot), write_slot)
    if overlap_slot is not None:
        overlap_slot = torch.where(is_hold, torch.full_like(overlap_slot, hold_slot), overlap_slot)
    return write_slot, overlap_slot


def paged_prefill_compress_data(
    req_to_token,
    full_to_swa,
    req_idx: int,
    ready_len: int,
    seq_len: int,
    ring: int,
    ratio: int = 4,
    page_size: int = DSV4_SWA_PAGE_SIZE,
    overlap: bool = True,
):
    """单请求 prefill chunk 的 (indices, extra_data, plan): 与 sglang 同走
    triton_create_paged_compress_data(按请求产出,内核经 plan 逐 token 步进)。
    三者都与层无关,同一 forward 内可跨全部 c4 层复用。"""
    mod, _ = _load_sglang_compressor()
    fn = _load_paged_compress_data_fn()
    device = req_to_token.device
    n_new = seq_len - ready_len
    write_loc, extra_data = fn(
        compress_ratio=ratio,
        is_overlap=overlap,
        swa_page_size=page_size,
        ring_size=ring,
        req_pool_indices=torch.tensor([req_idx], device=device, dtype=torch.int64),
        seq_lens=torch.tensor([seq_len], device=device, dtype=torch.int64),
        extend_seq_lens=torch.tensor([n_new], device=device, dtype=torch.int64),
        req_to_token=req_to_token,
        full_to_swa_index_mapping=full_to_swa,
    )
    plan = mod.CompressorPrefillPlan.generate(
        ratio,
        n_new,
        torch.tensor([seq_len], dtype=torch.int64),
        torch.tensor([n_new], dtype=torch.int64),
        device,
    )
    return write_loc, extra_data, plan


def compressor_paged_prefill(
    x,
    wkv_w,
    wgate_w,
    norm_w,
    ape,
    head_dim,
    cos_table,
    sin_table,
    eps,
    state_buffer,
    compress_data,
    ready_len,
    seq_len,
    ratio: int = 4,
):
    """单请求 prefill/extend chunk(paged): x [n_new, dim] 为位置 [ready, seq) 的 hidden,
    state 写到 swa 派生的分组槽(compress_data 来自 paged_prefill_compress_data,跨层复用)。
    返回本 chunk 完结组的压缩条目 [seq//ratio - ready//ratio, head_dim](rope 已施加)。"""
    mod, _ = _load_sglang_compressor()
    kv_score = _project_kv_score(x, wkv_w, wgate_w)
    pool = state_buffer.view(-1, ratio, state_buffer.shape[-1])
    write_loc, extra_data, plan = compress_data
    kwargs = {"extra_data": extra_data} if extra_data is not None else {}
    out = mod.compress_forward(
        pool,
        kv_score,
        _sglang_ape(ape.float(), ratio, head_dim),
        write_loc,
        plan,
        head_dim=head_dim,
        compress_ratio=ratio,
        **kwargs,
    )
    ncomp = seq_len // ratio - ready_len // ratio
    if ncomp == 0:
        return x.new_zeros(0, head_dim)
    mod.compress_fused_norm_rope_inplace(out, norm_w.float(), eps, _freq_cis(cos_table, sin_table), plan)
    ragged_ids = plan.compress_plan.view(torch.int32)[:ncomp, 0].long()
    return out.index_select(0, ragged_ids).to(x.dtype)


def compressor_paged_decode_batch(
    x_new,
    wkv_w,
    wgate_w,
    norm_w,
    ape,
    head_dim,
    cos_table,
    sin_table,
    eps,
    state_buffer,
    write_slot,
    overlap_slot,
    b_seq_len,
    ratio: int = 4,
):
    """批量 decode 一步(paged): state 槽位为 swa 派生分组槽(paged_decode_state_slots,
    可跨层复用)。返回 (entries [bs, head_dim], should_compress [bs])。"""
    mod, _ = _load_sglang_compressor()
    kv_score = _project_kv_score(x_new, wkv_w, wgate_w)
    pool = state_buffer.view(-1, ratio, state_buffer.shape[-1])
    seq_lens = b_seq_len.to(torch.int32).contiguous()
    plan = mod.CompressorDecodePlan(ratio, seq_lens)
    kwargs = {"extra_data": overlap_slot.view(-1, 1)} if overlap_slot is not None else {}
    out = mod.compress_forward(
        pool,
        kv_score,
        _sglang_ape(ape.float(), ratio, head_dim),
        write_slot,
        plan,
        head_dim=head_dim,
        compress_ratio=ratio,
        **kwargs,
    )
    should_compress = (seq_lens % ratio) == 0
    mod.compress_fused_norm_rope_inplace(out, norm_w.float(), eps, _freq_cis(cos_table, sin_table), plan)
    return out.to(x_new.dtype), should_compress
