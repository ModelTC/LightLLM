import torch
import triton
import triton.language as tl
from triton.language.extra import libdevice


@triton.jit
def _add_ape_to_kv_score_kernel(
    kv_score,
    kv_score_stride0,
    kv_score_stride1,
    ape,
    ape_stride0,
    position_ids,
    CANDIDATE_DIM: tl.constexpr,
    COMPRESS_RATIO: tl.constexpr,
    BLOCK: tl.constexpr,
):
    token_idx = tl.program_id(0)
    offs = tl.arange(0, BLOCK)
    mask = offs < CANDIDATE_DIM

    position = tl.load(position_ids + token_idx)
    ape_row = position % COMPRESS_RATIO
    score = tl.load(kv_score + token_idx * kv_score_stride0 + (CANDIDATE_DIM + offs) * kv_score_stride1, mask=mask)
    ape_value = tl.load(ape + ape_row * ape_stride0 + offs, mask=mask)
    tl.store(
        kv_score + token_idx * kv_score_stride0 + (CANDIDATE_DIM + offs) * kv_score_stride1,
        score + ape_value,
        mask=mask,
    )
    return


@triton.jit
def _save_compressor_state_kernel(
    kv_score,
    kv_score_stride0,
    kv_score_stride1,
    position_ids,
    token_to_batch_idx,
    b_req_idx,
    b_seq_len,
    req_to_compressor_state,
    CANDIDATE_DIM: tl.constexpr,
    STATE_DIM: tl.constexpr,
    IS_PREFILL: tl.constexpr,
    STATE_CAPACITY_TOKENS: tl.constexpr,
    BLOCK: tl.constexpr,
):
    token_idx = tl.program_id(0)
    batch_idx = tl.load(token_to_batch_idx + token_idx) if IS_PREFILL else token_idx
    position = tl.load(position_ids + token_idx)
    seq_len = tl.load(b_seq_len + batch_idx)

    # 只保留请求尾部，绝对位置对 ring 容量取模后写入对应 req_to_*_state。
    if position + STATE_CAPACITY_TOKENS < seq_len:
        return
    req_idx = tl.load(b_req_idx + batch_idx).to(tl.int64)
    state_slot = req_idx * STATE_CAPACITY_TOKENS + position % STATE_CAPACITY_TOKENS

    offs = tl.arange(0, BLOCK)
    mask = offs < CANDIDATE_DIM
    kv = tl.load(kv_score + token_idx * kv_score_stride0 + offs * kv_score_stride1, mask=mask)
    score = tl.load(kv_score + token_idx * kv_score_stride0 + (CANDIDATE_DIM + offs) * kv_score_stride1, mask=mask)
    state_base = req_to_compressor_state + state_slot * STATE_DIM
    tl.store(state_base + offs, kv, mask=mask)
    tl.store(state_base + CANDIDATE_DIM + offs, score, mask=mask)
    return


@triton.jit
def _fused_compress_norm_rope_kernel(
    kv_score,
    kv_score_stride0,
    kv_score_stride1,
    req_to_compressor_state,
    position_ids,
    token_to_batch_idx,
    b_req_idx,
    b_mtp_index,
    b_seq_len,
    b_ready_cache_len,
    b_q_start_loc,
    compressed_kv_slots,
    norm_weight,
    rms_eps,
    cos_table,
    cos_stride0,
    cos_stride1,
    sin_table,
    sin_stride0,
    sin_stride1,
    out_buffer,
    HEAD_DIM: tl.constexpr,
    CANDIDATE_DIM: tl.constexpr,
    STATE_DIM: tl.constexpr,
    COMPRESS_RATIO: tl.constexpr,
    WINDOW_SIZE: tl.constexpr,
    IS_C4: tl.constexpr,
    IS_PREFILL: tl.constexpr,
    STATE_CAPACITY_TOKENS: tl.constexpr,
    ROPE_HEAD_DIM: tl.constexpr,
    FP8_MAX: tl.constexpr,
    AMAX_MIN: tl.constexpr,
    NOPE_DIM: tl.constexpr,
    QUANT_BLOCK: tl.constexpr,
    SCALE_BYTES: tl.constexpr,
    PAGE_SIZE: tl.constexpr,
    BYTES_PER_PAGE: tl.constexpr,
    BLOCK: tl.constexpr,
    IS_IN_INDEXER: tl.constexpr,
):
    token_idx = tl.program_id(0)
    out_slot = tl.load(compressed_kv_slots + token_idx).to(tl.int64)
    if out_slot < 0:
        return

    position = tl.load(position_ids + token_idx)
    batch_idx = tl.load(token_to_batch_idx + token_idx) if IS_PREFILL else token_idx
    req_idx = tl.load(b_req_idx + batch_idx).to(tl.int64)
    seq_len = tl.load(b_seq_len + batch_idx)
    if IS_PREFILL:
        ready_len = tl.load(b_ready_cache_len + batch_idx)
        q_start = tl.load(b_q_start_loc + batch_idx)
    else:
        mtp_index = tl.load(b_mtp_index + batch_idx)
        ready_len = position - mtp_index
        q_start = token_idx - mtp_index

    token_offsets = tl.arange(0, WINDOW_SIZE)
    start = position - WINDOW_SIZE + 1
    gather_pos = start + token_offsets
    valid_pos = (gather_pos >= 0) & (gather_pos < seq_len)
    use_current = (gather_pos >= ready_len) & valid_pos
    current_idx = q_start + (gather_pos - ready_len)
    state_valid = valid_pos & ~use_current
    state_slot = req_idx * STATE_CAPACITY_TOKENS + gather_pos % STATE_CAPACITY_TOKENS
    if IS_C4:
        head_offset = tl.where(token_offsets >= COMPRESS_RATIO, HEAD_DIM, 0)
    else:
        head_offset = token_offsets * 0

    offs = tl.arange(0, BLOCK)
    dim_mask = offs < HEAD_DIM
    current_mask = use_current[:, None] & dim_mask[None, :]
    state_mask = state_valid[:, None] & dim_mask[None, :]

    cur_ptrs = (
        kv_score + current_idx[:, None] * kv_score_stride0 + (head_offset[:, None] + offs[None, :]) * kv_score_stride1
    )
    cur_kv = tl.load(
        cur_ptrs,
        mask=current_mask,
        other=0.0,
    )
    cur_score = tl.load(
        cur_ptrs + CANDIDATE_DIM * kv_score_stride1,
        mask=current_mask,
        other=float("-inf"),
    )

    state_ptrs = req_to_compressor_state + state_slot[:, None] * STATE_DIM + head_offset[:, None] + offs[None, :]
    state_kv = tl.load(
        state_ptrs,
        mask=state_mask,
        other=0.0,
    )
    state_score = tl.load(
        state_ptrs + CANDIDATE_DIM,
        mask=state_mask,
        other=float("-inf"),
    )

    kv = tl.where(current_mask, cur_kv, state_kv)
    score = tl.where(current_mask, cur_score, state_score)
    score = tl.softmax(score, dim=0)
    compressed_kv = tl.sum(kv * score, axis=0)
    # rms_norm
    rms_w = tl.load(norm_weight + offs, mask=dim_mask, other=0.0)
    variance = tl.sum(compressed_kv * compressed_kv, axis=0) / HEAD_DIM
    rrms = tl.rsqrt(variance + rms_eps)
    normed = compressed_kv * rrms * rms_w

    # rope
    num_pairs: tl.constexpr = BLOCK // 2
    nope_pairs: tl.constexpr = NOPE_DIM // 2
    pair_2d = tl.reshape(normed, (num_pairs, 2))
    even, odd = tl.split(pair_2d)
    pair_idx = tl.arange(0, num_pairs)
    rope_pair_local = pair_idx - nope_pairs
    is_rope_pair = rope_pair_local >= 0
    cs_idx = tl.maximum(rope_pair_local, 0)
    compressed_pos = (position // COMPRESS_RATIO) * COMPRESS_RATIO
    cos_v = tl.load(cos_table + compressed_pos * cos_stride0 + cs_idx * cos_stride1, mask=is_rope_pair, other=1.0)
    sin_v = tl.load(sin_table + compressed_pos * sin_stride0 + cs_idx * sin_stride1, mask=is_rope_pair, other=0.0)
    new_even = even * cos_v - odd * sin_v
    new_odd = odd * cos_v + even * sin_v
    rotated = tl.interleave(new_even, new_odd)

    if IS_IN_INDEXER:
        # indexer K 输出 BF16，由 destindex_copy_indexer_k 写入 indexer_k_buffer。
        # 它将整条向量量化为 FP8，并保存一个 FP32 scale。
        tl.store(out_buffer + token_idx * HEAD_DIM + offs, rotated.to(tl.bfloat16), mask=dim_mask)
        return

    page = out_slot // PAGE_SIZE
    token_in_page = out_slot % PAGE_SIZE
    data_base = page * BYTES_PER_PAGE + token_in_page * (NOPE_DIM + ROPE_HEAD_DIM * 2)
    scale_base = page * BYTES_PER_PAGE + PAGE_SIZE * (NOPE_DIM + ROPE_HEAD_DIM * 2) + token_in_page * SCALE_BYTES

    n_quant_blocks: tl.constexpr = BLOCK // QUANT_BLOCK
    n_nope_blocks: tl.constexpr = NOPE_DIM // QUANT_BLOCK
    # Preserve BF16 rounding before FP8 packing.
    quant_input = normed.to(tl.bfloat16).to(tl.float32)
    quant_2d = tl.reshape(quant_input, (n_quant_blocks, QUANT_BLOCK))
    abs_2d = tl.abs(quant_2d)
    block_absmax = tl.maximum(tl.max(abs_2d, axis=1), AMAX_MIN)
    scale_exp = tl.ceil(libdevice.log2(block_absmax / FP8_MAX)).to(tl.int32)
    scale = ((scale_exp + 127) << 23).to(tl.float32, bitcast=True)
    kv_fp8 = tl.clamp(quant_2d / scale[:, None], -FP8_MAX, FP8_MAX).to(tl.float8e4nv)
    kv_u8 = tl.reshape(kv_fp8.to(tl.uint8, bitcast=True), (BLOCK,))
    tl.store(out_buffer + data_base + offs, kv_u8, mask=offs < NOPE_DIM)

    # ue8m0
    scale_idx = tl.arange(0, SCALE_BYTES)
    scale_bytes = tl.where(scale_idx < n_nope_blocks, scale_exp + 127, 0).to(tl.uint8)
    tl.store(out_buffer + scale_base + scale_idx, scale_bytes)

    rope_local = offs - NOPE_DIM
    rope_mask = (offs >= NOPE_DIM) & dim_mask
    rope_ptr = (out_buffer + data_base + NOPE_DIM).to(tl.pointer_type(tl.bfloat16))
    tl.store(rope_ptr + rope_local, rotated.to(tl.bfloat16), mask=rope_mask)
    return


def apply_ape(
    *,
    kv_score: torch.Tensor,
    position_ids: torch.Tensor,
    ape: torch.Tensor,
    compress_ratio: int,
):
    candidate_dim = kv_score.shape[-1] // 2
    _add_ape_to_kv_score_kernel[(kv_score.shape[0],)](
        kv_score,
        kv_score.stride(0),
        kv_score.stride(1),
        ape,
        ape.stride(0),
        position_ids,
        CANDIDATE_DIM=candidate_dim,
        COMPRESS_RATIO=compress_ratio,
        BLOCK=triton.next_power_of_2(candidate_dim),
        num_warps=4,
    )
    return
