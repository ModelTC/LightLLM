import torch
import triton
import triton.language as tl
from triton.language.extra import libdevice

from lightllm.common.kv_cache_mem_manager.deepseek4_mem_manager import DSV4_SWA_PAGE_SIZE


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

    if IS_C4:
        full_slot = tl.load(mem_index + token_idx).to(tl.int64)
        swa_slot = tl.load(full_to_swa + full_slot).to(tl.int64)
        if swa_slot < 0:
            return
        state_row = (swa_slot // SWA_PAGE_SIZE) * STATE_RING + (swa_slot % STATE_RING)
    else:
        req_idx = tl.load(b_req_idx + batch_idx).to(tl.int64)
        state_row = req_idx * STATE_RING + position % STATE_RING

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
    b_mtp_index,
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
    cos_stride1,
    sin_table,
    sin_stride0,
    sin_stride1,
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
        mtp_index = tl.load(b_mtp_index + batch_idx)
        ready_len = position - mtp_index
        q_start = token_idx - mtp_index

    token_offsets = tl.arange(0, WINDOW_SIZE)
    # position 11, start 4
    start = position - WINDOW_SIZE + 1
    # 最近的window个索引 [4, 5, 6, 7, 8, 9, 10, 11]
    gather_pos = start + token_offsets
    # 都在[0, 12)之间，valid [1, 1, 1, 1, 1, 1, 1, 1]
    valid_pos = (gather_pos >= 0) & (gather_pos < seq_len)
    # 新计算的mask ready_len=8 [0, 0, 0, 0, 1, 1, 1, 1]
    use_current = (gather_pos >= ready_len) & valid_pos
    # 现在的kvscore的索引 [-4, -3, -2, -1, 0, 1, 2, 3]
    current_idx = q_start + (gather_pos - ready_len)
    # cache的mask [1, 1, 1, 1, 0, 0, 0, 0]
    valid_pos = valid_pos & (~use_current)
    cache_pos = valid_pos

    if IS_C4:
        full_slot = tl.load(
            req_to_token + req_idx * req_to_token_stride0 + gather_pos,
            mask=cache_pos,
            other=0,
        ).to(tl.int64)
        swa_slot = tl.load(full_to_swa + full_slot, mask=cache_pos, other=-1).to(tl.int64)
        state_row = (swa_slot // SWA_PAGE_SIZE) * STATE_RING + (swa_slot % STATE_RING)
        state_valid = cache_pos & (swa_slot >= 0)
        head_offset = tl.where(token_offsets >= COMPRESS_RATIO, HEAD_DIM, 0)
    else:
        state_row = req_idx * STATE_RING + gather_pos % STATE_RING
        state_valid = cache_pos
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
        cur_ptrs + STATE_WIDTH * kv_score_stride1,
        mask=current_mask,
        other=float("-inf"),
    )

    state_ptrs = state_buffer + state_row[:, None] * STATE_LAST_DIM + head_offset[:, None] + offs[None, :]
    state_kv = tl.load(
        state_ptrs,
        mask=state_mask,
        other=0.0,
    )
    state_score = tl.load(
        state_ptrs + STATE_WIDTH,
        mask=state_mask,
        other=float("-inf"),
    )

    kv = tl.where(current_mask, cur_kv, state_kv)
    score = tl.where(current_mask, cur_score, state_score)
    score = tl.softmax(score, dim=0)
    compressed_kv = tl.sum(kv * score, axis=0)
    # 以上得到compressed_kv

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
        # indexer-K path: emit the post-rope full HEAD_DIM vector as dense bf16 (token-indexed),
        # leaving the fp8 single-amax pack to destindex_copy_indexer_k_dsv4 (the c4_indexer_pool
        # ABI differs from the latent slab: whole-vector fp8 + one fp32 scale, no bf16 rope tail).
        tl.store(out_buffer + token_idx * HEAD_DIM + offs, rotated.to(tl.bfloat16), mask=dim_mask)
        return

    page = out_slot // PAGE_SIZE
    token_in_page = out_slot % PAGE_SIZE
    data_base = page * BYTES_PER_PAGE + token_in_page * (NOPE_DIM + ROPE_HEAD_DIM * 2)
    scale_base = page * BYTES_PER_PAGE + PAGE_SIZE * (NOPE_DIM + ROPE_HEAD_DIM * 2) + token_in_page * SCALE_BYTES

    n_quant_blocks: tl.constexpr = BLOCK // QUANT_BLOCK
    n_nope_blocks: tl.constexpr = NOPE_DIM // QUANT_BLOCK
    # 有待检验是否有必要
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
    state_width = kv_score.shape[-1] // 2
    _add_ape_to_kv_score_kernel[(kv_score.shape[0],)](
        kv_score,
        kv_score.stride(0),
        kv_score.stride(1),
        ape,
        ape.stride(0),
        position_ids,
        STATE_WIDTH=state_width,
        COMPRESS_RATIO=compress_ratio,
        BLOCK=triton.next_power_of_2(state_width),
        num_warps=4,
    )
    return


def fused_compress(
    *,
    kv_score: torch.Tensor,
    infer_state,
    layer_idx: int,
    norm_weight: torch.Tensor,
    eps: float,
    head_dim: int,
    qk_rope_head_dim: int,
    compress_ratio: int,
    cos_table: torch.Tensor,
    sin_table: torch.Tensor,
    is_in_indexer: bool = False,
    out_buffer: torch.Tensor = None,
):
    mem_manager = infer_state.mem_manager
    if is_in_indexer:
        assert compress_ratio == 4, "只有 c4(CSA) 层有 indexer-K"
        out_slots = mem_manager.full_to_c4_indexs[infer_state.mem_index.reshape(-1)]
        state_buffer = mem_manager.get_c4_indexer_state_buffer(layer_idx)
        state_ring = mem_manager.c4_state_ring
        if out_buffer is None:
            out_buffer = torch.empty(
                (infer_state.mem_index.numel(), mem_manager.indexer_head_dim),
                dtype=torch.bfloat16,
                device=infer_state.mem_index.device,
            )
        out_page_size = 1
    else:
        if compress_ratio == 4:
            out_slots = mem_manager.full_to_c4_indexs[infer_state.mem_index.reshape(-1)]
            state_buffer = mem_manager.get_c4_state_buffer(layer_idx)
            state_ring = mem_manager.c4_state_ring
            out_page_size = mem_manager.c4_pool.page_size
        elif compress_ratio == 128:
            out_slots = mem_manager.full_to_c128_indexs[infer_state.mem_index.reshape(-1)]
            state_buffer = mem_manager.get_c128_state_buffer(layer_idx)
            state_ring = mem_manager.c128_state_ring
            out_page_size = mem_manager.c128_pool.page_size
        out_buffer = mem_manager.get_compressed_kv_buffer(layer_idx)

    if kv_score.shape[0] == 0:
        return out_buffer

    state_width = kv_score.shape[-1] // 2
    state_last_dim = state_buffer.shape[-1]
    is_c4 = compress_ratio == 4
    block_state = triton.next_power_of_2(state_width)
    block_head = triton.next_power_of_2(head_dim)
    token_to_batch_idx = infer_state._dsv4_token_to_batch_idx if infer_state.is_prefill else infer_state.b_req_idx
    ready_cache_len = (
        infer_state.b_ready_cache_len if infer_state.b_ready_cache_len is not None else infer_state.b_seq_len
    )
    q_start_loc = infer_state.b_q_start_loc if infer_state.b_q_start_loc is not None else infer_state.b_seq_len
    req_to_token_indexs = infer_state.req_manager.req_to_token_indexs

    _fused_compress_norm_rope_insert_kernel[(kv_score.shape[0],)](
        kv_score,
        kv_score.stride(0),
        kv_score.stride(1),
        state_buffer,
        infer_state.position_ids,
        token_to_batch_idx,
        infer_state.b_req_idx,
        infer_state.b_mtp_index,
        infer_state.b_seq_len,
        ready_cache_len,
        q_start_loc,
        req_to_token_indexs,
        req_to_token_indexs.stride(0),
        mem_manager.full_to_swa_indexs,
        out_slots,
        norm_weight,
        eps,
        cos_table,
        cos_table.stride(0),
        cos_table.stride(1),
        sin_table,
        sin_table.stride(0),
        sin_table.stride(1),
        out_buffer,
        HEAD_DIM=head_dim,
        STATE_WIDTH=state_width,
        STATE_LAST_DIM=state_last_dim,
        COMPRESS_RATIO=compress_ratio,
        WINDOW_SIZE=compress_ratio * (2 if is_c4 else 1),
        IS_C4=is_c4,
        IS_PREFILL=infer_state.is_prefill,
        SWA_PAGE_SIZE=DSV4_SWA_PAGE_SIZE,
        STATE_RING=state_ring,
        ROPE_HEAD_DIM=qk_rope_head_dim,
        FP8_MAX=torch.finfo(torch.float8_e4m3fn).max,
        AMAX_MIN=1e-4,
        NOPE_DIM=head_dim - qk_rope_head_dim,
        QUANT_BLOCK=64,
        SCALE_BYTES=(head_dim - qk_rope_head_dim) // 64 + 1,
        PAGE_SIZE=out_page_size,
        BYTES_PER_PAGE=out_buffer.shape[-1],
        BLOCK=block_head,
        IS_IN_INDEXER=is_in_indexer,
        num_warps=4,
    )

    _save_partial_states_kernel[(kv_score.shape[0],)](
        kv_score,
        kv_score.stride(0),
        kv_score.stride(1),
        infer_state.position_ids,
        token_to_batch_idx,
        infer_state.b_req_idx,
        infer_state.b_seq_len,
        infer_state.mem_index,
        mem_manager.full_to_swa_indexs,
        state_buffer,
        STATE_WIDTH=state_width,
        STATE_LAST_DIM=state_last_dim,
        COMPRESS_RATIO=compress_ratio,
        IS_C4=is_c4,
        IS_PREFILL=infer_state.is_prefill,
        SWA_PAGE_SIZE=DSV4_SWA_PAGE_SIZE,
        STATE_RING=state_ring,
        BLOCK=block_state,
        num_warps=4,
    )
    return out_buffer
