import torch
import triton
import triton.language as tl


# Bound the temporary BF16 gather buffer to 2 GiB (2097152 × 1 × 512 × 2 bytes).
KV_GATHER_CHUNK_SIZE = 2097152


@triton.jit
def _direct_smallq_packed_kernel(
    Q,
    PackedKV,
    RaggedMemIndex,
    PrefillMemIndex,
    PrefillKV,
    KVStarts,
    Topk,
    PartialO,
    PartialLse,
    ragged_n,
    q_stride_s: tl.constexpr,
    q_stride_h: tl.constexpr,
    q_stride_d: tl.constexpr,
    packed_stride_s: tl.constexpr,
    prefill_stride_s: tl.constexpr,
    prefill_stride_d: tl.constexpr,
    topk_stride_s: tl.constexpr,
    topk_stride_d: tl.constexpr,
    HEADS_: tl.constexpr,
    PREFILL_N: tl.constexpr,
    SPLITS_: tl.constexpr,
    BLOCKS_: tl.constexpr,
    SLOTS_: tl.constexpr,
    softmax_scale: tl.constexpr,
):
    qid = tl.program_id(0)
    split = tl.program_id(1)
    h = tl.arange(0, 16)
    d = tl.arange(0, 512)
    hm = h < HEADS_
    q = tl.load(
        Q + qid * q_stride_s + h[:, None] * q_stride_h + d[None, :] * q_stride_d,
        hm[:, None],
        0.0,
    ).to(tl.bfloat16)
    acc = tl.zeros((16, 512), tl.float32)
    m = tl.full((16,), -float("inf"), tl.float32)
    l = tl.zeros((16,), tl.float32)
    start = tl.load(KVStarts + qid).to(tl.int64)
    for b in tl.range(0, BLOCKS_, num_stages=1, loop_unroll_factor=1, disallow_acc_multi_buffer=True):
        n = tl.arange(0, 64)
        slot = split * SLOTS_ + b * 64 + n
        top = tl.load(
            Topk + qid * topk_stride_s + slot * topk_stride_d,
            slot < 2176,
            -1,
        ).to(tl.int64)
        ragged_pos = start + top
        valid = (top >= 0) & (ragged_pos >= 0) & (ragged_pos < ragged_n)
        mem = tl.load(RaggedMemIndex + ragged_pos, valid, 0).to(tl.int64)
        current_row = tl.full((64,), -1, tl.int32)
        for j in tl.static_range(PREFILL_N):
            current_row = tl.where(valid & (mem == tl.load(PrefillMemIndex + j).to(tl.int64)), j, current_row)
        cached = valid & (current_row < 0)
        current = valid & (current_row >= 0)
        packed_row = PackedKV + mem[:, None] * packed_stride_s
        fp8 = tl.load(
            packed_row.to(tl.pointer_type(tl.float8e4nv)) + d[None, :],
            cached[:, None] & (d[None, :] < 448),
            0.0,
        ).to(tl.float32)
        scales = tl.load(
            packed_row + 576 + d[None, :] // 64,
            cached[:, None] & (d[None, :] < 448),
            127,
        )
        fp8 = fp8 * (scales.to(tl.uint32) << 23).to(tl.float32, bitcast=True)
        tail = tl.load(
            (packed_row + 448).to(tl.pointer_type(tl.bfloat16)) + d[None, :] - 448,
            cached[:, None] & (d[None, :] >= 448),
            0.0,
        ).to(tl.float32)
        cached_v = tl.where(d[None, :] < 448, fp8, tail)
        current_v = tl.load(
            PrefillKV + current_row[:, None] * prefill_stride_s + d[None, :] * prefill_stride_d,
            current[:, None],
            0.0,
        ).to(tl.float32)
        kv = tl.where(current[:, None], current_v, cached_v).to(tl.bfloat16)
        score = tl.dot(q, tl.trans(kv), out_dtype=tl.float32) * softmax_scale
        score = tl.where(valid[None, :], score, -float("inf"))
        block_m = tl.max(score, axis=1)
        new_m = tl.maximum(m, block_m)
        alpha = tl.where(m == -float("inf"), 0.0, tl.exp(m - new_m))
        p = tl.where(valid[None, :], tl.exp(score - new_m[:, None]), 0.0)
        l = l * alpha + tl.sum(p, axis=1)
        acc = acc * alpha[:, None] + tl.dot(p.to(tl.bfloat16), kv, out_dtype=tl.float32)
        m = new_m
    has = l > 0.0
    out = tl.where(has[:, None], acc / l[:, None], 0.0)
    tl.store(
        PartialO + ((qid * SPLITS_ + split) * HEADS_ + h[:, None]) * 512 + d[None, :],
        out,
        hm[:, None],
    )
    tl.store(
        PartialLse + (qid * SPLITS_ + split) * HEADS_ + h,
        tl.where(has, m + tl.log(l), -float("inf")),
        hm,
    )


@triton.jit
def _reduce_direct_smallq_kernel(
    PartialO,
    PartialLse,
    Out,
    HEADS_: tl.constexpr,
    SPLITS_: tl.constexpr,
    SPAD_: tl.constexpr,
):
    qid = tl.program_id(0)
    h = tl.program_id(1)
    d = tl.program_id(2) * 128 + tl.arange(0, 128)
    s = tl.arange(0, SPAD_)
    sm = s < SPLITS_
    lse = tl.load(PartialLse + (qid * SPLITS_ + s) * HEADS_ + h, sm, -float("inf"))
    maximum = tl.max(lse, axis=0)
    safe_maximum = tl.where(maximum == -float("inf"), 0.0, maximum)
    weight = tl.where(sm & (lse > -float("inf")), tl.exp(lse - safe_maximum), 0.0)
    denominator = tl.sum(weight, axis=0)
    partial = tl.load(
        PartialO + ((qid * SPLITS_ + s[:, None]) * HEADS_ + h) * 512 + d[None, :],
        sm[:, None],
        0.0,
    ).to(tl.float32)
    value = tl.sum(partial * weight[:, None], axis=0) / tl.where(denominator > 0.0, denominator, 1.0)
    tl.store(Out + (qid * HEADS_ + h) * 512 + d, value)


def _can_use_direct_smallq(q, packed_kv, ragged_mem_index, prefill_mem_index, prefill_kv, topk_indices):
    return (
        q.ndim == 3
        and 1 <= q.shape[0] <= 8
        and q.dtype == torch.bfloat16
        and q.shape[1:] == (8, 512)
        and packed_kv.ndim == 3
        and packed_kv.shape[1] == 1
        and packed_kv.shape[2] >= 584
        and ragged_mem_index.dtype == torch.int32
        and ragged_mem_index.numel() >= 65536
        and prefill_mem_index.dtype == torch.int32
        and prefill_mem_index.numel() == q.shape[0]
        and prefill_kv.dtype == torch.bfloat16
        and prefill_kv.ndim == 3
        and prefill_kv.shape[0] == q.shape[0]
        and prefill_kv.shape[1:] == (1, 512)
        and topk_indices.dtype == torch.int32
        and topk_indices.ndim == 2
        and topk_indices.shape == (q.shape[0], 2176)
    )


@torch.no_grad()
def _context_attention_fwd_fp8_direct_smallq(
    q,
    packed_kv,
    ragged_mem_index,
    prefill_mem_index,
    prefill_kv,
    kv_starts,
    topk_indices,
    softmax_scale,
):
    qn = q.shape[0]
    # 2176 selected keys ÷ 64: Q1 uses 34 one-tile splits; Q2..8 use
    # 17 two-tile splits.  The restricted dispatch keeps this fixed layout.
    splits, blocks, slots = (34, 1, 64) if qn == 1 else (17, 2, 128)
    partial_o = torch.empty((qn, splits, 8, 512), dtype=torch.float32, device=q.device)
    partial_lse = torch.empty((qn, splits, 8), dtype=torch.float32, device=q.device)
    # Reducer writes a contiguous [Q, H, 512] layout; do not inherit a
    # transpose/view stride from q.
    output = torch.empty(q.shape, dtype=q.dtype, device=q.device)
    _direct_smallq_packed_kernel[(qn, splits)](
        q,
        packed_kv,
        ragged_mem_index,
        prefill_mem_index,
        prefill_kv,
        kv_starts,
        topk_indices,
        partial_o,
        partial_lse,
        ragged_mem_index.numel(),
        q.stride(0),
        q.stride(1),
        q.stride(2),
        packed_kv.stride(0),
        prefill_kv.stride(0),
        prefill_kv.stride(2),
        topk_indices.stride(0),
        topk_indices.stride(1),
        HEADS_=8,
        PREFILL_N=qn,
        SPLITS_=splits,
        BLOCKS_=blocks,
        SLOTS_=slots,
        softmax_scale=softmax_scale,
        num_warps=8,
    )
    _reduce_direct_smallq_kernel[(qn, 8, 4)](
        partial_o,
        partial_lse,
        output,
        HEADS_=8,
        SPLITS_=splits,
        SPAD_=triton.next_power_of_2(splits),
        num_warps=4,
    )
    return output


@triton.jit
def _build_prefill_row_table_kernel(PrefillMemIndex, RowTable, num_tokens, BLOCK: tl.constexpr):
    row = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mem_index = tl.load(PrefillMemIndex + row, row < num_tokens, 0)
    tl.store(RowTable + mem_index, row, row < num_tokens)


@triton.jit
def _gather_prefill_kv_kernel(
    PackedKV,
    MemIndex,
    PrefillRowTable,
    PrefillKV,
    Output,
    num_tokens,
    packed_stride_s: tl.constexpr,
    prefill_stride_s: tl.constexpr,
    prefill_stride_d: tl.constexpr,
    BLOCK_ROWS: tl.constexpr,
):
    row = tl.program_id(0) * BLOCK_ROWS + tl.arange(0, BLOCK_ROWS)
    dim = tl.arange(0, 512)
    mem_index = tl.load(MemIndex + row, row < num_tokens, 0).to(tl.int64)
    prefill_row = tl.load(PrefillRowTable + mem_index, row < num_tokens, -1)
    cached = (row < num_tokens) & (prefill_row < 0)
    current = (row < num_tokens) & (prefill_row >= 0)
    packed_row = PackedKV + mem_index[:, None] * packed_stride_s

    quantized = tl.load(
        packed_row.to(tl.pointer_type(tl.float8e4nv)) + dim[None, :],
        cached[:, None] & (dim[None, :] < 448),
        0.0,
    ).to(tl.float32)
    scales = tl.load(packed_row + 576 + dim[None, :] // 64, cached[:, None] & (dim[None, :] < 448), 127)
    values = quantized * (scales.to(tl.uint32) << 23).to(tl.float32, bitcast=True)
    tail = tl.load(
        (packed_row + 448).to(tl.pointer_type(tl.bfloat16)) + dim[None, :] - 448,
        cached[:, None] & (dim[None, :] >= 448),
        0,
    ).to(tl.float32)
    values = tl.where(dim[None, :] < 448, values, tail)
    current_values = tl.load(
        PrefillKV + prefill_row[:, None] * prefill_stride_s + dim[None, :] * prefill_stride_d,
        current[:, None],
        0,
    ).to(tl.float32)
    tl.store(
        Output + row[:, None] * 512 + dim[None, :],
        tl.where(current[:, None], current_values, values),
        row[:, None] < num_tokens,
    )


@triton.jit
def _compact_topk_indices_kernel(
    TopkIndices,
    KVStarts,
    ChunkIndices,
    ChunkLengths,
    chunk_start,
    chunk_end,
    TOPK: tl.constexpr,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    lane = tl.arange(0, BLOCK)
    index = tl.load(TopkIndices + row * TOPK + lane, lane < TOPK, -1)
    ragged_index = index + tl.load(KVStarts + row)
    valid = (lane < TOPK) & (index >= 0) & (ragged_index >= chunk_start) & (ragged_index < chunk_end)
    compact_index = tl.cumsum(valid.to(tl.int32)) - 1
    tl.store(ChunkIndices + row * TOPK + compact_index, ragged_index - chunk_start, valid)
    tl.store(ChunkLengths + row, tl.sum(valid.to(tl.int32)))


@triton.jit
def _merge_att_output_kernel(
    Output,
    Lse,
    ChunkOutput,
    ChunkLse,
    chunk_lse_stride_s: tl.constexpr,
    chunk_lse_stride_h: tl.constexpr,
    NUM_HEADS: tl.constexpr,
    NUM_Q_TOKENS: tl.constexpr,
    BLOCK_HEADS: tl.constexpr,
):
    row = tl.program_id(0)
    head = tl.program_id(1) * BLOCK_HEADS + tl.arange(0, BLOCK_HEADS)
    dim = tl.arange(0, 512)
    previous_lse = tl.load(Lse + head * NUM_Q_TOKENS + row, head < NUM_HEADS, -float("inf"))
    chunk_lse = tl.load(
        ChunkLse + row * chunk_lse_stride_s + head * chunk_lse_stride_h, head < NUM_HEADS, -float("inf")
    )
    # FA3 uses natural-log LSE and returns +inf for an empty row.
    previous_lse = tl.where(previous_lse == float("inf"), -float("inf"), previous_lse)
    chunk_lse = tl.where(chunk_lse == float("inf"), -float("inf"), chunk_lse)
    maximum = tl.maximum(previous_lse, chunk_lse)
    maximum = tl.where(maximum == -float("inf"), 0, maximum)
    previous_weight = tl.exp(previous_lse - maximum)
    chunk_weight = tl.exp(chunk_lse - maximum)
    denominator = previous_weight + chunk_weight
    offset = (row * NUM_HEADS + head[:, None]) * 512 + dim[None, :]
    previous = tl.load(Output + offset, (head[:, None] < NUM_HEADS) & (previous_weight[:, None] > 0), 0)
    current = tl.load(ChunkOutput + offset, (head[:, None] < NUM_HEADS) & (chunk_weight[:, None] > 0), 0).to(tl.float32)
    result = (previous * previous_weight[:, None] + current * chunk_weight[:, None]) / tl.where(
        denominator > 0, denominator, 1
    )[:, None]
    tl.store(Output + offset, result, head[:, None] < NUM_HEADS)
    tl.store(
        Lse + head * NUM_Q_TOKENS + row,
        tl.where(denominator > 0, maximum + tl.log(denominator), -float("inf")),
        head < NUM_HEADS,
    )


@torch.no_grad()
def context_attention_fwd_fp8(
    q: torch.Tensor,
    packed_kv: torch.Tensor,
    ragged_mem_index: torch.Tensor,
    prefill_mem_index: torch.Tensor,
    prefill_kv: torch.Tensor,
    kv_starts: torch.Tensor,
    topk_indices: torch.Tensor,
    softmax_scale: float,
):
    """Run NoPE sparse prefill with a bounded gather buffer and BF16 current KV."""
    # Direct packed attention is intentionally restricted to the validated
    # small-Q GLM-5.3 shape; all other cases retain the bounded-gather path.
    if _can_use_direct_smallq(q, packed_kv, ragged_mem_index, prefill_mem_index, prefill_kv, topk_indices):
        return _context_attention_fwd_fp8_direct_smallq(
            q, packed_kv, ragged_mem_index, prefill_mem_index, prefill_kv, kv_starts, topk_indices, softmax_scale
        )
    from sgl_kernel.flash_attn import flash_attn_with_kvcache

    num_q_tokens, num_heads, _ = q.shape
    prefill_row_table = torch.full((packed_kv.shape[0],), -1, dtype=torch.int32, device=q.device)
    _build_prefill_row_table_kernel[(triton.cdiv(prefill_mem_index.numel(), 256),)](
        prefill_mem_index, prefill_row_table, prefill_mem_index.numel(), BLOCK=256
    )
    gather_chunk_size = min(KV_GATHER_CHUNK_SIZE, ragged_mem_index.numel())
    gathered_kv = q.new_empty((gather_chunk_size, 1, 512))
    chunk_indices = torch.empty_like(topk_indices)
    chunk_lengths = torch.empty(num_q_tokens, dtype=torch.int32, device=q.device)
    cu_seqlens_q = torch.arange(num_q_tokens + 1, dtype=torch.int32, device=q.device)
    output = lse = None
    for chunk_start in range(0, ragged_mem_index.numel(), gather_chunk_size):
        num_chunk_tokens = min(gather_chunk_size, ragged_mem_index.numel() - chunk_start)
        _gather_prefill_kv_kernel[(triton.cdiv(num_chunk_tokens, 4),)](
            packed_kv,
            ragged_mem_index[chunk_start : chunk_start + num_chunk_tokens],
            prefill_row_table,
            prefill_kv,
            gathered_kv,
            num_chunk_tokens,
            packed_kv.stride(0),
            prefill_kv.stride(0),
            prefill_kv.stride(2),
            BLOCK_ROWS=4,
        )
        _compact_topk_indices_kernel[(num_q_tokens,)](
            topk_indices,
            kv_starts,
            chunk_indices,
            chunk_lengths,
            chunk_start,
            chunk_start + num_chunk_tokens,
            TOPK=topk_indices.shape[1],
            BLOCK=triton.next_power_of_2(topk_indices.shape[1]),
        )
        chunk_kv = gathered_kv[:num_chunk_tokens].view(-1, 1, 1, 512)
        chunk_output, chunk_lse, *_ = flash_attn_with_kvcache(
            # only_qv ignores Q/K; these views avoid allocating placeholder tensors.
            q=q[..., :64],
            k_cache=chunk_kv[..., :64],
            qv=q,
            v_cache=chunk_kv,
            page_table=chunk_indices,
            cache_seqlens=chunk_lengths,
            cu_seqlens_q=cu_seqlens_q,
            max_seqlen_q=1,
            softmax_scale=softmax_scale,
            causal=False,
            only_qv=True,
            num_splits=1,
            return_softmax_lse=True,
        )
        if chunk_start == 0:
            output, lse = chunk_output, chunk_lse
        else:
            # Keep the online reduction in FP32 across all remaining chunks.
            output = output.float()
            _merge_att_output_kernel[(num_q_tokens, triton.cdiv(num_heads, 4))](
                output,
                lse,
                chunk_output,
                chunk_lse,
                chunk_lse.stride(1),
                chunk_lse.stride(0),
                NUM_HEADS=num_heads,
                NUM_Q_TOKENS=num_q_tokens,
                BLOCK_HEADS=4,
            )
        del chunk_output, chunk_lse
    return output.to(q.dtype)
