import torch
import triton
import triton.language as tl


# Bound the temporary BF16 gather buffer to 2 GiB (2097152 × 1 × 512 × 2 bytes).
KV_GATHER_CHUNK_SIZE = 2097152


@triton.jit
def _direct_smallq_packed_kernel(
    Q,
    PackedKV,
    PrefillMemIndex,
    PrefillKV,
    Topk,
    PartialO,
    PartialLse,
    packed_rows,
    q_stride_s: tl.constexpr,
    q_stride_h: tl.constexpr,
    q_stride_d: tl.constexpr,
    packed_stride_s: tl.constexpr,
    prefill_stride_s: tl.constexpr,
    prefill_stride_d: tl.constexpr,
    topk_stride_s: tl.constexpr,
    topk_stride_d: tl.constexpr,
    HEADS_: tl.constexpr,
    BLOCK_H_: tl.constexpr,
    PREFILL_N: tl.constexpr,
    SPLITS_: tl.constexpr,
    BLOCKS_: tl.constexpr,
    SLOTS_: tl.constexpr,
    softmax_scale: tl.constexpr,
):
    qid = tl.program_id(0)
    split = tl.program_id(1)
    if HEADS_ == 8:
        # H8 fits in one fixed 16-lane tile; the head mask excludes inactive lanes.
        h = tl.arange(0, 16)
    else:
        h = tl.program_id(2) * BLOCK_H_ + tl.arange(0, BLOCK_H_)
    d = tl.arange(0, 512)
    hm = h < HEADS_
    q = tl.load(
        Q + qid * q_stride_s + h[:, None] * q_stride_h + d[None, :] * q_stride_d,
        hm[:, None],
        0.0,
    ).to(tl.bfloat16)
    if HEADS_ == 8:
        acc = tl.zeros((16, 512), tl.float32)
        m = tl.full((16,), -float("inf"), tl.float32)
        denominator = tl.zeros((16,), tl.float32)
    else:
        acc = tl.zeros((BLOCK_H_, 512), tl.float32)
        m = tl.full((BLOCK_H_,), -float("inf"), tl.float32)
        denominator = tl.zeros((BLOCK_H_,), tl.float32)
    for b in tl.range(0, BLOCKS_, num_stages=1, loop_unroll_factor=1, disallow_acc_multi_buffer=True):
        n = tl.arange(0, 64)
        slot = split * SLOTS_ + b * 64 + n
        top = tl.load(
            Topk + qid * topk_stride_s + slot * topk_stride_d,
            slot < 2176,
            -1,
        ).to(tl.int64)
        valid = (top >= 0) & (top < packed_rows)
        mem = tl.where(valid, top, 0)
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
        denominator = denominator * alpha + tl.sum(p, axis=1)
        acc = acc * alpha[:, None] + tl.dot(p.to(tl.bfloat16), kv, out_dtype=tl.float32)
        m = new_m
    has = denominator > 0.0
    out = tl.where(has[:, None], acc / denominator[:, None], 0.0)
    tl.store(
        PartialO + ((qid * SPLITS_ + split) * HEADS_ + h[:, None]) * 512 + d[None, :],
        out,
        hm[:, None],
    )
    tl.store(
        PartialLse + (qid * SPLITS_ + split) * HEADS_ + h,
        tl.where(has, m + tl.log(denominator), -float("inf")),
        hm,
    )


@triton.jit
def reduce_direct_smallq_kernel(
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


def can_use_direct_smallq(q, packed_kv, prefill_mem_index, prefill_kv, topk_mem_indices, total_kv_tokens):
    return (
        q.ndim == 3
        and 1 <= q.shape[0] <= 8
        and q.dtype == torch.bfloat16
        and (q.shape[1:] == (8, 512) or (q.shape[0] == 1 and q.shape[1:] == (64, 512)))
        and packed_kv.ndim == 3
        and packed_kv.shape[1] == 1
        and packed_kv.shape[2] >= 584
        and total_kv_tokens >= 65536
        and prefill_mem_index.dtype == torch.int32
        and prefill_mem_index.numel() == q.shape[0]
        and prefill_kv.dtype == torch.bfloat16
        and prefill_kv.shape == (q.shape[0], 1, 512)
        and topk_mem_indices.dtype == torch.int32
        and topk_mem_indices.shape == (q.shape[0], 2176)
    )


def allocate_direct_smallq_workspace(q):
    qn, heads, _ = q.shape
    splits = 34 if qn == 1 else 17
    return {
        "partial_o": torch.empty((qn, splits, heads, q.shape[-1]), dtype=torch.float32, device=q.device),
        "partial_lse": torch.empty((qn, splits, heads), dtype=torch.float32, device=q.device),
        # The reducer requires a contiguous [Q, H, 512] destination.
        "output": torch.empty(q.shape, dtype=q.dtype, device=q.device),
    }


@torch.no_grad()
def _context_attention_fwd_fp8_direct_smallq(
    q, packed_kv, prefill_mem_index, prefill_kv, topk_mem_indices, softmax_scale, direct_workspace=None
):
    qn = q.shape[0]
    heads = q.shape[1]
    # Q1 uses 34 64-key splits; H64 uses two H32 head tiles per split.
    splits, blocks, slots = (34, 1, 64) if qn == 1 else (17, 2, 128)
    block_h = 32 if heads == 64 else 16
    if heads == 8:
        # H8 uses local buffers because its direct path has no reusable workspace.
        partial_o = torch.empty((qn, splits, 8, 512), dtype=torch.float32, device=q.device)
        partial_lse = torch.empty((qn, splits, 8), dtype=torch.float32, device=q.device)
        output = torch.empty(q.shape, dtype=q.dtype, device=q.device)
        launch_grid = (qn, splits)
    else:
        workspace = direct_workspace or allocate_direct_smallq_workspace(q)
        partial_o = workspace["partial_o"]
        partial_lse = workspace["partial_lse"]
        output = workspace["output"]
        launch_grid = (qn, splits, triton.cdiv(heads, block_h))
    _direct_smallq_packed_kernel[launch_grid](
        q,
        packed_kv,
        prefill_mem_index,
        prefill_kv,
        topk_mem_indices,
        partial_o,
        partial_lse,
        packed_kv.shape[0],
        q.stride(0),
        q.stride(1),
        q.stride(2),
        packed_kv.stride(0),
        prefill_kv.stride(0),
        prefill_kv.stride(2),
        topk_mem_indices.stride(0),
        topk_mem_indices.stride(1),
        HEADS_=heads,
        BLOCK_H_=block_h,
        PREFILL_N=qn,
        SPLITS_=splits,
        BLOCKS_=blocks,
        SLOTS_=slots,
        softmax_scale=softmax_scale,
        num_warps=8,
    )
    reduce_direct_smallq_kernel[(qn, heads, 4)](
        partial_o,
        partial_lse,
        output,
        HEADS_=heads,
        SPLITS_=splits,
        SPAD_=triton.next_power_of_2(splits),
        num_warps=4,
    )
    return output


@triton.jit
def _gather_prefill_kv_kernel(
    PackedKV,
    PrefillKV,
    Output,
    ReqToToken,
    ReqIdx,
    CuKV,
    CuQ,
    chunk_start,
    num_tokens,
    packed_stride_s: tl.constexpr,
    prefill_stride_s: tl.constexpr,
    prefill_stride_d: tl.constexpr,
    REQ_STRIDE: tl.constexpr,
    TOKEN_STRIDE: tl.constexpr,
    BATCH: tl.constexpr,
    BLOCK_ROWS: tl.constexpr,
):
    block = tl.program_id(0)
    lanes = tl.arange(0, triton.next_power_of_2(BATCH))
    ends = tl.load(CuKV + lanes + 1, lanes < BATCH, 0)
    ends = tl.minimum(tl.maximum(ends - chunk_start, 0), num_tokens)
    block_ends = ends // BLOCK_ROWS + lanes + 1
    batch = tl.sum(((block >= block_ends) & (lanes < BATCH)).to(tl.int32), 0)
    start = tl.load(CuKV + batch)
    copy_start = tl.minimum(tl.maximum(start - chunk_start, 0), num_tokens)
    copy_end = tl.minimum(tl.maximum(tl.load(CuKV + batch + 1) - chunk_start, 0), num_tokens)
    local_block = block - (copy_start // BLOCK_ROWS + batch)
    row = copy_start + local_block * BLOCK_ROWS + tl.arange(0, BLOCK_ROWS)
    live = row < copy_end
    req = tl.load(ReqIdx + batch).to(tl.int64)
    position = chunk_start + row - start
    q_start = tl.load(CuQ + batch)
    q_len = tl.load(CuQ + batch + 1) - q_start
    cache_len = tl.load(CuKV + batch + 1) - start - q_len
    prefill_row = tl.where(position >= cache_len, q_start + position - cache_len, -1)
    mem_index = tl.load(ReqToToken + req * REQ_STRIDE + position * TOKEN_STRIDE, live & (prefill_row < 0), 0).to(
        tl.int64
    )
    dim = tl.arange(0, 512)
    cached = live & (prefill_row < 0)
    current = live & (prefill_row >= 0)
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
        live[:, None],
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
    COMPACT: tl.constexpr = True,
):
    row = tl.program_id(0)
    lane = tl.arange(0, BLOCK)
    index = tl.load(TopkIndices + row * TOPK + lane, lane < TOPK, -1)
    kv_index = index + tl.load(KVStarts + row)
    valid = (lane < TOPK) & (index >= 0) & (kv_index >= chunk_start) & (kv_index < chunk_end)
    if COMPACT:
        compact_index = tl.cumsum(valid.to(tl.int32)) - 1
        tl.store(ChunkIndices + row * TOPK + compact_index, kv_index - chunk_start, valid)
        tl.store(ChunkLengths + row, tl.sum(valid.to(tl.int32)))
    else:
        tl.store(
            ChunkIndices + row * TOPK + lane,
            tl.where(valid, kv_index - chunk_start, -1),
            lane < TOPK,
        )


@triton.jit
def _merge_att_output_kernel(
    Output,
    Lse,
    ChunkOutput,
    ChunkLse,
    chunk_lse_stride_s: tl.constexpr,
    chunk_lse_stride_h,
    NUM_HEADS: tl.constexpr,
    NUM_Q_TOKENS,
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


@triton.jit
def _selected_prefill_compact_kernel(
    Topk,
    CompactPhysical,
    Lengths,
    PageTable,
    CuQ,
    packed_rows: tl.constexpr,
    topk_stride_s: tl.constexpr,
    topk_stride_d: tl.constexpr,
    BATCH: tl.constexpr,
    BLOCK: tl.constexpr,
):
    qid = tl.program_id(0)
    lane = tl.arange(0, BLOCK)
    local = tl.load(Topk + qid * topk_stride_s + lane * topk_stride_d, lane < 2176, -1).to(tl.int64)
    valid = (lane < 2176) & (local >= 0)
    physical = local
    valid = valid & (physical >= 0) & (physical < packed_rows)
    rank = tl.cumsum(valid.to(tl.int32), axis=0) - 1
    tl.store(CompactPhysical + qid * 2176 + rank, physical, valid)
    # Every selected slot has a stable contiguous page; invalid tail pages are
    # harmless because cache_seqlens limits FA3 to Lengths[qid].
    tl.store(PageTable + qid * 2176 + lane, qid * 2176 + lane, lane < 2176)
    if qid == 0:
        tl.store(CuQ + lane, lane.to(tl.int32), lane <= BATCH)
    tl.store(Lengths + qid, tl.sum(valid.to(tl.int32), axis=0))


@triton.jit
def _selected_prefill_gather_kernel(
    CompactPhysical,
    Lengths,
    PrefillMemIndex,
    PrefillKV,
    PackedKV,
    SelectedKV,
    CuK,
    packed_stride_s: tl.constexpr,
    prefill_stride_s: tl.constexpr,
    prefill_stride_d: tl.constexpr,
    BATCH: tl.constexpr,
    BATCH_PAD: tl.constexpr,
):
    qid = tl.program_id(0)
    row = tl.program_id(1) * 4 + tl.arange(0, 4)
    length = tl.load(Lengths + qid)
    live = row < length
    physical = tl.load(CompactPhysical + qid * 2176 + row, live, 0).to(tl.int64)
    current_row = tl.full((4,), -1, tl.int32)
    for j in tl.static_range(BATCH_PAD):
        current = j < BATCH
        current_mem = tl.load(PrefillMemIndex + j, current, 0).to(tl.int64)
        current_row = tl.where(live & current & (physical == current_mem), j, current_row)
    cached = live & (current_row < 0)
    current = live & (current_row >= 0)
    dim = tl.arange(0, 512)
    packed_row = PackedKV + physical[:, None] * packed_stride_s
    group = tl.arange(0, 8)
    raw_scales = tl.load(packed_row + 576 + group[None, :], cached[:, None] & (group[None, :] < 7), 127)
    scales = (raw_scales.to(tl.uint32) << 23).to(tl.float32, bitcast=True)
    scales = tl.broadcast_to(scales[:, :, None], (4, 8, 64)).reshape(4, 512)
    fp8 = (
        tl.load(
            packed_row.to(tl.pointer_type(tl.float8e4nv)) + dim[None, :],
            cached[:, None] & (dim[None, :] < 448),
            0.0,
        ).to(tl.float32)
        * scales
    )
    tail = tl.load(
        (packed_row + 448).to(tl.pointer_type(tl.bfloat16)) + dim[None, :] - 448,
        cached[:, None] & (dim[None, :] >= 448),
        0.0,
    ).to(tl.float32)
    cached_value = tl.where(dim[None, :] < 448, fp8, tail)
    current_value = tl.load(
        PrefillKV + current_row[:, None] * prefill_stride_s + dim[None, :] * prefill_stride_d,
        current[:, None],
        0.0,
    ).to(tl.float32)
    value = tl.where(current[:, None], current_value, cached_value).to(tl.bfloat16)
    tl.store(SelectedKV + (qid * 2176 + row[:, None]) * 512 + dim[None, :], value, row[:, None] < 2176)
    if (qid == 0) & (tl.program_id(1) == 0):
        x = tl.arange(0, BATCH_PAD)
        lengths = tl.load(Lengths + x, x < BATCH, 0)
        tl.store(CuK, 0)
        tl.store(CuK + 1 + x, tl.cumsum(lengths, axis=0), x < BATCH)


def can_use_selected_prefill(q, packed_kv, prefill_mem_index, prefill_kv, topk_mem_indices, total_kv_tokens):
    return (
        q.ndim == 3
        and 2 <= q.shape[0] <= 8
        and q.shape[1:] == (64, 512)
        and q.dtype == torch.bfloat16
        and q.is_cuda
        and q.stride(-1) == 1
        and packed_kv.shape[1:] == (1, 584)
        and packed_kv.dtype == torch.uint8
        and packed_kv.device == q.device
        and total_kv_tokens >= 65536
        and prefill_mem_index.dtype == torch.int32
        and prefill_mem_index.device == q.device
        and prefill_mem_index.shape == (q.shape[0],)
        and prefill_kv.dtype == torch.bfloat16
        and prefill_kv.device == q.device
        and prefill_kv.shape == (q.shape[0], 1, 512)
        and topk_mem_indices.dtype == torch.int32
        and topk_mem_indices.device == q.device
        and topk_mem_indices.shape == (q.shape[0], 2176)
    )


def allocate_selected_prefill_workspace(q, batch_capacity=None):
    batch = q.shape[0] if batch_capacity is None else batch_capacity
    return {
        "compact": torch.empty((batch, 2176), dtype=torch.int64, device=q.device),
        "lengths": torch.empty(batch, dtype=torch.int32, device=q.device),
        "page_table": torch.empty((batch, 2176), dtype=torch.int32, device=q.device),
        "cu_q": torch.empty(batch + 1, dtype=torch.int32, device=q.device),
        "cu_k": torch.empty(batch + 1, dtype=torch.int32, device=q.device),
        "selected_kv": torch.empty((batch * 2176, 1, q.shape[-1]), dtype=q.dtype, device=q.device),
    }


def allocate_generic_prefill_workspace(q, topk_indices, query_capacity=None):
    capacity = q.shape[0] if query_capacity is None else query_capacity
    return {
        "chunk_indices": torch.empty((capacity, topk_indices.shape[1]), dtype=topk_indices.dtype, device=q.device),
        "chunk_lengths": torch.empty(capacity, dtype=torch.int32, device=q.device),
        "cu_seqlens_q": torch.arange(capacity + 1, dtype=torch.int32, device=q.device),
    }


@torch.no_grad()
def _context_attention_fwd_fp8_selected_prefill(
    q, packed_kv, prefill_mem_index, prefill_kv, topk_mem_indices, softmax_scale, selected_workspace=None
):
    from sgl_kernel.flash_attn import flash_attn_with_kvcache

    batch = q.shape[0]
    workspace = selected_workspace or allocate_selected_prefill_workspace(q)
    batch_pad = triton.next_power_of_2(batch)
    _selected_prefill_compact_kernel[(batch,)](
        topk_mem_indices,
        workspace["compact"],
        workspace["lengths"],
        workspace["page_table"],
        workspace["cu_q"],
        packed_kv.shape[0],
        topk_mem_indices.stride(0),
        topk_mem_indices.stride(1),
        BATCH=batch,
        BLOCK=4096,
        num_warps=8,
    )
    _selected_prefill_gather_kernel[(batch, triton.cdiv(2176, 4))](
        workspace["compact"],
        workspace["lengths"],
        prefill_mem_index,
        prefill_kv,
        packed_kv,
        workspace["selected_kv"],
        workspace["cu_k"],
        packed_kv.stride(0),
        prefill_kv.stride(0),
        prefill_kv.stride(2),
        BATCH=batch,
        BATCH_PAD=batch_pad,
        num_warps=4,
    )
    return flash_attn_with_kvcache(
        q=None,
        qv=q,
        k_cache=None,
        v_cache=workspace["selected_kv"].view(-1, 1, 1, 512),
        page_table=workspace["page_table"],
        cache_seqlens=workspace["lengths"],
        cu_seqlens_q=workspace["cu_q"],
        cu_seqlens_k_new=workspace["cu_k"],
        max_seqlen_q=1,
        softmax_scale=softmax_scale,
        causal=False,
        only_qv=True,
    )


@torch.no_grad()
def context_attention_fwd_fp8(
    q,
    packed_kv,
    prefill_mem_index,
    prefill_kv,
    kv_starts,
    topk_indices,
    topk_mem_indices,
    req_to_token_index,
    b_req_idx,
    cu_kv_seq_len,
    cu_q_seq_len,
    total_kv_tokens,
    softmax_scale,
    selected_workspace=None,
    direct_workspace=None,
    generic_workspace=None,
):
    """Run NoPE sparse prefill from request mappings and physical top-k slots."""
    eligibility = (q, packed_kv, prefill_mem_index, prefill_kv, topk_mem_indices, total_kv_tokens)
    if can_use_direct_smallq(*eligibility):
        return _context_attention_fwd_fp8_direct_smallq(
            q, packed_kv, prefill_mem_index, prefill_kv, topk_mem_indices, softmax_scale, direct_workspace
        )
    if can_use_selected_prefill(*eligibility):
        return _context_attention_fwd_fp8_selected_prefill(
            q, packed_kv, prefill_mem_index, prefill_kv, topk_mem_indices, softmax_scale, selected_workspace
        )
    from sgl_kernel.flash_attn import flash_attn_with_kvcache

    num_q_tokens, num_heads, _ = q.shape
    num_kv_tokens = total_kv_tokens
    gather_chunk_size = min(KV_GATHER_CHUNK_SIZE, num_kv_tokens)
    workspace = generic_workspace or allocate_generic_prefill_workspace(q, topk_indices)
    chunk_indices = workspace["chunk_indices"]
    chunk_lengths = workspace["chunk_lengths"]
    cu_seqlens_q = workspace["cu_seqlens_q"]
    # Keep this potentially 2 GiB buffer local to a DSA layer.
    gathered_kv = q.new_empty((gather_chunk_size, 1, 512))
    output = lse = None
    for chunk_start in range(0, num_kv_tokens, gather_chunk_size):
        num_chunk_tokens = min(gather_chunk_size, num_kv_tokens - chunk_start)
        _gather_prefill_kv_kernel[(num_chunk_tokens // 4 + b_req_idx.numel(),)](
            packed_kv,
            prefill_kv,
            gathered_kv,
            req_to_token_index,
            b_req_idx,
            cu_kv_seq_len,
            cu_q_seq_len,
            chunk_start,
            num_chunk_tokens,
            packed_kv.stride(0),
            prefill_kv.stride(0),
            prefill_kv.stride(2),
            REQ_STRIDE=req_to_token_index.stride(0),
            TOKEN_STRIDE=req_to_token_index.stride(1),
            BATCH=b_req_idx.numel(),
            BLOCK_ROWS=4,
        )
        chunk_kv = gathered_kv[:num_chunk_tokens].view(-1, 1, 1, 512)
        if (
            num_q_tokens > 8
            and num_heads == 64
            and num_kv_tokens > num_q_tokens
            and num_kv_tokens <= KV_GATHER_CHUNK_SIZE
            and q.dtype == torch.bfloat16
            and q.is_contiguous()
            and packed_kv.dtype == torch.uint8
            and packed_kv.shape[1:] == (1, 584)
            and topk_indices.dtype == torch.int32
            and topk_indices.shape[1] == 2176
            and topk_indices.is_contiguous()
        ):
            from sgl_kernel.flash_mla import flash_mla_sparse_fwd

            _compact_topk_indices_kernel[(num_q_tokens,)](
                topk_indices,
                kv_starts,
                chunk_indices,
                chunk_lengths,
                chunk_start,
                chunk_start + num_chunk_tokens,
                TOPK=topk_indices.shape[1],
                BLOCK=triton.next_power_of_2(topk_indices.shape[1]),
                COMPACT=False,
            )
            return flash_mla_sparse_fwd(
                q=q,
                kv=gathered_kv[:num_chunk_tokens],
                indices=chunk_indices.unsqueeze(1),
                sm_scale=softmax_scale,
                d_v=512,
            )[0]
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
