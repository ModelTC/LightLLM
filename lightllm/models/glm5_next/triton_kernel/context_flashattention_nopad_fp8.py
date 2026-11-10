import torch
import triton
import triton.language as tl


# Bound the temporary BF16 gather buffer to 2 GiB (2097152 × 1 × 512 × 2 bytes).
KV_GATHER_CHUNK_SIZE = 2097152


@triton.jit
def _direct_prefill_packed_kernel(
    q,
    packed_kv,
    mem_index,
    prefill_cache_kv,
    topk_mem_indices,
    partial_output,
    partial_softmax_lse,
    kv_cache_capacity,
    q_stride_s: tl.constexpr,
    q_stride_h: tl.constexpr,
    q_stride_d: tl.constexpr,
    packed_kv_stride_s: tl.constexpr,
    prefill_cache_kv_stride_s: tl.constexpr,
    prefill_cache_kv_stride_d: tl.constexpr,
    topk_mem_indices_stride_s: tl.constexpr,
    topk_mem_indices_stride_d: tl.constexpr,
    q_head_num: tl.constexpr,
    BLOCK_HEADS: tl.constexpr,
    q_token_num: tl.constexpr,
    NUM_SPLITS: tl.constexpr,
    KV_BLOCKS_PER_SPLIT: tl.constexpr,
    KV_SLOTS_PER_SPLIT: tl.constexpr,
    softmax_scale: tl.constexpr,
):
    qid = tl.program_id(0)
    split = tl.program_id(1)
    if q_head_num == 8:
        # H8 fits in one fixed 16-lane tile; the head mask excludes inactive lanes.
        h = tl.arange(0, 16)
    else:
        h = tl.program_id(2) * BLOCK_HEADS + tl.arange(0, BLOCK_HEADS)
    d = tl.arange(0, 512)
    hm = h < q_head_num
    q_block = tl.load(
        q + qid * q_stride_s + h[:, None] * q_stride_h + d[None, :] * q_stride_d,
        hm[:, None],
        0.0,
    ).to(tl.bfloat16)
    if q_head_num == 8:
        acc = tl.zeros((16, 512), tl.float32)
        m = tl.full((16,), -float("inf"), tl.float32)
        denominator = tl.zeros((16,), tl.float32)
    else:
        acc = tl.zeros((BLOCK_HEADS, 512), tl.float32)
        m = tl.full((BLOCK_HEADS,), -float("inf"), tl.float32)
        denominator = tl.zeros((BLOCK_HEADS,), tl.float32)
    for b in tl.range(0, KV_BLOCKS_PER_SPLIT, num_stages=1, loop_unroll_factor=1, disallow_acc_multi_buffer=True):
        n = tl.arange(0, 64)
        slot = split * KV_SLOTS_PER_SPLIT + b * 64 + n
        top = tl.load(
            topk_mem_indices + qid * topk_mem_indices_stride_s + slot * topk_mem_indices_stride_d,
            slot < 2176,
            -1,
        ).to(tl.int64)
        valid = (top >= 0) & (top < kv_cache_capacity)
        kv_mem_index = tl.where(valid, top, 0)
        current_row = tl.full((64,), -1, tl.int32)
        for j in tl.static_range(q_token_num):
            current_row = tl.where(valid & (kv_mem_index == tl.load(mem_index + j).to(tl.int64)), j, current_row)
        cached = valid & (current_row < 0)
        current = valid & (current_row >= 0)
        packed_row = packed_kv + kv_mem_index[:, None] * packed_kv_stride_s
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
            prefill_cache_kv
            + current_row[:, None] * prefill_cache_kv_stride_s
            + d[None, :] * prefill_cache_kv_stride_d,
            current[:, None],
            0.0,
        ).to(tl.float32)
        kv = tl.where(current[:, None], current_v, cached_v).to(tl.bfloat16)
        score = tl.dot(q_block, tl.trans(kv), out_dtype=tl.float32) * softmax_scale
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
        partial_output + ((qid * NUM_SPLITS + split) * q_head_num + h[:, None]) * 512 + d[None, :],
        out,
        hm[:, None],
    )
    tl.store(
        partial_softmax_lse + (qid * NUM_SPLITS + split) * q_head_num + h,
        tl.where(has, m + tl.log(denominator), -float("inf")),
        hm,
    )


@triton.jit
def reduce_sparse_attention_kernel(
    partial_output,
    partial_softmax_lse,
    output,
    q_head_num: tl.constexpr,
    NUM_SPLITS: tl.constexpr,
    BLOCK_SPLITS: tl.constexpr,
):
    qid = tl.program_id(0)
    h = tl.program_id(1)
    d = tl.program_id(2) * 128 + tl.arange(0, 128)
    s = tl.arange(0, BLOCK_SPLITS)
    sm = s < NUM_SPLITS
    softmax_lse = tl.load(partial_softmax_lse + (qid * NUM_SPLITS + s) * q_head_num + h, sm, -float("inf"))
    maximum = tl.max(softmax_lse, axis=0)
    safe_maximum = tl.where(maximum == -float("inf"), 0.0, maximum)
    weight = tl.where(sm & (softmax_lse > -float("inf")), tl.exp(softmax_lse - safe_maximum), 0.0)
    denominator = tl.sum(weight, axis=0)
    partial = tl.load(
        partial_output + ((qid * NUM_SPLITS + s[:, None]) * q_head_num + h) * 512 + d[None, :],
        sm[:, None],
        0.0,
    ).to(tl.float32)
    value = tl.sum(partial * weight[:, None], axis=0) / tl.where(denominator > 0.0, denominator, 1.0)
    tl.store(output + (qid * q_head_num + h) * 512 + d, value)


def can_use_direct_prefill(q, packed_kv, mem_index, prefill_cache_kv, topk_mem_indices, total_token_num):
    return (
        q.ndim == 3
        and 1 <= q.shape[0] <= 8
        and q.dtype == torch.bfloat16
        and (q.shape[1:] == (8, 512) or (q.shape[0] == 1 and q.shape[1:] == (64, 512)))
        and packed_kv.ndim == 3
        and packed_kv.shape[1] == 1
        and packed_kv.shape[2] >= 584
        and total_token_num >= 65536
        and mem_index.dtype == torch.int32
        and mem_index.numel() == q.shape[0]
        and prefill_cache_kv.dtype == torch.bfloat16
        and prefill_cache_kv.shape == (q.shape[0], 1, 512)
        and topk_mem_indices.dtype == torch.int32
        and topk_mem_indices.shape == (q.shape[0], 2176)
    )


def allocate_direct_prefill_workspace(q):
    q_token_num, q_head_num, _ = q.shape
    num_splits = 34 if q_token_num == 1 else 17
    return {
        "partial_output": torch.empty(
            (q_token_num, num_splits, q_head_num, q.shape[-1]), dtype=torch.float32, device=q.device
        ),
        "partial_softmax_lse": torch.empty((q_token_num, num_splits, q_head_num), dtype=torch.float32, device=q.device),
        # The reducer requires a contiguous [Q, H, 512] destination.
        "output": torch.empty(q.shape, dtype=q.dtype, device=q.device),
    }


@torch.no_grad()
def _context_attention_fwd_fp8_direct_prefill(
    q, packed_kv, mem_index, prefill_cache_kv, topk_mem_indices, softmax_scale, direct_workspace=None
):
    q_token_num = q.shape[0]
    q_head_num = q.shape[1]
    # Q1 uses 34 64-key splits; H64 uses two H32 head tiles per split.
    num_splits, kv_blocks_per_split, kv_slots_per_split = (34, 1, 64) if q_token_num == 1 else (17, 2, 128)
    block_heads = 32 if q_head_num == 64 else 16
    if q_head_num == 8:
        # H8 uses local buffers because its direct path has no reusable workspace.
        partial_output = torch.empty((q_token_num, num_splits, 8, 512), dtype=torch.float32, device=q.device)
        partial_softmax_lse = torch.empty((q_token_num, num_splits, 8), dtype=torch.float32, device=q.device)
        output = torch.empty(q.shape, dtype=q.dtype, device=q.device)
        launch_grid = (q_token_num, num_splits)
    else:
        workspace = direct_workspace or allocate_direct_prefill_workspace(q)
        partial_output = workspace["partial_output"]
        partial_softmax_lse = workspace["partial_softmax_lse"]
        output = workspace["output"]
        launch_grid = (q_token_num, num_splits, triton.cdiv(q_head_num, block_heads))
    _direct_prefill_packed_kernel[launch_grid](
        q,
        packed_kv,
        mem_index,
        prefill_cache_kv,
        topk_mem_indices,
        partial_output,
        partial_softmax_lse,
        packed_kv.shape[0],
        q.stride(0),
        q.stride(1),
        q.stride(2),
        packed_kv.stride(0),
        prefill_cache_kv.stride(0),
        prefill_cache_kv.stride(2),
        topk_mem_indices.stride(0),
        topk_mem_indices.stride(1),
        q_head_num=q_head_num,
        BLOCK_HEADS=block_heads,
        q_token_num=q_token_num,
        NUM_SPLITS=num_splits,
        KV_BLOCKS_PER_SPLIT=kv_blocks_per_split,
        KV_SLOTS_PER_SPLIT=kv_slots_per_split,
        softmax_scale=softmax_scale,
        num_warps=8,
    )
    reduce_sparse_attention_kernel[(q_token_num, q_head_num, 4)](
        partial_output,
        partial_softmax_lse,
        output,
        q_head_num=q_head_num,
        NUM_SPLITS=num_splits,
        BLOCK_SPLITS=triton.next_power_of_2(num_splits),
        num_warps=4,
    )
    return output


@triton.jit
def _gather_prefill_kv_kernel(
    packed_kv,
    prefill_cache_kv,
    output,
    req_to_token_indexs,
    b_req_idx,
    b1_cu_kv_seq_len,
    b1_cu_q_seq_len,
    chunk_start,
    chunk_token_num,
    packed_kv_stride_s: tl.constexpr,
    prefill_cache_kv_stride_s: tl.constexpr,
    prefill_cache_kv_stride_d: tl.constexpr,
    req_to_token_indexs_stride_b: tl.constexpr,
    req_to_token_indexs_stride_s: tl.constexpr,
    batch_size: tl.constexpr,
    BLOCK_ROWS: tl.constexpr,
):
    block = tl.program_id(0)
    lanes = tl.arange(0, triton.next_power_of_2(batch_size))
    ends = tl.load(b1_cu_kv_seq_len + lanes + 1, lanes < batch_size, 0)
    ends = tl.minimum(tl.maximum(ends - chunk_start, 0), chunk_token_num)
    block_ends = ends // BLOCK_ROWS + lanes + 1
    batch = tl.sum(((block >= block_ends) & (lanes < batch_size)).to(tl.int32), 0)
    start = tl.load(b1_cu_kv_seq_len + batch)
    copy_start = tl.minimum(tl.maximum(start - chunk_start, 0), chunk_token_num)
    copy_end = tl.minimum(tl.maximum(tl.load(b1_cu_kv_seq_len + batch + 1) - chunk_start, 0), chunk_token_num)
    local_block = block - (copy_start // BLOCK_ROWS + batch)
    row = copy_start + local_block * BLOCK_ROWS + tl.arange(0, BLOCK_ROWS)
    live = row < copy_end
    req = tl.load(b_req_idx + batch).to(tl.int64)
    position = chunk_start + row - start
    q_start = tl.load(b1_cu_q_seq_len + batch)
    q_len = tl.load(b1_cu_q_seq_len + batch + 1) - q_start
    cache_len = tl.load(b1_cu_kv_seq_len + batch + 1) - start - q_len
    prefill_row = tl.where(position >= cache_len, q_start + position - cache_len, -1)
    kv_mem_index = tl.load(
        req_to_token_indexs + req * req_to_token_indexs_stride_b + position * req_to_token_indexs_stride_s,
        live & (prefill_row < 0),
        0,
    ).to(tl.int64)
    dim = tl.arange(0, 512)
    cached = live & (prefill_row < 0)
    current = live & (prefill_row >= 0)
    packed_row = packed_kv + kv_mem_index[:, None] * packed_kv_stride_s

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
        prefill_cache_kv + prefill_row[:, None] * prefill_cache_kv_stride_s + dim[None, :] * prefill_cache_kv_stride_d,
        current[:, None],
        0,
    ).to(tl.float32)
    tl.store(
        output + row[:, None] * 512 + dim[None, :],
        tl.where(current[:, None], current_values, values),
        live[:, None],
    )


@triton.jit
def _compact_topk_indices_kernel(
    topk_indices,
    q_kv_start_loc,
    chunk_topk_indices,
    chunk_cache_seqlens,
    chunk_start,
    chunk_end,
    TOPK: tl.constexpr,
    BLOCK: tl.constexpr,
    COMPACT: tl.constexpr = True,
):
    row = tl.program_id(0)
    lane = tl.arange(0, BLOCK)
    index = tl.load(topk_indices + row * TOPK + lane, lane < TOPK, -1)
    kv_index = index + tl.load(q_kv_start_loc + row)
    valid = (lane < TOPK) & (index >= 0) & (kv_index >= chunk_start) & (kv_index < chunk_end)
    if COMPACT:
        compact_index = tl.cumsum(valid.to(tl.int32)) - 1
        tl.store(chunk_topk_indices + row * TOPK + compact_index, kv_index - chunk_start, valid)
        tl.store(chunk_cache_seqlens + row, tl.sum(valid.to(tl.int32)))
    else:
        tl.store(
            chunk_topk_indices + row * TOPK + lane,
            tl.where(valid, kv_index - chunk_start, -1),
            lane < TOPK,
        )


@triton.jit
def _merge_att_output_kernel(
    output,
    softmax_lse,
    chunk_output,
    chunk_softmax_lse,
    chunk_lse_stride_s: tl.constexpr,
    chunk_lse_stride_h,
    q_head_num: tl.constexpr,
    q_token_num,
    BLOCK_HEADS: tl.constexpr,
):
    row = tl.program_id(0)
    head = tl.program_id(1) * BLOCK_HEADS + tl.arange(0, BLOCK_HEADS)
    dim = tl.arange(0, 512)
    previous_lse = tl.load(softmax_lse + head * q_token_num + row, head < q_head_num, -float("inf"))
    chunk_lse_block = tl.load(
        chunk_softmax_lse + row * chunk_lse_stride_s + head * chunk_lse_stride_h, head < q_head_num, -float("inf")
    )
    # FA3 uses natural-log LSE and returns +inf for an empty row.
    previous_lse = tl.where(previous_lse == float("inf"), -float("inf"), previous_lse)
    chunk_lse_block = tl.where(chunk_lse_block == float("inf"), -float("inf"), chunk_lse_block)
    maximum = tl.maximum(previous_lse, chunk_lse_block)
    maximum = tl.where(maximum == -float("inf"), 0, maximum)
    previous_weight = tl.exp(previous_lse - maximum)
    chunk_weight = tl.exp(chunk_lse_block - maximum)
    denominator = previous_weight + chunk_weight
    offset = (row * q_head_num + head[:, None]) * 512 + dim[None, :]
    previous = tl.load(output + offset, (head[:, None] < q_head_num) & (previous_weight[:, None] > 0), 0)
    current = tl.load(chunk_output + offset, (head[:, None] < q_head_num) & (chunk_weight[:, None] > 0), 0).to(
        tl.float32
    )
    result = (previous * previous_weight[:, None] + current * chunk_weight[:, None]) / tl.where(
        denominator > 0, denominator, 1
    )[:, None]
    tl.store(output + offset, result, head[:, None] < q_head_num)
    tl.store(
        softmax_lse + head * q_token_num + row,
        tl.where(denominator > 0, maximum + tl.log(denominator), -float("inf")),
        head < q_head_num,
    )


@triton.jit
def _selected_prefill_compact_kernel(
    topk_mem_indices,
    compact_topk_offsets,
    cache_seqlens,
    kv_cache_capacity: tl.constexpr,
    topk_mem_indices_stride_s: tl.constexpr,
    topk_mem_indices_stride_d: tl.constexpr,
    BLOCK: tl.constexpr,
):
    qid = tl.program_id(0)
    lane = tl.arange(0, BLOCK)
    physical = tl.load(
        topk_mem_indices + qid * topk_mem_indices_stride_s + lane * topk_mem_indices_stride_d, lane < 2176, -1
    ).to(tl.int64)
    valid = (lane < 2176) & (physical >= 0) & (physical < kv_cache_capacity)
    rank = tl.cumsum(valid.to(tl.int32), axis=0) - 1
    tl.store(compact_topk_offsets + qid * 2176 + rank, lane, valid)
    tl.store(cache_seqlens + qid, tl.sum(valid.to(tl.int32), axis=0))


@triton.jit
def _selected_prefill_gather_kernel(
    compact_topk_offsets,
    cache_seqlens,
    topk_indices,
    topk_mem_indices,
    b1_cu_q_seq_len,
    b1_cu_kv_seq_len,
    prefill_cache_kv,
    packed_kv,
    selected_kv,
    packed_kv_stride_s: tl.constexpr,
    prefill_cache_kv_stride_s: tl.constexpr,
    prefill_cache_kv_stride_d: tl.constexpr,
    topk_indices_stride_s: tl.constexpr,
    topk_indices_stride_d: tl.constexpr,
    topk_mem_indices_stride_s: tl.constexpr,
    topk_mem_indices_stride_d: tl.constexpr,
    batch_size: tl.constexpr,
    BLOCK_ROWS: tl.constexpr,
):
    qid = tl.program_id(0)
    row = tl.program_id(1) * BLOCK_ROWS + tl.arange(0, BLOCK_ROWS)
    length = tl.load(cache_seqlens + qid)
    live = row < length
    offset = tl.load(compact_topk_offsets + qid * 2176 + row, live, 0)
    physical = tl.load(
        topk_mem_indices + qid * topk_mem_indices_stride_s + offset * topk_mem_indices_stride_d, live, 0
    ).to(tl.int64)
    position = tl.load(topk_indices + qid * topk_indices_stride_s + offset * topk_indices_stride_d, live, 0)
    if batch_size == 1:
        batch = 0
    else:
        lower, upper = 0, batch_size
        while lower < upper:
            middle = (lower + upper) // 2
            q_end = tl.load(b1_cu_q_seq_len + middle + 1)
            before = qid < q_end
            upper = tl.where(before, middle, upper)
            lower = tl.where(before, lower, middle + 1)
        batch = lower
    q_start = tl.load(b1_cu_q_seq_len + batch)
    q_end = tl.load(b1_cu_q_seq_len + batch + 1)
    kv_start = tl.load(b1_cu_kv_seq_len + batch)
    kv_end = tl.load(b1_cu_kv_seq_len + batch + 1)
    cache_len = kv_end - kv_start - (q_end - q_start)
    current = live & (position >= cache_len)
    cached = live & (position < cache_len)
    current_row = q_start + position - cache_len
    dim = tl.arange(0, 512)
    packed_row = packed_kv + physical[:, None] * packed_kv_stride_s
    group = tl.arange(0, 8)
    raw_scales = tl.load(packed_row + 576 + group[None, :], cached[:, None] & (group[None, :] < 7), 127)
    scales = (raw_scales.to(tl.uint32) << 23).to(tl.float32, bitcast=True)
    scales = tl.broadcast_to(scales[:, :, None], (BLOCK_ROWS, 8, 64)).reshape(BLOCK_ROWS, 512)
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
        prefill_cache_kv + current_row[:, None] * prefill_cache_kv_stride_s + dim[None, :] * prefill_cache_kv_stride_d,
        current[:, None],
        0.0,
    ).to(tl.float32)
    value = tl.where(current[:, None], current_value, cached_value).to(tl.bfloat16)
    tl.store(selected_kv + (qid * 2176 + row[:, None]) * 512 + dim[None, :], value, row[:, None] < 2176)


def can_use_selected_prefill(q, packed_kv, mem_index, prefill_cache_kv, topk_mem_indices, total_token_num):
    # Bound the BF16 scratch space and avoid gathering more KV than the full context.
    return (
        q.ndim == 3
        and 2 <= q.shape[0] <= 128
        and q.shape[1:] == (64, 512)
        and q.dtype == torch.bfloat16
        and q.is_cuda
        and q.stride(-1) == 1
        and packed_kv.shape[1:] == (1, 584)
        and packed_kv.dtype == torch.uint8
        and packed_kv.device == q.device
        and total_token_num >= max(65536 if q.shape[0] <= 8 else 131072, q.shape[0] * 2048)
        and mem_index.dtype == torch.int32
        and mem_index.device == q.device
        and mem_index.shape == (q.shape[0],)
        and prefill_cache_kv.dtype == torch.bfloat16
        and prefill_cache_kv.device == q.device
        and prefill_cache_kv.shape == (q.shape[0], 1, 512)
        and topk_mem_indices.dtype == torch.int32
        and topk_mem_indices.device == q.device
        and topk_mem_indices.shape == (q.shape[0], 2176)
    )


def allocate_selected_prefill_workspace(q):
    q_token_num = q.shape[0]
    return {
        "compact_topk_offsets": torch.empty((q_token_num, 2176), dtype=torch.int32, device=q.device),
        "cache_seqlens": torch.empty(q_token_num, dtype=torch.int32, device=q.device),
        "selected_kv": torch.empty((q_token_num, 2176, 1, 512), dtype=q.dtype, device=q.device),
    }


def allocate_generic_prefill_workspace(q, topk_indices, q_token_capacity=None):
    capacity = q.shape[0] if q_token_capacity is None else q_token_capacity
    return {
        "chunk_topk_indices": torch.empty((capacity, topk_indices.shape[1]), dtype=topk_indices.dtype, device=q.device),
        "chunk_cache_seqlens": torch.empty(capacity, dtype=torch.int32, device=q.device),
        "cu_seqlens_q": torch.arange(capacity + 1, dtype=torch.int32, device=q.device),
    }


@torch.no_grad()
def _context_attention_fwd_fp8_selected_prefill(
    q,
    packed_kv,
    prefill_cache_kv,
    topk_indices,
    topk_mem_indices,
    b1_cu_q_seq_len,
    b1_cu_kv_seq_len,
    softmax_scale,
):
    from sgl_kernel.flash_attn import flash_attn_with_kvcache

    q_token_num = q.shape[0]
    workspace = allocate_selected_prefill_workspace(q)
    _selected_prefill_compact_kernel[(q_token_num,)](
        topk_mem_indices,
        workspace["compact_topk_offsets"],
        workspace["cache_seqlens"],
        packed_kv.shape[0],
        topk_mem_indices.stride(0),
        topk_mem_indices.stride(1),
        BLOCK=4096,
        num_warps=8,
    )
    _selected_prefill_gather_kernel[(q_token_num, triton.cdiv(2176, 4))](
        workspace["compact_topk_offsets"],
        workspace["cache_seqlens"],
        topk_indices,
        topk_mem_indices,
        b1_cu_q_seq_len,
        b1_cu_kv_seq_len,
        prefill_cache_kv,
        packed_kv,
        workspace["selected_kv"],
        packed_kv.stride(0),
        prefill_cache_kv.stride(0),
        prefill_cache_kv.stride(2),
        topk_indices_stride_s=topk_indices.stride(0),
        topk_indices_stride_d=topk_indices.stride(1),
        topk_mem_indices_stride_s=topk_mem_indices.stride(0),
        topk_mem_indices_stride_d=topk_mem_indices.stride(1),
        batch_size=b1_cu_q_seq_len.numel() - 1,
        BLOCK_ROWS=4,
        num_warps=4,
    )
    qv = q.unsqueeze(1)
    selected_kv = workspace["selected_kv"]
    return flash_attn_with_kvcache(
        q=qv[..., :64],
        qv=qv,
        k_cache=selected_kv[..., :64],
        v_cache=selected_kv,
        cache_seqlens=workspace["cache_seqlens"],
        softmax_scale=softmax_scale,
        causal=False,
        only_qv=True,
    ).squeeze(1)


@torch.no_grad()
def context_attention_fwd_fp8(
    q,
    packed_kv,
    mem_index,
    prefill_cache_kv,
    q_kv_start_loc,
    topk_indices,
    topk_mem_indices,
    req_to_token_indexs,
    b_req_idx,
    b1_cu_kv_seq_len,
    b1_cu_q_seq_len,
    total_token_num,
    softmax_scale,
    direct_workspace=None,
    generic_workspace=None,
):
    """Run NoPE sparse prefill from request mappings and physical top-k slots."""
    eligibility = (q, packed_kv, mem_index, prefill_cache_kv, topk_mem_indices, total_token_num)
    if can_use_direct_prefill(*eligibility):
        return _context_attention_fwd_fp8_direct_prefill(
            q, packed_kv, mem_index, prefill_cache_kv, topk_mem_indices, softmax_scale, direct_workspace
        )
    if can_use_selected_prefill(*eligibility):
        return _context_attention_fwd_fp8_selected_prefill(
            q,
            packed_kv,
            prefill_cache_kv,
            topk_indices,
            topk_mem_indices,
            b1_cu_q_seq_len,
            b1_cu_kv_seq_len,
            softmax_scale,
        )
    from sgl_kernel.flash_attn import flash_attn_with_kvcache

    q_token_num, q_head_num, _ = q.shape
    gather_chunk_size = min(KV_GATHER_CHUNK_SIZE, total_token_num)
    workspace = generic_workspace or allocate_generic_prefill_workspace(q, topk_indices)
    chunk_topk_indices = workspace["chunk_topk_indices"]
    chunk_cache_seqlens = workspace["chunk_cache_seqlens"]
    cu_seqlens_q = workspace["cu_seqlens_q"]
    # Keep this potentially 2 GiB buffer local to a DSA layer.
    gathered_kv = q.new_empty((gather_chunk_size, 1, 512))
    output = softmax_lse = None
    for chunk_start in range(0, total_token_num, gather_chunk_size):
        chunk_token_num = min(gather_chunk_size, total_token_num - chunk_start)
        _gather_prefill_kv_kernel[(chunk_token_num // 4 + b_req_idx.numel(),)](
            packed_kv,
            prefill_cache_kv,
            gathered_kv,
            req_to_token_indexs,
            b_req_idx,
            b1_cu_kv_seq_len,
            b1_cu_q_seq_len,
            chunk_start,
            chunk_token_num,
            packed_kv.stride(0),
            prefill_cache_kv.stride(0),
            prefill_cache_kv.stride(2),
            req_to_token_indexs_stride_b=req_to_token_indexs.stride(0),
            req_to_token_indexs_stride_s=req_to_token_indexs.stride(1),
            batch_size=b_req_idx.numel(),
            BLOCK_ROWS=4,
        )
        chunk_kv = gathered_kv[:chunk_token_num].view(-1, 1, 1, 512)
        if (
            q_token_num > 8
            and q_head_num == 64
            and total_token_num > q_token_num
            and total_token_num <= KV_GATHER_CHUNK_SIZE
            and q.dtype == torch.bfloat16
            and q.is_contiguous()
            and packed_kv.dtype == torch.uint8
            and packed_kv.shape[1:] == (1, 584)
            and topk_indices.dtype == torch.int32
            and topk_indices.shape[1] == 2176
            and topk_indices.is_contiguous()
        ):
            from sgl_kernel.flash_mla import flash_mla_sparse_fwd

            _compact_topk_indices_kernel[(q_token_num,)](
                topk_indices,
                q_kv_start_loc,
                chunk_topk_indices,
                chunk_cache_seqlens,
                chunk_start,
                chunk_start + chunk_token_num,
                TOPK=topk_indices.shape[1],
                BLOCK=triton.next_power_of_2(topk_indices.shape[1]),
                COMPACT=False,
            )
            return flash_mla_sparse_fwd(
                q=q,
                kv=gathered_kv[:chunk_token_num],
                indices=chunk_topk_indices.unsqueeze(1),
                sm_scale=softmax_scale,
                d_v=512,
            )[0]
        _compact_topk_indices_kernel[(q_token_num,)](
            topk_indices,
            q_kv_start_loc,
            chunk_topk_indices,
            chunk_cache_seqlens,
            chunk_start,
            chunk_start + chunk_token_num,
            TOPK=topk_indices.shape[1],
            BLOCK=triton.next_power_of_2(topk_indices.shape[1]),
        )
        chunk_output, chunk_softmax_lse, *_ = flash_attn_with_kvcache(
            # only_qv ignores Q/K; these views avoid allocating placeholder tensors.
            q=q[..., :64],
            k_cache=chunk_kv[..., :64],
            qv=q,
            v_cache=chunk_kv,
            page_table=chunk_topk_indices,
            cache_seqlens=chunk_cache_seqlens,
            cu_seqlens_q=cu_seqlens_q,
            max_seqlen_q=1,
            softmax_scale=softmax_scale,
            causal=False,
            only_qv=True,
            num_splits=1,
            return_softmax_lse=True,
        )
        if chunk_start == 0:
            output, softmax_lse = chunk_output, chunk_softmax_lse
        else:
            # Keep the online reduction in FP32 across all remaining chunks.
            output = output.float()
            _merge_att_output_kernel[(q_token_num, triton.cdiv(q_head_num, 4))](
                output,
                softmax_lse,
                chunk_output,
                chunk_softmax_lse,
                chunk_softmax_lse.stride(1),
                chunk_softmax_lse.stride(0),
                q_head_num=q_head_num,
                q_token_num=q_token_num,
                BLOCK_HEADS=4,
            )
        del chunk_output, chunk_softmax_lse
    return output.to(q.dtype)
