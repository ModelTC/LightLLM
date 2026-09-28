import torch
import triton
import triton.language as tl

from lightllm.models.deepseek3_2.triton_kernel.hadamard_transform import _butterfly_stage
from lightllm.utils.device_utils import get_device_sm_count


@triton.jit
def _get_query_block(CuQLens, BATCH: tl.constexpr, BLOCK: tl.constexpr):
    block = tl.program_id(0)
    # Request b owns [cu[b] // BLOCK + b, cu[b + 1] // BLOCK + b + 1).
    # The extra block covers unaligned boundaries without padding to max_q_len.
    if BATCH == 1:
        batch = 0
    else:
        batches = tl.arange(0, triton.next_power_of_2(BATCH))
        q_ends = tl.load(CuQLens + batches + 1, batches < BATCH, 0)
        block_ends = q_ends // BLOCK + batches + 1
        batch = tl.sum(((block >= block_ends) & (batches < BATCH)).to(tl.int32), 0)
    q_start = tl.load(CuQLens + batch)
    q_end = tl.load(CuQLens + batch + 1)
    local_block = block - (q_start // BLOCK + batch)
    return batch, q_start, q_end, local_block


@triton.jit
def _compress_pools(
    Raw,
    Tail,
    Packed,
    Ape,
    Lengths,
    Starts,
    Ragged,
    ReqIdx,
    CuQLens,
    MtpIndex,
    RAW_STRIDE: tl.constexpr,
    TAIL_REQ_STRIDE: tl.constexpr,
    TAIL_SLOT_STRIDE: tl.constexpr,
    PACKED_STRIDE: tl.constexpr,
    HOLD_REQ: tl.constexpr,
    BATCH: tl.constexpr,
    SINGLE_QUERY: tl.constexpr,
    TAIL_SIZE: tl.constexpr,
    HAS_MTP_INDEX: tl.constexpr,
):
    if SINGLE_QUERY:
        batch = tl.program_id(0)
        row = batch
        length = tl.load(Lengths + row)
        valid = length > 0 and length % 4 == 0
        q_start = row
        if HAS_MTP_INDEX:
            q_start -= tl.load(MtpIndex + row)
    else:
        batch, q_start, q_end, pool_index = _get_query_block(CuQLens, BATCH, 4)
        prefix = tl.load(Lengths + q_start, q_start < q_end, 1) - 1
        # Only process pool-closing rows, including a pool spanning the old tail.
        length = (prefix // 4 + pool_index + 1) * 4
        row = q_start + length - prefix - 1
        valid = row < q_end
    req = tl.load(ReqIdx + batch)
    if valid and req != HOLD_REQ:
        start = tl.load(Starts + row)
        pool = tl.arange(0, 4)
        cols = tl.arange(0, 128)
        raw_rows = row - 3 + pool
        from_chunk = raw_rows >= q_start
        chunk_ptr = Raw + raw_rows[:, None] * RAW_STRIDE + cols[None, :]
        tail_slots = (length - 4 + pool) % TAIL_SIZE
        tail_ptr = Tail + req * TAIL_REQ_STRIDE + tail_slots[:, None] * TAIL_SLOT_STRIDE + cols[None, :]
        raw = tl.where(
            from_chunk[:, None],
            tl.load(chunk_ptr, from_chunk[:, None], 0),
            tl.load(tail_ptr, ~from_chunk[:, None], 0),
        ).to(tl.float32)
        score = tl.where(
            from_chunk[:, None],
            tl.load(chunk_ptr + 128, from_chunk[:, None], 0),
            tl.load(tail_ptr + 128, ~from_chunk[:, None], 0),
        ).to(tl.float32)
        score += tl.load(Ape + pool[:, None] * 128 + cols[None, :])
        score = tl.exp(score - tl.max(score, 0)[None, :])
        weights = score / tl.sum(score, 0)[None, :]
        key = tl.sum(raw * weights, 0).to(tl.bfloat16).to(tl.float32).reshape(1, 128)
        for step in tl.static_range(7):
            key = _butterfly_stage(key, 64 >> step, 1 << step, 1, 128)
        key = (key * (128 ** -0.5)).to(tl.bfloat16).to(tl.float32)
        scale = tl.exp2(tl.ceil(tl.log2(tl.maximum(tl.max(tl.abs(key), 1), 1e-4) / 448.0)))
        key = tl.minimum(tl.maximum(key / scale[:, None], -448.0), 448.0).to(tl.float8e4nv)
        loc = tl.load(Ragged + start + length - 1).to(tl.int64)
        dest = Packed + loc * PACKED_STRIDE
        tl.store(dest + cols, key.reshape(128).to(tl.uint8, bitcast=True))
        tl.store((dest + 128).to(tl.pointer_type(tl.float32)), tl.sum(scale, 0))


@triton.jit
def _save_pool_tails(
    Raw,
    Tail,
    ReqIdx,
    CuQLens,
    SeqLens,
    RAW_STRIDE: tl.constexpr,
    TAIL_REQ_STRIDE: tl.constexpr,
    TAIL_SLOT_STRIDE: tl.constexpr,
    HOLD_REQ: tl.constexpr,
    TAIL_SIZE: tl.constexpr,
):
    batch = tl.program_id(0)
    req = tl.load(ReqIdx + batch)
    if req != HOLD_REQ:
        length = tl.load(SeqLens + batch)
        q_start = tl.load(CuQLens + batch)
        q_end = tl.load(CuQLens + batch + 1)
        slots = tl.arange(0, triton.next_power_of_2(TAIL_SIZE))
        cols = tl.arange(0, 256)
        positions = length - TAIL_SIZE + slots
        raw_rows = q_end - length + positions
        # Absolute positions address a short ring, preserving the raw history
        # needed after rejecting candidates or revisiting draft positions.
        mask = ((slots < TAIL_SIZE) & (positions >= 0) & (raw_rows >= q_start))[:, None]
        raw = tl.load(Raw + raw_rows[:, None] * RAW_STRIDE + cols[None, :], mask, 0)
        tl.store(
            Tail + req * TAIL_REQ_STRIDE + (positions % TAIL_SIZE)[:, None] * TAIL_SLOT_STRIDE + cols[None, :],
            raw,
            mask,
        )


def compress_pools(
    raw, tail, packed_buffer, ape, lengths, starts, ragged, req_idx, cu_q_lens, seq_lens, max_q_len, mtp_index=None
):
    batch = req_idx.numel()
    single_query = max_q_len == 1 and lengths.numel() == batch
    blocks = batch if single_query else lengths.numel() // 4 + batch
    _compress_pools[(blocks,)](
        raw,
        tail,
        packed_buffer,
        ape,
        lengths,
        starts,
        ragged,
        req_idx,
        cu_q_lens,
        mtp_index,
        raw.stride(0),
        tail.stride(0),
        tail.stride(1),
        packed_buffer.stride(0),
        tail.shape[0] - 1,
        batch,
        single_query,
        tail.shape[1],
        mtp_index is not None,
        num_warps=4,
    )
    # Complete all boundary pools before replacing the previous chunk's tail.
    _save_pool_tails[(req_idx.numel(),)](
        raw,
        tail,
        req_idx,
        cu_q_lens,
        seq_lens,
        raw.stride(0),
        tail.stride(0),
        tail.stride(1),
        tail.shape[0] - 1,
        tail.shape[1],
        num_warps=4,
    )


@triton.jit
def _get_pool_ranges(
    Lengths,
    CuQLens,
    Starts,
    Ends,
    PoolLengths,
    BATCH: tl.constexpr,
    TOKENS: tl.constexpr,
    MAX_POOLS: tl.constexpr,
    SINGLE_QUERY: tl.constexpr,
    BLOCK: tl.constexpr,
):
    if SINGLE_QUERY:
        row = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
        batch = row
        valid = row < TOKENS
    else:
        batch, q_start, q_end, local_block = _get_query_block(CuQLens, BATCH, BLOCK)
        row = q_start + local_block * BLOCK + tl.arange(0, BLOCK)
        valid = row < q_end
    length = tl.load(Lengths + row, valid, 0) // 4
    start = batch * MAX_POOLS
    tl.store(Starts + row, start, valid)
    tl.store(Ends + row, start + length, valid)
    tl.store(PoolLengths + row, length, valid)


def get_pool_ranges(lengths, cu_q_lens, max_q_len, max_pools):
    """Build per-query pool bounds directly from the packed request boundaries."""
    starts, ends, pool_lengths = [torch.empty_like(lengths) for _ in range(3)]
    batch, tokens = cu_q_lens.numel() - 1, lengths.numel()
    single_query = max_q_len == 1 and tokens == batch
    block = 256
    blocks = triton.cdiv(tokens, block) if single_query else tokens // block + batch
    _get_pool_ranges[(blocks,)](
        lengths, cu_q_lens, starts, ends, pool_lengths, batch, tokens, max_pools, single_query, block, num_warps=4
    )
    return starts, ends, pool_lengths


@triton.jit
def _gather_pools(
    Packed,
    ReqTable,
    ReqIdx,
    SeqLen,
    K,
    Scale,
    PACKED_STRIDE: tl.constexpr,
    REQ_STRIDE: tl.constexpr,
    POOLS: tl.constexpr,
):
    pool, batch = tl.program_id(0), tl.program_id(1)
    length = tl.load(SeqLen + batch)
    req = tl.load(ReqIdx + batch)
    valid = pool < length // 4
    loc = tl.load(ReqTable + req * REQ_STRIDE + pool * 4 + 3, valid, 0).to(tl.int64)
    cols = tl.arange(0, 128)
    packed = tl.load(Packed + loc * PACKED_STRIDE + cols, valid, 0)
    scale = tl.load((Packed + loc * PACKED_STRIDE + 128).to(tl.pointer_type(tl.float32)), valid, 1.0)
    row = batch.to(tl.int64) * POOLS + pool
    tl.store(K + row * 128 + cols, packed.to(tl.float8e4nv, bitcast=True))
    tl.store(Scale + row, scale)


def gather_pools(packed_buffer, req_table, req_idx, seq_len, max_pools):
    keys = torch.empty((req_idx.numel() * max_pools, 128), device=packed_buffer.device, dtype=torch.float8_e4m3fn)
    scales = torch.empty((keys.shape[0],), device=keys.device, dtype=torch.float32)
    # Long contexts can exceed grid Y's 65535-block limit; put pools on X.
    _gather_pools[(max_pools, req_idx.numel())](
        packed_buffer,
        req_table,
        req_idx,
        seq_len,
        keys,
        scales,
        packed_buffer.stride(0),
        req_table.stride(0),
        max_pools,
        num_warps=4,
    )
    return keys, scales


@triton.jit
def _gather_paged_pools(
    Packed,
    ReqTable,
    ReqIdx,
    PoolLengths,
    Pages,
    PACKED_STRIDE: tl.constexpr,
    REQ_STRIDE: tl.constexpr,
    MAX_PAGES: tl.constexpr,
):
    batch = tl.program_id(1)
    req = tl.load(ReqIdx + batch)
    length = tl.load(PoolLengths + batch)
    rows = tl.arange(0, 64)
    cols = tl.arange(0, 128)
    # A fixed grid is replayable at 1M; the GPU length bounds the work.
    for page in range(tl.program_id(0), tl.cdiv(length, 64), tl.num_programs(0)):
        pools = page * 64 + rows
        valid = pools < length
        locs = tl.load(ReqTable + req * REQ_STRIDE + pools * 4 + 3, valid, 0).to(tl.int64)
        keys = tl.load(Packed + locs[:, None] * PACKED_STRIDE + cols[None, :], valid[:, None], 0)
        scales = tl.load((Packed + locs * PACKED_STRIDE + 128).to(tl.pointer_type(tl.float32)), valid, 1.0)
        dest = Pages + (batch.to(tl.int64) * MAX_PAGES + page) * (64 * 132)
        # DeepGEMM stores 64 FP8 keys followed by their 64 FP32 scales.
        tl.store(dest + rows[:, None] * 128 + cols[None, :], keys)
        tl.store((dest + 64 * 128).to(tl.pointer_type(tl.float32)) + rows, scales)


def gather_paged_pools(packed_buffer, req_table, req_idx, pool_lengths, max_pools):
    """Pack valid pools into DeepGEMM pages; unused pages remain unread."""
    batch = req_idx.numel()
    max_pages = triton.cdiv(max_pools, 64)
    pages = torch.empty((batch * max_pages, 64, 1, 132), device=packed_buffer.device, dtype=torch.uint8)
    block_table = torch.arange(batch * max_pages, device=pages.device, dtype=torch.int32).view(batch, max_pages)
    blocks = min(max_pages, triton.cdiv(get_device_sm_count() * 4, batch))
    _gather_paged_pools[(blocks, batch)](
        packed_buffer,
        req_table,
        req_idx,
        pool_lengths,
        pages,
        packed_buffer.stride(0),
        req_table.stride(0),
        max_pages,
        num_warps=4,
    )
    return pages, block_table


@triton.jit
def _expand_topk(
    Groups,
    Lengths,
    Starts,
    Ragged,
    Out,
    Relative,
    TOPK: tl.constexpr,
    WIDTH: tl.constexpr,
    DENSE: tl.constexpr,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    lane = tl.arange(0, BLOCK)
    length = tl.load(Lengths + row)
    start = tl.load(Starts + row)
    if DENSE:
        token = lane
        valid = lane < length
    else:
        closed_tokens = tl.minimum(length // 4 * 4, TOPK)
        group = tl.load(Groups + row * (TOPK // 4) + lane // 4, lane < closed_tokens, -1)
        token = tl.where(lane < closed_tokens, group * 4 + lane % 4, length // 4 * 4 + lane - closed_tokens)
        valid = (lane < closed_tokens + length % 4) & (token >= 0)
    mem = tl.load(Ragged + start + token, valid & (lane < WIDTH), -1)
    tl.store(Out + row * WIDTH + lane, mem, lane < WIDTH)
    tl.store(Relative + row * WIDTH + lane, tl.where(valid, token, -1), lane < WIDTH)


def expand_topk(groups, lengths, starts, ragged, topk, dense=False):
    width = triton.cdiv(topk + 3, 128) * 128
    out = torch.empty((lengths.numel(), width), dtype=torch.int32, device=lengths.device)
    relative = torch.empty_like(out)
    _expand_topk[(lengths.numel(),)](
        groups,
        lengths,
        starts,
        ragged,
        out,
        relative,
        topk,
        width,
        dense,
        triton.next_power_of_2(width),
        num_warps=4,
    )
    return out, relative
