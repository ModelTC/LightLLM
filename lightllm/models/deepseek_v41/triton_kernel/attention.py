"""V4.1 pair compression, FP8 history writes and paged sparse-indexer scoring."""

import torch
import triton
import triton.language as tl
from triton.language.extra import libdevice
from lightllm.common.basemodel.triton_kernel.norm.rmsnorm import rmsnorm_num_warps


@triton.jit
def _pool_and_norm_pairs(
    KV,
    SCORE,
    REQ,
    POS,
    STATE,
    OUT,
    NORM,
    EPS,
    D: tl.constexpr,
    STATE_STRIDE: tl.constexpr,
    KV_STRIDE: tl.constexpr,
    SCORE_STRIDE: tl.constexpr,
    SKIP_OPEN_GROUPS: tl.constexpr,
):
    row = tl.program_id(0)
    cols = tl.arange(0, D)
    pos = tl.load(POS + row)
    pair_end = pos % 2 == 1
    if SKIP_OPEN_GROUPS:
        if not pair_end:
            # The dense latent still feeds the indexer projection. Match the
            # signed zero produced by the normal RMSNorm weight multiply.
            weight = tl.load(NORM + cols).to(tl.float32)
            tl.store(OUT + row * D + cols, 0.0 * weight)
            return
    req = tl.load(REQ + row)
    previous_req = tl.load(REQ + row - 1, pair_end & (row > 0), -1)
    in_batch = pair_end & (row > 0) & (previous_req == req)
    previous_kv = tl.load(KV + (row - 1) * KV_STRIDE + cols, in_batch, 0.0)
    previous_score = tl.load(SCORE + (row - 1) * SCORE_STRIDE + cols, in_batch, 0.0)
    previous_state = STATE + req * STATE_STRIDE
    from_state = pair_end & ~in_batch
    previous_kv += tl.load(previous_state + cols, from_state, 0.0)
    previous_score += tl.load(previous_state + D + cols, from_state, 0.0)
    current_kv = tl.load(KV + row * KV_STRIDE + cols, pair_end, 0.0)
    current_score = tl.load(SCORE + row * SCORE_STRIDE + cols, pair_end, 0.0)
    gate = tl.sigmoid(current_score - previous_score)
    pooled = previous_kv + (current_kv - previous_kv) * gate
    # Match the BF16 intermediate and separately rounded FP32 squares of RMSNorm.
    pooled = pooled.to(tl.bfloat16).to(tl.float32)
    variance = tl.sum(libdevice.mul_rn(pooled, pooled), axis=0) / D
    rstd = 1 / tl.sqrt(variance + EPS)
    weight = tl.load(NORM + cols).to(tl.float32)
    tl.store(OUT + row * D + cols, (pooled * rstd) * weight)


@triton.jit
def _commit_pair_state(
    KV,
    SCORE,
    REQ,
    POS,
    STATE,
    T,
    D: tl.constexpr,
    HOLD: tl.constexpr,
    STATE_STRIDE: tl.constexpr,
    KV_STRIDE: tl.constexpr,
    SCORE_STRIDE: tl.constexpr,
):
    row = tl.program_id(0)
    req = tl.load(REQ + row)
    pos = tl.load(POS + row)
    # Only a request ending mid-pair needs to retain its final candidate for the next forward.
    next_req = tl.load(REQ + row + 1, row + 1 < T, -1)
    commit = (req != HOLD) & (next_req != req) & (pos % 2 == 0)
    cols = tl.arange(0, D)
    dest = STATE + req * STATE_STRIDE
    tl.store(dest + cols, tl.load(KV + row * KV_STRIDE + cols), commit)
    tl.store(dest + D + cols, tl.load(SCORE + row * SCORE_STRIDE + cols), commit)


def pool_and_norm_pairs(kv, score, req, positions, state, hold_request, norm_weight, eps):
    out = torch.empty_like(kv, dtype=torch.bfloat16)
    _pool_and_norm_pairs[(kv.shape[0],)](
        kv,
        score,
        req,
        positions,
        state,
        out,
        norm_weight,
        eps,
        kv.shape[1],
        state.stride(0),
        kv.stride(0),
        score.stride(0),
        SKIP_OPEN_GROUPS=kv.shape[0] >= 1024,
        num_warps=rmsnorm_num_warps,
    )
    # Commit after scoring so current rows cannot overwrite another query's previous-step history.
    _commit_pair_state[(kv.shape[0],)](
        kv,
        score,
        req,
        positions,
        state,
        kv.shape[0],
        kv.shape[1],
        hold_request,
        state.stride(0),
        kv.stride(0),
        score.stride(0),
    )
    return out


@triton.jit
def _fwd_kernel_destindex_copy_kv_dsv41(
    KV,
    FREQS,
    FULL_SLOT,
    POS,
    CACHE,
    RATIO: tl.constexpr,
    PAGE: tl.constexpr,
    STRIDE: tl.constexpr,
):
    row = tl.program_id(0)
    pos = tl.load(POS + row)
    if (pos + 1) % RATIO != 0:
        return
    slot = tl.load(FULL_SLOT + row).to(tl.int64) // RATIO
    page, offset = slot // PAGE, slot % PAGE
    d = tl.arange(0, 512)
    pair = tl.arange(0, 256)
    real = tl.load(KV + row * 512 + pair * 2).to(tl.float32)
    imag = tl.load(KV + row * 512 + pair * 2 + 1).to(tl.float32)
    freq = FREQS + (pos - RATIO + 1) * 64 + (pair - 224) * 2
    cos = tl.load(freq, pair >= 224, 1.0)
    sin = tl.load(freq + 1, pair >= 224, 0.0)
    value = tl.reshape(tl.join(real * cos - imag * sin, real * sin + imag * cos), (512,))
    value = value.to(tl.bfloat16).to(tl.float32)
    grouped = tl.reshape(value, (8, 64))
    scale = tl.maximum(tl.max(tl.abs(grouped), 1), 1e-4) / 448
    bits = scale.to(tl.uint32, bitcast=True)
    exponent = ((bits >> 23) & 255) + ((bits & 0x7FFFFF) != 0).to(tl.uint32)
    inverse = ((254 - exponent) << 23).to(tl.float32, bitcast=True)
    quant = tl.reshape(grouped * inverse[:, None], (512,))
    rd = tl.arange(0, 64)
    rope = tl.gather(value, 448 + rd, axis=0)
    groups = tl.arange(0, 8)
    base = CACHE + page * STRIDE
    tl.store((base + offset * 576).to(tl.pointer_type(tl.float8e4nv)) + d, quant, d < 448)
    tl.store((base + offset * 576 + 448).to(tl.pointer_type(tl.bfloat16)) + rd, rope)
    tl.store(base + PAGE * 576 + offset * 8 + groups, exponent, groups < 7)


def destindex_copy_kv_dsv41(kv, freqs, full_slots, positions, cache, ratio, page_size):
    """Fuse RoPE, FP8 quantization and history scatter into Hopper FlashMLA cache."""
    _fwd_kernel_destindex_copy_kv_dsv41[(kv.shape[0],)](
        kv,
        torch.view_as_real(freqs).flatten(-2),
        full_slots,
        positions,
        cache,
        ratio,
        page_size,
        cache.stride(0),
        enable_fp_fusion=False,
    )


@triton.jit
def _prepare_indexer_logits(LOGITS, LENGTHS, MASK, SCORES, STRIDE, WIDTH, BLOCKS: tl.constexpr):
    row = tl.program_id(0)
    blocks = tl.program_id(1) * BLOCKS + tl.arange(0, BLOCKS)
    cols = blocks[:, None] * 8 + tl.arange(0, 8)[None, :]
    length = tl.load(LENGTHS + row)
    valid = cols < length
    if MASK is not None:
        keep = tl.load(MASK + row * (WIDTH // 8) + blocks, blocks < WIDTH // 8, False)
        valid &= keep[:, None]
    values = tl.load(LOGITS + row * STRIDE + cols, valid, -float("inf"))
    tl.store(LOGITS + row * STRIDE + cols, values, cols < WIDTH)
    if SCORES is not None:
        latest = tl.maximum(length - 1, 0) // 8
        scores = tl.max(values, 1)
        scores = tl.where(tl.max(libdevice.isnan(values).to(tl.int32), 1) != 0, float("nan"), scores)
        scores = tl.where(blocks == latest, float("inf"), scores)
        tl.store(SCORES + row * (WIDTH // 8) + blocks, scores, blocks < WIDTH // 8)


@triton.jit
def _indexer_topk_transform(
    LOGITS,
    RAW,
    PAGES,
    OUT,
    VALID_LENGTHS,
    LENGTHS,
    STRIDE,
    PAGE_STRIDE,
    PAGE_SIZE: tl.constexpr,
    TOPK: tl.constexpr,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    col = tl.arange(0, BLOCK)
    length = tl.load(VALID_LENGTHS + row)
    index = col
    if RAW is not None:
        if length > TOPK:
            index = tl.load(RAW + row * TOPK + col, col < TOPK, 0).to(tl.int32)
    valid = (col < TOPK) & (index < length)
    score = tl.load(LOGITS + row * STRIDE + index, valid, -float("inf"))
    valid &= tl.abs(score) < float("inf")
    if RAW is not None:
        if length > TOPK:
            index = tl.sort(tl.where(valid, index, 2147483647))
            valid = index != 2147483647
    rank = tl.cumsum(valid.to(tl.int32)) - 1
    count = tl.sum(valid.to(tl.int32))
    page = tl.load(PAGES + row * PAGE_STRIDE + index // PAGE_SIZE, valid, 0)
    tl.store(OUT + row * TOPK + col, -1, (col < TOPK) & (col >= count))
    tl.store(OUT + row * TOPK + rank, page * PAGE_SIZE + index % PAGE_SIZE, valid)
    tl.store(LENGTHS + row, count)


def select_indexer_topk(logits, lengths, pages, slots, selected_lengths, page_size, candidate_mask, candidate_topk):
    """Keep the latest candidate block, then pack finite top-k entries in logical order."""
    rows, width = logits.shape
    topk = slots.shape[1]
    # A candidate budget covering the entire row offers every block to subsequent layers.
    if candidate_topk is not None and candidate_topk >= width // 8:
        candidate_mask.fill_(True)
        candidate_mask = None
        candidate_topk = None
    if candidate_mask is not None or width > topk:
        scores = logits.new_empty((rows, width // 8)) if candidate_topk is not None else None
        _prepare_indexer_logits[(rows, triton.cdiv(width, 1024))](
            logits, lengths, candidate_mask if scores is None else None, scores, logits.stride(0), width, 128
        )
        if scores is not None:
            selected = scores.topk(candidate_topk, sorted=False).indices
            candidate_mask.zero_().scatter_(1, selected, True)
    indices = logits.topk(topk, sorted=False).indices if width > topk else None
    _indexer_topk_transform[(rows,)](
        logits,
        indices,
        pages,
        slots,
        lengths,
        selected_lengths,
        logits.stride(0),
        pages.stride(0),
        page_size,
        topk,
        triton.next_power_of_2(topk),
    )


@triton.jit
def _candidate_scores(LOGITS, LENGTHS, OUT, TOPK_LENGTHS, STRIDE, WIDTH, TOPK: tl.constexpr, BLOCK: tl.constexpr):
    row, tile = tl.program_id(0), tl.program_id(1)
    length = tl.load(LENGTHS + row)
    block_length = tl.cdiv(length, 8)
    needs_topk = block_length > TOPK
    tl.store(TOPK_LENGTHS + row, tl.where(needs_topk, block_length, 0), tile == 0)
    if needs_topk & (tile * BLOCK < block_length):
        block = tile * BLOCK + tl.arange(0, BLOCK)
        col = block[:, None] * 8 + tl.arange(0, 8)[None, :]
        value = tl.load(LOGITS + row * STRIDE + col, col < length, -float("inf"))
        score = tl.max(value, 1)
        score = tl.where(block == block_length - 1, float("inf"), score)
        tl.store(OUT + row * WIDTH + block, score, (block < WIDTH) & (block < block_length))


@triton.jit
def _sort_candidates(INDICES, LENGTHS, TOPK: tl.constexpr, BLOCK: tl.constexpr):
    row = tl.program_id(0)
    col = tl.arange(0, BLOCK)
    block_length = tl.cdiv(tl.load(LENGTHS + row), 8)
    if block_length <= TOPK:
        tl.store(INDICES + row * TOPK + col, tl.where(col < block_length, col, -1), col < TOPK)
    else:
        index = tl.load(INDICES + row * TOPK + col, col < TOPK, -1)
        valid = (index >= 0) & (index < block_length)
        index = tl.sort(tl.where(valid, index, 2147483647))
        valid = (index != 2147483647) & ((col == 0) | (index != tl.gather(index, tl.maximum(col - 1, 0), axis=0)))
        rank = tl.cumsum(valid.to(tl.int32)) - 1
        count = tl.sum(valid.to(tl.int32))
        tl.store(INDICES + row * TOPK + col, -1, (col < TOPK) & (col >= count))
        tl.store(INDICES + row * TOPK + rank, index, valid)


# Prefill changes the page-table stride as the context grows.
@triton.jit(do_not_specialize=["PAGE_ROW"])
def _candidate_indexer_logits(
    Q,
    CACHE,
    WEIGHTS,
    PAGES,
    PAGE_ROWS,
    CANDIDATES,
    LENGTHS,
    LOGITS,
    Q_ROW: tl.constexpr,
    Q_HEAD: tl.constexpr,
    CACHE_PAGE: tl.constexpr,
    WEIGHT_ROW: tl.constexpr,
    PAGE_ROW,
    CANDIDATE_ROW: tl.constexpr,
    HEADS: tl.constexpr,
    BLOCK_H: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    row, tile = tl.program_id(0), tl.program_id(1)
    col = tile * BLOCK_N + tl.arange(0, BLOCK_N)
    # The producer sorts valid blocks and pads their tail with -1. Test one
    # scalar before materializing the dot's vector inputs and live ranges.
    first = tl.load(CANDIDATES + row * CANDIDATE_ROW + tile * (BLOCK_N // 8), tile * (BLOCK_N // 8) < CANDIDATE_ROW, -1)
    length = tl.load(LENGTHS + row)
    if (first >= 0) & (first * 8 < length):
        block = tl.load(CANDIDATES + row * CANDIDATE_ROW + col // 8, col < CANDIDATE_ROW * 8, -1)
        position = block * 8 + col % 8
        valid = (block >= 0) & (position < length)
        page_row = tl.load(PAGE_ROWS + row) if PAGE_ROWS is not None else row
        page = tl.load(PAGES + page_row * PAGE_ROW + position // 64, valid, 0).to(tl.int64)
        base = CACHE + page * CACHE_PAGE
        d = tl.arange(0, 128)
        h = tl.arange(0, BLOCK_H)
        q = tl.load(Q + row * Q_ROW + h[:, None] * Q_HEAD + d[None, :], h[:, None] < HEADS, 0.0)
        k = tl.load(
            (base[None, :] + (position % 64)[None, :] * 128 + d[:, None]).to(tl.pointer_type(tl.float8e4nv)),
            valid[None, :],
            0.0,
        )
        scale = tl.load(
            (base + 64 * 128 + position % 64 * 4).to(tl.pointer_type(tl.float32)),
            valid,
            0,
        )
        weights = tl.load(WEIGHTS + row * WEIGHT_ROW + h, h < HEADS, 0)
        scores = tl.sum(tl.maximum(tl.dot(q, k), 0.0) * weights[:, None], axis=0) * scale
        tl.store(
            LOGITS + row * CANDIDATE_ROW * 8 + col, tl.where(valid, scores, -float("inf")), col < CANDIDATE_ROW * 8
        )
    else:
        tl.store(LOGITS + row * CANDIDATE_ROW * 8 + col, -float("inf"), col < CANDIDATE_ROW * 8)


def candidate_indexer_logits(query, cache, weights, pages, candidates, lengths, *, page_rows=None):
    """Score the producer's sorted, -1-padded eight-key blocks in FP8 slabs."""
    assert query.shape[-1] == 128
    block_n = 256 if query.shape[0] >= 32 else 64
    logits = torch.empty((query.shape[0], candidates.shape[1] * 8), dtype=torch.float32, device=query.device)
    _candidate_indexer_logits[(query.shape[0], triton.cdiv(logits.shape[1], block_n))](
        query,
        cache,
        weights,
        pages,
        page_rows,
        candidates,
        lengths,
        logits,
        query.stride(0),
        query.stride(1),
        cache.stride(0),
        weights.stride(0),
        pages.stride(0),
        candidates.stride(0),
        query.shape[1],
        max(16, triton.next_power_of_2(query.shape[1])),
        block_n,
        num_warps=4,
    )
    return logits


@triton.jit
def _paged_topk_transform(
    LOGITS,
    RAW,
    PAGES,
    PAGE_ROWS,
    OUT,
    VALID_LENGTHS,
    LENGTHS,
    LOGIT_STRIDE,
    PAGE_STRIDE,
    PAGE_SIZE: tl.constexpr,
    TOPK: tl.constexpr,
    BLOCK: tl.constexpr,
    CANDIDATES,
    CANDIDATE_STRIDE: tl.constexpr,
    SPARSE: tl.constexpr,
):
    row = tl.program_id(0)
    col = tl.arange(0, BLOCK)
    length = tl.load(VALID_LENGTHS + row)
    width = tl.minimum(length, CANDIDATE_STRIDE * 8) if SPARSE else length
    fits = width <= TOPK
    if fits:
        index = col
    else:
        index = tl.load(RAW + row * TOPK + col, col < TOPK, -1)
    valid = (col < TOPK) & (index >= 0) & (index < width)
    score = tl.load(LOGITS + row * LOGIT_STRIDE + index, valid, -float("inf"))
    valid &= tl.abs(score) < float("inf")
    if SPARSE:
        block = tl.load(CANDIDATES + row * CANDIDATE_STRIDE + index // 8, valid, -1)
        index = block * 8 + index % 8
        valid &= (block >= 0) & (index < length)
    # Dense keys and the sorted, unique candidate blocks are already in logical
    # order when every score fits. The selected top-k can contain dummy repeats.
    if not fits:
        index = tl.sort(tl.where(valid, index, 2147483647))
        previous = tl.gather(index, tl.maximum(col - 1, 0), axis=0)
        valid = (index != 2147483647) & ((col == 0) | (index != previous))
    rank = tl.cumsum(valid.to(tl.int32)) - 1
    count = tl.sum(valid.to(tl.int32))
    page_row = tl.load(PAGE_ROWS + row) if PAGE_ROWS is not None else row
    page = tl.load(PAGES + page_row * PAGE_STRIDE + index // PAGE_SIZE, valid, 0)
    tl.store(OUT + row * TOPK + col, -1, (col < TOPK) & (col >= count))
    tl.store(OUT + row * TOPK + rank, page * PAGE_SIZE + index % PAGE_SIZE, valid)
    tl.store(LENGTHS + row, count)


def select_indexer_candidates(logits, lengths, indices):
    from lightllm.utils.vllm_utils import vllm_ops

    topk = indices.shape[1]
    scores = torch.empty((logits.shape[0], triton.cdiv(logits.shape[1], 8)), dtype=torch.float32, device=logits.device)
    # Rows that fit are written directly by _sort_candidates. Report zero valid
    # scores to the top-k operator for those rows.
    topk_lengths = torch.empty_like(lengths)
    _candidate_scores[(logits.shape[0], triton.cdiv(scores.shape[1], 128))](
        logits,
        lengths,
        scores,
        topk_lengths,
        logits.stride(0),
        scores.shape[1],
        topk,
        128,
    )
    vllm_ops.top_k_per_row_decode(
        scores,
        1,
        topk_lengths,
        indices,
        scores.shape[0],
        scores.stride(0),
        1,
        topk,
    )
    _sort_candidates[(logits.shape[0],)](
        indices,
        lengths,
        topk,
        triton.next_power_of_2(topk),
    )


def select_paged_topk(logits, lengths, pages, slots, selected_lengths, candidate_blocks=None, *, page_rows=None):
    from lightllm.utils.vllm_utils import vllm_ops

    raw = torch.empty_like(slots)
    topk = slots.shape[1]
    vllm_ops.top_k_per_row_decode(
        logits,
        1,
        lengths if candidate_blocks is None else lengths.clamp_max(logits.shape[1]),
        raw,
        logits.shape[0],
        logits.stride(0),
        1,
        topk,
    )
    _paged_topk_transform[(logits.shape[0],)](
        logits,
        raw,
        pages,
        page_rows,
        slots,
        lengths,
        selected_lengths,
        logits.stride(0),
        pages.stride(0),
        64,
        topk,
        triton.next_power_of_2(topk),
        candidate_blocks,
        candidate_blocks.stride(0) if candidate_blocks is not None else 0,
        candidate_blocks is not None,
    )


@triton.jit
def _gather_indexer_k(
    CACHE,
    REQS,
    LENGTHS,
    STARTS,
    TABLE,
    VALUES,
    SCALES,
    CACHE_STRIDE: tl.constexpr,
    TABLE_STRIDE: tl.constexpr,
    ENTRIES: tl.constexpr,
    RATIO: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    PAGE: tl.constexpr,
):
    batch, block = tl.program_id(0), tl.program_id(1)
    length = tl.load(LENGTHS + batch)
    if block * PAGE < length:
        key = block * PAGE + tl.arange(0, PAGE)
        req = tl.load(REQS + batch).to(tl.int64)
        full = key * RATIO + RATIO - 1
        slot = tl.load(TABLE + req * TABLE_STRIDE + full, key < length, 0).to(tl.int64) // RATIO
        page = slot // ENTRIES
        within = slot % ENTRIES
        base = page * CACHE_STRIDE + (within // PAGE) * PAGE * (HEAD_DIM + 4)
        offset = within % PAGE
        dim = tl.arange(0, HEAD_DIM)
        value = tl.load(
            CACHE.to(tl.pointer_type(tl.float8e4nv)) + base[:, None] + offset[:, None] * HEAD_DIM + dim[None, :],
            key[:, None] < length,
            0.0,
        )
        scale = tl.load(
            CACHE.to(tl.pointer_type(tl.float32)) + (base + PAGE * HEAD_DIM) // 4 + offset,
            key < length,
            0.0,
        )
        out = tl.load(STARTS + batch) + key
        tl.store(VALUES + out[:, None] * HEAD_DIM + dim[None, :], value, key[:, None] < length)
        tl.store(SCALES + out, scale, key < length)


def gather_indexer_k(packed, ratio, reqs, lengths, starts, table, token_capacity, capacity):
    """Gather each prefill request's keys once for DeepGEMM's dense MQA scorer."""
    import torch

    values = torch.empty((token_capacity, 128), dtype=torch.float8_e4m3fn, device=packed.device)
    scales = torch.empty((token_capacity,), dtype=torch.float32, device=packed.device)
    _gather_indexer_k[(reqs.shape[0], triton.cdiv(capacity, 64))](
        packed,
        reqs,
        lengths,
        starts,
        table,
        values,
        scales,
        packed.stride(0),
        table.stride(0),
        64,
        ratio,
        128,
        64,
        num_warps=4,
    )
    return values, scales
