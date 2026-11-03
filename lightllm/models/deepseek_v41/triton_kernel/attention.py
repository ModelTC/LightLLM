"""V4.1 pair compression, FP8 history writes and paged sparse-indexer scoring."""

import torch
import triton
import triton.language as tl


@triton.jit
def _bound_swa_replay(
    INDICES,
    LENGTHS,
    POS,
    STARTS,
    IMAGE_LEFT,
    IMAGE_RIGHT,
    STRIDE: tl.constexpr,
    WIDTH: tl.constexpr,
    WINDOW: tl.constexpr,
    HAS_IMAGE: tl.constexpr,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    col = tl.arange(0, BLOCK)
    pos, start = tl.load(POS + row), tl.load(STARTS + row)
    length = tl.load(LENGTHS + row)
    shift = 0
    if HAS_IMAGE:
        left, right = tl.load(IMAGE_LEFT + row), tl.load(IMAGE_RIGHT + row)
        is_image = (left > 0) | (right > 0)
        first = tl.maximum(pos - (WINDOW - 1) - tl.maximum(left - (WINDOW - 1), 0), 0)
        shift = tl.where(is_image, tl.maximum(start - first, 0), 0)
        length = tl.where(is_image, tl.maximum(length - shift, 0), tl.minimum(length, pos - start + 1))
    else:
        length = tl.minimum(length, pos - start + 1)
    value = tl.load(INDICES + row * STRIDE + col + shift, (col < length) & (col + shift < WIDTH), -1)
    tl.store(INDICES + row * STRIDE + col, value, col < WIDTH)
    tl.store(LENGTHS + row, length)


def bound_swa_replay(state, starts):
    width = state.dsv4_swa_indices.shape[1]
    _bound_swa_replay[(state.position_ids.numel(),)](
        state.dsv4_swa_indices,
        state.dsv4_swa_lengths,
        state.position_ids,
        starts,
        state.dsv4_image_left,
        state.dsv4_image_right,
        state.dsv4_swa_indices.stride(0),
        width,
        state.req_manager.cache_config.sliding_window,
        state.dsv4_image_left is not None,
        triton.next_power_of_2(width),
    )


@triton.jit
def _pool_pairs(
    KV,
    SCORE,
    REQ,
    POS,
    STATE,
    OUT,
    D: tl.constexpr,
    STATE_STRIDE: tl.constexpr,
    STATE_SLOT_STRIDE: tl.constexpr,
    STATE_SLOTS: tl.constexpr,
    KV_STRIDE: tl.constexpr,
    SCORE_STRIDE: tl.constexpr,
):
    row = tl.program_id(0)
    cols = tl.arange(0, D)
    req = tl.load(REQ + row)
    pos = tl.load(POS + row)
    previous_req = tl.load(REQ + row - 1, row > 0, -1)
    in_batch = (row > 0) & (previous_req == req)
    previous_kv = tl.load(KV + (row - 1) * KV_STRIDE + cols, in_batch, 0.0)
    previous_score = tl.load(SCORE + (row - 1) * SCORE_STRIDE + cols, in_batch, 0.0)
    previous_state = STATE + req * STATE_STRIDE + (pos - 1) % STATE_SLOTS * STATE_SLOT_STRIDE
    previous_kv += tl.load(previous_state + cols, ~in_batch & (pos % 2 == 1), 0.0)
    previous_score += tl.load(previous_state + D + cols, ~in_batch & (pos % 2 == 1), 0.0)
    current_kv = tl.load(KV + row * KV_STRIDE + cols)
    current_score = tl.load(SCORE + row * SCORE_STRIDE + cols)
    gate = tl.sigmoid(current_score - previous_score)
    pooled = previous_kv + (current_kv - previous_kv) * gate
    tl.store(OUT + row * D + cols, tl.where((pos + 1) % 2 == 0, pooled, 0.0))


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
    STATE_SLOT_STRIDE: tl.constexpr,
    STATE_SLOTS: tl.constexpr,
    KV_STRIDE: tl.constexpr,
    SCORE_STRIDE: tl.constexpr,
):
    row = tl.program_id(0)
    req = tl.load(REQ + row)
    pos = tl.load(POS + row)
    # Keep the newest even position for each ring slot. The period doubles for odd-sized rings.
    period: tl.constexpr = STATE_SLOTS if STATE_SLOTS % 2 == 0 else 2 * STATE_SLOTS
    next_req = tl.load(REQ + row + period, row + period < T, -1)
    commit = (req != HOLD) & (next_req != req) & (pos % 2 == 0)
    cols = tl.arange(0, D)
    dest = STATE + req * STATE_STRIDE + pos % STATE_SLOTS * STATE_SLOT_STRIDE
    tl.store(dest + cols, tl.load(KV + row * KV_STRIDE + cols), commit)
    tl.store(dest + D + cols, tl.load(SCORE + row * SCORE_STRIDE + cols), commit)


def pool_pairs(kv, score, req, positions, state, hold_request):
    out = torch.empty_like(kv, dtype=torch.bfloat16)
    _pool_pairs[(kv.shape[0],)](
        kv,
        score,
        req,
        positions,
        state,
        out,
        kv.shape[1],
        state.stride(0),
        state.stride(1),
        state.shape[1],
        kv.stride(0),
        score.stride(0),
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
        state.stride(1),
        state.shape[1],
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
    slot = tl.load(FULL_SLOT + row).to(tl.int64)
    if slot < 0:
        return
    slot //= RATIO
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
def _candidate_scores(LOGITS, LENGTHS, OUT, STRIDE, WIDTH, BLOCK: tl.constexpr):
    row, tile = tl.program_id(0), tl.program_id(1)
    length = tl.load(LENGTHS + row)
    if tile * BLOCK * 8 < length:
        block = tile * BLOCK + tl.arange(0, BLOCK)
        col = block[:, None] * 8 + tl.arange(0, 8)[None, :]
        value = tl.load(LOGITS + row * STRIDE + col, col < length, -float("inf"))
        score = tl.max(value, 1)
        score = tl.where(block == (length - 1) // 8, float("inf"), score)
        tl.store(OUT + row * WIDTH + block, score, (block < WIDTH) & (block < tl.cdiv(length, 8)))


@triton.jit
def _sort_candidates(INDICES, LENGTHS, TOPK: tl.constexpr, BLOCK: tl.constexpr):
    row = tl.program_id(0)
    col = tl.arange(0, BLOCK)
    index = tl.load(INDICES + row * TOPK + col, col < TOPK, -1)
    valid = (index >= 0) & (index < tl.cdiv(tl.load(LENGTHS + row), 8))
    index = tl.sort(tl.where(valid, index, 2147483647))
    valid = (index != 2147483647) & ((col == 0) | (index != tl.gather(index, tl.maximum(col - 1, 0), axis=0)))
    rank = tl.cumsum(valid.to(tl.int32)) - 1
    count = tl.sum(valid.to(tl.int32))
    tl.store(INDICES + row * TOPK + col, -1, (col < TOPK) & (col >= count))
    tl.store(INDICES + row * TOPK + rank, index, valid)


@triton.jit
def _candidate_indexer_logits(
    Q,
    CACHE,
    WEIGHTS,
    PAGES,
    CANDIDATES,
    LENGTHS,
    LOGITS,
    Q_ROW: tl.constexpr,
    Q_HEAD: tl.constexpr,
    CACHE_PAGE: tl.constexpr,
    WEIGHT_ROW: tl.constexpr,
    PAGE_ROW: tl.constexpr,
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
        page = tl.load(PAGES + row * PAGE_ROW + position // 64, valid, 0).to(tl.int64)
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


def candidate_indexer_logits(query, cache, weights, pages, candidates, lengths):
    """Score the producer's sorted, -1-padded eight-key blocks in Hopper FP8 slabs."""
    assert query.shape[-1] == 128
    logits = torch.empty((query.shape[0], candidates.shape[1] * 8), dtype=torch.float32, device=query.device)
    _candidate_indexer_logits[(query.shape[0], triton.cdiv(logits.shape[1], 64))](
        query,
        cache,
        weights,
        pages,
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
        64,
        num_warps=4,
    )
    return logits


@triton.jit
def _indexer_topk_transform(
    LOGITS,
    RAW,
    PAGES,
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
    index = tl.load(RAW + row * TOPK + col, col < TOPK, -1)
    width = CANDIDATE_STRIDE * 8 if SPARSE else tl.load(VALID_LENGTHS + row)
    valid = (col < TOPK) & (index >= 0) & (index < width)
    score = tl.load(LOGITS + row * LOGIT_STRIDE + index, valid, -float("inf"))
    valid &= score > -float("inf")
    if SPARSE:
        block = tl.load(CANDIDATES + row * CANDIDATE_STRIDE + index // 8, valid, -1)
        index = block * 8 + index % 8
        valid &= (block >= 0) & (index < tl.load(VALID_LENGTHS + row))
    # Native top-k can repeat its dummy index when fewer than TOPK scores are finite.
    index = tl.sort(tl.where(valid, index, 2147483647))
    previous = tl.gather(index, tl.maximum(col - 1, 0), axis=0)
    valid = (index != 2147483647) & ((col == 0) | (index != previous))
    rank = tl.cumsum(valid.to(tl.int32)) - 1
    count = tl.sum(valid.to(tl.int32))
    page = tl.load(PAGES + row * PAGE_STRIDE + index // PAGE_SIZE, valid, 0)
    tl.store(OUT + row * TOPK + col, -1, (col < TOPK) & (col >= count))
    tl.store(OUT + row * TOPK + rank, page * PAGE_SIZE + index % PAGE_SIZE, valid)
    tl.store(LENGTHS + row, count)


def select_indexer_candidates(logits, lengths, indices):
    from lightllm.utils.vllm_utils import vllm_ops

    topk = indices.shape[1]
    scores = torch.empty((logits.shape[0], triton.cdiv(logits.shape[1], 8)), dtype=torch.float32, device=logits.device)
    _candidate_scores[(logits.shape[0], triton.cdiv(scores.shape[1], 128))](
        logits,
        lengths,
        scores,
        logits.stride(0),
        scores.shape[1],
        128,
    )
    block_lengths = (lengths + 7) // 8
    vllm_ops.top_k_per_row_decode(
        scores,
        1,
        block_lengths,
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


def select_indexer_topk(logits, lengths, pages, slots, selected_lengths, candidate_blocks=None):
    from lightllm.utils.vllm_utils import vllm_ops
    from lightllm.models.deepseek_v4.triton_kernel.cache_io import INDEXER_PAGE_SIZE

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
    _indexer_topk_transform[(logits.shape[0],)](
        logits,
        raw,
        pages,
        slots,
        lengths,
        selected_lengths,
        logits.stride(0),
        pages.stride(0),
        INDEXER_PAGE_SIZE,
        topk,
        triton.next_power_of_2(topk),
        candidate_blocks,
        candidate_blocks.stride(0) if candidate_blocks is not None else 0,
        candidate_blocks is not None,
    )
