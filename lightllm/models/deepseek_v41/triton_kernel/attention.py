"""V4.1 pair compression, FP8 history writes and paged sparse-indexer scoring."""

import torch
import triton
import triton.language as tl


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
def _mark_candidates(INDICES, MASK, WIDTH, TOPK: tl.constexpr, BLOCK: tl.constexpr):
    row = tl.program_id(0)
    col = tl.arange(0, BLOCK)
    index = tl.load(INDICES + row * TOPK + col, col < TOPK, -1)
    tl.store(MASK + row * WIDTH + index, True, (col < TOPK) & (index >= 0))


@triton.jit
def _mask_candidates(LOGITS, LENGTHS, MASK, STRIDE, WIDTH, BLOCK: tl.constexpr):
    row, tile = tl.program_id(0), tl.program_id(1)
    length = tl.load(LENGTHS + row)
    if tile * BLOCK < length:
        col = tile * BLOCK + tl.arange(0, BLOCK)
        selected = tl.load(MASK + row * WIDTH + col // 8, col < length, False)
        value = tl.load(LOGITS + row * STRIDE + col, col < length, -float("inf"))
        tl.store(LOGITS + row * STRIDE + col, tl.where(selected, value, -float("inf")), col < length)


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
):
    row = tl.program_id(0)
    col = tl.arange(0, BLOCK)
    index = tl.load(RAW + row * TOPK + col, col < TOPK, -1)
    # Native top-k can repeat its dummy index when fewer than TOPK scores are finite.
    index = tl.sort(index, descending=False)
    previous = tl.gather(index, tl.maximum(col - 1, 0), axis=0)
    valid = (col < TOPK) & (index >= 0) & (index < tl.load(VALID_LENGTHS + row))
    valid &= (col == 0) | (index != previous)
    score = tl.load(LOGITS + row * LOGIT_STRIDE + index, valid, -float("inf"))
    valid &= score > -float("inf")
    rank = tl.cumsum(valid.to(tl.int32)) - 1
    count = tl.sum(valid.to(tl.int32))
    page = tl.load(PAGES + row * PAGE_STRIDE + index // PAGE_SIZE, valid, 0)
    tl.store(OUT + row * TOPK + col, -1, (col < TOPK) & (col >= count))
    tl.store(OUT + row * TOPK + rank, page * PAGE_SIZE + index % PAGE_SIZE, valid)
    tl.store(LENGTHS + row, count)


def select_indexer_candidates(logits, lengths, mask, topk):
    from lightllm.utils.vllm_utils import vllm_ops

    topk = min(topk, mask.shape[1])
    scores = torch.empty(mask.shape, dtype=torch.float32, device=logits.device)
    _candidate_scores[(logits.shape[0], triton.cdiv(mask.shape[1], 128))](
        logits,
        lengths,
        scores,
        logits.stride(0),
        scores.shape[1],
        128,
    )
    block_lengths = (lengths + 7) // 8
    indices = torch.empty((logits.shape[0], topk), dtype=torch.int32, device=logits.device)
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
    _mark_candidates[(logits.shape[0],)](
        indices,
        mask,
        mask.stride(0),
        topk,
        triton.next_power_of_2(topk),
    )


def mask_indexer_candidates(logits, lengths, mask):
    _mask_candidates[(logits.shape[0], triton.cdiv(logits.shape[1], 1024))](
        logits,
        lengths,
        mask,
        logits.stride(0),
        mask.stride(0),
        1024,
    )


def select_indexer_topk(logits, lengths, pages, slots, selected_lengths):
    from lightllm.utils.vllm_utils import vllm_ops
    from lightllm.models.deepseek_v4.triton_kernel.cache_io import INDEXER_PAGE_SIZE

    raw = torch.empty_like(slots)
    topk = slots.shape[1]
    vllm_ops.top_k_per_row_decode(
        logits,
        1,
        lengths,
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
    )
