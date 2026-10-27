"""CSA2 operations missing from V4's overlapping C4/C128 compressor."""

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
def _index_logits(
    Q,
    WEIGHTS,
    CACHE,
    REQ,
    POS,
    TABLE,
    CANDIDATE_BLOCKS,
    OUT,
    TABLE_STRIDE: tl.constexpr,
    CACHE_STRIDE: tl.constexpr,
    N,
    H: tl.constexpr,
    BH: tl.constexpr,
    RATIO: tl.constexpr,
    PAGE: tl.constexpr,
    CANDIDATE: tl.constexpr,
    CANDIDATE_WIDTH,
    BLOCK: tl.constexpr,
):
    row, block = tl.program_id(0), tl.program_id(1)
    col = block * BLOCK + tl.arange(0, BLOCK)
    if CANDIDATE:
        key_pos = tl.load(CANDIDATE_BLOCKS + row * CANDIDATE_WIDTH + col // 8, col < N, -1) * 8 + col % 8
    else:
        key_pos = col
    req = tl.load(REQ + row).to(tl.int64)
    visible = (tl.load(POS + row) + 1) // RATIO
    valid = (col < N) & (key_pos >= 0) & (key_pos < visible)
    full_pos = key_pos * RATIO + RATIO - 1
    slot = tl.load(TABLE + req * TABLE_STRIDE + full_pos, valid, 0).to(tl.int64) // RATIO
    pg, off = slot // PAGE, slot % PAGE
    d = tl.arange(0, 128)
    h = tl.arange(0, BH)
    k = tl.load(
        CACHE.to(tl.pointer_type(tl.float8e4nv)) + pg[None, :] * CACHE_STRIDE + off[None, :] * 128 + d[:, None],
        valid[None, :],
        0.0,
    )
    scale_k = tl.load(
        CACHE.to(tl.pointer_type(tl.float32)) + (pg * CACHE_STRIDE + PAGE * 128) // 4 + off,
        valid,
        0.0,
    )
    q = tl.load(Q + row * H * 128 + h[:, None] * 128 + d[None, :], h[:, None] < H, 0.0)
    scores = tl.dot(q, k, max_num_imprecise_acc=32).to(tl.float32) * scale_k[None, :]
    weights = tl.load(WEIGHTS + row * H + h, h < H, 0.0)
    logits = tl.sum(tl.maximum(scores, 0.0) * weights[:, None], 0)
    tl.store(OUT + row * N + col, tl.where(valid, logits, -float("inf")), col < N)


def index_logits(q, weights, cache, req, positions, req_to_token, ratio, count, candidate_blocks=None):
    logits = torch.empty((q.shape[0], count), dtype=torch.float32, device=q.device)
    if count:
        _index_logits[(q.shape[0], triton.cdiv(count, 32))](
            q,
            weights.contiguous(),
            cache,
            req,
            positions,
            req_to_token,
            candidate_blocks,
            logits,
            req_to_token.shape[1],
            cache.stride(0),
            count,
            q.shape[1],
            # SM90 FP8 warpgroup MMA requires at least 64 rows; unused heads are zero.
            max(64, triton.next_power_of_2(q.shape[1])),
            ratio,
            256 // ratio,
            candidate_blocks is not None,
            candidate_blocks.shape[1] if candidate_blocks is not None else 0,
            32,
            num_warps=4,
        )
    return logits
