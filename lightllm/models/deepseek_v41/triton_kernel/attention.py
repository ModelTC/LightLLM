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
    previous_state = STATE + req * STATE_STRIDE
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


def select_indexer_topk(logits, lengths, pages, slots, selected_lengths, page_size, candidate_mask, candidate_topk):
    """Keep the latest candidate block, then map finite causal top-k entries to physical slots."""
    columns = torch.arange(logits.shape[1], device=logits.device)
    logits.masked_fill_(columns >= lengths[:, None], -torch.inf)
    if candidate_mask is not None:
        blocks = logits.unflatten(-1, (-1, 8))
        if candidate_topk is not None:
            scores = blocks.amax(-1)
            recent = ((lengths.long() - 1).clamp_min(0) // 8).unsqueeze(1)
            scores.scatter_(1, recent, torch.inf)
            selected = scores.topk(min(candidate_topk, scores.shape[1]), sorted=False).indices
            candidate_mask.zero_().scatter_(1, selected, True)
        else:
            blocks.masked_fill_(~candidate_mask[..., None], -torch.inf)
    scores, indices = logits.topk(min(slots.shape[1], logits.shape[1]), sorted=False)
    indices = indices.masked_fill(~scores.isfinite(), logits.shape[1]).sort(-1).values
    valid = indices < logits.shape[1]
    indices = indices.masked_fill(~valid, 0)
    physical = pages.gather(1, indices // page_size) * page_size + indices % page_size
    slots.fill_(-1)
    slots[:, : indices.shape[1]] = physical.masked_fill(~valid, -1)
    selected_lengths.copy_(valid.sum(-1))
