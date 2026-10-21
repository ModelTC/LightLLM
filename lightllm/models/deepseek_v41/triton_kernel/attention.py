"""CSA2 operations missing from V4's overlapping C4/C128 compressor."""

import torch
import triton
import triton.language as tl
from triton.language.extra import libdevice


@triton.jit
def _decode_e2m1(code):
    magnitude = code & 7
    value = tl.where(
        magnitude < 4, magnitude * 0.5, tl.where(magnitude < 6, magnitude - 2.0, tl.where(magnitude == 6, 4.0, 6.0))
    )
    return (value.to(tl.uint32, bitcast=True) | ((code.to(tl.uint32) & 8) << 28)).to(tl.float32, bitcast=True)


@triton.jit
def _rotary_quantize_fp4(X, FREQS, POS, OUT, SF, D: tl.constexpr, H: tl.constexpr, GROUP: tl.constexpr):
    row = tl.program_id(0)
    pair = tl.arange(0, D // 2)
    real = tl.load(X + row * D + pair * 2).to(tl.float32)
    imag = tl.load(X + row * D + pair * 2 + 1).to(tl.float32)
    position = tl.load(POS + row // H)
    frequency = FREQS + position * 64 + (pair - (D - 64) // 2) * 2
    cos = tl.load(frequency, pair >= (D - 64) // 2, 1.0)
    sin = tl.load(frequency + 1, pair >= (D - 64) // 2, 0.0)
    value = tl.reshape(tl.join(real * cos - imag * sin, real * sin + imag * cos), (D,)).to(tl.bfloat16).to(tl.float32)
    grouped = tl.reshape(value, (D // GROUP, GROUP))
    amax = tl.max(tl.abs(grouped), 1)
    if GROUP == 16:
        scale = tl.clamp(amax / 6.0, 2.0 ** -9, 448.0).to(tl.float8e4nv).to(tl.float32)
        encoded_scale = scale.to(tl.float8e4nv).to(tl.uint8, bitcast=True)
    else:
        scale = tl.exp2(tl.ceil(tl.log2(tl.maximum(amax, 6.0 * 2.0 ** -126) / 6.0)))
        encoded_scale = ((scale.to(tl.uint32, bitcast=True) >> 23) & 255).to(tl.uint8)
    normalized = tl.clamp(tl.div_rn(grouped, scale[:, None]), -6.0, 6.0)
    absolute = tl.abs(normalized)
    step = tl.where(absolute < 2, 0.5, tl.where(absolute < 4, 1.0, 2.0))
    rounded = libdevice.nearbyint(absolute / step) * step
    magnitude = tl.where(rounded <= 2, rounded * 2, tl.where(rounded <= 4, rounded + 2, 7)).to(tl.uint8)
    code = tl.reshape(magnitude | tl.where(normalized < 0, 8, 0).to(tl.uint8), (D // 2, 2))
    lo, hi = tl.split(code)
    tl.store(OUT + row * (D // 2) + pair, lo | (hi << 4))
    tl.store(SF + row * (D // GROUP) + tl.arange(0, D // GROUP), encoded_scale)


def rotary_quantize_fp4(x, freqs, positions, group_size=32):
    """RoPE plus packed E2M1: per-16 E4M3 scales for main KV, per-32 UE8M0 for index Q/K."""
    dim = x.shape[-1]
    heads = x.shape[1] if x.ndim == 3 else 1
    data = torch.empty((*x.shape[:-1], dim // 2), dtype=torch.uint8, device=x.device)
    scales = torch.empty((*x.shape[:-1], dim // group_size), dtype=torch.uint8, device=x.device)
    _rotary_quantize_fp4[(x.numel() // dim,)](
        x,
        torch.view_as_real(freqs).flatten(-2),
        positions,
        data,
        scales,
        dim,
        heads,
        group_size,
        num_warps=4,
        enable_fp_fusion=False,
    )
    return data, scales


@triton.jit
def _pool_pairs(KV, SCORE, REQ, POS, STATE, OUT, D: tl.constexpr, STATE_STRIDE: tl.constexpr):
    row = tl.program_id(0)
    cols = tl.arange(0, D)
    req = tl.load(REQ + row)
    pos = tl.load(POS + row)
    previous_req = tl.load(REQ + row - 1, row > 0, -1)
    in_batch = (row > 0) & (previous_req == req)
    previous_kv = tl.load(KV + (row - 1) * D + cols, in_batch, 0.0)
    previous_score = tl.load(SCORE + (row - 1) * D + cols, in_batch, 0.0)
    previous_kv += tl.load(STATE + req * STATE_STRIDE + cols, ~in_batch, 0.0)
    previous_score += tl.load(STATE + req * STATE_STRIDE + D + cols, ~in_batch, 0.0)
    current_kv = tl.load(KV + row * D + cols)
    current_score = tl.load(SCORE + row * D + cols)
    gate = tl.sigmoid(current_score - previous_score)
    pooled = previous_kv + (current_kv - previous_kv) * gate
    tl.store(OUT + row * D + cols, pooled, (pos + 1) % 2 == 0)


@triton.jit
def _commit_pair_tail(KV, SCORE, REQ, POS, STATE, T, D: tl.constexpr, HOLD: tl.constexpr, STATE_STRIDE: tl.constexpr):
    row = tl.program_id(0)
    req = tl.load(REQ + row)
    pos = tl.load(POS + row)
    next_req = tl.load(REQ + row + 1, row + 1 < T, -1)
    commit = (req != HOLD) & (next_req != req) & (pos % 2 == 0)
    cols = tl.arange(0, D)
    tl.store(STATE + req * STATE_STRIDE + cols, tl.load(KV + row * D + cols), commit)
    tl.store(STATE + req * STATE_STRIDE + D + cols, tl.load(SCORE + row * D + cols), commit)


def pool_pairs(kv, score, req, positions, state, hold_request):
    kv, score = kv.contiguous(), score.contiguous()
    out = torch.zeros_like(kv, dtype=torch.bfloat16)
    _pool_pairs[(kv.shape[0],)](kv, score, req, positions, state, out, kv.shape[1], state.stride(0))
    # Separate launch: a chunk's last even row cannot overwrite its first odd row's history.
    _commit_pair_tail[(kv.shape[0],)](
        kv, score, req, positions, state, kv.shape[0], kv.shape[1], hold_request, state.stride(0)
    )
    return out


@triton.jit
def _fwd_kernel_destindex_copy_kv_dsv41(
    KV,
    SF,
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
    packed = tl.load(KV + row * 256 + d // 2)
    code = (packed >> ((d % 2) * 4)) & 15
    scale_fp4 = tl.load(SF.to(tl.pointer_type(tl.float8e4nv)) + row * 32 + d // 16).to(tl.float32)
    value = (_decode_e2m1(code) * scale_fp4).to(tl.bfloat16).to(tl.float32)
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


def destindex_copy_kv_dsv41(kv, full_slots, positions, cache, ratio, page_size):
    """Convert packed FP4 to Hopper FlashMLA's FP8/BF16 cache while scattering history."""
    data, scale = kv
    _fwd_kernel_destindex_copy_kv_dsv41[(data.shape[0],)](
        data,
        scale,
        full_slots,
        positions,
        cache,
        ratio,
        page_size,
        cache.stride(0),
    )


@triton.jit
def _index_logits(
    Q,
    Q_SF,
    WEIGHTS,
    CACHE,
    REQ,
    POS,
    TABLE,
    CANDIDATES,
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
        key_pos = tl.load(CANDIDATES + row * CANDIDATE_WIDTH + col // 8, col < N, -1) * 8 + col % 8
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
    packed_k = tl.load(
        CACHE + pg[None, :] * CACHE_STRIDE + off[None, :] * 64 + d[:, None] // 2,
        valid[None, :],
        0,
    )
    code_k = (packed_k >> ((d[:, None] % 2) * 4)) & 15
    scale_k = tl.load(
        CACHE + pg[None, :] * CACHE_STRIDE + PAGE * 64 + off[None, :] * 4 + d[:, None] // 32,
        valid[None, :],
        127,
    )
    k = (_decode_e2m1(code_k) * tl.exp2(scale_k.to(tl.float32) - 127)).to(tl.bfloat16)
    packed_q = tl.load(Q + row * H * 64 + h[:, None] * 64 + d[None, :] // 2, h[:, None] < H, 0)
    code_q = (packed_q >> ((d[None, :] % 2) * 4)) & 15
    scale_q = tl.load(Q_SF + row * H * 4 + h[:, None] * 4 + d[None, :] // 32, h[:, None] < H, 127)
    q = (_decode_e2m1(code_q) * tl.exp2(scale_q.to(tl.float32) - 127)).to(tl.bfloat16)
    scores = tl.dot(q, k).to(tl.float32)
    weights = tl.load(WEIGHTS + row * H + h, h < H, 0.0)
    logits = tl.sum(tl.maximum(scores, 0.0) * weights[:, None], 0)
    tl.store(OUT + row * N + col, tl.where(valid, logits, -float("inf")), col < N)


def index_logits(q, weights, cache, req, positions, req_to_token, ratio, count, candidates=None):
    data, scale = q
    logits = torch.empty((data.shape[0], count), dtype=torch.float32, device=data.device)
    if count:
        _index_logits[(data.shape[0], triton.cdiv(count, 32))](
            data,
            scale,
            weights.contiguous(),
            cache,
            req,
            positions,
            req_to_token,
            candidates,
            logits,
            req_to_token.shape[1],
            cache.stride(0),
            count,
            data.shape[1],
            max(16, triton.next_power_of_2(data.shape[1])),
            ratio,
            256 // ratio,
            candidates is not None,
            candidates.shape[1] if candidates is not None else 0,
            32,
            num_warps=4,
        )
    return logits
