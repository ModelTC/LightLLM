"""NVFP4 writes into pages with separate KV-data and block-scale regions."""

import triton
import triton.language as tl

from .nvfp4 import encode_e2m1


@triton.jit
def store_packed_nvfp4(
    value,
    rotated,
    out,
    slot,
    valid,
    DIM: tl.constexpr,
    ROPE: tl.constexpr,
    PAGE_TOKENS: tl.constexpr,
    PAGE_BYTES: tl.constexpr,
    BLOCK: tl.constexpr,
):
    """Encode NoPE in block16 E2M1/E4M3 and retain the rotated BF16 tail."""
    NOPE: tl.constexpr = DIM - ROPE
    DATA_BYTES: tl.constexpr = NOPE // 2 + ROPE * 2
    SCALE_BYTES: tl.constexpr = triton.cdiv(NOPE // 16, 8) * 8
    page, token = slot.to(tl.int64) // PAGE_TOKENS, slot % PAGE_TOKENS
    data = out + page * PAGE_BYTES + token * DATA_BYTES
    scales = out + page * PAGE_BYTES + PAGE_TOKENS * DATA_BYTES + token * SCALE_BYTES
    pairs = tl.reshape(value, (BLOCK // 16, 8, 2))
    x0, x1 = tl.split(pairs)
    amax = tl.max(tl.maximum(tl.abs(x0), tl.abs(x1)), 1)
    scale = tl.minimum(amax / 6.0, 448.0).to(tl.float8e4nv)
    inverse = tl.where(amax != 0, tl.minimum(tl.div_rn(1.0, scale.to(tl.float32)), 3.4028234663852886e38), 0.0)
    packed = encode_e2m1(x0 * inverse[:, None]) | (encode_e2m1(x1 * inverse[:, None]) << 4)
    group = tl.arange(0, BLOCK // 16)
    pair = tl.arange(0, 8)
    tl.store(data + group[:, None] * 8 + pair[None, :], packed, valid & (group[:, None] < NOPE // 16))
    # Scale padding is part of the byte contract and must not retain old data.
    tl.store(scales + group, scale.to(tl.uint8, bitcast=True), valid & (group < NOPE // 16))
    padding = NOPE // 16 + tl.arange(0, 8)
    tl.store(scales + padding, 0, valid & (padding < SCALE_BYTES))
    dim = tl.arange(0, BLOCK)
    tail = (data + NOPE // 2).to(tl.pointer_type(tl.bfloat16))
    tl.store(tail + dim - NOPE, rotated.to(tl.bfloat16), valid & (dim >= NOPE) & (dim < DIM))
