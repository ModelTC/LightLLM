"""Fused NVFP4 block16 quantization and indexed KV writes."""

import torch
import triton
import triton.language as tl


@triton.jit
def encode_e2m1(value):
    value = value.to(tl.float32)
    magnitude = tl.abs(value)
    code = (
        (magnitude > 0.25).to(tl.int32)
        + (magnitude >= 0.75).to(tl.int32)
        + (magnitude > 1.25).to(tl.int32)
        + (magnitude >= 1.75).to(tl.int32)
        + (magnitude > 2.5).to(tl.int32)
        + (magnitude >= 3.5).to(tl.int32)
        + (magnitude > 5.0).to(tl.int32)
    )
    sign = (value.to(tl.uint32, bitcast=True) >> 28) & 8
    return (code | sign).to(tl.uint8)


@triton.jit
def decode_e2m1(packed, feature_offsets):
    code = (packed >> ((feature_offsets & 1) * 4)) & 15
    magnitude = code & 7
    bits = tl.where(magnitude < 2, magnitude * 0x3F00, (magnitude << 6) + 0x3F00)
    bits = bits | ((code & 8).to(tl.uint16) << 12)
    return bits.to(tl.uint16).to(tl.bfloat16, bitcast=True)


@triton.jit(do_not_specialize=["count"])
def _quantize(
    KV,
    INDICES,
    K,
    V,
    count,
    IS: tl.constexpr,
    KS: tl.constexpr,
    VS: tl.constexpr,
    XS: tl.constexpr,
    HEADS: tl.constexpr,
    DIM: tl.constexpr,
    GROUPS: tl.constexpr,
):
    token, head = tl.program_id(0).to(tl.int64), tl.program_id(1)
    group = tl.arange(0, GROUPS)
    pair = tl.arange(0, 8)
    d = group[:, None] * 16 + pair[None, :] * 2
    valid = (token < count) & (d < DIM)
    offset = token * XS[0] + head * XS[1] + d * XS[2]
    k0 = tl.load(KV + offset, valid, other=0).to(tl.float32)
    k1 = tl.load(KV + offset + XS[2], valid, other=0).to(tl.float32)
    v0 = tl.load(KV + offset + HEADS * XS[1], valid, other=0).to(tl.float32)
    v1 = tl.load(KV + offset + HEADS * XS[1] + XS[2], valid, other=0).to(tl.float32)
    ka = tl.max(tl.maximum(tl.abs(k0), tl.abs(k1)), 1)
    va = tl.max(tl.maximum(tl.abs(v0), tl.abs(v1)), 1)
    # Saturate E4M3 block scales, then normalize using a finite FP32 reciprocal.
    ksf = tl.minimum(ka * (1.0 / 6.0), 448.0).to(tl.float8e4nv)
    vsf = tl.minimum(va * (1.0 / 6.0), 448.0).to(tl.float8e4nv)
    ki = tl.where(ka != 0, tl.minimum(tl.div_rn(1.0, ksf.to(tl.float32)), 3.4028234663852886e38), 0.0)
    vi = tl.where(va != 0, tl.minimum(tl.div_rn(1.0, vsf.to(tl.float32)), 3.4028234663852886e38), 0.0)
    ks, vs = ksf.to(tl.uint8, bitcast=True), vsf.to(tl.uint8, bitcast=True)
    kp = encode_e2m1(k0 * ki[:, None]) | (encode_e2m1(k1 * ki[:, None]) << 4)
    vp = encode_e2m1(v0 * vi[:, None]) | (encode_e2m1(v1 * vi[:, None]) << 4)
    slot = tl.load(INDICES + token * IS).to(tl.int64)
    tl.store(K + slot * KS[0] + head * KS[1] + (d // 2) * KS[2], kp, valid)
    tl.store(V + slot * VS[0] + head * VS[1] + (d // 2) * VS[2], vp, valid)
    tl.store(K + slot * KS[0] + head * KS[1] + (DIM // 2 + group) * KS[2], ks, group < DIM // 16)
    tl.store(V + slot * VS[0] + head * VS[1] + (DIM // 2 + group) * VS[2], vs, group < DIM // 16)


@torch.no_grad()
def quantize_nvfp4_kv(kv, token_indices, k_out, v_out):
    assert kv.is_cuda and token_indices.is_cuda and k_out.is_cuda and v_out.is_cuda
    assert kv.ndim == 3 and kv.dtype in (torch.bfloat16, torch.float16)
    assert token_indices.ndim == 1 and token_indices.dtype in (torch.int32, torch.int64)
    count, both_heads, dim = kv.shape
    heads = k_out.shape[1]
    assert dim % 16 == 0 and both_heads == 2 * heads and count == token_indices.numel()
    assert k_out.shape == v_out.shape and k_out.shape[2] == dim // 2 + dim // 16
    assert k_out.dtype == v_out.dtype == torch.uint8
    if count:
        _quantize[(count, heads)](
            kv,
            token_indices,
            k_out,
            v_out,
            count,
            token_indices.stride(0),
            k_out.stride(),
            v_out.stride(),
            kv.stride(),
            heads,
            dim,
            triton.next_power_of_2(dim // 16),
            num_warps=4,
        )
