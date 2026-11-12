"""Indexed NVFP4 latent writes; RoPE and the packed indexer are preserved."""

import torch
import triton
import triton.language as tl

from .nvfp4 import encode_e2m1


@triton.jit
def _quantize_mla(
    X,
    Indices,
    Out,
    XS: tl.constexpr,
    OS: tl.constexpr,
    IS: tl.constexpr,
    DIM: tl.constexpr,
    ROPE: tl.constexpr,
    GROUPS: tl.constexpr,
    RB: tl.constexpr,
):
    token = tl.program_id(0).to(tl.int64)
    group = tl.arange(0, GROUPS)
    pair = tl.arange(0, 8)
    d = group[:, None] * 16 + pair[None, :] * 2
    x0 = tl.load(X + token * XS[0] + d * XS[2], d < DIM, 0).to(tl.float32)
    x1 = tl.load(X + token * XS[0] + (d + 1) * XS[2], d < DIM, 0).to(tl.float32)
    amax = tl.max(tl.maximum(tl.abs(x0), tl.abs(x1)), 1)
    scale = tl.minimum(amax / 6.0, 448.0).to(tl.float8e4nv)
    inverse = tl.where(amax != 0, tl.minimum(tl.div_rn(1.0, scale.to(tl.float32)), 3.4028234663852886e38), 0.0)
    packed = encode_e2m1(x0 * inverse[:, None]) | (encode_e2m1(x1 * inverse[:, None]) << 4)
    slot = tl.load(Indices + token * IS).to(tl.int64)
    row = Out + slot * OS
    tl.store(row + d // 2, packed, d < DIM)
    tl.store(row + DIM // 2 + group, scale.to(tl.uint8, bitcast=True), group < DIM // 16)
    if ROPE:
        r = tl.arange(0, RB)
        value = tl.load(X + token * XS[0] + (DIM + r) * XS[2], r < ROPE, 0)
        rope_out = (row + DIM // 2 + DIM // 16).to(tl.pointer_type(X.dtype.element_ty))
        tl.store(rope_out + r, value, r < ROPE)


@torch.no_grad()
def quantize_nvfp4_mla(kv, token_indices, out, rope_dim=0):
    assert kv.is_cuda and token_indices.is_cuda and out.is_cuda
    assert kv.device == token_indices.device == out.device
    assert kv.ndim == out.ndim == 3 and kv.shape[1] == out.shape[1] == 1
    assert kv.dtype in (torch.bfloat16, torch.float16) and out.dtype == torch.uint8
    assert token_indices.ndim == 1 and token_indices.dtype in (torch.int32, torch.int64)
    assert kv.shape[0] == token_indices.numel() and out.stride(-1) == 1
    dim = kv.shape[-1] - rope_dim
    assert dim > 0 and dim % 32 == 0 and rope_dim >= 0
    assert out.shape[-1] == dim // 2 + dim // 16 + rope_dim * 2
    assert out.data_ptr() % 2 == out.stride(0) % 2 == 0
    if token_indices.numel():
        _quantize_mla[(kv.shape[0],)](
            kv,
            token_indices,
            out,
            kv.stride(),
            out.stride(0),
            token_indices.stride(0),
            dim,
            rope_dim,
            triton.next_power_of_2(dim // 16),
            triton.next_power_of_2(max(rope_dim, 1)),
        )
