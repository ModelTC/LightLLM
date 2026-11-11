"""Gather visible NVFP4 KV once into a request-packed, temporary compute buffer."""

import torch
import triton
import triton.language as tl

from .nvfp4 import decode_e2m1


@triton.jit
def _dequantize(
    K,
    V,
    KO,
    VO,
    TABLE,
    REQ,
    LENGTH,
    CU,
    KS: tl.constexpr,
    VS: tl.constexpr,
    TS: tl.constexpr,
    HEADS: tl.constexpr,
    DIM: tl.constexpr,
    BT: tl.constexpr,
):
    tile, batch, head = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    length = tl.load(LENGTH + batch)
    if tile * BT >= length:
        return
    token = tile * BT + tl.arange(0, BT)
    dim = tl.arange(0, DIM)
    request = tl.load(REQ + batch)
    slot = tl.load(TABLE + request * TS[0] + token * TS[1], token < length, other=0).to(tl.int64)
    km = slot[:, None] * KS[0] + head * KS[1]
    vm = slot[:, None] * VS[0] + head * VS[1]
    mask = token[:, None] < length
    kd = decode_e2m1(tl.load(K + km + (dim[None, :] // 2) * KS[2], mask, other=0), dim[None, :])
    vd = decode_e2m1(tl.load(V + vm + (dim[None, :] // 2) * VS[2], mask, other=0), dim[None, :])
    ks = tl.load(K + km + (DIM // 2 + dim[None, :] // 16) * KS[2], mask, other=0)
    vs = tl.load(V + vm + (DIM // 2 + dim[None, :] // 16) * VS[2], mask, other=0)
    start = tl.load(CU + batch).to(tl.int64)
    dst = ((start + token[:, None]) * HEADS + head) * DIM + dim[None, :]
    tl.store(KO + dst, kd.to(tl.float32) * ks.to(tl.float8e4nv, bitcast=True).to(tl.float32), mask)
    tl.store(VO + dst, vd.to(tl.float32) * vs.to(tl.float8e4nv, bitcast=True).to(tl.float32), mask)


def dequantize_nvfp4_kv(k, v, table, requests, lengths, cu_seqlens, max_length, k_out, v_out):
    """Decode all visible tokens, including the quantized current prefill chunk.

    The caller supplies CPU-known bounds; this operation never reads lengths back
    to the host. Output rows are concatenated in request order for varlen FA3.
    """
    assert k.dtype == v.dtype == torch.uint8
    assert k_out.shape == v_out.shape and k_out.is_contiguous() and v_out.is_contiguous()
    heads, dim = k_out.shape[1:]
    assert k.shape[1:] == v.shape[1:] == (heads, dim // 2 + dim // 16)
    if requests.numel() and max_length:
        _dequantize[(triton.cdiv(max_length, 16), requests.numel(), heads)](
            k,
            v,
            k_out,
            v_out,
            table,
            requests,
            lengths,
            cu_seqlens,
            k.stride(),
            v.stride(),
            table.stride(),
            heads,
            dim,
            16,
            num_warps=4,
        )
