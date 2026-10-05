"""Fused routed SwiGLU activation and non-UE8M0 FP8 quantization."""

from typing import Callable, Tuple

import torch
from triton.experimental import gluon
from triton.experimental.gluon import language as gl

from lightllm.utils.sgl_utils import HAS_SGL_KERNEL


_GROUPS_PER_CTA = 8
_NUM_WARPS = 4


@gluon.jit
def _routed_silu_mul_quant_fp8_kernel(
    input_ptr,
    q_ptr,
    scale_ptr,
    input_stride_m,
    q_stride_m,
    scale_stride_m,
    scale_stride_n,
    fp8_min: gl.constexpr,
    fp8_max: gl.constexpr,
    eps: gl.constexpr,
    limit: gl.constexpr,
    groups_per_cta: gl.constexpr,
):
    parent: gl.constexpr = gl.BlockedLayout(
        size_per_thread=[1, 4], threads_per_warp=[1, 32], warps_per_cta=[gl.num_warps(), 1], order=[1, 0]
    )
    groups_layout: gl.constexpr = gl.SliceLayout(1, parent)
    columns_layout: gl.constexpr = gl.SliceLayout(0, parent)
    group_ids = gl.arange(0, groups_per_cta, layout=groups_layout) + gl.program_id(0) * groups_per_cta
    columns = gl.arange(0, 128, layout=columns_layout)
    row = group_ids // 16
    group = group_ids % 16
    offsets = (
        gl.expand_dims(row, 1) * input_stride_m
        + gl.expand_dims(group * 128, 1)
        + gl.expand_dims(columns, 0)
    )
    gate = gl.load(input_ptr + offsets).to(gl.float32)
    up = gl.load(input_ptr + offsets + 2048)
    gate = gl.minimum(gate, limit)
    up = gl.minimum(gl.maximum(up, -limit), limit)
    gate = (gate / (1.0 + gl.exp(-gate))).to(input_ptr.dtype.element_ty)
    product = (up * gate).to(input_ptr.dtype.element_ty).to(gl.float32)
    amax = gl.maximum(gl.max(gl.abs(product), axis=1), eps)
    scale = amax / fp8_max
    quant = gl.minimum(
        gl.maximum(product * (fp8_max / gl.expand_dims(amax, 1)), fp8_min), fp8_max
    ).to(q_ptr.dtype.element_ty)
    gl.store(
        q_ptr + gl.expand_dims(row, 1) * q_stride_m + gl.expand_dims(group * 128, 1) + gl.expand_dims(columns, 0),
        quant,
    )
    gl.store(scale_ptr + row * scale_stride_m + group * scale_stride_n, scale)


def routed_silu_mul_quant_fp8(
    input: torch.Tensor,
    output_q: torch.Tensor,
    output_scale: torch.Tensor,
    *,
    limit: float = 10.0,
    eps: float = 1e-10,
) -> None:
    """Fill production-layout FP8 and column-major non-UE8M0 scale buffers."""
    if input.dtype != torch.bfloat16 or input.ndim != 2 or input.shape[1] != 4096 or not input.is_contiguous():
        raise ValueError("expected contiguous BF16 [M, 4096] input")
    if output_q.dtype != torch.float8_e4m3fn or tuple(output_q.shape) != (input.shape[0], 2048):
        raise ValueError("expected FP8 [M, 2048] output")
    if output_scale.dtype != torch.float32 or tuple(output_scale.shape) != (input.shape[0], 16):
        raise ValueError("expected [M, 16] scale view")
    if not (input.is_cuda and output_q.is_cuda and output_scale.is_cuda):
        raise ValueError("all tensors must be CUDA tensors")
    if (input.shape[0] * 16) % _GROUPS_PER_CTA:
        raise ValueError("rows must form complete fused CTA groups")
    fp8 = torch.finfo(torch.float8_e4m3fn)
    _routed_silu_mul_quant_fp8_kernel[(input.shape[0] * 16 // _GROUPS_PER_CTA,)](
        input,
        output_q,
        output_scale,
        input.stride(0),
        output_q.stride(0),
        output_scale.stride(0),
        output_scale.stride(1),
        fp8_min=fp8.min,
        fp8_max=fp8.max,
        eps=eps,
        limit=limit,
        groups_per_cta=_GROUPS_PER_CTA,
        num_warps=_NUM_WARPS,
    )


def alloc_routed_silu_mul_quant_fp8(
    input: torch.Tensor,
    alloc_func: Callable,
    *,
    limit: float = 10.0,
    eps: float = 1e-10,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Allocate TMA-aligned routed quant buffers through the caller allocator."""
    if not HAS_SGL_KERNEL:
        raise RuntimeError("routed fused quant requires installed SGL non-UE8M0 quant semantics")
    rows = input.shape[0]
    output_q = alloc_func((rows, 2048), dtype=torch.float8_e4m3fn, device=input.device)
    aligned_rows = (rows + 3) // 4 * 4
    scale_storage = alloc_func((16, aligned_rows), dtype=torch.float32, device=input.device)
    output_scale = scale_storage.permute(1, 0)[:rows, :]
    routed_silu_mul_quant_fp8(input, output_q, output_scale, limit=limit, eps=eps)
    return output_q, output_scale
