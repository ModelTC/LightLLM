"""Fused shared-expert SwiGLU activation and FP8 block quantization."""

import torch
import triton
import triton.language as tl

from lightllm.common.quantization.deepgemm import DeepGEMMFP8w8a8B128QuantizationMethod


HIDDEN = 4096
INTERMEDIATE = 2048
GROUP_SIZE = 128
NUM_GROUPS = INTERMEDIATE // GROUP_SIZE
SWIGLU_LIMIT = 10.0


@triton.jit
def _ceil_ue8m0(value):
    bits = value.to(tl.float32).to(tl.int32, bitcast=True)
    exponent = ((bits >> 23) & 255) + ((bits & 0x7FFFFF) != 0)
    exponent = tl.maximum(tl.minimum(exponent, 254), 1)
    return (exponent << 23).to(tl.float32, bitcast=True)


@triton.jit
def _silu_mul_quant_fp8_kernel(
    gate_up,
    quantized,
    scales,
    scale_ld,
    fp8_max: tl.constexpr,
    limit: tl.constexpr,
    hidden: tl.constexpr,
    intermediate: tl.constexpr,
    group_size: tl.constexpr,
    num_groups: tl.constexpr,
):
    program = tl.program_id(0)
    row = program // num_groups
    group = program % num_groups
    columns = group * group_size + tl.arange(0, group_size)
    gate = tl.load(gate_up + row * hidden + columns).to(tl.float32)
    up = tl.load(gate_up + row * hidden + intermediate + columns).to(tl.float32)
    gate = tl.minimum(gate, limit)
    up = tl.minimum(tl.maximum(up, -limit), limit)
    # Match the existing path's BF16 rounding before multiplication and quantization.
    gate = (gate / (1.0 + tl.exp(-gate))).to(gate_up.dtype.element_ty)
    product = (up * gate).to(gate_up.dtype.element_ty).to(tl.float32)
    scale = _ceil_ue8m0(tl.maximum(tl.max(tl.abs(product), axis=0), 1e-4) / fp8_max)
    tl.store(
        quantized + row * intermediate + columns,
        (product / scale).to(quantized.dtype.element_ty),
    )
    tl.store(scales + row + group * scale_ld, scale)


def can_fuse_shared_silu_fp8(gate_up: torch.Tensor, down_proj, swiglu_limit: float) -> bool:
    mm_param = getattr(down_proj, "mm_param", None)
    weight = getattr(mm_param, "weight", None)
    weight_scale = getattr(mm_param, "weight_scale", None)
    quant_method = getattr(down_proj, "quant_method", None)
    return (
        gate_up.ndim == 2
        and gate_up.is_cuda
        and gate_up.dtype is torch.bfloat16
        and gate_up.is_contiguous()
        and gate_up.shape[0] > 0
        and gate_up.shape[1] == HIDDEN
        and float(swiglu_limit) == SWIGLU_LIMIT
        and getattr(down_proj, "bias", None) is None
        and type(quant_method) is DeepGEMMFP8w8a8B128QuantizationMethod
        and weight is not None
        and weight.dtype is torch.float8_e4m3fn
        and tuple(weight.shape) == (HIDDEN, INTERMEDIATE)
        and weight_scale is not None and weight_scale.dtype is torch.float32
    )


def fused_silu_mul_quant_fp8(gate_up: torch.Tensor, alloc_func):
    if gate_up.ndim != 2 or gate_up.shape[0] == 0 or gate_up.shape[1] != HIDDEN:
        raise ValueError(f"expected contiguous [M, {HIDDEN}] gate/up tensor")
    if gate_up.dtype is not torch.bfloat16 or not gate_up.is_contiguous():
        raise ValueError("expected contiguous BF16 gate/up tensor")
    rows = gate_up.shape[0]
    quantized = alloc_func(
        (rows, INTERMEDIATE), dtype=torch.float8_e4m3fn, device=gate_up.device
    )
    scale_ld = triton.cdiv(rows, 4) * 4
    scale_storage = alloc_func((NUM_GROUPS, scale_ld), dtype=torch.float32, device=gate_up.device)
    scales = scale_storage.t()[:rows, :]
    _silu_mul_quant_fp8_kernel[(rows * NUM_GROUPS,)](
        gate_up,
        quantized,
        scales,
        scale_ld,
        fp8_max=torch.finfo(torch.float8_e4m3fn).max,
        limit=SWIGLU_LIMIT,
        hidden=HIDDEN,
        intermediate=INTERMEDIATE,
        group_size=GROUP_SIZE,
        num_groups=NUM_GROUPS,
        num_warps=1,
    )
    return quantized, scales
