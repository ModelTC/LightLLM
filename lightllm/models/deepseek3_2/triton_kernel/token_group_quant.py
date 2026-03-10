# Adapted from https://github.com/sgl-project/sglang/blob/main/python/sglang/srt/layers/quantization/fp8_kernel.py

import triton
import triton.language as tl
import torch
from typing import Tuple

fp8_min = -448.0
fp8_max = 448.0
fp8_dtype = torch.float8_e4m3fn


@triton.jit
def _per_token_group_quant_mla_deep_gemm_masked_fp8(
    y_ptr,
    y_q_ptr,
    y_s_ptr,
    masked_m_ptr,
    group_size,
    y_stride_b,
    y_stride_t,
    y_q_stride_b,
    y_q_stride_t,
    y_s_stride_b,
    y_s_stride_g,
    eps,
    fp8_min,
    fp8_max,
    NUM_GROUP: tl.constexpr,
    BLOCK: tl.constexpr,
):
    """A Triton-accelerated function to perform per-token-group
    quantization on a tensor for deep_gemm grouped_gemm_masked.
    This function converts the tensor values into float8 values.
    y and y_q: (b, t, k)
    y_s: (b, k//group_size, t)
    """
    t_id = tl.program_id(0)
    b_id = tl.program_id(1)

    y_ptr += b_id * y_stride_b + t_id * y_stride_t
    y_q_ptr += b_id * y_q_stride_b + t_id * y_q_stride_t
    y_s_ptr += b_id * y_s_stride_b + t_id

    if t_id == 0:
        tl.store(masked_m_ptr + b_id, tl.num_programs(0))

    cols = tl.arange(0, BLOCK)  # group_size <= BLOCK
    mask = cols < group_size

    for gid in range(NUM_GROUP):
        y = tl.load(y_ptr + gid * group_size + cols, mask=mask, other=0.0).to(tl.float32)
        _absmax = tl.maximum(tl.max(tl.abs(y)), eps)
        y_s = _absmax / fp8_max
        y_q = tl.clamp(y / y_s, fp8_min, fp8_max).to(y_q_ptr.dtype.element_ty)

        tl.store(y_q_ptr + gid * group_size + cols, y_q, mask=mask)
        tl.store(y_s_ptr + gid * y_s_stride_g, y_s)


def per_token_group_quant_mla_deep_gemm_masked_fp8(
    x: torch.Tensor,
    group_size: int = 128,
    eps: float = 1e-12,
    dtype: torch.dtype = fp8_dtype,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    This function quantizes input values to float8 values with per-token-group-quantization
    for deep_gemm grouped_gemm_masked and specialized for mla absorbed case.
    """
    assert x.dim() == 3, "`x` is not a 3d-tensor"

    b, m, k = x.shape
    aligned_m = (m + 255) // 256 * 256  # 256 is the max block_m of the gemm kernel
    num_tiles_k = k // group_size
    assert num_tiles_k * group_size == k, f"k % {group_size} must be zero"

    x_q = x.new_empty((b, aligned_m, k), dtype=dtype)
    x_s = x.new_empty((b, num_tiles_k, aligned_m), dtype=torch.float32)
    masked_m = x.new_empty((b,), dtype=torch.int32)

    BLOCK_SIZE = triton.next_power_of_2(group_size)
    grid = (m, b)

    _per_token_group_quant_mla_deep_gemm_masked_fp8[grid](
        x,
        x_q,
        x_s,
        masked_m,
        group_size,
        x.stride(0),
        x.stride(1),
        x_q.stride(0),
        x_q.stride(1),
        x_s.stride(0),
        x_s.stride(1),
        eps,
        -fp8_max,
        fp8_max,
        num_tiles_k,
        BLOCK_SIZE,
    )

    return x_q, x_s.transpose(1, 2), masked_m, m, aligned_m
