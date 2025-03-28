import torch
import triton
import triton.language as tl

from lightllm.common.kernel_config import KernelConfigs
from frozendict import frozendict
from functools import lru_cache
from typing import Any, Dict, List, Optional, Tuple


# Adapted from https://github.com/sgl-project/sglang/blob/main/python/sglang/srt/layers/quantization/fp8_kernel.py
@triton.jit
def _per_token_group_quant_fp8(
    y_ptr,
    y_q_ptr,
    y_s_ptr,
    y_stride,
    N,
    eps,
    fp8_min,
    fp8_max,
    xs_m,
    xs_n,
    xs_row_major: tl.constexpr,
    BLOCK: tl.constexpr,
):
    g_id = tl.program_id(0)
    y_ptr += g_id * y_stride
    y_q_ptr += g_id * y_stride
    if xs_row_major:
        y_s_ptr += g_id
    else:
        row_id = g_id // xs_n
        col_id = g_id % xs_n
        y_s_ptr += col_id * xs_m + row_id  # col major

    cols = tl.arange(0, BLOCK)  # N <= BLOCK
    mask = cols < N

    y = tl.load(y_ptr + cols, mask=mask, other=0.0).to(tl.float32)
    # Quant
    _absmax = tl.maximum(tl.max(tl.abs(y)), eps)
    y_s = _absmax / fp8_max
    y_q = tl.clamp(y / y_s, fp8_min, fp8_max).to(y_q_ptr.dtype.element_ty)

    tl.store(y_q_ptr + cols, y_q, mask=mask)
    tl.store(y_s_ptr, y_s)


def per_token_group_quant_fp8(
    x: torch.Tensor,
    group_size: int,
    x_q: torch.Tensor,
    x_s: torch.Tensor,
    eps: float = 1e-10,
    dtype: torch.dtype = torch.float8_e4m3fn,
):
    """group-wise, per-token quantization on input tensor `x`.
    Args:
        x: The input tenosr with ndim >= 2.
        group_size: The group size used for quantization.
        x_q: the tensor to save the quantized result of x.
        x_s: the tensor to save the scale of x.
        eps: The minimum to avoid dividing zero.
        dtype: The dype of output tensor. Note that only `torch.float8_e4m3fn` is supported for now.
    """
    assert x.shape[-1] % group_size == 0, "the last dimension of `x` cannot be divisible by `group_size`"
    assert x.is_contiguous(), "`x` is not contiguous"

    xs_row_major = x_s.is_contiguous()
    xs_m, xs_n = x_s.shape
    finfo = torch.finfo(dtype)
    fp8_max = finfo.max

    fp8_min = -fp8_max

    M = x.numel() // group_size
    N = group_size
    BLOCK = triton.next_power_of_2(N)
    # heuristics for number of warps
    num_warps = min(max(BLOCK // 256, 1), 8)
    num_stages = 1
    _per_token_group_quant_fp8[(M,)](
        x,
        x_q,
        x_s,
        group_size,
        N,
        eps,
        fp8_min=fp8_min,
        fp8_max=fp8_max,
        xs_m=xs_m,
        xs_n=xs_n,
        xs_row_major=xs_row_major,
        BLOCK=BLOCK,
        num_warps=num_warps,
        num_stages=num_stages,
    )

    return


def torch_quant(x, group_size, dtype=torch.float8_e4m3fn):
    M, N = x.shape
    x_q = torch.randn((M, N)).cuda().to(torch.float8_e4m3fn)
    x_s = torch.randn((M, N // group_size), dtype=torch.float32).cuda()
    x = x.reshape(-1, group_size)
    finfo = torch.finfo(dtype)
    fp8_max = finfo.max
    fp8_min = -fp8_max

    x_s = x.to(torch.float32).abs().max(-1)[0] / fp8_max
    x_q = x.to(torch.float32) / x_s.reshape(-1, 1)
    x_q = x_q.clamp(fp8_min, fp8_max).to(dtype)
    return x_q.reshape(M, N), x_s


if __name__ == "__main__":
    group_size = 128
    x = torch.randn((1024, 8192), dtype=torch.bfloat16).cuda()

    x_q = torch.randn((1024, 8192)).cuda().to(torch.float8_e4m3fn)
    # x_s = torch.randn((1024, 8192 // group_size), dtype=torch.float32).cuda()
    x_s = torch.randn((8192 // group_size, 1024 + 10), dtype=torch.float32).cuda().t()
    per_token_group_quant_fp8(x, group_size, x_q, x_s)
    x_s = x_s[:1024]
    th_x_q, th_x_s = torch_quant(x, group_size)
    print("th_x_s - x_s", torch.abs(th_x_s - x_s.reshape(-1)).max())
    print("th_x_q - x_q", torch.abs(th_x_q.to(torch.float32) - x_q.to(torch.float32)).max())
