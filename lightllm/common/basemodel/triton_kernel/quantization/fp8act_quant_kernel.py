import torch
import triton
import triton.language as tl

from lightllm.common.kernel_config import KernelConfigs
from lightllm.utils.sgl_utils import HAS_SGL_KERNEL, sgl_ops
from frozendict import frozendict
from functools import lru_cache
from typing import Any, Callable, Dict, List, Optional, Tuple

try:
    from deep_gemm import ceil_div
except:
    pass


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


def lightllm_per_token_group_quant_fp8(
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


def per_token_group_quant_fp8(
    x: torch.Tensor,
    group_size: int,
    eps: float = 1e-10,
    dtype: torch.dtype = torch.float8_e4m3fn,
    column_major_scales: bool = False,
    scale_tma_aligned: bool = False,
    alloc_func: Callable = torch.empty,
):
    x_q = alloc_func(x.shape, dtype=dtype, device=x.device)
    x_s = None
    # Adapted from
    # https://github.com/sgl-project/sglang/blob/7e257cd666c0d639626487987ea8e590da1e9395/python/sglang/srt/layers/quantization/fp8_kernel.py#L290
    if HAS_SGL_KERNEL:
        finfo = torch.finfo(dtype)
        fp8_max, fp8_min = finfo.max, finfo.min

        # 创建scale张量
        if column_major_scales:
            if scale_tma_aligned:
                # 对齐到4 * sizeof(float)
                aligned_size = (x.shape[-2] + 3) // 4 * 4
                x_s = alloc_func(
                    x.shape[:-2] + (x.shape[-1] // group_size, aligned_size),
                    device=x.device,
                    dtype=torch.float32,
                ).permute(-1, -2)[: x.shape[-2], :]
            else:
                x_s = alloc_func(
                    (x.shape[-1] // group_size,) + x.shape[:-1],
                    device=x.device,
                    dtype=torch.float32,
                ).permute(-1, -2)
        else:
            x_s = alloc_func(
                x.shape[:-1] + (x.shape[-1] // group_size,),
                device=x.device,
                dtype=torch.float32,
            )

        # 使用SGL kernel进行量化
        sgl_ops.sgl_per_token_group_quant_fp8(x, x_q, x_s, group_size, 1e-10, fp8_min, fp8_max, False)
    else:
        # 使用LightLLM kernel进行量化
        x_s = alloc_func(
            x.shape[:-1] + (x.shape[-1] // group_size,),
            device=x.device,
            dtype=torch.float32,
        )
        lightllm_per_token_group_quant_fp8(x, group_size, x_q, x_s, eps=1e-10, dtype=torch.float8_e4m3fn)
        if column_major_scales and scale_tma_aligned:
            x_s = tma_align_input_scale(x_s)
    return x_q, x_s


# copy from
# https://github.com/deepseek-ai/DeepGEMM/blob/bd2a77552886b98c205af12f8d7d2d61247c4b27/deep_gemm/jit_kernels/utils.py#L58
def get_tma_aligned_size(x: int, element_size: int) -> int:
    """
    Global memory address of TMA must be 16-byte aligned.
    Since we use column-major layout for the LHS scaling tensor,
        the M-axis of the LHS scaling tensor needs to be padded to a multiple of 16 bytes.

    Arguments:
        x: original M-axis shape of the LHS scaling tensor.
        element_size: element size of the LHS scaling tensor.

    Returns:
        M-axis shape of the LHS scaling tensor after padding.
    """
    tma_alignment_bytes = 16
    assert tma_alignment_bytes % element_size == 0
    alignment = tma_alignment_bytes // element_size
    return ceil_div(x, alignment) * alignment


@triton.jit
def _tma_align_input_scale_kernel(
    input_scale_ptr,
    output_ptr,
    m,
    k_div_block_size,
    input_scale_stride_m,
    input_scale_stride_k,
    output_stride_m,
    output_stride_k,
    BLOCK_SIZE_K: tl.constexpr,
):
    pid_m = tl.program_id(axis=0)
    grid_m = tl.num_programs(0)
    k_offsets = tl.arange(0, BLOCK_SIZE_K)

    for m_base in range(pid_m, m, grid_m):
        input_offset = input_scale_ptr + m_base * input_scale_stride_m + k_offsets * input_scale_stride_k
        input_data = tl.load(input_offset, mask=k_offsets < k_div_block_size)

        output_offset = output_ptr + k_offsets * output_stride_k + m_base * output_stride_m
        tl.store(output_offset, input_data, mask=k_offsets < k_div_block_size)


def tma_align_input_scale(input_scale: torch.Tensor):
    assert input_scale.dim() == 2
    m, k_div_block_size = input_scale.shape
    padd_m = get_tma_aligned_size(m, input_scale.element_size())
    output = torch.empty((k_div_block_size, padd_m), dtype=input_scale.dtype, device=input_scale.device)

    grid_m = min(m, 8192)
    BLOCK_SIZE_K = triton.next_power_of_2(k_div_block_size)

    _tma_align_input_scale_kernel[(grid_m,)](
        input_scale_ptr=input_scale,
        output_ptr=output,
        m=m,
        k_div_block_size=k_div_block_size,
        input_scale_stride_m=input_scale.stride(0),
        input_scale_stride_k=input_scale.stride(1),
        output_stride_m=output.stride(1),  # Note: these are swapped
        output_stride_k=output.stride(0),  # for column-major
        BLOCK_SIZE_K=BLOCK_SIZE_K,
    )
    return output.t()[:m]


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


def test_tma_align():
    m = 576
    k = 8192
    x = torch.randn((m, k // 128), dtype=torch.float32).cuda()

    for _ in range(10):
        x_padded = tma_align_input_scale(x)
    import time

    torch.cuda.synchronize()
    start = time.time()
    for _ in range(100):
        x_padded = tma_align_input_scale(x)
    torch.cuda.synchronize()
    print("Time:", time.time() - start)
    x_padded = tma_align_input_scale(x)
    print(torch.abs(x_padded - x).max())


def test_per_token_group_quant_fp8():
    group_size = 128
    x = torch.randn((1024, 8192), dtype=torch.bfloat16).cuda()
    # x_s = torch.randn((1024, 8192 // group_size), dtype=torch.float32).cuda()
    # x_s = torch.randn((8192 // group_size, 1024 + 10), dtype=torch.float32).cuda().t()
    x_q, x_s = per_token_group_quant_fp8(x, group_size, column_major_scales=True, scale_tma_aligned=True)
    x_s = x_s[:1024]
    th_x_q, th_x_s = torch_quant(x, group_size)
    print("th_x_s - x_s", torch.abs(th_x_s - x_s.reshape(-1)).max())
    print("th_x_q - x_q", torch.abs(th_x_q.to(torch.float32) - x_q.to(torch.float32)).max())


if __name__ == "__main__":
    test_per_token_group_quant_fp8()
    test_tma_align()
