import torch
import triton
import triton.language as tl

from lightllm.common.basemodel.triton_kernel.norm.rmsnorm import rmsnorm_num_warps
from lightllm.common.kernel_config import KernelConfigs
from lightllm.utils.sgl_utils import HAS_SGL_KERNEL, sgl_ops
from frozendict import frozendict
from functools import lru_cache
from typing import Any, Callable, Dict, List, Optional, Tuple

try:
    from deep_gemm import ceil_div
except:
    pass


@triton.jit
def _ceil_to_ue8m0(x):
    bits = x.to(tl.float32).to(tl.int32, bitcast=True)
    exp = ((bits >> 23) & 0xFF) + ((bits & 0x7FFFFF) != 0)
    exp = tl.maximum(tl.minimum(exp, 254), 1)
    return (exp << 23).to(tl.float32, bitcast=True)


@triton.jit
def _pack_ue8m0_scales(scales, valid_groups, GROUPS_PER_CTA: tl.constexpr):
    groups = tl.arange(0, GROUPS_PER_CTA)
    exponents = (scales.to(tl.int32, bitcast=True) >> 23) & 255
    bytes_ = tl.where(valid_groups, exponents << ((groups % 4) * 8), 0)
    return tl.sum(tl.reshape(bytes_, (GROUPS_PER_CTA // 4, 4)), axis=1)


# Adapted from https://github.com/sgl-project/sglang/blob/main/python/sglang/srt/layers/quantization/fp8_kernel.py
@triton.jit
def _per_token_group_quant_fp8(
    y_ptr,
    y_q_ptr,
    y_s_ptr,
    y_stride,
    M,
    N,
    eps,
    fp8_min,
    fp8_max,
    xs_n,
    xs_stride_m,
    xs_stride_n,
    topk_ids_ptr,
    topk_weights_ptr,
    topk_ids_out_ptr,
    topk_weights_out_ptr,
    num_topk,
    topk_row_stride,
    topk_out_row_stride,
    BLOCK: tl.constexpr,
    TOPK_BLOCK: tl.constexpr,
    NEED_MASK: tl.constexpr,
    USE_UE8M0_SCALE: tl.constexpr,
    COPY_TOPK: tl.constexpr,
    GROUPS_PER_CTA: tl.constexpr,
    USE_PACKED_UE8M0: tl.constexpr,
):
    if USE_PACKED_UE8M0:
        row_id = tl.program_id(0)
        groups = tl.arange(0, GROUPS_PER_CTA)
        group_ids = tl.program_id(1) * GROUPS_PER_CTA + groups
        cols = tl.arange(0, BLOCK)
        offsets = (row_id * xs_n + group_ids[:, None]) * y_stride + cols[None, :]
        mask = (group_ids[:, None] < xs_n) & (cols[None, :] < N)
        y = tl.load(y_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
        y_s = tl.maximum(tl.max(tl.abs(y), axis=1), eps) / fp8_max
        y_s = _ceil_to_ue8m0(y_s)
        y_q = tl.clamp(y / y_s[:, None], fp8_min, fp8_max).to(y_q_ptr.dtype.element_ty)
        tl.store(y_q_ptr + offsets, y_q, mask=mask)
        packs_per_cta: tl.constexpr = GROUPS_PER_CTA // 4
        packed = _pack_ue8m0_scales(y_s, group_ids < xs_n, GROUPS_PER_CTA)
        packed_col = tl.program_id(1) * packs_per_cta + tl.arange(0, packs_per_cta)
        scale_offset = row_id * xs_stride_m + packed_col * xs_stride_n
        tl.store(y_s_ptr + scale_offset, packed, mask=packed_col < tl.cdiv(xs_n, 4))
        if COPY_TOPK:
            if tl.program_id(1) == 0:
                topk_cols = tl.arange(0, TOPK_BLOCK)
                topk_mask = topk_cols < num_topk
                topk_offsets = row_id * topk_row_stride + topk_cols
                topk_out_offsets = row_id * topk_out_row_stride + topk_cols
                topk_ids = tl.load(topk_ids_ptr + topk_offsets, mask=topk_mask)
                topk_weights = tl.load(topk_weights_ptr + topk_offsets, mask=topk_mask)
                tl.store(topk_ids_out_ptr + topk_out_offsets, topk_ids, mask=topk_mask)
                tl.store(topk_weights_out_ptr + topk_out_offsets, topk_weights, mask=topk_mask)
        return

    g_id = tl.program_id(0) * GROUPS_PER_CTA + tl.arange(0, GROUPS_PER_CTA)
    y_ptr += g_id[:, None] * y_stride
    y_q_ptr += g_id[:, None] * y_stride
    row_id = g_id // xs_n
    col_id = g_id % xs_n
    y_s_ptr += row_id * xs_stride_m + col_id * xs_stride_n

    cols = tl.arange(0, BLOCK)[None, :]  # N <= BLOCK

    if NEED_MASK or GROUPS_PER_CTA > 1:
        mask = (g_id[:, None] < M) & (cols < N)
        other = 0.0
    else:
        mask = None
        other = None

    y = tl.load(y_ptr + cols, mask=mask, other=other).to(tl.float32)
    # Quant
    _absmax = tl.max(tl.abs(y), axis=1)
    if USE_UE8M0_SCALE:
        y_s = _ceil_to_ue8m0(tl.maximum(_absmax, eps) / fp8_max)
    else:
        y_s = tl.maximum(_absmax, eps) / fp8_max
    y_q = tl.clamp(y / y_s[:, None], fp8_min, fp8_max).to(y_q_ptr.dtype.element_ty)

    tl.store(y_q_ptr + cols, y_q, mask=mask)
    tl.store(y_s_ptr, y_s, mask=g_id < M)

    if COPY_TOPK:
        topk_cols = tl.arange(0, TOPK_BLOCK)[None, :]
        topk_mask = (g_id[:, None] < M) & (col_id[:, None] == 0) & (topk_cols < num_topk)
        topk_offsets = row_id[:, None] * topk_row_stride + topk_cols
        topk_out_offsets = row_id[:, None] * topk_out_row_stride + topk_cols
        topk_ids = tl.load(topk_ids_ptr + topk_offsets, mask=topk_mask)
        topk_weights = tl.load(topk_weights_ptr + topk_offsets, mask=topk_mask)
        tl.store(topk_ids_out_ptr + topk_out_offsets, topk_ids, mask=topk_mask)
        tl.store(topk_weights_out_ptr + topk_out_offsets, topk_weights, mask=topk_mask)


def lightllm_per_token_group_quant_fp8(
    x: torch.Tensor,
    group_size: int,
    x_q: torch.Tensor,
    x_s: torch.Tensor,
    eps: float = 1e-10,
    dtype: torch.dtype = torch.float8_e4m3fn,
    use_ue8m0_scales: bool = False,
    topk_ids: Optional[torch.Tensor] = None,
    topk_weights: Optional[torch.Tensor] = None,
    topk_ids_out: Optional[torch.Tensor] = None,
    topk_weights_out: Optional[torch.Tensor] = None,
    use_packed_ue8m0: bool = False,
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

    xs_n = x.shape[-1] // group_size
    x_s = x_s.view(-1, x_s.shape[-1])
    xs_stride_m, xs_stride_n = x_s.stride()
    finfo = torch.finfo(dtype)
    fp8_max = finfo.max

    fp8_min = -fp8_max

    M = x.numel() // group_size
    N = group_size
    BLOCK = triton.next_power_of_2(N)
    # heuristics for number of warps
    num_warps = min(max(BLOCK // 256, 1), 8)
    num_stages = 1
    copy_topk = topk_ids is not None
    if copy_topk:
        num_topk = topk_ids.shape[-1]
        topk_block = triton.next_power_of_2(num_topk)
        topk_row_stride = topk_ids.stride(0)
        topk_out_row_stride = topk_ids_out.stride(0)
    else:
        topk_ids = topk_weights = topk_ids_out = topk_weights_out = x
        topk_block = 1
        num_topk = topk_row_stride = topk_out_row_stride = 0
    # Large Mega inputs otherwise launch one CTA per 128-element group.
    if use_packed_ue8m0:
        groups_per_cta = 32 if M >= 4096 else 4
    else:
        groups_per_cta = 16 if copy_topk and M >= 4096 else 1
    if groups_per_cta >= 16:
        num_warps = 4
    if use_packed_ue8m0:
        grid = (x.shape[-2], triton.cdiv(xs_n, groups_per_cta))
    else:
        grid = (triton.cdiv(M, groups_per_cta),)
    _per_token_group_quant_fp8[grid](
        x,
        x_q,
        x_s,
        group_size,
        M,
        N,
        eps,
        fp8_min=fp8_min,
        fp8_max=fp8_max,
        xs_n=xs_n,
        xs_stride_m=xs_stride_m,
        xs_stride_n=xs_stride_n,
        topk_ids_ptr=topk_ids,
        topk_weights_ptr=topk_weights,
        topk_ids_out_ptr=topk_ids_out,
        topk_weights_out_ptr=topk_weights_out,
        num_topk=num_topk,
        topk_row_stride=topk_row_stride,
        topk_out_row_stride=topk_out_row_stride,
        BLOCK=BLOCK,
        TOPK_BLOCK=topk_block,
        NEED_MASK=BLOCK != group_size,
        USE_UE8M0_SCALE=use_ue8m0_scales,
        COPY_TOPK=copy_topk,
        GROUPS_PER_CTA=groups_per_cta,
        USE_PACKED_UE8M0=use_packed_ue8m0,
        num_warps=num_warps,
        num_stages=num_stages,
    )
    return


@triton.jit
def _rmsnorm_quant_mxfp8(
    X, W, Y, Q, S, x_stride_m, x_stride_n, scale_stride, width, eps, quant_eps, BLOCK_SIZE: tl.constexpr
):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK_SIZE)
    x = tl.load(X + row * x_stride_m + cols * x_stride_n, cols < width, 0).to(tl.float32)
    w = tl.load(W + cols, cols < width, 0).to(tl.float32)
    var = tl.sum(x * x, axis=0) / width
    rstd = 1 / tl.sqrt(var + eps)
    # Quantize the same rounded values that the indexer reads.
    normalized = ((x * rstd) * w).to(Y.dtype.element_ty)
    tl.store(Y + row * width + cols, normalized, cols < width)

    groups: tl.constexpr = BLOCK_SIZE // 32
    grouped = tl.reshape(normalized.to(tl.float32), (groups, 32))
    scales = _ceil_to_ue8m0(tl.maximum(tl.max(tl.abs(grouped), axis=1), quant_eps) / 448.0)
    quant = tl.clamp(grouped / scales[:, None], -448.0, 448.0)
    tl.store(Q + row * width + cols, tl.reshape(quant, (BLOCK_SIZE,)), cols < width)
    packed = _pack_ue8m0_scales(scales, tl.arange(0, groups) < width // 32, groups)
    pack_ids = tl.arange(0, groups // 4)
    tl.store(S + row + pack_ids * scale_stride, packed, pack_ids < tl.cdiv(width, 128))


def _alloc_packed_ue8m0_scales(x, group_size, alloc_func):
    rows, width = x.shape[-2:]
    aligned_rows = triton.cdiv(rows, 4) * 4
    return alloc_func((triton.cdiv(width // group_size, 4), aligned_rows), device=x.device, dtype=torch.int32).t()[
        :rows
    ]


def rmsnorm_quant_mxfp8(x, weight, eps, quant_eps, alloc_func):
    rows, width = x.shape
    normalized = torch.empty(x.shape, dtype=x.dtype, device=x.device)
    quant = alloc_func(x.shape, dtype=torch.float8_e4m3fn, device=x.device)
    scale = _alloc_packed_ue8m0_scales(x, 32, alloc_func)
    _rmsnorm_quant_mxfp8[(rows,)](
        x,
        weight,
        normalized,
        quant,
        scale,
        x.stride(0),
        x.stride(1),
        scale.stride(1),
        width,
        eps,
        quant_eps,
        BLOCK_SIZE=max(128, triton.next_power_of_2(width)),
        num_warps=rmsnorm_num_warps,
    )
    return normalized, quant, scale


def per_token_group_quant_fp8(
    x: torch.Tensor,
    group_size: int,
    eps: float = 1e-10,
    dtype: torch.dtype = torch.float8_e4m3fn,
    column_major_scales: bool = False,
    scale_tma_aligned: bool = False,
    alloc_func: Callable = torch.empty,
    use_ue8m0_scales: bool = False,
    use_packed_ue8m0: bool = False,
):
    x_q = alloc_func(x.shape, dtype=dtype, device=x.device)
    if use_packed_ue8m0:
        x_s = _alloc_packed_ue8m0_scales(x, group_size, alloc_func)
    elif column_major_scales:
        if scale_tma_aligned:
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

    # Adapted from
    # https://github.com/sgl-project/sglang/blob/7e257cd666c0d639626487987ea8e590da1e9395/python/sglang/srt/layers/quantization/fp8_kernel.py#L290
    if HAS_SGL_KERNEL and (not use_ue8m0_scales or (use_packed_ue8m0 and eps == 1e-10)):
        finfo = torch.finfo(dtype)
        fp8_max, fp8_min = finfo.max, finfo.min
        # 使用SGL kernel进行量化
        sgl_ops.sgl_per_token_group_quant_fp8(
            x, x_q, x_s, group_size, eps, fp8_min, fp8_max, scale_ue8m0=use_packed_ue8m0, enable_v2=True
        )
    else:
        # 使用LightLLM kernel进行量化
        lightllm_per_token_group_quant_fp8(
            x,
            group_size,
            x_q,
            x_s,
            eps=eps,
            dtype=dtype,
            use_ue8m0_scales=use_ue8m0_scales,
            use_packed_ue8m0=use_packed_ue8m0,
        )
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
