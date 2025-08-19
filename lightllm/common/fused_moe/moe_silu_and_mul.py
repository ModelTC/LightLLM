import torch

import triton
import triton.language as tl
from .moe_silu_and_mul_config import MoeSiluAndMulKernelConfig
from typing import Any, Callable, Dict, Optional, Tuple
from lightllm.common.triton.autotuner import autotune, nearest_power_of_2

@triton.jit
def _silu_and_mul_kernel(
    input_ptr,
    output_ptr,
    stride_input_m,
    stride_input_n,
    stride_output_m,
    stride_output_n,
    size_m,
    size_n,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    stride_input_m = tl.cast(stride_input_m, dtype=tl.int64)
    stride_output_m = tl.cast(stride_output_m, dtype=tl.int64)

    tid = tl.program_id(0)
    input_m_offsets = tid * BLOCK_M + tl.arange(0, BLOCK_M)
    output_m_offsets = tid * BLOCK_M + tl.arange(0, BLOCK_M)

    pid = tl.program_id(1)
    input_n_offsets = pid * BLOCK_N + tl.arange(0, BLOCK_N)
    output_n_offsets = pid * BLOCK_N + tl.arange(0, BLOCK_N)

    up_offsets = input_m_offsets[:, None] * stride_input_m + (input_n_offsets[None, :] + size_n)
    gate_offsets = input_m_offsets[:, None] * stride_input_m + input_n_offsets[None, :]
    res_offsets = output_m_offsets[:, None] * stride_output_m + output_n_offsets[None, :]

    up = tl.load(
        input_ptr + up_offsets,
        mask=(input_n_offsets < size_n)[None, :] * (input_m_offsets < size_m)[:, None],
        other=0.0,
    )
    gate = tl.load(
        input_ptr + gate_offsets,
        mask=(input_n_offsets < size_n)[None, :] * (input_m_offsets < size_m)[:, None],
        other=0.0,
    ).to(tl.float32)

    gate = gate / (1 + tl.exp(-gate))
    gate = gate.to(input_ptr.dtype.element_ty)

    tl.store(
        output_ptr + res_offsets,
        up * gate,
        mask=(output_n_offsets < size_n)[None, :] * (output_m_offsets < size_m)[:, None],
    )

@autotune(  
    configs=[
         {"BLOCK_M": bm, "BLOCK_N": bn, "num_warps": nw, "num_stages": ns} 
         for ns in [1, 2, 4] for nw in [1, 4, 8] for bm in [32, 64, 128, 256] for bn in [32, 64, 128, 256] 
    ],
    default_config={"BLOCK_M": 128, "BLOCK_N": 128, "num_warps": 4, "num_stages": 1},
    static_key_func=lambda input, output : f"N={input.shape[-1] // 2},out_dtype={output.dtype}",
    run_key_func=lambda input : str(nearest_power_of_2(input.shape[0])),
)
def silu_and_mul_fwd(input: torch.Tensor, output: torch.Tensor, run_config: Dict=None):
    assert input.is_contiguous()
    assert output.is_contiguous()

    stride_input_m = input.stride(0)
    stride_input_n = input.stride(1)
    stride_output_m = output.stride(0)
    stride_output_n = output.stride(1)
    size_m = input.shape[0]
    size_n = input.shape[-1] // 2

    BLOCK_M = run_config["BLOCK_M"]
    BLOCK_N = run_config["BLOCK_N"]
    num_warps = run_config["num_warps"]
    num_stages = run_config["num_stages"]

    grid = (
        triton.cdiv(size_m, BLOCK_M),
        triton.cdiv(size_n, BLOCK_N),
    )
    _silu_and_mul_kernel[grid](
        input,
        output,
        stride_input_m,
        stride_input_n,
        stride_output_m,
        stride_output_n,
        size_m,
        size_n,
        BLOCK_M=BLOCK_M,
        BLOCK_N=BLOCK_N,
        num_warps=num_warps,
        num_stages=num_stages,
    )
    return
