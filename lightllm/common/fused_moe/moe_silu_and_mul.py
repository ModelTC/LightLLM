import torch

import triton
import triton.language as tl
from .moe_silu_and_mul_config import MoeSiluAndMulKernelConfig
from lightllm.common.triton_utils.autotuner import autotune


@triton.jit
def _silu_and_mul_kernel_fast(
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
    NUM_STAGES: tl.constexpr,
    NEED_MASK: tl.constexpr,
):
    stride_input_m = tl.cast(stride_input_m, dtype=tl.int64)
    stride_output_m = tl.cast(stride_output_m, dtype=tl.int64)

    n_block_index = tl.program_id(0)
    m_block_index = tl.program_id(1)
    n_offsets = n_block_index * BLOCK_N + tl.arange(0, BLOCK_N)
    m_start_index = m_block_index * BLOCK_M
    m_end_index = (m_block_index + 1) * BLOCK_M
    m_end_index = tl.where(m_end_index < size_m, m_end_index, size_m)
    if NEED_MASK:
        mask = n_offsets[None, :] < size_n
        other = 0.0
    else:
        mask = None
        other = None

    for m_index in tl.range(m_start_index, m_end_index, num_stages=NUM_STAGES):
        gate_offsets = m_index * stride_input_m + n_offsets[None, :]
        up_offsets = m_index * stride_input_m + (n_offsets[None, :] + size_n)
        out_offsets = m_index * stride_output_m + n_offsets[None, :]

        up = tl.load(
            input_ptr + up_offsets,
            mask=mask,
            other=other,
        )
        gate = tl.load(
            input_ptr + gate_offsets,
            mask=mask,
            other=other,
        ).to(tl.float32)

        gate = gate / (1 + tl.exp(-gate))
        gate = gate.to(input_ptr.dtype.element_ty)

        tl.store(
            output_ptr + out_offsets,
            up * gate,
            mask=mask,
        )


def _get_silu_and_mul_configs():
    return [
        {"BLOCK_M": bm, "BLOCK_N": bn, "num_warps": nw, "NUM_STAGES": ns}
        for ns in [1, 2, 4]
        for nw in [1, 4, 8]
        for bm in [1, 8, 32, 64, 128, 256]
        for bn in [32, 64, 128, 256]
    ]


def _get_silu_and_mul_static_key(input: torch.Tensor, output: torch.Tensor):
    return {"N": input.shape[-1] // 2, "out_dtype": str(output.dtype)}


@autotune(
    kernel_name="silu_and_mul_fwd:v1",
    configs_gen_func=_get_silu_and_mul_configs,
    static_key_func=_get_silu_and_mul_static_key,
    run_key_func=lambda input: input.shape[0],
    mutates_args=["output"],
)
def silu_and_mul_fwd(input: torch.Tensor, output: torch.Tensor, run_config=None):
    assert input.is_contiguous()
    assert output.is_contiguous()

    stride_input_m = input.stride(0)
    stride_input_n = input.stride(1)
    stride_output_m = output.stride(0)
    stride_output_n = output.stride(1)
    size_m = input.shape[0]
    size_n = input.shape[-1] // 2

    if not run_config:
        run_config = MoeSiluAndMulKernelConfig.try_to_get_best_config(M=size_m, N=size_n, out_dtype=str(output.dtype))

    BLOCK_M = run_config["BLOCK_M"]
    BLOCK_N = run_config["BLOCK_N"]
    num_warps = run_config["num_warps"]
    NUM_STAGES = run_config["NUM_STAGES"]
    # limit the grid size to avoid the invalid argument error of triton
    while triton.cdiv(size_m, BLOCK_M) > 8192:
        BLOCK_M *= 2

    grid = (
        triton.cdiv(size_n, BLOCK_N),
        triton.cdiv(size_m, BLOCK_M),
    )
    NEED_MASK = (size_n % BLOCK_N) != 0
    _silu_and_mul_kernel_fast[grid](
        input_ptr=input,
        output_ptr=output,
        stride_input_m=stride_input_m,
        stride_input_n=stride_input_n,
        stride_output_m=stride_output_m,
        stride_output_n=stride_output_n,
        size_m=size_m,
        size_n=size_n,
        BLOCK_M=BLOCK_M,
        BLOCK_N=BLOCK_N,
        NUM_STAGES=NUM_STAGES,
        NEED_MASK=NEED_MASK,
        num_warps=num_warps,
    )
    return
