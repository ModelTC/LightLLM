import torch

import triton
import triton.language as tl
from .moe_sum_recude_config import MoeSumReduceKernelConfig
from typing import Any, Callable, Dict, Optional, Tuple
from lightllm.common.triton_utils.autotuner import autotune


@triton.jit
def _moe_sum_reduce_kernel(
    input_ptr,
    input_stride_0,
    input_stride_1,
    input_stride_2,
    output_ptr,
    output_stride_0,
    output_stride_1,
    token_num: int,
    topk_num: int,
    hidden_dim: int,
    BLOCK_M: tl.constexpr,
    BLOCK_DIM: tl.constexpr,
    NUM_STAGE: tl.constexpr,
):
    input_stride_0 = tl.cast(input_stride_0, dtype=tl.int64)
    input_stride_1 = tl.cast(input_stride_1, dtype=tl.int64)
    output_stride_0 = tl.cast(output_stride_0, dtype=tl.int64)

    token_block_id = tl.program_id(0)
    dim_block_id = tl.program_id(1)

    token_start = token_block_id * BLOCK_M
    token_end = min((token_block_id + 1) * BLOCK_M, token_num)

    dim_start = dim_block_id * BLOCK_DIM
    dim_end = min((dim_block_id + 1) * BLOCK_DIM, hidden_dim)

    offs_dim = dim_start + tl.arange(0, BLOCK_DIM)

    for token_index in range(token_start, token_end):
        accumulator = tl.zeros((BLOCK_DIM,), dtype=tl.float32)
        input_t_ptr = input_ptr + token_index * input_stride_0 + offs_dim
        for i in tl.range(0, topk_num, num_stages=NUM_STAGE):
            tmp = tl.load(input_t_ptr + i * input_stride_1, mask=offs_dim < dim_end, other=0.0)
            accumulator += tmp
        store_t_ptr = output_ptr + token_index * output_stride_0 + offs_dim
        tl.store(store_t_ptr, accumulator.to(input_ptr.dtype.element_ty), mask=offs_dim < dim_end)


def _get_moe_sum_reduce_static_key(input: torch.Tensor, output: torch.Tensor):
    return {"topk_num": input.shape[1], "hidden_dim": input.shape[2], "out_dtype": str(output.dtype)}


def _get_moe_sum_reduce_configs():
    return [
        {"BLOCK_M": bm, "BLOCK_DIM": bd, "NUM_STAGE": ns, "num_warps": nw}
        for ns in [1, 2, 4]
        for nw in [1, 2, 4, 8, 16]
        for bm in [1, 2, 4, 8, 16, 32]
        for bd in [64, 128, 256, 512, 1024]
    ]


@autotune(
    kernel_name="moe_sum_reduce:v1",
    configs_gen_func=_get_moe_sum_reduce_configs,
    static_key_func=_get_moe_sum_reduce_static_key,
    run_key_func=lambda input: input.shape[0],
    mutates_args=["output"],
)
def moe_sum_reduce(input: torch.Tensor, output: torch.Tensor, run_config: Dict = None):
    assert input.is_contiguous()
    assert output.is_contiguous()

    token_num, topk_num, hidden_dim = input.shape
    assert output.shape[0] == token_num and output.shape[1] == hidden_dim

    if not run_config:
        run_config = MoeSumReduceKernelConfig.try_to_get_best_config(
            M=token_num, topk_num=topk_num, hidden_dim=hidden_dim, out_dtype=str(output.dtype)
        )

    BLOCK_M = run_config["BLOCK_M"]
    BLOCK_DIM = run_config["BLOCK_DIM"]
    NUM_STAGE = run_config["NUM_STAGE"]
    num_warps = run_config["num_warps"]

    grid = (
        triton.cdiv(token_num, BLOCK_M),
        triton.cdiv(hidden_dim, BLOCK_DIM),
    )

    _moe_sum_reduce_kernel[grid](
        input,
        *input.stride(),
        output,
        *output.stride(),
        token_num=token_num,
        topk_num=topk_num,
        hidden_dim=hidden_dim,
        BLOCK_M=BLOCK_M,
        BLOCK_DIM=BLOCK_DIM,
        NUM_STAGE=NUM_STAGE,
        num_warps=num_warps,
    )
    return
