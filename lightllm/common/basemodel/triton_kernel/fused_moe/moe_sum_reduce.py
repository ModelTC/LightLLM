import torch
import triton
import triton.language as tl
from typing import Dict
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
    shared_ptr,
    shared_stride_0,
    shared_stride_1,
    gate_ptr,
    gate_stride_0,
    gate_stride_1,
    token_num: int,
    topk_num: int,
    hidden_dim: int,
    BLOCK_M: tl.constexpr,
    BLOCK_DIM: tl.constexpr,
    NUM_STAGE: tl.constexpr,
    HAS_SHARED_GATE: tl.constexpr,
    GATE_DIM: tl.constexpr,
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
        if HAS_SHARED_GATE:
            shared = tl.load(
                shared_ptr + token_index * shared_stride_0 + offs_dim * shared_stride_1,
                mask=offs_dim < dim_end,
                other=0.0,
            ).to(tl.float32)
            if GATE_DIM == 1:
                gate = tl.load(gate_ptr + token_index * gate_stride_0).to(tl.float32) + tl.zeros(
                    (BLOCK_DIM,), dtype=tl.float32
                )
            else:
                gate = tl.load(
                    gate_ptr + token_index * gate_stride_0 + offs_dim * gate_stride_1,
                    mask=offs_dim < dim_end,
                    other=0.0,
                ).to(tl.float32)
            gate = 1.0 / (1.0 + tl.exp(-gate))
            accumulator += shared * gate
        store_t_ptr = output_ptr + token_index * output_stride_0 + offs_dim
        tl.store(store_t_ptr, accumulator.to(input_ptr.dtype.element_ty), mask=offs_dim < dim_end)


def _get_moe_sum_reduce_static_key(
    input: torch.Tensor, output: torch.Tensor, shared: torch.Tensor = None, gate: torch.Tensor = None
):
    return {
        "topk_num": input.shape[1],
        "hidden_dim": input.shape[2],
        "out_dtype": str(output.dtype),
        "has_shared_gate": shared is not None,
        "gate_dim": 0 if gate is None else gate.shape[-1],
    }


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
def moe_sum_reduce(input: torch.Tensor, output: torch.Tensor, shared=None, gate=None, run_config: Dict = None):
    assert input.is_contiguous()
    assert output.is_contiguous()

    token_num, topk_num, hidden_dim = input.shape
    assert output.shape[0] == token_num and output.shape[1] == hidden_dim
    has_shared_gate = shared is not None
    if has_shared_gate:
        assert gate is not None
        shared = shared.view(token_num, hidden_dim)
        gate = gate.view(token_num, gate.shape[-1])
        assert shared.is_contiguous()
        assert gate.is_contiguous()
        assert gate.shape[1] in (1, hidden_dim)

    if not run_config:
        run_config = {
            "BLOCK_M": 1,
            "BLOCK_DIM": 128,
            "NUM_STAGE": 1,
            "num_warps": 2,
        }

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
        shared if has_shared_gate else output,
        shared.stride(0) if has_shared_gate else 0,
        shared.stride(1) if has_shared_gate else 0,
        gate if has_shared_gate else output,
        gate.stride(0) if has_shared_gate else 0,
        gate.stride(1) if has_shared_gate else 0,
        token_num=token_num,
        topk_num=topk_num,
        hidden_dim=hidden_dim,
        BLOCK_M=BLOCK_M,
        BLOCK_DIM=BLOCK_DIM,
        NUM_STAGE=NUM_STAGE,
        HAS_SHARED_GATE=has_shared_gate,
        GATE_DIM=gate.shape[1] if has_shared_gate else 0,
        num_warps=num_warps,
    )
    return
