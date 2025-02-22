"""Fused MoE kernel."""
# Adapted from
# https://github.com/vllm-project/vllm/blob/v0.6.4.post1/vllm/model_executor/layers/fused_moe/fused_moe.py
# of the vllm-project/vllm GitHub repository.
#
# Copyright 2023 ModelTC Team
# Copyright 2023 vLLM Team
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import torch
import triton
import triton.language as tl
from typing import Any, Callable, Dict, Optional, Tuple
from lightllm.utils.log_utils import init_logger
from lightllm.common.vllm_kernel import _custom_ops as ops
from lightllm.utils.device_utils import (
    get_device_sm_count,
    get_device_sm_regs_num,
    get_device_sm_shared_mem_num,
    get_device_warp_size,
)
from .moe_kernel_configs import MoeGroupedGemmKernelConfig
from .moe_silu_and_mul import silu_and_mul_fwd
from .moe_sum_reduce import moe_sum_reduce
from lightllm.common.quantization.triton_quant.fp8.fp8act_quant_kernel import per_token_group_quant_fp8

FFN_MOE_CHUNK_SIZE = 8 * 1024

logger = init_logger(__name__)


@triton.jit
def moe_align_kernel(
    topk_ids_ptr,
    topk_m,
    topk_n,
    out_ptr,
    out_stride_m,
    out_stride_n,
    TOPK_BLOCK_M: tl.constexpr,
):
    pid = tl.program_id(axis=0)

    m_range = pid * TOPK_BLOCK_M + tl.arange(0, TOPK_BLOCK_M)

    topk_ptr = topk_ids_ptr + m_range
    topk_datas = tl.load(topk_ptr, mask=m_range < topk_m * topk_n, other=-1)
    write_datas = tl.where(topk_datas != -1, 1, 0)

    tl.store(
        out_ptr + topk_datas * out_stride_m + m_range,
        write_datas,
        mask=m_range < topk_m * topk_n,
    )


def moe_align(topk_ids: torch.Tensor, out: torch.Tensor):
    """
    topk_ids is tensor like [[0, 1, 2], [0, 3, 1], [3, 1, 4]] shape is [token_num, topk_num],
    the topk_ids needs to be in contiguous memory.
    out is tensor is shape with [expert_num, token_num * topk_num]
    out need fill 0 first, and then, fill the value to 1 in selected token loc.
    when expert_num is 5 and token_num is 3. topk_num is 3.
    topk_ids = [[0, 1, 2], [0, 3, 1], [3, 1, 4]]
    out tensor will be:
    [
    [1, 0, 0, 1, 0, 0, 0, 0, 0],
    [0, 1, 0, 0, 0, 1, 0, 1, 0],
    [0, 0, 1, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 1, 0, 1, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 1],
    ]
    """
    TOPK_BLOCK_M = 256

    token_num, topk = topk_ids.shape
    assert out.shape[1] == token_num * topk, f"out shape {out.shape} topk_ids shape {topk_ids.shape} "
    assert topk_ids.is_contiguous()
    out.fill_(0)
    grid = (triton.cdiv(token_num * topk, TOPK_BLOCK_M),)
    moe_align_kernel[grid](
        topk_ids,
        token_num,
        topk,
        out,
        out.stride(0),
        out.stride(1),
        TOPK_BLOCK_M=TOPK_BLOCK_M,
        num_warps=4,
        num_stages=1,
    )


# one grid handle a expert data
@triton.jit
def moe_align1_kernel(
    experts_info_ptr,  # [expert_num, token_num * topk_num]
    experts_info_stride0,
    experts_info_stride1,
    experts_info_m,
    experts_info_n,
    topk_weights,  # [token_num * topk_num,]
    expert_token_num_ptr,
    experts_topk_weight,  # [expert_num, token_num * topk_num]
    experts_topk_weight_stride0,
    experts_topk_weight_stride1,
    TOKEN_BLOCK_SIZE: tl.constexpr,
    NUM_STAGE: tl.constexpr,
):

    expert_id = tl.program_id(axis=0)

    off_n = tl.arange(0, TOKEN_BLOCK_SIZE)

    pre_sum = 0

    for start_loc in tl.range(0, experts_info_n, TOKEN_BLOCK_SIZE, num_stages=NUM_STAGE):
        n_range = start_loc + off_n
        topk_weights_data = tl.load(topk_weights + n_range, mask=n_range < experts_info_n, other=0)
        expert_data = tl.load(
            experts_info_ptr + expert_id * experts_info_stride0 + n_range, mask=n_range < experts_info_n, other=0
        )
        cumsum_expert_data = tl.cumsum(expert_data) + pre_sum
        pre_sum = tl.max(cumsum_expert_data)
        tl.store(
            experts_info_ptr + expert_id * experts_info_stride0 + cumsum_expert_data - 1,
            n_range,
            mask=(expert_data == 1) & (n_range < experts_info_n),
        )
        tl.store(
            experts_topk_weight + expert_id * experts_topk_weight_stride0 + cumsum_expert_data - 1,
            topk_weights_data,
            mask=(expert_data == 1) & (n_range < experts_info_n),
        )

    tl.store(expert_token_num_ptr + expert_id, pre_sum)
    return


def moe_align1(
    experts_info: torch.Tensor,
    topk_weights: torch.Tensor,
    experts_weight_info: torch.Tensor,
    exports_token_num: torch.Tensor,
    topk: int,
):
    """
    experts_info is tensor shape [expert_num, token_num * topk],
    topk_weights is tensor shape [token_num, topk]
    experts_weight_info is tensor shape [expert_num, token_num * topk]
    exports_token_num is out tensor, will get expert need handle token num.

    experts_info will change inplace.
    demo:
    topids = [[0, 1], [1, 3]]
    topk_weights = [[0.3, 0.7], [0.2, 0.8]]
    expert_num = 4, token_num = 2, topk = 2
    experts_info = [
        [1, 0, 0, 0],
        [0, 1, 1, 0],
        [0, 0, 0, 1],
        [0, 0, 0, 0]
    ]
    will get:
    experts_info = [
        [0, x, x, x],  # mark x， data will not be used future
        [1, 2, x, x],
        [3, x, x, x],
        [x, x, x, x]
    ]
    experts_weight_info = [
        [0.3,  x, x, x],
        [0.7, 0.2, x, x],
        [0.8, x, x, x],
        [x,  x,  x,  x]
    ]

    exports_token_num = [1, 2, 1, 0]
    """
    expert_num, token_num_mul_topk = experts_info.shape
    topk_num = topk_weights.shape[1]
    assert token_num_mul_topk <= FFN_MOE_CHUNK_SIZE * topk_num, "need split to handle seq len too long"
    assert exports_token_num.shape[0] == expert_num
    assert topk_weights.is_contiguous()
    if token_num_mul_topk <= 512:
        TOKEN_BLOCK_SIZE = 256
    else:
        TOKEN_BLOCK_SIZE = 512 if token_num_mul_topk <= 4 * 1024 else 2048

    grid = (expert_num,)
    moe_align1_kernel[grid](
        experts_info,
        experts_info.stride(0),
        experts_info.stride(1),
        expert_num,
        token_num_mul_topk,
        topk_weights,
        exports_token_num,
        experts_weight_info,
        experts_weight_info.stride(0),
        experts_weight_info.stride(1),
        TOKEN_BLOCK_SIZE=TOKEN_BLOCK_SIZE,
        NUM_STAGE=4,
        num_warps=8,
        num_stages=1,
    )


@triton.jit
def grouped_matmul_kernel(
    expert_token_limit,  # int,
    k,  # int
    n,  # int
    expert_num,  # int
    topk_num,  # int
    token_scale_ptr,  # [1,] for per tensor quant, or [token_num, hidden_dim // block_size] for per token, group quant
    weight_scale_ptr,  # [expert_num,] or [export_num, n // block_size_n, k // block_size_k]
    weight_scale_stride0,
    weight_scale_stride1,
    weight_scale_stride2,
    token_ptr,  # [token_num, hidden_dim]
    token_stride_0,
    token_stride_1,
    weights_ptr,  # [expert_num, N, K]
    weight_stride_0,
    weight_stride_1,
    weight_stride_2,
    expert_to_weights_ptr,  # [expert_num, token_num * topk]
    expert_to_weights_stride0,
    expert_to_weights_stride1,
    expert_to_token_num,  # [expert_num]
    expert_to_token_index,  # [expert_num, token_num * topk_num]
    expert_to_token_index_stride_0,
    out_ptr,  # [token_num * topk_num, n]
    out_stride_0,
    out_stride_1,
    # number of virtual SM
    num_sm,  # int
    compute_type: tl.constexpr,
    use_fp8_w8a8: tl.constexpr,
    block_size_n: tl.constexpr,
    block_size_k: tl.constexpr,
    # tile sizes
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr,
    MUL_ROUTED_WEIGHT: tl.constexpr = False,
):
    tile_idx = tl.program_id(0)

    for expert_id in range(expert_num):
        # get the gemm size of the current problem
        cur_m = tl.load(expert_to_token_num + expert_id, eviction_policy="evict_last")
        cur_m = tl.where(cur_m <= expert_token_limit, cur_m, 0)
        num_m_tiles = tl.cdiv(cur_m, BLOCK_SIZE_M)
        num_n_tiles = tl.cdiv(n, BLOCK_SIZE_N)
        num_tiles = num_m_tiles * num_n_tiles

        # iterate through the tiles in the current gemm problem
        while tile_idx < num_tiles:
            tile_idx_in_gemm = tile_idx
            # better super-grouping for L2 Cache Optimizations
            pid = tile_idx_in_gemm
            num_pid_m = num_m_tiles
            num_pid_n = num_n_tiles
            num_pid_in_group = GROUP_SIZE_M * num_pid_n
            group_id = pid // num_pid_in_group
            first_pid_m = group_id * GROUP_SIZE_M
            group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
            in_group_index = pid % num_pid_in_group
            back_mark = (in_group_index // group_size_m) % 2
            back_mark1 = -1 * (2 * back_mark - 1)
            pid_m = first_pid_m + back_mark * (group_size_m - 1) + back_mark1 * (in_group_index % group_size_m)
            pid_n = (pid % num_pid_in_group) // group_size_m

            tile_m_idx = pid_m
            tile_n_idx = pid_n

            # do regular gemm here
            offs_am = tile_m_idx * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
            a_m_index = tl.load(
                expert_to_token_index + expert_id * expert_to_token_index_stride_0 + offs_am,
                mask=offs_am < cur_m,
                other=0,
            )
            if MUL_ROUTED_WEIGHT:
                a_m_scale = tl.load(
                    expert_to_weights_ptr + expert_id * expert_to_weights_stride0 + offs_am,
                    mask=offs_am < cur_m,
                    other=0.0,
                )

            offs_bn = tile_n_idx * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
            offs_k = tl.arange(0, BLOCK_SIZE_K)

            if use_fp8_w8a8:
                if block_size_k > 0 and block_size_n > 0:
                    a_scale_ptrs = token_scale_ptr + (a_m_index // topk_num) * (token_stride_0 // block_size_k)
                    offs_bsn = offs_bn // block_size_n
                    b_scale_ptrs = weight_scale_ptr + expert_id * weight_scale_stride0 + offs_bsn * weight_scale_stride1
                else:
                    a_scale = tl.load(token_scale_ptr, eviction_policy="evict_last")
                    b_scale = tl.load(weight_scale_ptr + expert_id, eviction_policy="evict_last")
                    ab_scale = a_scale * b_scale

            if use_fp8_w8a8:
                a_ptrs = token_ptr + (a_m_index // topk_num)[None, :] * token_stride_0 + offs_k[:, None]
                b_ptrs = (
                    weights_ptr + weight_stride_0 * expert_id + offs_k[None, :] + offs_bn[:, None] * weight_stride_1
                )
                accumulator = tl.zeros((BLOCK_SIZE_N, BLOCK_SIZE_M), dtype=tl.float32)
            else:
                a_ptrs = token_ptr + (a_m_index // topk_num)[:, None] * token_stride_0 + offs_k[None, :]
                b_ptrs = (
                    weights_ptr + weight_stride_0 * expert_id + offs_k[:, None] + offs_bn[None, :] * weight_stride_1
                )
                accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)

            for step_k in range(0, tl.cdiv(k, BLOCK_SIZE_K)):
                # hint to Triton compiler to do proper loop pipelining
                # tl.multiple_of(a_ptrs, [16, 16])
                tl.multiple_of(b_ptrs, [16, 16])

                if use_fp8_w8a8:
                    a = tl.load(a_ptrs, mask=(offs_am[None, :] < cur_m) & (offs_k[:, None] < k))
                    b = tl.load(b_ptrs, mask=(offs_bn[:, None] < n) & (offs_k[None, :] < k))
                else:
                    a = tl.load(a_ptrs, mask=(offs_am[:, None] < cur_m) & (offs_k[None, :] < k))
                    b = tl.load(b_ptrs, mask=(offs_bn[None, :] < n) & (offs_k[:, None] < k))

                if use_fp8_w8a8:
                    if block_size_k > 0 and block_size_n > 0:
                        offs_ks = step_k * BLOCK_SIZE_K // block_size_k
                        a_scale = tl.load(a_scale_ptrs + offs_ks, mask=offs_am < cur_m, other=0.0)
                        b_scale = tl.load(b_scale_ptrs + offs_ks * weight_scale_stride2)
                        accumulator += tl.dot(b, a) * b_scale[:, None] * a_scale[None, :]
                    else:
                        accumulator = tl.dot(b, a, acc=accumulator)
                else:
                    accumulator += tl.dot(a, b)

                a_ptrs += BLOCK_SIZE_K
                b_ptrs += BLOCK_SIZE_K
                offs_k += BLOCK_SIZE_K

            if use_fp8_w8a8:
                if block_size_k > 0 and block_size_n > 0:
                    accumulator = accumulator.T
                else:
                    accumulator = accumulator.T
                    accumulator *= ab_scale

            if MUL_ROUTED_WEIGHT:
                accumulator *= a_m_scale[:, None]

            c = accumulator.to(compute_type)

            offs_cn = tile_n_idx * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
            c_ptrs = out_ptr + a_m_index[:, None] * out_stride_0 + offs_cn[None, :]
            tl.store(c_ptrs, c, mask=(offs_am[:, None] < cur_m) & (offs_cn[None, :] < n))
            tile_idx += num_sm

        tile_idx -= num_tiles
    return


def grouped_matmul(
    token_inputs: torch.Tensor,
    token_input_scale: torch.Tensor,  # for fp8
    expert_to_token_num: torch.Tensor,
    expert_to_token_index: torch.Tensor,
    expert_to_weights: torch.Tensor,
    expert_weights: torch.Tensor,
    expert_to_weights_scale: torch.Tensor,  # for fp8
    topk_num: int,
    out: torch.Tensor,
    expert_token_limit: int,
    mul_routed_weight: bool,
    use_fp8_w8a8: bool,
    alloc_tensor_func=torch.empty,
    **run_config,
):
    """
    token_inputs is tensor shape [token_num, hidden_dim],
    token_input_scale is tensor shape [1,], when use_fp8_w8a8 is False, it must be None
    expert_to_token_num is tensor shape [expert_num],
    expert_to_token_index is tensor shape [expert_num, token_num * topk_num],
    expert_weights is tensor shape [expert_num, out_dim, hidden_dim]
    expert_to_weights_scale is tensor shape [expert_num] or
    [expert_num, out_dim // block_size_, hidden_dim // block_size_k],
    when use_fp8_w8a8 is False, it must be None
    expert_token_limit use to limit handles token per expert.
    out is tensor shape [token_num * topk_num, out_dim]
    """
    compute_type = tl.bfloat16 if out.dtype == torch.bfloat16 else tl.float16
    expert_num, n, k = expert_weights.shape
    assert token_inputs.shape[1] == k
    assert expert_to_token_index.shape == expert_to_weights.shape
    assert token_inputs.is_contiguous()
    assert expert_to_token_num.is_contiguous()
    assert expert_to_weights.is_contiguous()
    assert expert_weights.is_contiguous()

    # for deepseek_v3 block-wise quant
    block_size_n = 0
    block_size_k = 0
    if use_fp8_w8a8:
        if expert_to_weights_scale.ndim == 3:
            block_size_n = expert_weights.shape[1] // expert_to_weights_scale.shape[1]
            block_size_k = expert_weights.shape[2] // expert_to_weights_scale.shape[2]
    if not run_config:
        run_config = MoeGroupedGemmKernelConfig.try_to_get_best_config(
            M=token_inputs.shape[0],
            N=n,
            K=k,
            topk_num=topk_num,
            expert_num=expert_num,
            mul_routed_weight=mul_routed_weight,
            use_fp8_w8a8=use_fp8_w8a8,
            out_dtype=str(out.dtype),
        )
    BLOCK_SIZE_M = run_config["BLOCK_SIZE_M"]
    BLOCK_SIZE_N = run_config["BLOCK_SIZE_N"]
    BLOCK_SIZE_K = run_config["BLOCK_SIZE_K"]
    GROUP_SIZE_M = run_config["GROUP_SIZE_M"]
    num_warps = run_config["num_warps"]
    num_stages = run_config["num_stages"]

    if block_size_k != 0:
        # 如果使用了 block wise 量化，分块大小不能超过 block size
        BLOCK_SIZE_K = min(BLOCK_SIZE_K, block_size_k)
        assert BLOCK_SIZE_K == triton.next_power_of_2(BLOCK_SIZE_K)

    if use_fp8_w8a8:
        # 当权重使用 block wise 量化时，激活也使用 per token， group size 量化
        if block_size_k == 0:
            token_inputs, token_input_scale = ops.scaled_fp8_quant(token_inputs, token_input_scale)
        else:
            _m, _k = token_inputs.shape
            assert _k % block_size_k == 0
            input_scale = alloc_tensor_func((_m, _k // block_size_k), dtype=torch.float32, device=token_inputs.device)
            qinput_tensor = alloc_tensor_func((_m, _k), dtype=expert_weights.dtype, device=token_inputs.device)
            per_token_group_quant_fp8(token_inputs, block_size_k, qinput_tensor, input_scale)
            token_inputs, token_input_scale = qinput_tensor, input_scale

    kernel = grouped_matmul_kernel.warmup(
        expert_token_limit,
        k,
        n,
        expert_num,
        topk_num,
        token_input_scale,
        expert_to_weights_scale,
        expert_to_weights_scale.stride(0)
        if expert_to_weights_scale is not None and expert_to_weights_scale.ndim == 2
        else 0,
        expert_to_weights_scale.stride(1)
        if expert_to_weights_scale is not None and expert_to_weights_scale.ndim == 2
        else 0,
        expert_to_weights_scale.stride(2)
        if expert_to_weights_scale is not None and expert_to_weights_scale.ndim == 3
        else 0,
        token_inputs,
        token_inputs.stride(0),
        token_inputs.stride(1),
        expert_weights,
        expert_weights.stride(0),
        expert_weights.stride(1),
        expert_weights.stride(2),
        expert_to_weights,
        expert_to_weights.stride(0),
        expert_to_weights.stride(1),
        expert_to_token_num,
        expert_to_token_index,
        expert_to_token_index.stride(0),
        out,
        out.stride(0),
        out.stride(1),
        num_sm=1,
        compute_type=compute_type,
        use_fp8_w8a8=use_fp8_w8a8,
        block_size_n=block_size_n,
        block_size_k=block_size_k,
        BLOCK_SIZE_M=BLOCK_SIZE_M,
        BLOCK_SIZE_N=BLOCK_SIZE_N,
        BLOCK_SIZE_K=BLOCK_SIZE_K,
        GROUP_SIZE_M=GROUP_SIZE_M,
        MUL_ROUTED_WEIGHT=mul_routed_weight,
        num_warps=num_warps,
        num_stages=num_stages,
        grid=(1,),
    )

    kernel._init_handles()
    n_regs = kernel.n_regs
    size_smem = kernel.metadata.shared

    sm_count = get_device_sm_count()
    max_regs = get_device_sm_regs_num()
    shared_mem_max = get_device_sm_shared_mem_num()
    warp_size = get_device_warp_size()

    occupancy = max_regs // (n_regs * warp_size * num_warps)
    occupancy = min(occupancy, shared_mem_max // size_smem)
    num_sm = sm_count * occupancy

    need_max_grid = triton.cdiv(n, BLOCK_SIZE_N) * triton.cdiv(
        expert_to_token_index.shape[1] + expert_num * (BLOCK_SIZE_M - 1), BLOCK_SIZE_M
    )
    num_sm = min(num_sm, need_max_grid)

    grid = (num_sm,)

    grouped_matmul_kernel[grid](
        expert_token_limit,
        k,
        n,
        expert_num,
        topk_num,
        token_input_scale,
        expert_to_weights_scale,
        expert_to_weights_scale.stride(0)
        if expert_to_weights_scale is not None and expert_to_weights_scale.ndim == 2
        else 0,
        expert_to_weights_scale.stride(1)
        if expert_to_weights_scale is not None and expert_to_weights_scale.ndim == 2
        else 0,
        expert_to_weights_scale.stride(2)
        if expert_to_weights_scale is not None and expert_to_weights_scale.ndim == 3
        else 0,
        token_inputs,
        token_inputs.stride(0),
        token_inputs.stride(1),
        expert_weights,
        expert_weights.stride(0),
        expert_weights.stride(1),
        expert_weights.stride(2),
        expert_to_weights,
        expert_to_weights.stride(0),
        expert_to_weights.stride(1),
        expert_to_token_num,
        expert_to_token_index,
        expert_to_token_index.stride(0),
        out,
        out.stride(0),
        out.stride(1),
        num_sm=num_sm,
        compute_type=compute_type,
        use_fp8_w8a8=use_fp8_w8a8,
        block_size_n=block_size_n,
        block_size_k=block_size_k,
        BLOCK_SIZE_M=BLOCK_SIZE_M,
        BLOCK_SIZE_N=BLOCK_SIZE_N,
        BLOCK_SIZE_K=BLOCK_SIZE_K,
        GROUP_SIZE_M=GROUP_SIZE_M,
        MUL_ROUTED_WEIGHT=mul_routed_weight,
        num_warps=num_warps,
        num_stages=num_stages,
    )
    return


def fused_experts_impl(
    hidden_states: torch.Tensor,
    w1: torch.Tensor,
    w2: torch.Tensor,
    topk_weights: torch.Tensor,
    topk_ids: torch.Tensor,
    inplace: bool = False,
    use_fp8_w8a8: bool = False,
    use_int8_w8a16: bool = False,
    w1_scale: Optional[torch.Tensor] = None,
    w2_scale: Optional[torch.Tensor] = None,
    a1_scale: Optional[torch.Tensor] = None,
    a2_scale: Optional[torch.Tensor] = None,
    alloc_tensor_func=torch.empty,
    **run_config,
):
    # Check constraints.
    assert hidden_states.shape[1] == w1.shape[2], "Hidden size mismatch"
    assert topk_weights.shape == topk_ids.shape, "topk shape mismatch"
    assert hidden_states.is_contiguous(), "Hidden_states must be contiguous"
    assert w1.is_contiguous(), "Expert weights1 must be contiguous"
    assert w2.is_contiguous(), "Expert weights2 must be contiguous"
    assert hidden_states.dtype in [torch.float32, torch.float16, torch.bfloat16]
    num_tokens, _ = hidden_states.shape
    E, N, _ = w1.shape
    CHUNK_SIZE = FFN_MOE_CHUNK_SIZE
    topk_num = topk_ids.shape[1]
    M = min(num_tokens, CHUNK_SIZE)

    intermediate_cache1 = alloc_tensor_func((M, topk_num, N), device=hidden_states.device, dtype=hidden_states.dtype)
    intermediate_cache2 = alloc_tensor_func(
        (M, topk_num, N // 2), device=hidden_states.device, dtype=hidden_states.dtype
    )
    intermediate_cache3 = alloc_tensor_func(
        (M, topk_num, w2.shape[1]), device=hidden_states.device, dtype=hidden_states.dtype
    )

    if inplace:
        out_hidden_states = hidden_states
    else:
        out_hidden_states = alloc_tensor_func(
            hidden_states.shape, device=hidden_states.device, dtype=hidden_states.dtype
        )

    for chunk in range(triton.cdiv(num_tokens, CHUNK_SIZE)):
        begin_chunk_idx, end_chunk_idx = (chunk * CHUNK_SIZE, min((chunk + 1) * CHUNK_SIZE, num_tokens))
        curr_hidden_states = hidden_states[begin_chunk_idx:end_chunk_idx]
        tokens_in_chunk, _ = curr_hidden_states.shape

        intermediate_cache1 = intermediate_cache1[:tokens_in_chunk]
        intermediate_cache2 = intermediate_cache2[:tokens_in_chunk]
        intermediate_cache3 = intermediate_cache3[:tokens_in_chunk]

        curr_topk_ids = topk_ids[begin_chunk_idx:end_chunk_idx]
        curr_topk_weights = topk_weights[begin_chunk_idx:end_chunk_idx]

        expert_to_tokens = torch.empty((E, topk_num * tokens_in_chunk), dtype=torch.int32, device="cuda")
        expert_to_weights = torch.empty((E, topk_num * tokens_in_chunk), dtype=torch.float32, device="cuda")
        moe_align(topk_ids=curr_topk_ids, out=expert_to_tokens)
        expert_to_token_num = torch.empty((E,), dtype=torch.int32, device="cuda")
        moe_align1(expert_to_tokens, curr_topk_weights, expert_to_weights, expert_to_token_num, topk=topk_num)

        grouped_matmul(
            curr_hidden_states,
            a1_scale,
            expert_to_token_num,
            expert_to_tokens,
            expert_to_weights=expert_to_weights,
            expert_weights=w1,
            expert_to_weights_scale=w1_scale,
            topk_num=topk_num,
            out=intermediate_cache1.view(-1, N),
            expert_token_limit=2 ** 31 - 1,
            mul_routed_weight=False,
            use_fp8_w8a8=use_fp8_w8a8,
            alloc_tensor_func=alloc_tensor_func,
            **run_config,
        )

        silu_and_mul_fwd(intermediate_cache1.view(-1, N), intermediate_cache2.view(-1, N // 2))

        grouped_matmul(
            intermediate_cache2.view(-1, N // 2),
            a2_scale,
            expert_to_token_num,
            expert_to_tokens,
            expert_to_weights=expert_to_weights,
            expert_weights=w2,
            expert_to_weights_scale=w2_scale,
            topk_num=1,
            out=intermediate_cache3.view(-1, w2.shape[1]),
            expert_token_limit=2 ** 31 - 1,
            mul_routed_weight=True,
            use_fp8_w8a8=use_fp8_w8a8,
            alloc_tensor_func=alloc_tensor_func,
            **run_config,
        )

        moe_sum_reduce(
            intermediate_cache3.view(*intermediate_cache3.shape), out_hidden_states[begin_chunk_idx:end_chunk_idx]
        )
    return out_hidden_states
