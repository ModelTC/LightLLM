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
from lightllm.utils.vllm_utils import vllm_ops
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
from lightllm.utils.torch_ops_utils import direct_register_custom_op
from lightllm.common.triton_utils.autotuner import autotune

FFN_MOE_CHUNK_SIZE = 32 * 1024

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
def moe_align_fused_kernel(
    topk_ids_ptr,  # [token_num, topk]
    topk_weights_ptr,  # [token_num, topk]
    expert_to_token_index_ptr,  # [expert_num, token_num * topk]
    expert_to_weight_ptr,  # [expert_num, token_num * topk]
    expert_token_num_ptr,  # [expert_num]
    token_num,
    topk_num: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    token_block = tl.program_id(0)
    offs = token_block * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offs < token_num * topk_num

    expert_ids = tl.load(topk_ids_ptr + offs, mask=mask, other=0)
    weights = tl.load(topk_weights_ptr + offs, mask=mask, other=0.0)

    # 用 atomic_add 给 expert 分配写位置
    write_pos = tl.atomic_add(expert_token_num_ptr + expert_ids, 1, mask=mask)

    # 按 token 顺序写 index 和 weight
    tl.store(
        expert_to_token_index_ptr + expert_ids * (token_num * topk_num) + write_pos,
        offs,
        mask=mask,
    )
    tl.store(
        expert_to_weight_ptr + expert_ids * (token_num * topk_num) + write_pos,
        weights,
        mask=mask,
    )


def _get_moe_align_fused_static_key(
    topk_weights: torch.Tensor,
) -> dict:
    topk_num = topk_weights.shape[1]
    return {
        "topk_num": topk_num,
    }


def _get_moe_align_fused_configs():
    return [
        {
            "BLOCK_SIZE": bt,
            "num_warps": nw,
        }
        for nw in [1, 2, 4, 8]
        for bt in [128, 256, 512, 1024, 2048]
    ]


@autotune(
    kernel_name="moe_align_fused:v1",
    configs_gen_func=_get_moe_align_fused_configs,
    static_key_func=_get_moe_align_fused_static_key,
    run_key_func=lambda topk_ids: topk_ids.shape[0],
    mutates_args=["expert_to_token_index", "expert_to_weight", "expert_token_num"],
)
def moe_align_fused(
    expert_to_token_index, expert_to_weight, expert_token_num, topk_ids, topk_weights, run_config: Optional[dict] = None
):
    token_num, topk_num = topk_ids.shape
    if run_config is None:
        run_config = {}
    BLOCK_SIZE = run_config.get("BLOCK_SIZE", 256)
    num_warps = run_config.get("num_warps", 4)

    grid = (triton.cdiv(token_num * topk_num, BLOCK_SIZE),)
    moe_align_fused_kernel[grid](
        topk_ids,
        topk_weights,
        expert_to_token_index,
        expert_to_weight,
        expert_token_num,
        token_num,
        topk_num,
        BLOCK_SIZE=BLOCK_SIZE,
        num_warps=num_warps,
    )
    return expert_to_token_index, expert_to_weight, expert_token_num


@triton.jit
def moe_align2_kernel(
    experts_token_num_ptr,  # [expert_num,]
    mblocks_to_expert_id,  # [max_num_m_blocks,]
    mblocks_to_m_index,  # [max_num_m_blocks,]
    expert_num,
    max_num_m_blocks,
    BLOCK_M: tl.constexpr,
    BLOCK_EXPERT: tl.constexpr,
):

    expert_id = tl.program_id(axis=0)
    off_expert = tl.arange(0, BLOCK_EXPERT)
    expert_to_token_num = tl.load(experts_token_num_ptr + off_expert, mask=off_expert < expert_num, other=0)
    expert_to_block_num = tl.cdiv(expert_to_token_num, BLOCK_M)
    block_starts = tl.cumsum(expert_to_block_num) - expert_to_block_num
    block_start = tl.sum(tl.where(off_expert == expert_id, block_starts, 0))

    cur_expert_token_num = tl.load(experts_token_num_ptr + expert_id)
    cur_block_num = tl.cdiv(cur_expert_token_num, BLOCK_M)

    block_off = tl.arange(0, 128)
    for start_loc in range(0, cur_block_num, 128):
        tl.store(
            mblocks_to_expert_id + block_start + start_loc + block_off,
            expert_id,
            mask=start_loc + block_off < cur_block_num,
        )
        tl.store(
            mblocks_to_m_index + block_start + start_loc + block_off,
            start_loc + block_off,
            mask=start_loc + block_off < cur_block_num,
        )

    if expert_id == expert_num - 1:
        for extra_fill_start in range(block_start + cur_block_num, max_num_m_blocks, 128):
            tl.store(
                mblocks_to_expert_id + extra_fill_start + block_off,
                -1,
                mask=extra_fill_start + block_off < max_num_m_blocks,
            )
    return


def moe_align2(token_num_mul_topk_num: int, exports_token_num: torch.Tensor, block_m: int):
    """
    exports_token_num is tensor shape [expert_num] , will get expert need handle token num.
    out tensor is a tensor that contain block schduel infos tensor.
    """
    max_num_tokens_padded = token_num_mul_topk_num + exports_token_num.shape[0] * (block_m - 1)
    max_num_m_blocks = triton.cdiv(max_num_tokens_padded, block_m)
    mblocks_to_expert_id = torch.empty((max_num_m_blocks,), dtype=torch.int32, device="cuda")
    mblocks_to_m_index = torch.empty((max_num_m_blocks,), dtype=torch.int32, device="cuda")
    expert_num = exports_token_num.shape[0]

    grid = (expert_num,)
    moe_align2_kernel[grid](
        exports_token_num,
        mblocks_to_expert_id,
        mblocks_to_m_index,
        expert_num,
        max_num_m_blocks,
        BLOCK_M=block_m,
        BLOCK_EXPERT=triton.next_power_of_2(expert_num),
        num_warps=4,
        num_stages=1,
    )

    return mblocks_to_expert_id, mblocks_to_m_index


@triton.jit
def grouped_matmul_kernel(
    mblocks_to_expert_id,  # [max_m_block_size]
    mblocks_to_m_index,  # [max_m_block_size]
    k,  # int
    n,  # int
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
    bias_ptr,  # [expert_num, n]
    bias_stride_0,
    bias_stride_1,
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
    m_block_num,  # int
    n_block_num,  # int
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
    NEED_K_MASK: tl.constexpr = True,
    NEED_TRANS: tl.constexpr = False,
    ADD_BIAS: tl.constexpr = False,
):
    pid = tl.program_id(0)

    # better super-grouping for L2 Cache Optimizations
    num_pid_m = m_block_num
    num_pid_n = n_block_num
    num_pid_in_group = GROUP_SIZE_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_SIZE_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
    in_group_index = pid % num_pid_in_group

    back_mark = (in_group_index // group_size_m) % 2
    back_mark1 = -1 * (2 * back_mark - 1)
    pid_m = first_pid_m + back_mark * (group_size_m - 1) + back_mark1 * (in_group_index % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m

    expert_id = tl.load(mblocks_to_expert_id + pid_m)

    if expert_id == -1:
        return

    tile_m_idx = tl.load(mblocks_to_m_index + pid_m)
    tile_n_idx = pid_n

    # get the gemm size of the current problem
    cur_m = tl.load(expert_to_token_num + expert_id)

    # do regular gemm here
    offs_am = tile_m_idx * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    token_mask = offs_am < cur_m
    a_m_index = tl.load(
        expert_to_token_index + expert_id * expert_to_token_index_stride_0 + offs_am,
        mask=token_mask,
        other=0,
    )
    offs_bn = (tile_n_idx * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)) % n
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

    if NEED_TRANS:
        a_ptrs = token_ptr + (a_m_index // topk_num)[None, :] * token_stride_0 + offs_k[:, None]
        b_ptrs = weights_ptr + weight_stride_0 * expert_id + offs_k[None, :] + offs_bn[:, None] * weight_stride_1
        accumulator = tl.zeros((BLOCK_SIZE_N, BLOCK_SIZE_M), dtype=tl.float32)
    else:
        a_ptrs = token_ptr + (a_m_index // topk_num)[:, None] * token_stride_0 + offs_k[None, :]
        b_ptrs = weights_ptr + weight_stride_0 * expert_id + offs_k[:, None] + offs_bn[None, :] * weight_stride_1
        accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)

    for step_k in range(0, tl.cdiv(k, BLOCK_SIZE_K)):
        # hint to Triton compiler to do proper loop pipelining
        # tl.multiple_of(a_ptrs, [16, 16])
        # tl.multiple_of(b_ptrs, [16, 16])

        if NEED_TRANS:
            if NEED_K_MASK:
                a = tl.load(
                    a_ptrs, mask=(token_mask[None, :]) & (offs_k[:, None] < k - step_k * BLOCK_SIZE_K), other=0.0
                )
                b = tl.load(b_ptrs, mask=(offs_k[None, :] < k), other=0.0)
            else:
                a = tl.load(a_ptrs, mask=(token_mask[None, :]), other=0.0)
                b = tl.load(b_ptrs)
        else:
            if NEED_K_MASK:
                a = tl.load(
                    a_ptrs, mask=(token_mask[:, None]) & (offs_k[None, :] < k - step_k * BLOCK_SIZE_K), other=0.0
                )
                b = tl.load(b_ptrs, mask=(offs_k[:, None] < k), other=0.0)
            else:
                a = tl.load(a_ptrs, mask=(token_mask[:, None]), other=0.0)
                b = tl.load(b_ptrs)

        if use_fp8_w8a8:
            if block_size_k > 0 and block_size_n > 0:
                offs_ks = step_k * BLOCK_SIZE_K // block_size_k
                a_scale = tl.load(a_scale_ptrs + offs_ks, mask=token_mask, other=0.0)
                b_scale = tl.load(b_scale_ptrs + offs_ks * weight_scale_stride2)
                if NEED_TRANS:
                    accumulator += tl.dot(b, a) * b_scale[:, None] * a_scale[None, :]
                else:
                    accumulator += tl.dot(a, b) * a_scale[:, None] * b_scale[None, :]
            else:
                if NEED_TRANS:
                    accumulator = tl.dot(b, a, acc=accumulator)
                else:
                    accumulator = tl.dot(a, b, acc=accumulator)
        else:
            accumulator += tl.dot(a, b)

        a_ptrs += BLOCK_SIZE_K
        b_ptrs += BLOCK_SIZE_K

    if NEED_TRANS:
        accumulator = accumulator.T

    if use_fp8_w8a8:
        if not (block_size_k > 0 and block_size_n > 0):
            accumulator *= ab_scale

    # Apply bias if requested
    if ADD_BIAS:
        offs_bn_bias = offs_bn  # [BLOCK_SIZE_N]
        bias_ptrs = bias_ptr + expert_id * bias_stride_0 + offs_bn_bias
        bias_vals = tl.load(bias_ptrs)  # [BLOCK_SIZE_N]
        accumulator += bias_vals[None, :]  # broadcast across M dimension

    if MUL_ROUTED_WEIGHT:
        a_m_scale = tl.load(
            expert_to_weights_ptr + expert_id * expert_to_weights_stride0 + offs_am,
            mask=token_mask,
            other=0.0,
        )
        accumulator *= a_m_scale[:, None]

    c = accumulator.to(compute_type)

    offs_cn = tile_n_idx * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    c_ptrs = out_ptr + a_m_index[:, None] * out_stride_0 + offs_cn[None, :]
    tl.store(c_ptrs, c, mask=(token_mask[:, None]) & (offs_cn[None, :] < n))

    return


def _get_grouped_matmul_static_key(
    expert_weights: torch.Tensor,
    topk_num: int,
    out: torch.Tensor,
    mul_routed_weight: bool,
    use_fp8_w8a8: bool,
) -> dict:
    expert_num, n, k = expert_weights.shape
    return {
        "N": n,
        "K": k,
        "topk_num": topk_num,
        "expert_num": expert_num,
        "mul_routed_weight": mul_routed_weight,
        "use_fp8_w8a8": use_fp8_w8a8,
        "out_dtype": str(out.dtype),
    }


def _get_grouped_matmul_configs():
    return [
        {
            "BLOCK_SIZE_M": bm,
            "BLOCK_SIZE_N": bn,
            "BLOCK_SIZE_K": bk,
            "GROUP_SIZE_M": gm,
            "num_warps": nw,
            "num_stages": ns,
            "NEED_TRANS": need_trans,
        }
        for ns in [2, 3, 4, 5]
        for gm in [1, 16, 32, 64]
        for nw in [4, 8]
        for bm in [16, 32, 64, 128]
        for bn in [16, 32, 64, 128]
        for bk in [32, 64, 128]
        for need_trans in [True, False]
    ]


@autotune(
    kernel_name="grouped_matmul:v1",
    configs_gen_func=_get_grouped_matmul_configs,
    static_key_func=_get_grouped_matmul_static_key,
    run_key_func=lambda token_inputs: token_inputs.shape[0],
    mutates_args=["out"],
)
def grouped_matmul(
    token_num_mul_topk_num: int,
    token_inputs: torch.Tensor,
    token_input_scale: torch.Tensor,  # for fp8
    expert_to_token_num: torch.Tensor,
    expert_to_token_index: torch.Tensor,
    expert_to_weights: torch.Tensor,
    expert_weights: torch.Tensor,
    expert_to_weights_scale: torch.Tensor,  # for fp8
    topk_num: int,
    out: torch.Tensor,
    mul_routed_weight: bool,
    use_fp8_w8a8: bool,
    alloc_tensor_func=torch.empty,
    reused_mblock_infos=None,
    run_config: Optional[dict] = None,
    bias: Optional[torch.Tensor] = None,  # per-expert bias [expert_num, N]
):
    """
    token_num_mul_topk_num is int equal token_num * topk_num,
    token_inputs is tensor shape [token_num, hidden_dim],
    token_input_scale is tensor shape [1,], when use_fp8_w8a8 is False, it must be None
    expert_to_token_num is tensor shape [expert_num],
    expert_to_token_index is tensor shape [expert_num, token_num * topk_num],
    expert_weights is tensor shape [expert_num, out_dim, hidden_dim]
    expert_to_weights_scale is tensor shape [expert_num] or
    [expert_num, out_dim // block_size_, hidden_dim // block_size_k],
    when use_fp8_w8a8 is False, it must be None
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

    if run_config is None:
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
    NEED_TRANS = run_config.get("NEED_TRANS", False)
    if not use_fp8_w8a8:
        assert NEED_TRANS is False, "only use_fp8_w8a8 mode can use NEED_TRANS to accelerate"

    if block_size_k != 0:
        # 如果使用了 block wise 量化，分块大小不能超过 block size
        BLOCK_SIZE_K = min(BLOCK_SIZE_K, block_size_k)
        assert BLOCK_SIZE_K == triton.next_power_of_2(BLOCK_SIZE_K)

    if use_fp8_w8a8:
        # 当权重使用 block wise 量化时，激活也使用 per token， group size 量化
        if block_size_k == 0:
            token_inputs, token_input_scale = vllm_ops.scaled_fp8_quant(token_inputs, token_input_scale)
        else:
            _m, _k = token_inputs.shape
            assert _k % block_size_k == 0
            token_inputs, token_input_scale = per_token_group_quant_fp8(
                token_inputs, block_size_k, dtype=expert_weights.dtype
            )

    if reused_mblock_infos is None:
        mblocks_to_expert_id, mblocks_to_m_index = moe_align2(token_num_mul_topk_num, expert_to_token_num, BLOCK_SIZE_M)
    else:
        # when up group gemm and down group gemm use same BLOCK_SIZE_M,
        # can reuse (mblocks_to_expert_id, mblocks_to_m_index) created by moe_align2 kernel.
        mblocks_to_expert_id, mblocks_to_m_index, reused_block_size_m = reused_mblock_infos
        if reused_block_size_m != BLOCK_SIZE_M:
            mblocks_to_expert_id, mblocks_to_m_index = moe_align2(
                token_num_mul_topk_num, expert_to_token_num, BLOCK_SIZE_M
            )

    block_num = triton.cdiv(n, BLOCK_SIZE_N) * mblocks_to_expert_id.shape[0]

    grid = (block_num,)

    NEED_K_MASK = (k % BLOCK_SIZE_K) != 0

    grouped_matmul_kernel[grid](
        mblocks_to_expert_id,
        mblocks_to_m_index,
        k,
        n,
        topk_num,
        token_input_scale,
        expert_to_weights_scale,
        expert_to_weights_scale.stride(0)
        if expert_to_weights_scale is not None and expert_to_weights_scale.ndim >= 1
        else 0,
        expert_to_weights_scale.stride(1)
        if expert_to_weights_scale is not None and expert_to_weights_scale.ndim >= 2
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
        bias,
        bias.stride(0) if bias is not None else 0,
        bias.stride(1) if bias is not None and bias.ndim >= 2 else 0,
        expert_to_weights,
        expert_to_weights.stride(0),
        expert_to_weights.stride(1),
        expert_to_token_num,
        expert_to_token_index,
        expert_to_token_index.stride(0),
        out,
        out.stride(0),
        out.stride(1),
        m_block_num=mblocks_to_expert_id.shape[0],
        n_block_num=triton.cdiv(n, BLOCK_SIZE_N),
        compute_type=compute_type,
        use_fp8_w8a8=use_fp8_w8a8,
        block_size_n=block_size_n,
        block_size_k=block_size_k,
        BLOCK_SIZE_M=BLOCK_SIZE_M,
        BLOCK_SIZE_N=BLOCK_SIZE_N,
        BLOCK_SIZE_K=BLOCK_SIZE_K,
        GROUP_SIZE_M=GROUP_SIZE_M,
        MUL_ROUTED_WEIGHT=mul_routed_weight,
        NEED_K_MASK=NEED_K_MASK,
        NEED_TRANS=NEED_TRANS,
        num_warps=num_warps,
        num_stages=num_stages,
        ADD_BIAS=bias is not None,
    )
    return (mblocks_to_expert_id, mblocks_to_m_index, BLOCK_SIZE_M)


def fused_experts_impl(
    hidden_states: torch.Tensor,
    w1: torch.Tensor,
    w2: torch.Tensor,
    w1_bias: Optional[torch.Tensor],
    w2_bias: Optional[torch.Tensor],
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
    run_config: Optional[dict] = None,
    layout="blocked",
    limit=None,
    alpha=None,
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

    intermediate_cache13_shared = alloc_tensor_func(
        (M, topk_num, max(N, w2.shape[1])), device=hidden_states.device, dtype=hidden_states.dtype
    )
    intermediate_cache1 = intermediate_cache13_shared.view(-1)[: (M * topk_num * N)].view(M, topk_num, N)
    intermediate_cache2 = alloc_tensor_func(
        (M, topk_num, N // 2), device=hidden_states.device, dtype=hidden_states.dtype
    )
    intermediate_cache3 = intermediate_cache13_shared.view(-1)[: (M * topk_num * w2.shape[1])].view(
        M, topk_num, w2.shape[1]
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
        expert_to_token_num = torch.zeros((E,), dtype=torch.int32, device="cuda")
        moe_align_fused(
            expert_to_token_index=expert_to_tokens,
            expert_to_weight=expert_to_weights,
            expert_token_num=expert_to_token_num,
            topk_ids=curr_topk_ids,
            topk_weights=curr_topk_weights,
        )

        reused_mblock_infos = grouped_matmul(
            curr_topk_ids.numel(),
            curr_hidden_states,
            a1_scale,
            expert_to_token_num,
            expert_to_tokens,
            expert_to_weights=expert_to_weights,
            expert_weights=w1,
            expert_to_weights_scale=w1_scale,
            topk_num=topk_num,
            out=intermediate_cache1.view(-1, N),
            mul_routed_weight=False,
            use_fp8_w8a8=use_fp8_w8a8,
            alloc_tensor_func=alloc_tensor_func,
            run_config=run_config,
            bias=w1_bias,
        )

        silu_and_mul_fwd(
            intermediate_cache1.view(-1, N),
            intermediate_cache2.view(-1, N // 2),
            limit=limit,
            alpha=alpha,
            layout=layout,
        )

        grouped_matmul(
            curr_topk_ids.numel(),
            intermediate_cache2.view(-1, N // 2),
            a2_scale,
            expert_to_token_num,
            expert_to_tokens,
            expert_to_weights=expert_to_weights,
            expert_weights=w2,
            expert_to_weights_scale=w2_scale,
            topk_num=1,
            out=intermediate_cache3.view(-1, w2.shape[1]),
            mul_routed_weight=True,
            use_fp8_w8a8=use_fp8_w8a8,
            alloc_tensor_func=alloc_tensor_func,
            reused_mblock_infos=reused_mblock_infos,
            run_config=run_config,
            bias=w2_bias,
        )

        moe_sum_reduce(
            intermediate_cache3.view(*intermediate_cache3.shape), out_hidden_states[begin_chunk_idx:end_chunk_idx]
        )
    return out_hidden_states


def inplace_fused_experts_impl(
    hidden_states: torch.Tensor,
    w1: torch.Tensor,
    w2: torch.Tensor,
    # optional bias for w1 and w2
    w1_bias: Optional[torch.Tensor],
    w2_bias: Optional[torch.Tensor],
    topk_weights: torch.Tensor,
    topk_ids: torch.Tensor,
    use_fp8_w8a8: bool = False,
    use_int8_w8a16: bool = False,
    w1_scale: Optional[torch.Tensor] = None,
    w2_scale: Optional[torch.Tensor] = None,
    a1_scale: Optional[torch.Tensor] = None,
    a2_scale: Optional[torch.Tensor] = None,
    layout: str = "blocked",
    alpha: Optional[float] = None,
    limit: Optional[float] = None,
) -> None:
    fused_experts_impl(
        hidden_states,
        w1,
        w2,
        w1_bias,
        w2_bias,
        topk_weights,
        topk_ids,
        True,
        use_fp8_w8a8,
        use_int8_w8a16,
        w1_scale,
        w2_scale,
        a1_scale,
        a2_scale,
        layout=layout,
        alpha=alpha,
        limit=limit,
    )


def inplace_fused_experts_impl_fake(
    hidden_states: torch.Tensor,
    w1: torch.Tensor,
    w2: torch.Tensor,
    # optional bias for w1 and w2
    w1_bias: Optional[torch.Tensor],
    w2_bias: Optional[torch.Tensor],
    topk_weights: torch.Tensor,
    topk_ids: torch.Tensor,
    use_fp8_w8a8: bool = False,
    use_int8_w8a16: bool = False,
    w1_scale: Optional[torch.Tensor] = None,
    w2_scale: Optional[torch.Tensor] = None,
    a1_scale: Optional[torch.Tensor] = None,
    a2_scale: Optional[torch.Tensor] = None,
    layout: str = "blocked",
    alpha: Optional[float] = None,
    limit: Optional[float] = None,
) -> None:
    pass


direct_register_custom_op(
    "inplace_fused_experts_impl",
    inplace_fused_experts_impl,
    ["hidden_states"],
    inplace_fused_experts_impl_fake,
)


def outplace_fused_experts_impl(
    hidden_states: torch.Tensor,
    w1: torch.Tensor,
    w2: torch.Tensor,
    # optional bias for w1 and w2
    w1_bias: Optional[torch.Tensor],
    w2_bias: Optional[torch.Tensor],
    topk_weights: torch.Tensor,
    topk_ids: torch.Tensor,
    use_fp8_w8a8: bool = False,
    use_int8_w8a16: bool = False,
    w1_scale: Optional[torch.Tensor] = None,
    w2_scale: Optional[torch.Tensor] = None,
    a1_scale: Optional[torch.Tensor] = None,
    a2_scale: Optional[torch.Tensor] = None,
    layout: str = "blocked",
    alpha: Optional[float] = None,
    limit: Optional[float] = None,
) -> None:
    return fused_experts_impl(
        hidden_states,
        w1,
        w2,
        w1_bias,
        w2_bias,
        topk_weights,
        topk_ids,
        False,
        use_fp8_w8a8,
        use_int8_w8a16,
        w1_scale,
        w2_scale,
        a1_scale,
        a2_scale,
        layout=layout,
        alpha=alpha,
        limit=limit,
    )


def outplace_fused_experts_impl_fake(
    hidden_states: torch.Tensor,
    w1: torch.Tensor,
    w2: torch.Tensor,
    # optional bias for w1 and w2
    w1_bias: Optional[torch.Tensor],
    w2_bias: Optional[torch.Tensor],
    topk_weights: torch.Tensor,
    topk_ids: torch.Tensor,
    use_fp8_w8a8: bool = False,
    use_int8_w8a16: bool = False,
    w1_scale: Optional[torch.Tensor] = None,
    w2_scale: Optional[torch.Tensor] = None,
    a1_scale: Optional[torch.Tensor] = None,
    a2_scale: Optional[torch.Tensor] = None,
    layout: str = "blocked",
    alpha: Optional[float] = None,
    limit: Optional[float] = None,
) -> None:
    return torch.empty_like(hidden_states)


direct_register_custom_op(
    "outplace_fused_experts_impl",
    outplace_fused_experts_impl,
    [],
    outplace_fused_experts_impl_fake,
)


def fused_experts(
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
    # optional bias for w1 and w2
    w1_bias: Optional[torch.Tensor] = None,
    w2_bias: Optional[torch.Tensor] = None,
    layout: str = "blocked",
    alpha: Optional[float] = None,
    limit: Optional[float] = None,
):
    if inplace:
        torch.ops.lightllm.inplace_fused_experts_impl(
            hidden_states,
            w1,
            w2,
            w1_bias,
            w2_bias,
            topk_weights,
            topk_ids,
            use_fp8_w8a8,
            use_int8_w8a16,
            w1_scale,
            w2_scale,
            a1_scale,
            a2_scale,
            layout=layout,
            alpha=alpha,
            limit=limit,
        )
        return hidden_states
    else:
        return torch.ops.lightllm.outplace_fused_experts_impl(
            hidden_states,
            w1,
            w2,
            w1_bias,
            w2_bias,
            topk_weights,
            topk_ids,
            use_fp8_w8a8,
            use_int8_w8a16,
            w1_scale,
            w2_scale,
            a1_scale,
            a2_scale,
            layout=layout,
            alpha=alpha,
            limit=limit,
        )
