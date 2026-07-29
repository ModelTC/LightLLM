import torch
import triton
import triton.language as tl

from .grouped_fused_moe import FFN_MOE_CHUNK_SIZE, moe_align2, moe_align_fused
from .moe_silu_and_mul import silu_and_mul_fwd
from .moe_sum_reduce import moe_sum_reduce


@triton.jit
def _grouped_matmul_mxfp4_kernel(
    mblocks_to_tuple_info,
    mblocks_stride,
    token_ptr,
    token_stride_0,
    weights_ptr,
    weight_stride_0,
    weight_stride_1,
    weight_stride_2,
    weight_scale_ptr,
    scale_stride_0,
    scale_stride_1,
    scale_stride_2,
    expert_to_weights_ptr,
    expert_to_weights_stride_0,
    expert_to_token_num,
    expert_to_token_index,
    expert_to_token_index_stride_0,
    out_ptr,
    out_stride_0,
    logical_k,
    n,
    topk_num,
    m_block_num,
    n_block_num,
    compute_type: tl.constexpr,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr,
    MUL_ROUTED_WEIGHT: tl.constexpr,
):
    pid = tl.program_id(0)
    num_pid_in_group = GROUP_SIZE_M * n_block_num
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_SIZE_M
    group_size_m = min(m_block_num - first_pid_m, GROUP_SIZE_M)
    in_group_index = pid % num_pid_in_group
    pid_m = first_pid_m + (in_group_index % group_size_m)
    pid_n = in_group_index // group_size_m

    expert_id = tl.load(mblocks_to_tuple_info + pid_m * mblocks_stride)
    if expert_id == -1:
        return

    tile_m_idx = tl.load(mblocks_to_tuple_info + pid_m * mblocks_stride + 1)
    cur_m = tl.load(expert_to_token_num + expert_id)
    offs_am = tile_m_idx * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    token_mask = offs_am < cur_m
    token_indices = tl.load(
        expert_to_token_index + expert_id * expert_to_token_index_stride_0 + offs_am,
        mask=token_mask,
        other=0,
    )
    offs_bn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    offs_k = tl.arange(0, BLOCK_SIZE_K)
    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)

    for k_start in range(0, logical_k, BLOCK_SIZE_K):
        logical_k_offsets = k_start + offs_k
        k_mask = logical_k_offsets < logical_k
        a = tl.load(
            token_ptr + (token_indices // topk_num)[:, None] * token_stride_0 + logical_k_offsets[None, :],
            mask=token_mask[:, None] & k_mask[None, :],
            other=0.0,
        )

        packed = tl.load(
            weights_ptr
            + expert_id * weight_stride_0
            + offs_bn[None, :] * weight_stride_1
            + (logical_k_offsets[:, None] // 2) * weight_stride_2,
            mask=k_mask[:, None] & (offs_bn[None, :] < n),
            other=0,
        )
        code = (packed >> ((logical_k_offsets[:, None] & 1) * 4)) & 0xF
        magnitude_code = code & 0x7
        magnitude = tl.where(
            magnitude_code <= 4,
            magnitude_code.to(tl.float32) * 0.5,
            tl.where(
                magnitude_code == 5,
                3.0,
                tl.where(magnitude_code == 6, 4.0, 6.0),
            ),
        )
        value = tl.where((code & 0x8) != 0, -magnitude, magnitude)

        scale_exp = tl.load(
            weight_scale_ptr
            + expert_id * scale_stride_0
            + offs_bn[None, :] * scale_stride_1
            + (logical_k_offsets[:, None] // 32) * scale_stride_2,
            mask=k_mask[:, None] & (offs_bn[None, :] < n),
            other=127,
        )
        scale = tl.exp2(scale_exp.to(tl.float32) - 127.0)
        b = (value * scale).to(compute_type)
        accumulator += tl.dot(a, b)

    if MUL_ROUTED_WEIGHT:
        routed_weight = tl.load(
            expert_to_weights_ptr + expert_id * expert_to_weights_stride_0 + offs_am,
            mask=token_mask,
            other=0.0,
        )
        accumulator *= routed_weight[:, None]

    tl.store(
        out_ptr + token_indices[:, None] * out_stride_0 + offs_bn[None, :],
        accumulator.to(compute_type),
        mask=token_mask[:, None] & (offs_bn[None, :] < n),
    )


def grouped_matmul_mxfp4(
    token_num_mul_topk_num,
    token_inputs,
    expert_to_token_num,
    expert_to_token_index,
    expert_to_weights,
    expert_weights,
    expert_weight_scales,
    topk_num,
    out,
    mul_routed_weight,
    reused_mblock_infos=None,
):
    expert_num, n, packed_k = expert_weights.shape
    logical_k = packed_k * 2
    assert token_inputs.shape[1] == logical_k
    assert expert_weight_scales.shape == (expert_num, n, logical_k // 32)
    assert expert_weights.dtype == torch.uint8
    assert expert_weight_scales.dtype == torch.uint8
    assert expert_weights.is_contiguous()
    assert expert_weight_scales.is_contiguous()

    block_m, block_n, block_k = 16, 32, 32
    if reused_mblock_infos is None:
        mblocks = moe_align2(token_num_mul_topk_num, expert_to_token_num, block_m)
    else:
        mblocks, reused_block_m = reused_mblock_infos
        if reused_block_m != block_m:
            mblocks = moe_align2(token_num_mul_topk_num, expert_to_token_num, block_m)

    n_block_num = triton.cdiv(n, block_n)
    grid = (n_block_num * mblocks.shape[0],)
    compute_type = tl.bfloat16 if out.dtype == torch.bfloat16 else tl.float16
    _grouped_matmul_mxfp4_kernel[grid](
        mblocks,
        mblocks.stride(0),
        token_inputs,
        token_inputs.stride(0),
        expert_weights,
        expert_weights.stride(0),
        expert_weights.stride(1),
        expert_weights.stride(2),
        expert_weight_scales,
        expert_weight_scales.stride(0),
        expert_weight_scales.stride(1),
        expert_weight_scales.stride(2),
        expert_to_weights,
        expert_to_weights.stride(0),
        expert_to_token_num,
        expert_to_token_index,
        expert_to_token_index.stride(0),
        out,
        out.stride(0),
        logical_k,
        n,
        topk_num,
        mblocks.shape[0],
        n_block_num,
        compute_type=compute_type,
        BLOCK_SIZE_M=block_m,
        BLOCK_SIZE_N=block_n,
        BLOCK_SIZE_K=block_k,
        GROUP_SIZE_M=1,
        MUL_ROUTED_WEIGHT=mul_routed_weight,
        num_warps=4,
        num_stages=1,
    )
    return mblocks, block_m


def fused_experts_mxfp4(
    hidden_states,
    w1,
    w2,
    topk_weights,
    topk_ids,
    w1_scale,
    w2_scale,
    activation="silu",
    activation_situ_beta=None,
    activation_situ_linear_beta=None,
):
    assert hidden_states.shape[1] == w1.shape[2] * 2
    assert w1.shape[1] == w2.shape[2] * 4
    assert topk_weights.shape == topk_ids.shape
    num_tokens = hidden_states.shape[0]
    expert_num, fused_intermediate, _ = w1.shape
    topk_num = topk_ids.shape[1]
    chunk_size = FFN_MOE_CHUNK_SIZE
    max_tokens = min(num_tokens, chunk_size)

    cache1 = torch.empty(
        (max_tokens, topk_num, fused_intermediate),
        dtype=hidden_states.dtype,
        device=hidden_states.device,
    )
    cache2 = torch.empty(
        (max_tokens, topk_num, fused_intermediate // 2),
        dtype=hidden_states.dtype,
        device=hidden_states.device,
    )
    cache3 = torch.empty(
        (max_tokens, topk_num, w2.shape[1]),
        dtype=hidden_states.dtype,
        device=hidden_states.device,
    )

    for chunk in range(triton.cdiv(num_tokens, chunk_size)):
        begin = chunk * chunk_size
        end = min(begin + chunk_size, num_tokens)
        token_count = end - begin
        x = hidden_states[begin:end]
        ids = topk_ids[begin:end]
        weights = topk_weights[begin:end]
        expert_to_tokens = torch.empty(
            (expert_num, token_count * topk_num), dtype=torch.int32, device=hidden_states.device
        )
        expert_to_weights = torch.empty(
            (expert_num, token_count * topk_num), dtype=torch.float32, device=hidden_states.device
        )
        expert_token_num = torch.empty((expert_num,), dtype=torch.int32, device=hidden_states.device)
        moe_align_fused(expert_to_tokens, expert_to_weights, expert_token_num, ids, weights)

        first_gemm = grouped_matmul_mxfp4(
            ids.numel(),
            x,
            expert_token_num,
            expert_to_tokens,
            expert_to_weights,
            w1,
            w1_scale,
            topk_num,
            cache1[:token_count].view(-1, fused_intermediate),
            False,
        )
        silu_and_mul_fwd(
            cache1[:token_count].view(-1, fused_intermediate),
            cache2[:token_count].view(-1, fused_intermediate // 2),
            activation=activation,
            activation_situ_beta=activation_situ_beta,
            activation_situ_linear_beta=activation_situ_linear_beta,
        )
        grouped_matmul_mxfp4(
            ids.numel(),
            cache2[:token_count].view(-1, fused_intermediate // 2),
            expert_token_num,
            expert_to_tokens,
            expert_to_weights,
            w2,
            w2_scale,
            1,
            cache3[:token_count].view(-1, w2.shape[1]),
            True,
            reused_mblock_infos=first_gemm,
        )
        moe_sum_reduce(cache3[:token_count], hidden_states[begin:end])

    return hidden_states
