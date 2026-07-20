import random
import torch
import torch.nn.functional as F
import triton
import triton.language as tl
from typing import Dict


@triton.jit
def _fwd_kernel_ep_scatter_1(
    num_recv_tokens_per_expert,
    expert_start_loc,
    m_indices,
    num_experts: tl.constexpr,
    BLOCK_E: tl.constexpr,
    BLOCK_EXPERT_NUM: tl.constexpr,
    ALIGN_COUNTS: tl.constexpr,
):
    cur_expert = tl.program_id(0)

    offset_cumsum = tl.arange(0, BLOCK_EXPERT_NUM)
    tokens_per_expert = tl.load(num_recv_tokens_per_expert + offset_cumsum, mask=offset_cumsum < num_experts, other=0)
    if ALIGN_COUNTS:
        tokens_per_expert = (tokens_per_expert + BLOCK_E - 1) // BLOCK_E * BLOCK_E
    cur_expert_start = tl.sum(tl.where(offset_cumsum < cur_expert, tokens_per_expert, 0))
    cur_expert_token_num = tl.load(num_recv_tokens_per_expert + cur_expert)
    if ALIGN_COUNTS:
        cur_expert_token_num = (cur_expert_token_num + BLOCK_E - 1) // BLOCK_E * BLOCK_E
    tl.store(expert_start_loc + cur_expert, cur_expert_start)

    m_indices_start_ptr = m_indices + cur_expert_start
    off_expert = tl.arange(0, BLOCK_E)

    for start_m in tl.range(0, cur_expert_token_num, BLOCK_E, num_stages=4):
        tl.store(
            m_indices_start_ptr + start_m + off_expert,
            cur_expert,
        )


@triton.jit
def _fwd_kernel_ep_scatter_2(
    total_token_num,
    expert_start_loc,
    recv_x,
    recv_x_stride0,
    recv_x_stride1,
    recv_x_scale,
    recv_x_scale_stride0,
    recv_x_scale_stride1,
    recv_topk,
    recv_topk_stride0,
    recv_topk_stride1,
    output_tensor,
    output_tensor_stride0,
    output_tensor_stride1,
    output_tensor_scale,
    output_tensor_scale_stride0,
    output_tensor_scale_stride1,
    output_index,
    output_index_stride0,
    output_index_stride1,
    topk_num: tl.constexpr,
    HIDDEN_SIZE: tl.constexpr,
    HIDDEN_SIZE_PAD: tl.constexpr,
    SCALE_HIDDEN_SIZE: tl.constexpr,
    SCALE_HIDDEN_SIZE_PAD: tl.constexpr,
):
    start_token_id = tl.program_id(0)
    grid_num = tl.num_programs(0)

    offset_in = tl.arange(0, HIDDEN_SIZE_PAD)
    mask = offset_in < HIDDEN_SIZE

    offset_in_s = tl.arange(0, SCALE_HIDDEN_SIZE_PAD)
    mask_s = offset_in_s < SCALE_HIDDEN_SIZE
    for token_id in range(start_token_id, total_token_num, grid_num):
        to_copy = tl.load(recv_x + token_id * recv_x_stride0 + offset_in, mask=mask)
        to_copy_s = tl.load(recv_x_scale + token_id * recv_x_scale_stride0 + offset_in_s, mask=mask_s)

        for topk_index in tl.range(0, topk_num, 1, num_stages=4):
            expert_id = tl.load(recv_topk + token_id * recv_topk_stride0 + topk_index)
            if expert_id >= 0:
                dest_token_index = tl.atomic_add(expert_start_loc + expert_id, 1)
                dest_token_index = dest_token_index.to(tl.int64)
                tl.store(output_index + token_id * output_index_stride0 + topk_index, dest_token_index)
                output_tensor_ptr = output_tensor + dest_token_index * output_tensor_stride0
                output_tensor_scale_ptr = output_tensor_scale + dest_token_index * output_tensor_scale_stride0
                tl.store(output_tensor_ptr + offset_in, to_copy, mask=mask)
                tl.store(output_tensor_scale_ptr + offset_in_s, to_copy_s, mask=mask_s)


@torch.no_grad()
def ep_scatter(
    recv_x: torch.Tensor,
    recv_x_scale: torch.Tensor,
    recv_topk: torch.Tensor,
    num_recv_tokens_per_expert: torch.Tensor,
    expert_start_loc: torch.Tensor,
    output_tensor: torch.Tensor,
    output_tensor_scale: torch.Tensor,
    m_indices: torch.Tensor,
    output_index: torch.Tensor,
):
    BLOCK_E = 128  # token num of per expert is aligned to 128
    BLOCK_D = 128  # block size of quantization
    num_warps = 8
    num_experts = num_recv_tokens_per_expert.shape[0]  # 获取num_recv_tokens_per_expert的元素个数
    hidden_size = recv_x.shape[1]
    # grid = (triton.cdiv(hidden_size, BLOCK_D), num_experts)
    grid = num_experts

    assert m_indices.shape[0] % BLOCK_E == 0

    _fwd_kernel_ep_scatter_1[(grid,)](
        num_recv_tokens_per_expert,
        expert_start_loc,
        m_indices,
        num_experts=num_experts,
        num_warps=num_warps,
        BLOCK_E=BLOCK_E,
        BLOCK_EXPERT_NUM=triton.next_power_of_2(num_experts),
        ALIGN_COUNTS=False,
    )

    grid = min(recv_topk.shape[0], 1024 * 8)

    _fwd_kernel_ep_scatter_2[(grid,)](
        recv_topk.shape[0],
        expert_start_loc,
        recv_x,
        recv_x.stride(0),
        recv_x.stride(1),
        recv_x_scale,
        recv_x_scale.stride(0),
        recv_x_scale.stride(1),
        recv_topk,
        recv_topk.stride(0),
        recv_topk.stride(1),
        output_tensor,
        output_tensor.stride(0),
        output_tensor.stride(1),
        output_tensor_scale,
        output_tensor_scale.stride(0),
        output_tensor_scale.stride(1),
        output_index,
        output_index.stride(0),
        output_index.stride(1),
        topk_num=recv_topk.shape[1],
        num_warps=num_warps,
        HIDDEN_SIZE=hidden_size,
        HIDDEN_SIZE_PAD=triton.next_power_of_2(hidden_size),
        SCALE_HIDDEN_SIZE=hidden_size // BLOCK_D,
        SCALE_HIDDEN_SIZE_PAD=triton.next_power_of_2(hidden_size // BLOCK_D),
    )
    return


@torch.no_grad()
def ep_fill_m_indices(
    num_unaligned_recv_tokens_per_expert: torch.Tensor,
    m_indices: torch.Tensor,
):
    """Build aligned expert offsets and DeepGEMM's expert index vector."""
    block_e = 128
    num_experts = num_unaligned_recv_tokens_per_expert.shape[0]
    assert m_indices.shape[0] % block_e == 0

    expert_start_loc = torch.empty_like(num_unaligned_recv_tokens_per_expert)
    _fwd_kernel_ep_scatter_1[(num_experts,)](
        num_unaligned_recv_tokens_per_expert,
        expert_start_loc,
        m_indices,
        num_experts=num_experts,
        num_warps=8,
        BLOCK_E=block_e,
        BLOCK_EXPERT_NUM=triton.next_power_of_2(num_experts),
        ALIGN_COUNTS=True,
    )
    return expert_start_loc


@triton.jit
def _zero_expanded_padding_kernel(
    recv_x,
    recv_x_stride_m,
    recv_x_stride_k,
    recv_x_scale,
    recv_x_scale_stride_m,
    recv_x_scale_stride_k,
    recv_topk_weights,
    num_unaligned_recv_tokens_per_expert,
    expert_start_loc,
    hidden_size: tl.constexpr,
    scale_hidden_size: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_K: tl.constexpr,
    BLOCK_SCALE_K: tl.constexpr,
):
    expert_id = tl.program_id(0)
    pad_block_id = tl.program_id(1)
    hidden_block_id = tl.program_id(2)
    expert_start = tl.load(expert_start_loc + expert_id)
    actual_count = tl.load(num_unaligned_recv_tokens_per_expert + expert_id)
    aligned_count = (actual_count + 127) // 128 * 128
    pad_offsets = pad_block_id * BLOCK_M + tl.arange(0, BLOCK_M)
    row_offsets = (expert_start + actual_count + pad_offsets).to(tl.int64)
    row_mask = pad_offsets < aligned_count - actual_count

    hidden_offsets = hidden_block_id * BLOCK_K + tl.arange(0, BLOCK_K)
    x_ptrs = recv_x + row_offsets[:, None] * recv_x_stride_m + hidden_offsets[None, :] * recv_x_stride_k
    tl.store(x_ptrs, 0.0, mask=row_mask[:, None] & (hidden_offsets[None, :] < hidden_size))
    if hidden_block_id == 0:
        scale_offsets = tl.arange(0, BLOCK_SCALE_K)
        scale_ptrs = (
            recv_x_scale + row_offsets[:, None] * recv_x_scale_stride_m + scale_offsets[None, :] * recv_x_scale_stride_k
        )
        tl.store(scale_ptrs, 0.0, mask=row_mask[:, None] & (scale_offsets[None, :] < scale_hidden_size))
        tl.store(recv_topk_weights + row_offsets, 0.0, mask=row_mask)


@torch.no_grad()
def ep_zero_expanded_padding(
    recv_x: torch.Tensor,
    recv_x_scale: torch.Tensor,
    recv_topk_weights: torch.Tensor,
    num_unaligned_recv_tokens_per_expert: torch.Tensor,
    expert_start_loc: torch.Tensor,
):
    block_m = 8
    block_k = 256
    scale_hidden_size = recv_x_scale.shape[1]
    grid = (
        num_unaligned_recv_tokens_per_expert.shape[0],
        triton.cdiv(127, block_m),
        triton.cdiv(recv_x.shape[1], block_k),
    )
    _zero_expanded_padding_kernel[grid](
        recv_x,
        recv_x.stride(0),
        recv_x.stride(1),
        recv_x_scale,
        recv_x_scale.stride(0),
        recv_x_scale.stride(1),
        recv_topk_weights,
        num_unaligned_recv_tokens_per_expert,
        expert_start_loc,
        hidden_size=recv_x.shape[1],
        scale_hidden_size=scale_hidden_size,
        BLOCK_M=block_m,
        BLOCK_K=block_k,
        BLOCK_SCALE_K=triton.next_power_of_2(scale_hidden_size),
        num_warps=4,
    )


@triton.jit
def _accumulate_expanded_chunk_kernel(
    total_recv_tokens,
    chunk,
    chunk_stride_m,
    chunk_stride_k,
    chunk_start,
    chunk_end,
    weights,
    recv_src_metadata,
    metadata_stride_m,
    metadata_stride_k,
    output,
    output_stride_m,
    output_stride_k,
    TOPK: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    hidden_block_id = tl.program_id(0)
    start_recv_token_id = tl.program_id(1)
    recv_token_grid_size = tl.num_programs(1)
    hidden_offsets = hidden_block_id * BLOCK_D + tl.arange(0, BLOCK_D)

    for recv_token_id in range(start_recv_token_id, total_recv_tokens, recv_token_grid_size):
        output_ptrs = output + recv_token_id * output_stride_m + hidden_offsets * output_stride_k
        accumulator = tl.load(output_ptrs).to(tl.float32)
        for topk_id in range(TOPK):
            slot = tl.load(recv_src_metadata + recv_token_id * metadata_stride_m + (topk_id + 2) * metadata_stride_k)
            if slot >= chunk_start and slot < chunk_end:
                local_row = (slot - chunk_start).to(tl.int64)
                value = tl.load(chunk + local_row * chunk_stride_m + hidden_offsets * chunk_stride_k)
                weight = tl.load(weights + slot)
                accumulator += value.to(tl.float32) * weight
        tl.store(output_ptrs, accumulator)


@torch.no_grad()
def ep_accumulate_expanded_chunk(
    chunk: torch.Tensor,
    chunk_start: int,
    weights: torch.Tensor,
    recv_src_metadata: torch.Tensor,
    output: torch.Tensor,
):
    """Accumulate one contiguous expanded W2 chunk into dense receive-token rows."""
    topk = recv_src_metadata.shape[1] - 2
    block_d = 1024
    assert chunk.shape[1] == output.shape[1] and output.shape[1] % block_d == 0
    grid = (triton.cdiv(output.shape[1], block_d), min(output.shape[0], 1024))
    _accumulate_expanded_chunk_kernel[grid](
        output.shape[0],
        chunk,
        chunk.stride(0),
        chunk.stride(1),
        chunk_start,
        chunk_start + chunk.shape[0],
        weights,
        recv_src_metadata,
        recv_src_metadata.stride(0),
        recv_src_metadata.stride(1),
        output,
        output.stride(0),
        output.stride(1),
        TOPK=topk,
        BLOCK_D=block_d,
        num_warps=2,
    )


@triton.jit
def _compact_expanded_metadata_kernel(
    recv_src_metadata,
    metadata_stride_m,
    metadata_stride_k,
    TOPK: tl.constexpr,
    BLOCK_TOPK: tl.constexpr,
):
    recv_token_id = tl.program_id(0)
    topk_id = tl.arange(0, BLOCK_TOPK)
    slot = tl.where(topk_id == 0, recv_token_id, -1)
    tl.store(
        recv_src_metadata + recv_token_id * metadata_stride_m + (topk_id + 2) * metadata_stride_k,
        slot,
        mask=topk_id < TOPK,
    )


@torch.no_grad()
def ep_compact_expanded_metadata(recv_src_metadata: torch.Tensor):
    """Point expanded combine metadata at pre-reduced dense token rows."""
    topk = recv_src_metadata.shape[1] - 2
    if recv_src_metadata.shape[0] == 0:
        return
    _compact_expanded_metadata_kernel[(recv_src_metadata.shape[0],)](
        recv_src_metadata,
        recv_src_metadata.stride(0),
        recv_src_metadata.stride(1),
        TOPK=topk,
        BLOCK_TOPK=triton.next_power_of_2(topk),
        num_warps=1,
    )


@triton.jit
def _fwd_kernel_ep_gather(
    total_token_num,
    input_tensor,
    input_tensor_stride0,
    input_tensor_stride1,
    recv_topk_ids,
    recv_topk_ids_stride0,
    recv_topk_ids_stride1,
    recv_topk_weight,
    recv_topk_weight_stride0,
    recv_topk_weight_stride1,
    input_index,
    input_index_stride0,
    input_index_stride1,
    output_tensor,
    output_tensor_stride0,
    output_tensor_stride1,
    topk_num: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    cur_block = tl.program_id(0)
    start_cur_token = tl.program_id(1)
    grid_num = tl.num_programs(1)

    for cur_token in range(start_cur_token, total_token_num, grid_num):
        off_d = tl.arange(0, BLOCK_D)
        accumulator = tl.zeros([BLOCK_D], dtype=tl.float32)
        for topk_index in range(0, topk_num):
            expert_id = tl.load(recv_topk_ids + cur_token * recv_topk_ids_stride0 + topk_index)
            if expert_id >= 0:
                source_token_index = tl.load(input_index + cur_token * input_index_stride0 + topk_index)
                acc_weight = tl.load(recv_topk_weight + cur_token * recv_topk_weight_stride0 + topk_index)
                tmp = tl.load(input_tensor + source_token_index * input_tensor_stride0 + cur_block * BLOCK_D + off_d)
                accumulator += tmp.to(tl.float32) * acc_weight

        tl.store(
            output_tensor + cur_token * output_tensor_stride0 + cur_block * BLOCK_D + off_d,
            accumulator.to(output_tensor.dtype.element_ty),
        )


@torch.no_grad()
def ep_gather(
    input_tensor: torch.Tensor,
    recv_topk_ids: torch.Tensor,
    recv_topk_weight: torch.Tensor,
    input_index: torch.Tensor,
    output_tensor: torch.Tensor,
):
    BLOCK_D = 1024  # block size of quantization
    num_warps = 2
    num_tokens = output_tensor.shape[0]
    hidden_size = input_tensor.shape[1]
    assert hidden_size % BLOCK_D == 0
    grid = (triton.cdiv(hidden_size, BLOCK_D), min(num_tokens, 1024))
    _fwd_kernel_ep_gather[grid](
        num_tokens,
        input_tensor,
        input_tensor.stride(0),
        input_tensor.stride(1),
        recv_topk_ids,
        recv_topk_ids.stride(0),
        recv_topk_ids.stride(1),
        recv_topk_weight,
        recv_topk_weight.stride(0),
        recv_topk_weight.stride(1),
        input_index,
        input_index.stride(0),
        input_index.stride(1),
        output_tensor,
        output_tensor.stride(0),
        output_tensor.stride(1),
        topk_num=recv_topk_ids.shape[1],
        num_warps=num_warps,
        BLOCK_D=BLOCK_D,
    )
    return
