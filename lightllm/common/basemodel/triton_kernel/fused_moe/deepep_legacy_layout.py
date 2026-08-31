import torch
import triton
import triton.language as tl


@triton.jit
def _ep_scatter_offsets(
    padded_tokens_per_expert,
    valid_tokens_per_expert,
    expert_start_loc,
    m_indices,
    num_experts: tl.constexpr,
    block_experts: tl.constexpr,
    block_tokens: tl.constexpr,
):
    expert_id = tl.program_id(0)
    expert_offsets = tl.arange(0, block_experts)
    counts = tl.load(
        padded_tokens_per_expert + expert_offsets,
        mask=expert_offsets < num_experts,
        other=0,
    )
    starts = tl.cumsum(counts) - counts
    tl.store(expert_start_loc + expert_offsets, starts, mask=expert_offsets < num_experts)

    expert_start = tl.load(expert_start_loc + expert_id)
    padded_count = tl.load(padded_tokens_per_expert + expert_id)
    valid_count = tl.load(valid_tokens_per_expert + expert_id)
    offsets = tl.arange(0, block_tokens)
    for token_start in tl.range(0, padded_count, block_tokens, num_stages=4):
        token_offsets = token_start + offsets
        tl.store(
            m_indices + expert_start + token_offsets,
            tl.where(token_offsets < valid_count, expert_id, -1),
        )


@triton.jit
def _ep_scatter_tokens(
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
    hidden_size: tl.constexpr,
    hidden_size_pad: tl.constexpr,
    scale_hidden_size: tl.constexpr,
    scale_hidden_size_pad: tl.constexpr,
):
    start_token_id = tl.program_id(0)
    grid_size = tl.num_programs(0)
    hidden_offsets = tl.arange(0, hidden_size_pad)
    hidden_mask = hidden_offsets < hidden_size
    scale_offsets = tl.arange(0, scale_hidden_size_pad)
    scale_mask = scale_offsets < scale_hidden_size

    for token_id_int32 in range(start_token_id, total_token_num, grid_size):
        token_id = token_id_int32.to(tl.int64)
        token = tl.load(
            recv_x + token_id * recv_x_stride0 + hidden_offsets * recv_x_stride1,
            mask=hidden_mask,
        )
        token_scale = tl.load(
            recv_x_scale + token_id * recv_x_scale_stride0 + scale_offsets * recv_x_scale_stride1,
            mask=scale_mask,
        )
        for topk_offset_int32 in tl.range(0, topk_num, 1, num_stages=4):
            topk_offset = topk_offset_int32.to(tl.int64)
            expert_id = tl.load(recv_topk + token_id * recv_topk_stride0 + topk_offset * recv_topk_stride1)
            if expert_id >= 0:
                destination_int32 = tl.atomic_add(expert_start_loc + expert_id, 1)
                destination = destination_int32.to(tl.int64)
                tl.store(
                    output_index + token_id * output_index_stride0 + topk_offset * output_index_stride1,
                    destination_int32,
                )
                tl.store(
                    output_tensor + destination * output_tensor_stride0 + hidden_offsets * output_tensor_stride1,
                    token,
                    mask=hidden_mask,
                )
                tl.store(
                    output_tensor_scale
                    + destination * output_tensor_scale_stride0
                    + scale_offsets * output_tensor_scale_stride1,
                    token_scale,
                    mask=scale_mask,
                )


@torch.no_grad()
def ep_scatter(
    recv_x: torch.Tensor,
    recv_x_scale: torch.Tensor,
    recv_topk: torch.Tensor,
    padded_tokens_per_expert: torch.Tensor,
    valid_tokens_per_expert: torch.Tensor,
    expert_start_loc: torch.Tensor,
    output_tensor: torch.Tensor,
    output_tensor_scale: torch.Tensor,
    m_indices: torch.Tensor,
    output_index: torch.Tensor,
):
    block_tokens = 128
    num_experts = padded_tokens_per_expert.shape[0]
    hidden_size = recv_x.shape[1]
    scale_hidden_size = recv_x_scale.shape[1]
    assert m_indices.shape[0] % block_tokens == 0

    _ep_scatter_offsets[(num_experts,)](
        padded_tokens_per_expert,
        valid_tokens_per_expert,
        expert_start_loc,
        m_indices,
        num_experts=num_experts,
        block_experts=triton.next_power_of_2(num_experts),
        block_tokens=block_tokens,
        num_warps=8,
    )
    _ep_scatter_tokens[(min(recv_topk.shape[0], 8192),)](
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
        hidden_size=hidden_size,
        hidden_size_pad=triton.next_power_of_2(hidden_size),
        scale_hidden_size=scale_hidden_size,
        scale_hidden_size_pad=triton.next_power_of_2(scale_hidden_size),
        num_warps=8,
    )


@triton.jit
def _ep_gather_kernel(
    total_token_num,
    input_tensor,
    input_tensor_stride0,
    input_tensor_stride1,
    recv_topk_ids,
    recv_topk_ids_stride0,
    recv_topk_ids_stride1,
    recv_topk_weights,
    recv_topk_weights_stride0,
    recv_topk_weights_stride1,
    input_index,
    input_index_stride0,
    input_index_stride1,
    output_tensor,
    output_tensor_stride0,
    output_tensor_stride1,
    topk_num: tl.constexpr,
    block_hidden: tl.constexpr,
):
    hidden_block_int32 = tl.program_id(0)
    hidden_block = hidden_block_int32.to(tl.int64)
    start_token_int32 = tl.program_id(1)
    grid_size = tl.num_programs(1)
    hidden_offsets = tl.arange(0, block_hidden)

    for token_int32 in range(start_token_int32, total_token_num, grid_size):
        token = token_int32.to(tl.int64)
        accumulator = tl.zeros([block_hidden], dtype=tl.float32)
        for topk_offset_int32 in range(0, topk_num):
            topk_offset = topk_offset_int32.to(tl.int64)
            expert_id = tl.load(recv_topk_ids + token * recv_topk_ids_stride0 + topk_offset * recv_topk_ids_stride1)
            if expert_id >= 0:
                source_int32 = tl.load(input_index + token * input_index_stride0 + topk_offset * input_index_stride1)
                source = source_int32.to(tl.int64)
                weight = tl.load(
                    recv_topk_weights + token * recv_topk_weights_stride0 + topk_offset * recv_topk_weights_stride1
                )
                value = tl.load(
                    input_tensor
                    + source * input_tensor_stride0
                    + hidden_block * block_hidden
                    + hidden_offsets * input_tensor_stride1
                )
                accumulator += value.to(tl.float32) * weight
        tl.store(
            output_tensor
            + token * output_tensor_stride0
            + hidden_block * block_hidden
            + hidden_offsets * output_tensor_stride1,
            accumulator.to(output_tensor.dtype.element_ty),
        )


@torch.no_grad()
def ep_gather(
    input_tensor: torch.Tensor,
    recv_topk_ids: torch.Tensor,
    recv_topk_weights: torch.Tensor,
    input_index: torch.Tensor,
    output_tensor: torch.Tensor,
):
    hidden_size = input_tensor.shape[1]
    block_hidden = 1024 if hidden_size % 1024 == 0 else 128
    assert hidden_size % block_hidden == 0
    grid = (triton.cdiv(hidden_size, block_hidden), min(output_tensor.shape[0], 1024))
    _ep_gather_kernel[grid](
        output_tensor.shape[0],
        input_tensor,
        input_tensor.stride(0),
        input_tensor.stride(1),
        recv_topk_ids,
        recv_topk_ids.stride(0),
        recv_topk_ids.stride(1),
        recv_topk_weights,
        recv_topk_weights.stride(0),
        recv_topk_weights.stride(1),
        input_index,
        input_index.stride(0),
        input_index.stride(1),
        output_tensor,
        output_tensor.stride(0),
        output_tensor.stride(1),
        topk_num=recv_topk_ids.shape[1],
        block_hidden=block_hidden,
        num_warps=2,
    )
