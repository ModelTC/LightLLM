import torch
import triton
import triton.language as tl


@triton.jit
def _ep_build_m_indices_kernel(
    num_unaligned_recv_tokens_per_expert,
    expert_start_loc,
    m_indices,
    num_experts: tl.constexpr,
    BLOCK_E: tl.constexpr,
    BLOCK_EXPERT_NUM: tl.constexpr,
):
    cur_expert = tl.program_id(0)

    offset_cumsum = tl.arange(0, BLOCK_EXPERT_NUM)
    tokens_per_expert = tl.load(
        num_unaligned_recv_tokens_per_expert + offset_cumsum,
        mask=offset_cumsum < num_experts,
        other=0,
    )
    tokens_per_expert = (tokens_per_expert + BLOCK_E - 1) // BLOCK_E * BLOCK_E
    cur_expert_start = tl.sum(tl.where(offset_cumsum < cur_expert, tokens_per_expert, 0))
    cur_expert_token_num = tl.load(num_unaligned_recv_tokens_per_expert + cur_expert)
    cur_expert_token_num = (cur_expert_token_num + BLOCK_E - 1) // BLOCK_E * BLOCK_E
    tl.store(expert_start_loc + cur_expert, cur_expert_start)

    m_indices_start_ptr = m_indices + cur_expert_start
    off_expert = tl.arange(0, BLOCK_E)

    for start_m in tl.range(0, cur_expert_token_num, BLOCK_E, num_stages=4):
        tl.store(
            m_indices_start_ptr + start_m + off_expert,
            cur_expert,
        )


@torch.no_grad()
def ep_build_m_indices(
    num_unaligned_recv_tokens_per_expert: torch.Tensor,  # [num_local_experts]
    m_indices: torch.Tensor,  # [num_expanded_tokens]
):
    """Build the 128-aligned expert layout used by contiguous grouped GEMM.

    Each expert's actual token count is rounded up to 128. ``m_indices`` is
    filled in-place with the owning expert ID for every real and padding row.

    Returns:
        ``expert_start_loc`` with shape ``[num_local_experts]``. Each value is
        the expert's starting row in the expanded tensors.
    """
    block_e = 128
    num_experts = num_unaligned_recv_tokens_per_expert.shape[0]
    assert m_indices.shape[0] % block_e == 0

    expert_start_loc = torch.empty_like(num_unaligned_recv_tokens_per_expert)
    _ep_build_m_indices_kernel[(num_experts,)](
        num_unaligned_recv_tokens_per_expert,
        expert_start_loc,
        m_indices,
        num_experts=num_experts,
        num_warps=8,
        BLOCK_E=block_e,
        BLOCK_EXPERT_NUM=triton.next_power_of_2(num_experts),
    )
    return expert_start_loc


@triton.jit
def _ep_zero_padding_kernel(
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
def ep_zero_padding(
    recv_x: torch.Tensor,  # [num_expanded_tokens, hidden_size]
    recv_x_scale: torch.Tensor,  # [num_expanded_tokens, scale_hidden_size]
    recv_topk_weights: torch.Tensor,  # [num_expanded_tokens]
    num_unaligned_recv_tokens_per_expert: torch.Tensor,  # [num_local_experts]
    expert_start_loc: torch.Tensor,  # [num_local_experts]
):
    """Zero the alignment-padding rows in DeepEP's expanded receive layout.

    For every expert, rows from its actual token count up to its 128-aligned
    count are cleared in-place in the FP8 activations, activation scales, and
    routing weights. ``recv_x_scale`` may use a column-major physical layout;
    its logical shape remains ``[num_expanded_tokens, scale_hidden_size]``.
    """
    block_m = 8
    block_k = 256
    scale_hidden_size = recv_x_scale.shape[1]
    grid = (
        num_unaligned_recv_tokens_per_expert.shape[0],
        triton.cdiv(127, block_m),
        triton.cdiv(recv_x.shape[1], block_k),
    )
    _ep_zero_padding_kernel[grid](
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
def _ep_gather_chunk_kernel(
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
def ep_gather_chunk(
    chunk: torch.Tensor,  # [chunk_rows, hidden_size]
    chunk_start: int,  # scalar expanded-row offset
    weights: torch.Tensor,  # [num_expanded_tokens]
    recv_src_metadata: torch.Tensor,  # [num_recv_tokens, topk + 2]
    output: torch.Tensor,  # [num_recv_tokens, hidden_size]
):
    """Accumulate one expanded W2-output chunk into dense receive-token rows.

    The last ``topk`` columns of ``recv_src_metadata`` map each dense receive
    token to global expanded-row IDs. Entries covered by this chunk are read,
    multiplied by ``weights``, and accumulated in-place into ``output``. This
    allows multiple chunks to contribute to the same dense output tensor.
    """
    topk = recv_src_metadata.shape[1] - 2
    block_d = 1024
    assert chunk.shape[1] == output.shape[1] and output.shape[1] % block_d == 0
    grid = (triton.cdiv(output.shape[1], block_d), min(output.shape[0], 1024))
    _ep_gather_chunk_kernel[grid](
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
def _ep_compact_metadata_kernel(
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
def ep_compact_metadata(
    recv_src_metadata: torch.Tensor,  # [num_recv_tokens, topk + 2]
):
    """Rewrite expanded routing metadata for a pre-reduced dense tensor.

    The operation preserves the first two metadata columns and updates the
    final ``topk`` columns in-place to ``[recv_token_id, -1, ...]``. DeepEP
    combine can then read each already-reduced dense row exactly once.
    """
    topk = recv_src_metadata.shape[1] - 2
    if recv_src_metadata.shape[0] == 0:
        return
    _ep_compact_metadata_kernel[(recv_src_metadata.shape[0],)](
        recv_src_metadata,
        recv_src_metadata.stride(0),
        recv_src_metadata.stride(1),
        TOPK=topk,
        BLOCK_TOPK=triton.next_power_of_2(topk),
        num_warps=1,
    )
