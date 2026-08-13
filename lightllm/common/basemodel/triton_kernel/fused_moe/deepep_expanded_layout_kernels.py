import torch
import triton
import triton.language as tl


@triton.jit
def _ep_build_m_indices_kernel(
    num_unaligned_recv_tokens_per_expert,
    m_indices,
    padding_mask,
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
    tokens_per_expert = tl.cdiv(tokens_per_expert, BLOCK_E) * BLOCK_E
    cur_expert_start = tl.sum(tl.where(offset_cumsum < cur_expert, tokens_per_expert, 0))
    cur_expert_token_num = tl.load(num_unaligned_recv_tokens_per_expert + cur_expert)
    cur_expert_aligned_token_num = tl.cdiv(cur_expert_token_num, BLOCK_E) * BLOCK_E

    m_indices_start_ptr = m_indices + cur_expert_start
    padding_mask_start_ptr = padding_mask + cur_expert_start
    off_expert = tl.arange(0, BLOCK_E)

    for start_m in tl.range(0, cur_expert_aligned_token_num, BLOCK_E, num_stages=4):
        tl.store(
            m_indices_start_ptr + start_m + off_expert,
            cur_expert,
        )
        tl.store(
            padding_mask_start_ptr + start_m + off_expert,
            tl.where(start_m + off_expert >= cur_expert_token_num, 1, 0),
        )


@torch.no_grad()
def ep_build_m_indices(
    num_unaligned_recv_tokens_per_expert: torch.Tensor,  # [num_local_experts]
    m_indices: torch.Tensor,  # [num_expanded_tokens]
    padding_mask: torch.Tensor,  # [num_expanded_tokens]
    expert_alignment: int,
):
    """Build the aligned expert layout used by contiguous grouped GEMM.

    Each expert's actual token count is rounded up to ``expert_alignment``.
    ``m_indices`` is filled in-place with the owning expert ID for every real
    and padding row. The alignment must match the value used by DeepEP
    dispatch.

    ``padding_mask`` is filled in-place with ``1`` for alignment-padding rows
    and ``0`` for real token rows.
    """
    assert expert_alignment >= 8, "expert_alignment must be at least the zero-padding BLOCK_M (8)"
    assert triton.next_power_of_2(expert_alignment) == expert_alignment, "expert_alignment must be a power of two"
    num_experts = num_unaligned_recv_tokens_per_expert.shape[0]
    assert m_indices.shape[0] % expert_alignment == 0
    assert padding_mask.dtype == torch.int32 and padding_mask.shape == m_indices.shape

    _ep_build_m_indices_kernel[(num_experts,)](
        num_unaligned_recv_tokens_per_expert,
        m_indices,
        padding_mask,
        num_experts=num_experts,
        num_warps=8,
        BLOCK_E=expert_alignment,
        BLOCK_EXPERT_NUM=triton.next_power_of_2(num_experts),
    )


@triton.jit
def _ep_zero_padding_kernel(
    recv_x,
    recv_x_stride_m,
    recv_x_stride_k,
    recv_x_scale,
    recv_x_scale_stride_m,
    recv_x_scale_stride_k,
    recv_topk_weights,
    padding_mask,
    hidden_size: tl.constexpr,
    scale_hidden_size: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_K: tl.constexpr,
    BLOCK_SCALE_K: tl.constexpr,
):
    row_block_id = tl.program_id(0)
    hidden_block_id = tl.program_id(1)
    row_offsets = row_block_id * BLOCK_M + tl.arange(0, BLOCK_M)
    row_mask = tl.load(padding_mask + row_offsets) == 1
    row_offsets = row_offsets.to(tl.int64)

    hidden_offsets = hidden_block_id * BLOCK_K + tl.arange(0, BLOCK_K)
    hidden_mask = hidden_offsets < hidden_size
    x_ptrs = recv_x + row_offsets[:, None] * recv_x_stride_m + hidden_offsets[None, :] * recv_x_stride_k
    tl.store(x_ptrs, 0.0, mask=row_mask[:, None] & hidden_mask[None, :])
    if hidden_block_id == 0:
        scale_offsets = tl.arange(0, BLOCK_SCALE_K)
        scale_mask = scale_offsets < scale_hidden_size
        scale_ptrs = (
            recv_x_scale + row_offsets[:, None] * recv_x_scale_stride_m + scale_offsets[None, :] * recv_x_scale_stride_k
        )
        tl.store(scale_ptrs, 0.0, mask=row_mask[:, None] & scale_mask[None, :])
        tl.store(recv_topk_weights + row_offsets, 0.0, mask=row_mask)


@torch.no_grad()
def ep_zero_padding(
    recv_x: torch.Tensor,  # [num_expanded_tokens, hidden_size]
    recv_x_scale: torch.Tensor,  # [num_expanded_tokens, scale_hidden_size]
    recv_topk_weights: torch.Tensor,  # [num_expanded_tokens]
    padding_mask: torch.Tensor,  # [num_expanded_tokens], 1 for padding rows
):
    """Zero the alignment-padding rows in DeepEP's expanded receive layout.

    Rows marked by ``padding_mask`` are cleared in-place in the FP8
    activations, activation scales, and routing weights. ``recv_x_scale`` may
    use a column-major physical layout; its logical shape remains
    ``[num_expanded_tokens, scale_hidden_size]``.
    """
    assert padding_mask.dtype == torch.int32 and padding_mask.shape == recv_topk_weights.shape
    block_m = 8
    block_k = 256
    assert padding_mask.shape[0] % block_m == 0, "padding_mask rows must be divisible by BLOCK_M (8)"
    scale_hidden_size = recv_x_scale.shape[1]
    grid = (
        triton.cdiv(padding_mask.shape[0], block_m),
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
        padding_mask,
        hidden_size=recv_x.shape[1],
        scale_hidden_size=scale_hidden_size,
        BLOCK_M=block_m,
        BLOCK_K=block_k,
        BLOCK_SCALE_K=triton.next_power_of_2(scale_hidden_size),
        num_warps=4,
    )


# TODO: 当前实现会为每个 chunk 重新遍历所有接收 token 的 top-k metadata，并反复
# 读取、累加和写回长期保留的 gather_out，仍有较大的性能提升空间。后续可以考虑
# 预先构建 chunk-local 的路由映射，减少无效 metadata 扫描和全局内存读写。
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
    hidden_size,
    TOPK: tl.constexpr,
    BLOCK_D: tl.constexpr,
    NEED_HIDDEN_MASK: tl.constexpr,
):
    hidden_block_id = tl.program_id(0)
    start_recv_token_id = tl.program_id(1)
    recv_token_grid_size = tl.num_programs(1)
    hidden_offsets = hidden_block_id * BLOCK_D + tl.arange(0, BLOCK_D)
    if NEED_HIDDEN_MASK:
        hidden_mask = hidden_offsets < hidden_size

    for recv_token_id in range(start_recv_token_id, total_recv_tokens, recv_token_grid_size):
        output_ptrs = output + recv_token_id * output_stride_m + hidden_offsets * output_stride_k
        if NEED_HIDDEN_MASK:
            accumulator = tl.load(output_ptrs, mask=hidden_mask, other=0.0).to(tl.float32)
        else:
            accumulator = tl.load(output_ptrs).to(tl.float32)
        for topk_id in range(TOPK):
            slot = tl.load(recv_src_metadata + recv_token_id * metadata_stride_m + (topk_id + 2) * metadata_stride_k)
            if slot >= chunk_start and slot < chunk_end:
                local_row = (slot - chunk_start).to(tl.int64)
                chunk_ptrs = chunk + local_row * chunk_stride_m + hidden_offsets * chunk_stride_k
                if NEED_HIDDEN_MASK:
                    value = tl.load(chunk_ptrs, mask=hidden_mask, other=0.0)
                else:
                    value = tl.load(chunk_ptrs)
                weight = tl.load(weights + slot)
                accumulator += value.to(tl.float32) * weight
        if NEED_HIDDEN_MASK:
            tl.store(output_ptrs, accumulator, mask=hidden_mask)
        else:
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
    hidden_size = output.shape[1]
    assert chunk.shape[1] == hidden_size
    grid = (triton.cdiv(output.shape[1], block_d), min(output.shape[0], 1024))
    _ep_gather_chunk_kernel[grid](
        total_recv_tokens=output.shape[0],
        chunk=chunk,
        chunk_stride_m=chunk.stride(0),
        chunk_stride_k=chunk.stride(1),
        chunk_start=chunk_start,
        chunk_end=chunk_start + chunk.shape[0],
        weights=weights,
        recv_src_metadata=recv_src_metadata,
        metadata_stride_m=recv_src_metadata.stride(0),
        metadata_stride_k=recv_src_metadata.stride(1),
        output=output,
        output_stride_m=output.stride(0),
        output_stride_k=output.stride(1),
        hidden_size=hidden_size,
        TOPK=topk,
        BLOCK_D=block_d,
        # 常见整块 hidden size 保持原来的无 mask kernel；仅尾块不完整时启用边界保护。
        NEED_HIDDEN_MASK=hidden_size % block_d != 0,
        num_warps=2,
    )


@triton.jit
def _ep_compact_metadata_kernel(
    recv_src_metadata,
    metadata_stride_m,
    metadata_stride_k,
    num_recv_tokens,
    TOPK: tl.constexpr,
    BLOCK_TOKEN: tl.constexpr,
    BLOCK_TOPK: tl.constexpr,
):
    token_offsets = tl.program_id(0) * BLOCK_TOKEN + tl.arange(0, BLOCK_TOKEN)
    topk_offsets = tl.arange(0, BLOCK_TOPK)
    metadata_offsets = token_offsets[:, None] * metadata_stride_m + (topk_offsets[None, :] + 2) * metadata_stride_k
    valid_token_mask = token_offsets < num_recv_tokens
    valid_topk_mask = topk_offsets < TOPK
    metadata_mask = valid_token_mask[:, None] & valid_topk_mask[None, :]

    # 每个 token 只保留其在稠密输出中的同序行号，其余 top-k 位置全部置为无效。
    slots = tl.where(topk_offsets[None, :] == 0, token_offsets[:, None], -1)
    tl.store(recv_src_metadata + metadata_offsets, slots, mask=metadata_mask)


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
    block_token = 128
    grid = (triton.cdiv(recv_src_metadata.shape[0], block_token),)
    _ep_compact_metadata_kernel[grid](
        recv_src_metadata=recv_src_metadata,
        metadata_stride_m=recv_src_metadata.stride(0),
        metadata_stride_k=recv_src_metadata.stride(1),
        num_recv_tokens=recv_src_metadata.shape[0],
        TOPK=topk,
        BLOCK_TOKEN=block_token,
        BLOCK_TOPK=triton.next_power_of_2(topk),
        num_warps=4,
    )
