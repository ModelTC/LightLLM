import torch

import triton
import triton.language as tl


@triton.jit
def _trans_topk_index_to_mem_index(
    topk_index,
    topk_index_stride_b,
    topk_index_stride_k,
    ragged_start_index,
    ragged_mem_index,
    topk_mem_index,
    topk_mem_index_stride_b,
    topk_mem_index_stride_k,
    topk_width,
    BLOCK_DMODEL: tl.constexpr,
):
    cur_index = tl.program_id(0)
    offs_d = tl.arange(0, BLOCK_DMODEL)
    mask = offs_d < topk_width
    topk_index_ptrs = topk_index + cur_index * topk_index_stride_b + offs_d * topk_index_stride_k
    topk_indices = tl.load(topk_index_ptrs, mask=mask, other=-1)
    ragged_start = tl.load(ragged_start_index + cur_index)
    topk_indices = tl.where(topk_indices != -1, topk_indices + ragged_start, -1)
    tl.store(topk_index_ptrs, topk_indices, mask=mask)

    dest_mem_index = ragged_mem_index + topk_indices
    mem_index = tl.load(dest_mem_index, mask=mask & (topk_indices != -1), other=-1)
    tl.store(
        topk_mem_index + cur_index * topk_mem_index_stride_b + offs_d * topk_mem_index_stride_k,
        mem_index,
        mask=mask,
    )


@torch.no_grad()
def trans_topk_index_to_mem_index(
    topk_index: torch.Tensor,
    ragged_start_index: torch.Tensor,
    ragged_mem_index: torch.Tensor,
):
    assert ragged_start_index.shape == (topk_index.shape[0],)

    grid = (topk_index.shape[0],)

    topk_mem_index = torch.empty_like(topk_index)

    _trans_topk_index_to_mem_index[grid](
        topk_index=topk_index,
        topk_index_stride_b=topk_index.stride(0),
        topk_index_stride_k=topk_index.stride(1),
        ragged_start_index=ragged_start_index,
        ragged_mem_index=ragged_mem_index,
        topk_mem_index=topk_mem_index,
        topk_mem_index_stride_b=topk_mem_index.stride(0),
        topk_mem_index_stride_k=topk_mem_index.stride(1),
        topk_width=topk_index.shape[1],
        BLOCK_DMODEL=triton.next_power_of_2(topk_index.shape[1]),
        num_warps=8,
    )
    return topk_mem_index
