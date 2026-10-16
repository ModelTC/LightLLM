"""Stream-ordered copies for pinned checkpoint pages."""

import torch
import triton
import triton.language as tl


@triton.jit
def _copy_checkpoint_pages(
    src_state,
    dst_state,
    src_mem_index,
    dst_mem_index,
    page_readies,
    src_stride_s,
    dst_stride_s,
    state_dim,
    page_num,
    SKIP_READY: tl.constexpr,
    BLOCK: tl.constexpr,
):
    for page in range(page_num):
        src_id = tl.load(src_mem_index + page).to(tl.int64)
        dst_id = tl.load(dst_mem_index + page).to(tl.int64)
        valid = (src_id >= 0) & (dst_id >= 0)
        if SKIP_READY:
            valid = valid & (tl.load(page_readies + page) == 0)
        if valid:
            for block in range(tl.program_id(0), tl.cdiv(state_dim, BLOCK), tl.num_programs(0)):
                offsets = block * BLOCK + tl.arange(0, BLOCK)
                data = tl.load(src_state + src_id * src_stride_s + offsets, offsets < state_dim, other=0)
                tl.store(dst_state + dst_id * dst_stride_s + offsets, data, offsets < state_dim, cache_modifier=".wt")


def copy_checkpoint_pages(sources, destinations, source_ids, destination_ids, page_readies=None):
    """Copy slot-indexed, contiguous rows; preserve each tensor's byte representation."""
    assert len(source_ids) == len(destination_ids)
    for source, destination in zip(sources, destinations):
        source = source.reshape(source.shape[0], -1).view(torch.uint8)
        destination = destination.reshape(destination.shape[0], -1).view(torch.uint8)
        assert source.shape[1] == destination.shape[1]
        _copy_checkpoint_pages[(12,)](
            source,
            destination,
            source_ids,
            destination_ids,
            page_readies,
            source.stride(0),
            destination.stride(0),
            source.shape[1],
            len(source_ids),
            SKIP_READY=page_readies is not None,
            BLOCK=4096,
        )
