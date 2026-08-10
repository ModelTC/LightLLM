import torch
import triton
import triton.language as tl


@triton.jit
def _scatter_swa_slots_kernel(
    mem_indexes_ptr,
    mem_index_stride,
    previous_full_slot_source_ptr,
    full_to_swa_ptr,
    swa_page_live_count_ptr,
    chunk_meta_ptr,
    new_chunk_count,
    PAGE_SIZE: tl.constexpr,
):
    chunk_idx = tl.program_id(0)
    meta = chunk_meta_ptr + chunk_idx * 4
    mem_offset = tl.load(meta)
    page_offset = tl.load(meta + 1)
    chunk_len = tl.load(meta + 2)
    page_or_previous_offset = tl.load(meta + 3)

    is_new_page = chunk_idx < new_chunk_count
    previous_full_slot_offset = tl.where(is_new_page, 0, page_or_previous_offset)
    prev_full_slot = tl.load(
        previous_full_slot_source_ptr + previous_full_slot_offset,
        mask=~is_new_page,
        other=0,
    ).to(tl.int64)
    prev_swa_slot = tl.load(
        full_to_swa_ptr + prev_full_slot,
        mask=~is_new_page,
        other=0,
    )
    page_idx = tl.where(is_new_page, page_or_previous_offset, prev_swa_slot // PAGE_SIZE)

    offsets = tl.arange(0, PAGE_SIZE)
    mask = offsets < chunk_len
    full_slots = tl.load(
        mem_indexes_ptr + (mem_offset + offsets) * mem_index_stride,
        mask=mask,
    ).to(tl.int64)
    swa_slots = page_idx * PAGE_SIZE + page_offset + offsets
    tl.store(full_to_swa_ptr + full_slots, swa_slots, mask=mask)
    tl.atomic_add(swa_page_live_count_ptr + page_idx, chunk_len.to(tl.int32))


def scatter_swa_slots(
    mem_indexes: torch.Tensor,
    previous_full_slot_source: torch.Tensor,
    full_to_swa_indexs: torch.Tensor,
    swa_page_live_count: torch.Tensor,
    chunk_meta: torch.Tensor,
    new_chunk_count: int,
    page_size: int,
) -> None:
    """Publish packed SWA page chunks and update each touched page once.

    The first ``new_chunk_count`` metadata rows end with a newly allocated physical page.
    Remaining rows end with the flat offset of the previous full slot that owns the page.
    """
    _scatter_swa_slots_kernel[(chunk_meta.shape[0],)](
        mem_indexes,
        mem_indexes.stride(0),
        previous_full_slot_source,
        full_to_swa_indexs,
        swa_page_live_count,
        chunk_meta,
        new_chunk_count,
        PAGE_SIZE=page_size,
        num_warps=4,
    )
    return
