import torch
import triton
import triton.language as tl


@triton.jit
def _build_dspark_swa_index_kernel(
    req_idx_ptr,
    pos_ptr,
    req_to_token_ptr,
    req_to_token_stride0,
    full_to_swa_ptr,
    scratch_pages_ptr,
    swa_index_ptr,
    swa_index_stride0,
    swa_length_ptr,
    swa_write_slot_ptr,
    HOLD_REQ_ID: tl.constexpr,
    HOLD_FULL_SLOT: tl.constexpr,
    HOLD_SWA_SLOT: tl.constexpr,
    WINDOW: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
    PAGE_SIZE: tl.constexpr,
    WIDTH: tl.constexpr,
    BLOCK_W: tl.constexpr,
):
    token_idx = tl.program_id(0)
    req_idx = tl.load(req_idx_ptr + token_idx).to(tl.int64)
    position = tl.load(pos_ptr + token_idx).to(tl.int64)
    is_hold = req_idx == HOLD_REQ_ID

    block_offset = token_idx % BLOCK_SIZE
    block_start = tl.maximum(position - block_offset, 0)
    history_len = tl.minimum(block_start, WINDOW)

    column = tl.arange(0, BLOCK_W)
    in_width = column < WIDTH
    is_history = column < history_len
    is_block = (column >= history_len) & (column < history_len + BLOCK_SIZE)
    history_position = block_start - 1 - column
    block_position = block_start + column - history_len
    source_position = tl.where(is_history, history_position, block_position)
    source_position = tl.where(is_hold | ~(is_history | is_block), 0, source_position)

    history_full_slot = tl.load(
        req_to_token_ptr + req_idx * req_to_token_stride0 + source_position,
        mask=in_width & is_history & ~is_hold,
        other=HOLD_FULL_SLOT,
    ).to(tl.int64)
    history_swa_slot = tl.load(
        full_to_swa_ptr + history_full_slot,
        mask=in_width & is_history & ~is_hold,
        other=HOLD_SWA_SLOT,
    )
    scratch_page = tl.load(
        scratch_pages_ptr + token_idx // BLOCK_SIZE,
        mask=~is_hold,
        other=0,
    ).to(tl.int64)
    block_swa_slot = scratch_page * PAGE_SIZE + column - history_len
    swa_slot = tl.where(is_history, history_swa_slot, block_swa_slot)
    valid = is_history | is_block
    output = tl.where(valid, swa_slot, -1)
    output = tl.where(is_hold, HOLD_SWA_SLOT, output).to(tl.int32)
    tl.store(
        swa_index_ptr + token_idx * swa_index_stride0 + column, output, mask=in_width
    )

    length = tl.where(is_hold, 1, history_len + BLOCK_SIZE).to(tl.int32)
    tl.store(swa_length_ptr + token_idx, length)
    write_slot = scratch_page * PAGE_SIZE + block_offset
    write_slot = tl.where(is_hold, HOLD_SWA_SLOT, write_slot).to(tl.int32)
    tl.store(swa_write_slot_ptr + token_idx, write_slot)


def build_dspark_swa_index(
    req_idx: torch.Tensor,
    positions: torch.Tensor,
    req_to_token_indexs: torch.Tensor,
    full_to_swa_indexs: torch.Tensor,
    scratch_pages: torch.Tensor,
    swa_index: torch.Tensor,
    swa_length: torch.Tensor,
    swa_write_slots: torch.Tensor,
    window: int,
    block_size: int,
    page_size: int,
    hold_req_id: int,
    hold_full_slot: int,
    hold_swa_slot: int,
):
    """Build ``history SWA + complete draft block`` indices for every DSpark query row."""

    token_num = positions.shape[0]
    width = swa_index.shape[1]
    assert token_num % block_size == 0
    assert scratch_pages is not None and scratch_pages.is_cuda
    assert window > 0 and width >= window + block_size
    if token_num == 0:
        return swa_index, swa_length

    _build_dspark_swa_index_kernel[(token_num,)](
        req_idx,
        positions,
        req_to_token_indexs,
        req_to_token_indexs.stride(0),
        full_to_swa_indexs,
        scratch_pages,
        swa_index,
        swa_index.stride(0),
        swa_length,
        swa_write_slots,
        HOLD_REQ_ID=hold_req_id,
        HOLD_FULL_SLOT=hold_full_slot,
        HOLD_SWA_SLOT=hold_swa_slot,
        WINDOW=window,
        BLOCK_SIZE=block_size,
        PAGE_SIZE=page_size,
        WIDTH=width,
        BLOCK_W=triton.next_power_of_2(width),
        num_warps=4,
    )
    return swa_index, swa_length
