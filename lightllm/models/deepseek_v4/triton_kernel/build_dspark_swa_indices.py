import torch
import triton
import triton.language as tl


@triton.jit
def _build_dspark_swa_indices_kernel(
    token_req_idx,
    position_ids,
    swa_indices,
    swa_indices_stride0,
    swa_lengths,
    swa_write_slots,
    SWA_CAPACITY_TOKENS: tl.constexpr,
    HOLD_REQUEST_ID: tl.constexpr,
    HOLD_SWA_SLOT: tl.constexpr,
    SLIDING_WINDOW: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
    WIDTH: tl.constexpr,
    BLOCK_W: tl.constexpr,
):
    token_idx = tl.program_id(0)
    req_idx = tl.load(token_req_idx + token_idx).to(tl.int64)
    position = tl.load(position_ids + token_idx).to(tl.int64)
    is_hold = req_idx == HOLD_REQUEST_ID

    block_offset = token_idx % BLOCK_SIZE
    block_start = tl.maximum(position - block_offset, 0)
    history_len = tl.minimum(block_start, SLIDING_WINDOW)

    column = tl.arange(0, BLOCK_W)
    in_width = column < WIDTH
    is_history = column < history_len
    is_block = (column >= history_len) & (column < history_len + BLOCK_SIZE)
    history_position = block_start - 1 - column
    block_position = block_start + column - history_len
    source_position = tl.where(is_history, history_position, block_position)
    source_position = tl.where(is_hold | ~(is_history | is_block), 0, source_position)

    valid = is_history | is_block
    swa_slot = req_idx * SWA_CAPACITY_TOKENS + source_position % SWA_CAPACITY_TOKENS
    output = tl.where(valid, swa_slot, -1)
    output = tl.where(is_hold, HOLD_SWA_SLOT, output).to(tl.int32)
    tl.store(swa_indices + token_idx * swa_indices_stride0 + column, output, mask=in_width)

    length = tl.where(is_hold, 1, history_len + BLOCK_SIZE).to(tl.int32)
    tl.store(swa_lengths + token_idx, length)
    write_slot = req_idx * SWA_CAPACITY_TOKENS + position % SWA_CAPACITY_TOKENS
    write_slot = tl.where(is_hold, HOLD_SWA_SLOT, write_slot).to(tl.int32)
    tl.store(swa_write_slots + token_idx, write_slot)


def build_dspark_swa_indices(
    token_req_idx: torch.Tensor,
    position_ids: torch.Tensor,
    req_manager,
    swa_indices: torch.Tensor,
    swa_lengths: torch.Tensor,
    swa_write_slots: torch.Tensor,
    sliding_window: int,
    block_size: int,
    hold_req_id: int,
    hold_swa_slot: int,
):
    """Build ``history SWA + complete draft block`` indices for every DSpark query row."""

    token_num = position_ids.shape[0]
    width = swa_indices.shape[1]
    assert token_num % block_size == 0
    assert sliding_window > 0 and width >= sliding_window + block_size
    if token_num == 0:
        return swa_indices, swa_lengths

    _build_dspark_swa_indices_kernel[(token_num,)](
        token_req_idx,
        position_ids,
        swa_indices,
        swa_indices.stride(0),
        swa_lengths,
        swa_write_slots,
        SWA_CAPACITY_TOKENS=req_manager.swa_capacity_tokens,
        HOLD_REQUEST_ID=hold_req_id,
        HOLD_SWA_SLOT=hold_swa_slot,
        SLIDING_WINDOW=sliding_window,
        BLOCK_SIZE=block_size,
        WIDTH=width,
        BLOCK_W=triton.next_power_of_2(width),
        num_warps=4,
    )
    return swa_indices, swa_lengths
