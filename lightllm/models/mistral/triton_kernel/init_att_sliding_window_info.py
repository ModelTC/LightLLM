import torch

import triton
import triton.language as tl


@triton.jit
def _fwd_kernel_init_att_window_info(
    b_seq_len,
    b_att_seq_len,
    batch_size,
    sliding_window,
    BLOCK_SIZE: tl.constexpr,
):
    cur_index = tl.program_id(0)
    cur_start = cur_index * BLOCK_SIZE
    offsets = cur_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < batch_size

    cur_seq_len = tl.load(b_seq_len + offsets, mask=mask)
    b_att_seq_len_data = tl.minimum(cur_seq_len, sliding_window)

    tl.store(b_att_seq_len + offsets, b_att_seq_len_data, mask=mask)
    return


@torch.no_grad()
def init_att_window_info_fwd(batch_size, b_seq_len, b_att_seq_len, sliding_window):
    # shape constraints
    assert batch_size == b_seq_len.shape[0] == b_att_seq_len.shape[0]

    BLOCK_SIZE = 32
    num_warps = 1
    grid = (triton.cdiv(batch_size, BLOCK_SIZE),)

    _fwd_kernel_init_att_window_info[grid](
        b_seq_len,
        b_att_seq_len,
        batch_size=batch_size,
        sliding_window=sliding_window,
        BLOCK_SIZE=BLOCK_SIZE,
        num_warps=num_warps,
        num_stages=1,
    )
    return
