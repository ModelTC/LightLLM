import torch

import triton
import triton.language as tl


@triton.jit
def _gen_cumsum_pad0_kernel(
    b_q_seq_len,
    b1_cu_q_seq_len,
    b_kv_seq_len,
    b1_cu_kv_seq_len,
    size,
    b_kv_seq_len_stride_0,
    BLOCK: tl.constexpr,  # num_warps
):
    offs = tl.arange(0, BLOCK)
    start_value = tl.cast(0, dtype=tl.int64)

    for start_index in range(0, size, BLOCK):
        current_offs = start_index + offs
        in_data = tl.load(b_q_seq_len + offs, mask=current_offs < size, other=0)
        in_data = tl.cumsum(in_data) + start_value
        start_value = tl.max(in_data, 0)
        tl.store(b1_cu_q_seq_len + current_offs + 1, in_data, mask=current_offs < size)

    # pad 0
    tl.store(b1_cu_q_seq_len + 0, 0)

    start_value = tl.cast(0, tl.int64)
    for start_index in range(0, size, BLOCK):
        current_offs = start_index + offs
        in_data = tl.load(b_kv_seq_len + offs * b_kv_seq_len_stride_0, mask=current_offs < size, other=0)
        in_data = tl.cumsum(in_data) + start_value
        start_value = tl.max(in_data, 0)
        tl.store(b1_cu_kv_seq_len + current_offs + 1, in_data, mask=current_offs < size)

    # pad 0
    tl.store(b1_cu_kv_seq_len + 0, 0)


@torch.no_grad()
def gen_cumsum_pad0_tensor(b_q_seq_len: torch.Tensor, b_kv_seq_len: torch.Tensor):
    assert len(b_q_seq_len.shape) == 1
    assert b_q_seq_len.shape == b_kv_seq_len.shape
    assert b_q_seq_len.is_contiguous()

    b1_cu_q_seq_len = torch.empty((b_q_seq_len.shape[0] + 1,), dtype=torch.int32, device="cuda")
    b1_cu_kv_seq_len = torch.empty((b_kv_seq_len.shape[0] + 1,), dtype=torch.int32, device="cuda")
    _gen_cumsum_pad0_kernel[(1,)](
        b_q_seq_len,
        b1_cu_q_seq_len,
        b_kv_seq_len,
        b1_cu_kv_seq_len,
        b_q_seq_len.shape[0],
        b_kv_seq_len.stride(0),
        BLOCK=1024,
        num_warps=4,
    )
    return b1_cu_q_seq_len, b1_cu_kv_seq_len


@triton.jit
def _gen_prefill_position(
    b_ready_cache_len,
    b_seq_len,
    b1_cu_q_seq_len,
    position_ids,
    RANGE_BLOCK: tl.constexpr,
):
    cur_batch = tl.program_id(0)
    ready_len = tl.load(b_ready_cache_len + cur_batch)
    seq_len = tl.load(b_seq_len + cur_batch)
    q_seq_len = seq_len - ready_len

    dest_start = tl.load(b1_cu_q_seq_len + cur_batch)

    for start in range(ready_len, seq_len, RANGE_BLOCK):
        write_loc = start + tl.arange(0, RANGE_BLOCK) - ready_len
        write_value = start + tl.arange(0, RANGE_BLOCK)
        tl.store(position_ids + dest_start + write_loc, write_value, mask=write_loc < q_seq_len)
    return


@torch.no_grad()
def gen_prefill_params(input_token_num: int, b_ready_cache_len: torch.Tensor, b_seq_len: torch.Tensor):
    batch_size = b_ready_cache_len.shape[0]
    position_ids = torch.empty((input_token_num,), dtype=torch.int32, device="cuda")
    assert b_ready_cache_len.shape[0] == b_seq_len.shape[0]
    b_q_seq_len = b_seq_len - b_ready_cache_len
    b1_cu_q_seq_len, b1_cu_kv_seq_len = gen_cumsum_pad0_tensor(b_q_seq_len, b_seq_len)
    grid = (batch_size,)
    num_warps = 4

    _gen_prefill_position[grid](
        b_ready_cache_len,
        b_seq_len,
        b1_cu_q_seq_len,
        position_ids,
        RANGE_BLOCK=1024,
        num_warps=num_warps,
        num_stages=1,
    )
    b_kv_seq_len = b_seq_len
    return b_q_seq_len, b1_cu_q_seq_len, b_kv_seq_len, b1_cu_kv_seq_len, position_ids
