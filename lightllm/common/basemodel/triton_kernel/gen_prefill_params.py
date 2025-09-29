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
        in_data = tl.load(b_kv_seq_len + offs, mask=current_offs < size, other=0)
        in_data = tl.cumsum(in_data) + start_value
        start_value = tl.max(in_data, 0)
        tl.store(b1_cu_kv_seq_len + current_offs + 1, in_data, mask=current_offs < size)

    # pad 0
    tl.store(b1_cu_kv_seq_len + 0, 0)


@torch.no_grad()
def gen_cumsum_pad0_tensor(b_q_seq_len: torch.Tensor, b_kv_seq_len: torch.Tensor):
    assert len(b_q_seq_len.shape) == 1
    assert b_q_seq_len.shape == b_kv_seq_len.shape

    b1_cu_q_seq_len = torch.empty((b_q_seq_len.shape[0] + 1,), dtype=torch.int32, device="cuda")
    b1_cu_kv_seq_len = torch.empty((b_kv_seq_len.shape[0] + 1,), dtype=torch.int32, device="cuda")
    _gen_cumsum_pad0_kernel[(1,)](
        b_q_seq_len,
        b1_cu_q_seq_len,
        b_kv_seq_len,
        b1_cu_kv_seq_len,
        b_q_seq_len.shape[0],
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


@triton.jit
def fill_req_to_token_indexes_kernel(
    req_to_token_indexs_ptr,  # [num_req, max_len]
    b_req_idx_ptr,  # [B]
    b_seq_len_ptr,  # [B]
    b_ready_cache_len_ptr,  # [B]
    b_start_loc_ptr,  # [B]
    alloc_mem_index_ptr,  # [total_new_tokens]
    req_to_token_indexs_stride0,
    req_to_token_indexs_stride1,
    BLOCK: tl.constexpr,
):
    pid = tl.program_id(0)  # batch id
    req_idx = tl.load(b_req_idx_ptr + pid)
    cur_seq_len = tl.load(b_seq_len_ptr + pid)
    cur_ready_cache_len = tl.load(b_ready_cache_len_ptr + pid)
    start_loc = tl.load(b_start_loc_ptr + pid)

    copy_len = cur_seq_len - cur_ready_cache_len
    if copy_len <= 0:
        return

    # 一次 BLOCK 个线程
    offs = tl.arange(0, BLOCK)
    for base in range(0, copy_len, BLOCK):
        idx = base + offs
        mask = idx < copy_len
        vals = tl.load(alloc_mem_index_ptr + start_loc + idx, mask=mask, other=0)

        out_ptrs = (
            req_to_token_indexs_ptr
            + req_idx * req_to_token_indexs_stride0
            + (cur_ready_cache_len + idx) * req_to_token_indexs_stride1
        )
        tl.store(out_ptrs, vals, mask=mask)


def init_req_to_token_indexes_triton(
    req_to_token_indexs: torch.Tensor,  # [num_req, max_len]
    b_req_idx: torch.Tensor,  # [B]
    b_seq_len: torch.Tensor,  # [B]
    b_ready_cache_len: torch.Tensor,  # [B]
    b_start_loc: torch.Tensor,  # [B], alloc_mem_index 的 prefix sum 起点
    alloc_mem_index: torch.Tensor,  # [total_new_tokens]
    max_q_seq_len: int,
):
    BLOCK = 128
    batch_size = b_seq_len.shape[0]
    grid = (batch_size,)
    fill_req_to_token_indexes_kernel[grid](
        req_to_token_indexs,
        b_req_idx,
        b_seq_len,
        b_ready_cache_len,
        b_start_loc,
        alloc_mem_index,
        req_to_token_indexs.stride(0),
        req_to_token_indexs.stride(1),
        BLOCK=BLOCK,
    )
