import torch
import triton
import triton.language as tl


_DYNAMIC_MTP_FA3_FAST_PATH_MAX_BATCH_SIZE = 1024
_DYNAMIC_MTP_FA3_COMPACT_BLOCK_SIZE = 256


@triton.jit
def page_table_copy_kernel(
    page_table_ptr,
    req_to_token_indexs_ptr,
    b_req_idx,
    max_seq_len_k,
    b_req_idx_stride_0,
    page_table_stride_0,
    page_table_stride_1,
    req_to_token_stride_0,
    req_to_token_stride_1,
    BLOCK_SIZE: tl.constexpr,
):
    cur_batch = tl.program_id(axis=0)
    cur_block = tl.program_id(axis=1)
    cur_req_idx = tl.load(b_req_idx + cur_batch * b_req_idx_stride_0)

    offs = cur_block * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offs < max_seq_len_k

    input_pos = cur_req_idx * req_to_token_stride_0 + offs * req_to_token_stride_1
    output_pos = cur_batch * page_table_stride_0 + offs * page_table_stride_1

    mem_index = tl.load(req_to_token_indexs_ptr + input_pos, mask=mask)
    tl.store(page_table_ptr + output_pos, mem_index, mask=mask)


def page_table_copy(
    page_table,  # destination tensor [batch, seq]
    req_to_token_indexs,  # source tensor [batch, seq]
    b_req_idx,  # request index to copy from
):
    assert page_table.dim() == 2, "page_table should be 2D"
    assert req_to_token_indexs.dim() == 2, "req_to_token_indexs should be 2D"

    max_seq_len_k = page_table.shape[1]
    batch_size = page_table.size(0)
    BLOCK_SIZE = 128

    grid = (batch_size, triton.cdiv(max_seq_len_k, BLOCK_SIZE))

    page_table_copy_kernel[grid](
        page_table_ptr=page_table,
        req_to_token_indexs_ptr=req_to_token_indexs,
        b_req_idx=b_req_idx,
        max_seq_len_k=max_seq_len_k,
        b_req_idx_stride_0=b_req_idx.stride(0),
        page_table_stride_0=page_table.stride(0),
        page_table_stride_1=page_table.stride(1),
        req_to_token_stride_0=req_to_token_indexs.stride(0),
        req_to_token_stride_1=req_to_token_indexs.stride(1),
        BLOCK_SIZE=BLOCK_SIZE,
    )


@triton.jit
def _build_dynamic_mtp_fa3_decode_params_kernel(
    b_req_idx,
    b_seq_len,
    b_mark_shared_group,
    out_b_q_seq_len,
    out_b_kv_seq_len,
    out_b_att_req_idx,
    out_b_att_seq_len,
    batch_size,
    hold_req_id,
    BLOCK_SIZE: tl.constexpr,
):
    offsets = tl.arange(0, BLOCK_SIZE)
    mask = offsets < batch_size

    mark = tl.load(b_mark_shared_group + offsets, mask=mask, other=0)
    is_group_end = mask & (mark > 0)
    dst_pos = tl.cumsum(tl.where(is_group_end, 1, 0), axis=0) - 1

    tl.store(out_b_q_seq_len + offsets, 0, mask=mask)
    tl.store(out_b_kv_seq_len + offsets, 0, mask=mask)
    tl.store(out_b_att_req_idx + offsets, hold_req_id, mask=mask)
    tl.store(out_b_att_seq_len + offsets, 0, mask=mask)

    cur_req_idx = tl.load(b_req_idx + offsets, mask=mask, other=hold_req_id)
    cur_seq_len = tl.load(b_seq_len + offsets, mask=mask, other=0)

    tl.store(out_b_q_seq_len + dst_pos, mark, mask=is_group_end)
    tl.store(out_b_kv_seq_len + dst_pos, cur_seq_len, mask=is_group_end)
    tl.store(out_b_att_req_idx + dst_pos, cur_req_idx, mask=is_group_end)
    tl.store(out_b_att_seq_len + dst_pos, cur_seq_len, mask=is_group_end)


@triton.jit
def _count_dynamic_mtp_fa3_decode_params_kernel(
    b_mark_shared_group,
    out_block_counts,
    batch_size,
    BLOCK_SIZE: tl.constexpr,
):
    block_id = tl.program_id(axis=0)
    offsets = block_id * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < batch_size

    mark = tl.load(b_mark_shared_group + offsets, mask=mask, other=0)
    block_count = tl.sum(tl.where(mask & (mark > 0), 1, 0), axis=0)
    tl.store(out_block_counts + block_id, block_count)


@triton.jit
def _compact_dynamic_mtp_fa3_decode_params_kernel(
    b_req_idx,
    b_seq_len,
    b_mark_shared_group,
    block_counts,
    block_offsets,
    out_b_q_seq_len,
    out_b_kv_seq_len,
    out_b_att_req_idx,
    out_b_att_seq_len,
    batch_size,
    hold_req_id,
    BLOCK_SIZE: tl.constexpr,
):
    block_id = tl.program_id(axis=0)
    offsets = block_id * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < batch_size

    mark = tl.load(b_mark_shared_group + offsets, mask=mask, other=0)
    is_group_end = mask & (mark > 0)
    local_pos = tl.cumsum(tl.where(is_group_end, 1, 0), axis=0) - 1
    block_count = tl.load(block_counts + block_id)
    block_end_offset = tl.load(block_offsets + block_id)
    block_start_offset = block_end_offset - block_count
    dst_pos = block_start_offset + local_pos

    cur_req_idx = tl.load(b_req_idx + offsets, mask=mask, other=hold_req_id)
    cur_seq_len = tl.load(b_seq_len + offsets, mask=mask, other=0)

    tl.store(out_b_q_seq_len + dst_pos, mark, mask=is_group_end)
    tl.store(out_b_kv_seq_len + dst_pos, cur_seq_len, mask=is_group_end)
    tl.store(out_b_att_req_idx + dst_pos, cur_req_idx, mask=is_group_end)
    tl.store(out_b_att_seq_len + dst_pos, cur_seq_len, mask=is_group_end)


@torch.no_grad()
def build_dynamic_mtp_fa3_decode_params(
    b_req_idx: torch.Tensor,
    b_seq_len: torch.Tensor,
    b_mark_shared_group: torch.Tensor,
    att_batch_size: int,
    hold_req_id: int,
):
    assert b_req_idx.is_cuda and b_seq_len.is_cuda and b_mark_shared_group.is_cuda
    assert b_req_idx.shape == b_seq_len.shape == b_mark_shared_group.shape
    assert b_req_idx.shape[0] == att_batch_size
    assert att_batch_size > 0

    if att_batch_size <= _DYNAMIC_MTP_FA3_FAST_PATH_MAX_BATCH_SIZE:
        b_q_seq_len = torch.empty((att_batch_size,), dtype=torch.int32, device=b_seq_len.device)
        b_kv_seq_len = torch.empty((att_batch_size,), dtype=torch.int32, device=b_seq_len.device)
        b_att_req_idx = torch.empty((att_batch_size,), dtype=torch.int32, device=b_req_idx.device)
        b_att_seq_len = torch.empty((att_batch_size,), dtype=torch.int32, device=b_seq_len.device)

        _build_dynamic_mtp_fa3_decode_params_kernel[(1,)](
            b_req_idx=b_req_idx,
            b_seq_len=b_seq_len,
            b_mark_shared_group=b_mark_shared_group,
            out_b_q_seq_len=b_q_seq_len,
            out_b_kv_seq_len=b_kv_seq_len,
            out_b_att_req_idx=b_att_req_idx,
            out_b_att_seq_len=b_att_seq_len,
            batch_size=att_batch_size,
            hold_req_id=hold_req_id,
            BLOCK_SIZE=triton.next_power_of_2(att_batch_size),
            num_warps=8,
            num_stages=1,
        )
        return b_q_seq_len, b_kv_seq_len, b_att_req_idx, b_att_seq_len

    b_q_seq_len = torch.zeros((att_batch_size,), dtype=torch.int32, device=b_seq_len.device)
    b_kv_seq_len = torch.zeros((att_batch_size,), dtype=torch.int32, device=b_seq_len.device)
    b_att_req_idx = torch.full((att_batch_size,), hold_req_id, dtype=torch.int32, device=b_req_idx.device)
    b_att_seq_len = torch.zeros((att_batch_size,), dtype=torch.int32, device=b_seq_len.device)

    block_size = _DYNAMIC_MTP_FA3_COMPACT_BLOCK_SIZE
    grid = (triton.cdiv(att_batch_size, block_size),)
    block_counts = torch.empty((grid[0],), dtype=torch.int32, device=b_mark_shared_group.device)

    _count_dynamic_mtp_fa3_decode_params_kernel[grid](
        b_mark_shared_group=b_mark_shared_group,
        out_block_counts=block_counts,
        batch_size=att_batch_size,
        BLOCK_SIZE=block_size,
        num_warps=8,
        num_stages=1,
    )
    block_offsets = torch.cumsum(block_counts, dim=0, dtype=torch.int32)

    _compact_dynamic_mtp_fa3_decode_params_kernel[grid](
        b_req_idx=b_req_idx,
        b_seq_len=b_seq_len,
        b_mark_shared_group=b_mark_shared_group,
        block_counts=block_counts,
        block_offsets=block_offsets,
        out_b_q_seq_len=b_q_seq_len,
        out_b_kv_seq_len=b_kv_seq_len,
        out_b_att_req_idx=b_att_req_idx,
        out_b_att_seq_len=b_att_seq_len,
        batch_size=att_batch_size,
        hold_req_id=hold_req_id,
        BLOCK_SIZE=block_size,
        num_warps=8,
        num_stages=1,
    )
    return b_q_seq_len, b_kv_seq_len, b_att_req_idx, b_att_seq_len
