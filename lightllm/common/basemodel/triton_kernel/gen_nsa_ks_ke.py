import torch
import torch.nn.functional as F
import triton
import triton.language as tl
from typing import Optional


@triton.jit
def _gen_nsa_ks_ke(
    b_seq_len,
    b_req_idx,
    b_q_seq_len,
    b_same_req_mark,
    ks,
    ke,
    lengths,
    req_to_token_index,
    strided_req_to_token_index_b,
    strided_req_to_token_index_s,
    ragged_mem_index,
    BLOCK_REQ: tl.constexpr,
    BLOCK_SEQ_SPLIT: tl.constexpr,
):
    cur_index = tl.program_id(0)
    # 不处于边界mark的最后一个req不进行处理。
    req_mark = tl.load(b_same_req_mark + cur_index)
    if req_mark == 0:
        return

    off = tl.arange(0, BLOCK_REQ)
    b_same_req_mark = tl.load(b_same_req_mark + off, off < cur_index, other=0)
    b_seq_len_data = tl.load(b_seq_len + off, (off < cur_index) & (b_same_req_mark != 0), other=0)
    pre_sum_seq_len = tl.sum(b_seq_len_data)

    # 兼容 prefill 和 decode 的情况， decode 可能存在 mtp 的情况，各个请求会共享一个req对象，其处理比较特殊
    q_seq_len = tl.load(b_q_seq_len + cur_index) + req_mark - 1
    cur_total_len = tl.load(b_seq_len + cur_index)
    cur_req_idx = tl.load(b_req_idx + cur_index)

    b_q_seq_len_data = tl.load(b_q_seq_len + off, (off < (cur_index - req_mark + 1)), other=0)
    store_start_index = tl.sum(b_q_seq_len_data)

    for block_index in range(tl.cdiv(q_seq_len, BLOCK_SEQ_SPLIT)):
        block_start = block_index * BLOCK_SEQ_SPLIT
        block_end = min(q_seq_len, (block_index + 1) * BLOCK_SEQ_SPLIT)
        ks_data = tl.zeros((BLOCK_SEQ_SPLIT,), dtype=tl.int32)
        ke_data = (cur_total_len - q_seq_len) + tl.arange(0, BLOCK_SEQ_SPLIT)

        tl.store(
            ks + store_start_index + block_start + tl.arange(0, BLOCK_SEQ_SPLIT),
            ks_data + pre_sum_seq_len,
            mask=block_start + tl.arange(0, BLOCK_SEQ_SPLIT) < block_end,
        )
        tl.store(
            ke + store_start_index + block_start + tl.arange(0, BLOCK_SEQ_SPLIT),
            ke_data + pre_sum_seq_len,
            mask=block_start + tl.arange(0, BLOCK_SEQ_SPLIT) < block_end,
        )
        tl.store(
            lengths + store_start_index + block_start + tl.arange(0, BLOCK_SEQ_SPLIT),
            ke_data - ks_data + 1,
            mask=block_start + tl.arange(0, BLOCK_SEQ_SPLIT) < block_end,
        )

    for block_index in range(tl.cdiv(cur_total_len, BLOCK_SEQ_SPLIT)):
        block_start = block_index * BLOCK_SEQ_SPLIT
        block_end = min(cur_total_len, (block_index + 1) * BLOCK_SEQ_SPLIT)
        mask = block_start + tl.arange(0, BLOCK_SEQ_SPLIT) < block_end

        src_mem_index_ptr = (
            req_to_token_index
            + strided_req_to_token_index_b * cur_req_idx
            + block_start
            + tl.arange(0, BLOCK_SEQ_SPLIT)
        )
        src_mem_index = tl.load(src_mem_index_ptr, mask=mask, other=-1)
        tl.store(
            ragged_mem_index + pre_sum_seq_len + block_start + tl.arange(0, BLOCK_SEQ_SPLIT), src_mem_index, mask=mask
        )
    return


@torch.no_grad()
def gen_nsa_ks_ke(
    b_seq_len: torch.Tensor,
    b_q_seq_len: torch.Tensor,
    b_req_idx: torch.Tensor,
    req_to_token_index: torch.Tensor,
    q_token_num: int,
    ragged_mem_index: torch.Tensor,
):
    batch_size = b_seq_len.shape[0]
    ks = torch.empty((q_token_num,), dtype=torch.int32, device=b_seq_len.device)
    ke = torch.empty((q_token_num,), dtype=torch.int32, device=b_seq_len.device)
    lengths = torch.empty((q_token_num,), dtype=torch.int32, device=b_seq_len.device)
    b_same_req_mark = gen_same_req_mark(b_req_idx)

    _gen_nsa_ks_ke[(batch_size,)](
        b_seq_len=b_seq_len,
        b_req_idx=b_req_idx,
        b_q_seq_len=b_q_seq_len,
        b_same_req_mark=b_same_req_mark,
        ks=ks,
        ke=ke,
        lengths=lengths,
        req_to_token_index=req_to_token_index,
        strided_req_to_token_index_b=req_to_token_index.stride(0),
        strided_req_to_token_index_s=req_to_token_index.stride(1),
        ragged_mem_index=ragged_mem_index,
        BLOCK_REQ=triton.next_power_of_2(batch_size),
        BLOCK_SEQ_SPLIT=256,
    )
    return ks, ke, lengths


@triton.jit
def _gen_same_req_mark(b_req_idx, b_same_req_mark, BLOCK_SIZE: tl.constexpr):
    cur_index = tl.program_id(0)
    cur_req_idx = tl.load(b_req_idx + cur_index)
    off = tl.arange(0, BLOCK_SIZE)
    pre_req_idxs = tl.load(b_req_idx + off, (off < cur_index) & (off < tl.num_programs(0)), other=-1)
    after_req_idxs = tl.load(b_req_idx + off, (off > cur_index) & (off < tl.num_programs(0)), other=-1)

    pre_idx_count = tl.sum(pre_req_idxs == cur_req_idx)
    after_idx_count = tl.sum(after_req_idxs == cur_req_idx)

    has_mark = tl.where(after_idx_count == 0, 1, 0)
    for _ in range(has_mark):
        tl.store(b_same_req_mark + cur_index, pre_idx_count + 1)
    return


@torch.no_grad()
def gen_same_req_mark(b_req_idx: torch.Tensor):
    """
    b_req_idx: torch.Tensor
    out: torch.Tensor

    demo:
    b_req_idx = [1, 1, 2, 3, 3, 3]
    out = [0, 2, 1, 0, 0, 3]
    """
    batch_size = b_req_idx.shape[0]
    b_same_req_mark = torch.empty((batch_size,), dtype=torch.int32, device=b_req_idx.device)
    b_same_req_mark.fill_(0)
    _gen_same_req_mark[(batch_size,)](
        b_req_idx=b_req_idx, b_same_req_mark=b_same_req_mark, BLOCK_SIZE=triton.next_power_of_2(batch_size)
    )
    return b_same_req_mark
