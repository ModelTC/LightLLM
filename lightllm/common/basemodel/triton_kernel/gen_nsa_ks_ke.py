import torch
import torch.nn.functional as F
import triton
import triton.language as tl
from typing import Optional


@triton.jit
def _gen_nsa_ks_ke(
    b_seq_len,
    b_q_seq_len,
    b_same_req_mark,
    ks,
    ke,
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

    return


@torch.no_grad()
def gen_nsa_ks_ke(
    b_seq_len: torch.Tensor,
    b_q_seq_len: torch.Tensor,
    b_same_req_mark: torch.Tensor,
    q_token_num: int,
):
    batch_size = b_seq_len.shape[0]
    ks = torch.empty((q_token_num,), dtype=torch.int32, device=b_seq_len.device)
    ke = torch.empty((q_token_num,), dtype=torch.int32, device=b_seq_len.device)

    _gen_nsa_ks_ke[(batch_size,)](
        b_seq_len=b_seq_len,
        b_q_seq_len=b_q_seq_len,
        b_same_req_mark=b_same_req_mark,
        ks=ks,
        ke=ke,
        BLOCK_REQ=triton.next_power_of_2(batch_size),
        BLOCK_SEQ_SPLIT=256,
    )
    return ks, ke
