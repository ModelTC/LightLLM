import torch
import torch.nn.functional as F
import triton
import triton.language as tl
from typing import Optional


@triton.jit
def _gen_mtp_new_input_ids(
    b1_cu_q_seq_len_ptr, old_input_ids_ptr, insert_tail_input_ids, new_input_ids_ptr, BLOCK: tl.constexpr
):
    batch_index = tl.program_id(0)
    start_index = tl.load(b1_cu_q_seq_len_ptr + batch_index)
    end_index = tl.load(b1_cu_q_seq_len_ptr + batch_index + 1)
    offs = tl.arange(0, BLOCK)

    for iter_start_index in tl.range(start_index + 1, end_index, BLOCK, num_stages=3):
        input_offs = iter_start_index + offs
        t_input_ids = tl.load(old_input_ids_ptr + input_offs, mask=input_offs < end_index, other=0)
        tl.store(new_input_ids_ptr + input_offs - 1, t_input_ids, mask=input_offs - 1 < end_index - 1)
    tail_token_id = tl.load(insert_tail_input_ids + batch_index)
    tl.store(new_input_ids_ptr + end_index - 1, tail_token_id)
    return


@torch.no_grad()
def gen_mtp_new_input_ids(
    input_ids: torch.Tensor,
    b_next_token_ids: torch.Tensor,
    b_seq_len: torch.Tensor,
    b_ready_cache_len: Optional[torch.Tensor] = None,
):
    assert len(b_seq_len.shape) == 1
    batch_size = b_seq_len.shape[0]
    if b_ready_cache_len is None:
        b_q_seq_len = b_seq_len
    else:
        b_q_seq_len = b_seq_len - b_ready_cache_len
    b1_cu_q_seq_len = F.pad(torch.cumsum(b_q_seq_len, dim=0, dtype=torch.int32), pad=(1, 0), mode="constant", value=0)
    new_input_ids = torch.empty_like(input_ids)
    BLOCK = 512
    num_warps = 4
    _gen_mtp_new_input_ids[(batch_size,)](
        b1_cu_q_seq_len_ptr=b1_cu_q_seq_len,
        old_input_ids_ptr=input_ids,
        insert_tail_input_ids=b_next_token_ids,
        new_input_ids_ptr=new_input_ids,
        BLOCK=BLOCK,
        num_warps=num_warps,
    )
    return new_input_ids
