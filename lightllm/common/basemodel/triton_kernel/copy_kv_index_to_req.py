import torch

import triton
import triton.language as tl


@triton.jit
def _fwd_kernel_copy_kv_index_to_req(
    req_to_token_indexs, b_req_idx, b_seq_len, memindex,
    stride_req_to_token_b, stride_req_to_token_s
):
    cur_index = tl.program_id(0)
    cur_req_idx = tl.load(b_req_idx + cur_index)
    cur_token_index = tl.load(memindex + cur_index)
    cur_seq_len = tl.load(b_seq_len + cur_index)
    dest_offset = req_to_token_indexs + cur_req_idx * stride_req_to_token_b + (cur_seq_len - 1) * stride_req_to_token_s
    tl.store(dest_offset, cur_token_index)
    return


@torch.no_grad()
def copy_kv_index_to_req(req_to_token_indexs, b_req_idx, b_seq_len, memindex):
    seq_len = b_seq_len.shape[0]
    assert b_seq_len.shape[0] == memindex.shape[0] and b_req_idx.shape[0] == b_seq_len.shape[0]
    grid = (seq_len,)
    num_warps = 1

    _fwd_kernel_copy_kv_index_to_req[grid](
        req_to_token_indexs, b_req_idx, b_seq_len, memindex,
        req_to_token_indexs.stride(0), req_to_token_indexs.stride(1),
        num_warps=num_warps,
        num_stages=1,
    )
    return
