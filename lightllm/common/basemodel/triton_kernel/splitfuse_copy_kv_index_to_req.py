import torch

import triton
import triton.language as tl


@triton.jit
def _fwd_kernel_copy_kv_index_to_req(
    req_to_token_indexs,
    b_req_idx,
    b_split_seq_len,
    cumsum_split_seq_len,
    b_seq_len,
    memindex,
    stride_req_to_token_b,
    stride_req_to_token_s,
    BLOCK_M: tl.constexpr,
):
    cur_index = tl.program_id(0)
    cur_req_idx = tl.load(b_req_idx + cur_index)
    q_split_len = tl.load(b_split_seq_len + cur_index)
    q_mem_end = tl.load(cumsum_split_seq_len + cur_index)
    q_mem_start = q_mem_end - q_split_len

    store_end = tl.load(b_seq_len + cur_index)
    store_start = store_end - q_split_len

    off_m = tl.arange(0, BLOCK_M)
    for block_start in range(0, q_split_len, BLOCK_M):
        read_index = tl.load(
            memindex + q_mem_start + block_start + off_m, mask=q_mem_start + block_start + off_m < q_mem_end, other=0
        )
        tl.store(
            req_to_token_indexs + cur_req_idx * stride_req_to_token_b + (block_start + store_start + off_m),
            read_index,
            mask=block_start + store_start + off_m < store_end,
        )
    return


@torch.no_grad()
def splitfuse_copy_kv_index_to_req(req_to_token_indexs, b_req_idx, b_ready_cache_len, b_seq_len, memindex):
    batch_size = b_seq_len.shape[0]
    grid = (batch_size,)
    num_warps = 1
    b_split_seq_len = b_seq_len - b_ready_cache_len
    cumsum_split_seq_len = torch.cumsum(b_split_seq_len, dim=0)
    _fwd_kernel_copy_kv_index_to_req[grid](
        req_to_token_indexs,
        b_req_idx,
        b_split_seq_len,
        cumsum_split_seq_len,
        b_seq_len,
        memindex,
        req_to_token_indexs.stride(0),
        req_to_token_indexs.stride(1),
        BLOCK_M=32,
        num_warps=num_warps,
        num_stages=1,
    )
    return
