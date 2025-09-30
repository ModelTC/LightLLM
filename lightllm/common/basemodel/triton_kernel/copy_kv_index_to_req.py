import torch

import triton
import triton.language as tl


@triton.jit
def _fwd_kernel_copy_kv_index_to_req(
    req_to_token_indexs, b_req_idx, b_seq_len, memindex, stride_req_to_token_b, stride_req_to_token_s
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
        req_to_token_indexs,
        b_req_idx,
        b_seq_len,
        memindex,
        req_to_token_indexs.stride(0),
        req_to_token_indexs.stride(1),
        num_warps=num_warps,
        num_stages=1,
    )
    return


@triton.jit
def _fwd_kernel_copy_kv_index_to_req_prefill(
    req_to_token_indexs,
    b_req_idx,
    b_seq_len,
    b_ready_cache_len,
    b_start_loc,
    memindex,
    stride_req_to_token_b,
    stride_req_to_token_s,
    BLOCK: tl.constexpr,
):

    block_index = tl.program_id(0)
    batch_index = tl.program_id(1)
    cur_req_idx = tl.load(b_req_idx + batch_index)
    cur_seq_len = tl.load(b_seq_len + batch_index)
    cur_ready_cache_len = tl.load(b_ready_cache_len + batch_index)
    cur_start_loc = tl.load(b_start_loc + batch_index)
    copy_len = cur_seq_len - cur_ready_cache_len

    block_range = block_index * BLOCK + tl.arange(0, BLOCK)
    block_mask = block_range < copy_len
    cur_token_index = tl.load(memindex + cur_start_loc + block_range, mask=block_mask)
    dest_offset = (
        req_to_token_indexs
        + cur_req_idx * stride_req_to_token_b
        + (cur_ready_cache_len + block_range) * stride_req_to_token_s
    )
    tl.store(dest_offset, cur_token_index, mask=block_mask)

    return


def get_triton_config(max_q_seq_len: int) -> tuple[int, int]:
    if max_q_seq_len <= 512:
        return 256, 2
    elif max_q_seq_len <= 4096:
        return 512, 4
    else:
        return 1024, 8


@torch.no_grad()
def copy_kv_index_to_req_prefill(
    req_to_token_indexs: torch.Tensor,
    b_req_idx: torch.Tensor,
    b_seq_len: torch.Tensor,
    b_ready_cache_len: torch.Tensor,
    b_start_loc: torch.Tensor,
    memindex: torch.Tensor,
    max_q_seq_len: int,
):
    batch_size = b_req_idx.shape[0]
    BLOCK, num_warps = get_triton_config(max_q_seq_len)
    grid = (triton.cdiv(max_q_seq_len, BLOCK), batch_size)
    num_warps = 1

    _fwd_kernel_copy_kv_index_to_req_prefill[grid](
        req_to_token_indexs,
        b_req_idx,
        b_seq_len,
        b_ready_cache_len,
        b_start_loc,
        memindex,
        req_to_token_indexs.stride(0),
        req_to_token_indexs.stride(1),
        BLOCK=BLOCK,
        num_warps=num_warps,
        num_stages=1,
    )
    return
