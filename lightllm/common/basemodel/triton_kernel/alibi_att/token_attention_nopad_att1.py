import torch

import triton
import triton.language as tl
import math


@triton.jit
def _fwd_kernel_token_att1(
    Q,
    K,
    sm_scale,
    Alibi,
    Req_to_tokens,
    B_req_idx,
    B_Start_Loc,
    B_Seqlen,  # B_Start_Loc 保存的是如果连续存储时候的累加输入和
    Att_Out,
    stride_req_to_tokens_b,
    stride_req_to_tokens_s,
    stride_qbs,
    stride_qh,
    stride_qd,
    stride_kbs,
    stride_kh,
    stride_kd,
    att_stride_h,
    att_stride_bs,
    BLOCK_DMODEL: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    cur_batch = tl.program_id(0)
    cur_head = tl.program_id(1)
    start_n = tl.program_id(2)

    offs_d = tl.arange(0, BLOCK_DMODEL)
    cur_batch_seq_len = tl.load(B_Seqlen + cur_batch)
    cur_batch_in_all_start_index = tl.load(B_Start_Loc + cur_batch)
    cur_batch_req_id = tl.load(B_req_idx + cur_batch)

    cur_batch_start_index = 0
    cur_batch_end_index = cur_batch_seq_len

    off_q = cur_batch * stride_qbs + cur_head * stride_qh + offs_d * stride_qd

    offs_n = start_n * BLOCK_N + tl.arange(0, BLOCK_N)

    block_stard_index = start_n * BLOCK_N
    block_mask = tl.where(block_stard_index < cur_batch_seq_len, 1, 0)

    stride_req_to_tokens_s = tl.cast(stride_req_to_tokens_s, tl.int64)
    cur_batch_req_id = tl.cast(cur_batch_req_id, tl.int64)
    stride_kbs = tl.cast(stride_kbs, tl.int64)

    for start_mark in range(0, block_mask, 1):  # 用来判断当前 mask 是否需要计算
        alibi_m = tl.load(Alibi + cur_head)
        q = tl.load(Q + off_q + start_mark)
        offs_n_new = cur_batch_start_index + offs_n
        k_loc = tl.load(
            Req_to_tokens + stride_req_to_tokens_b * cur_batch_req_id + stride_req_to_tokens_s * offs_n_new,
            mask=offs_n_new < cur_batch_end_index,
            other=0,
        )
        off_k = k_loc[:, None] * stride_kbs + cur_head * stride_kh + offs_d[None, :] * stride_kd
        k = tl.load(K + off_k, mask=offs_n_new[:, None] < cur_batch_end_index, other=0.0)
        att_value = tl.sum(q[None, :] * k, 1)
        att_value *= sm_scale
        att_value -= alibi_m * (cur_batch_seq_len - 1 - offs_n)
        off_o = cur_head * att_stride_h + (cur_batch_in_all_start_index + offs_n) * att_stride_bs
        tl.store(Att_Out + off_o, att_value, mask=offs_n_new < cur_batch_end_index)
    return


@torch.no_grad()
def token_att_fwd(q, k, att_out, alibi, Req_to_tokens, B_req_idx, B_Start_Loc, B_Seqlen, max_len_in_batch):
    BLOCK = 32
    # shape constraints
    Lq, Lk = q.shape[-1], k.shape[-1]
    assert Lq == Lk
    assert Lk in {16, 32, 64, 128}
    sm_scale = 1.0 / (Lk ** 0.5)

    batch, head_num = B_req_idx.shape[0], q.shape[1]

    grid = (batch, head_num, triton.cdiv(max_len_in_batch, BLOCK))

    num_warps = 4 if Lk <= 64 else 8
    num_warps = 2

    _fwd_kernel_token_att1[grid](
        q,
        k,
        sm_scale,
        alibi,
        Req_to_tokens,
        B_req_idx,
        B_Start_Loc,
        B_Seqlen,
        att_out,
        Req_to_tokens.stride(0),
        Req_to_tokens.stride(1),
        q.stride(0),
        q.stride(1),
        q.stride(2),
        k.stride(0),
        k.stride(1),
        k.stride(2),
        att_out.stride(0),
        att_out.stride(1),
        BLOCK_DMODEL=Lk,
        BLOCK_N=BLOCK,
        num_warps=num_warps,
        num_stages=1,
    )
    return


def torch_att(xq, xk, bs, seqlen, num_head, head_dim):
    xq = xq.view(bs, 1, num_head, head_dim)
    xk = xk.view(bs, seqlen, num_head, head_dim)
    keys = xk
    xq = xq.transpose(1, 2)
    keys = keys.transpose(1, 2)
    scores = (
        (torch.matmul(xq, keys.transpose(2, 3)) / math.sqrt(head_dim)).squeeze().transpose(0, 1).reshape(num_head, -1)
    )
    print("s  ", scores.shape)
    return scores


def torch_att1(xq, xk, seqlen, num_head, head_dim):
    xq = xq.view(1, num_head, head_dim)
    xk = xk.view(seqlen, num_head, head_dim)
    logics = torch.sum(xq * xk, dim=-1, keepdim=False)

    logics = logics.transpose(0, 1) / math.sqrt(head_dim)
    return logics
