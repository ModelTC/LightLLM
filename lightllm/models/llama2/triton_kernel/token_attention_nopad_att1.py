import torch

import triton
import triton.language as tl
import math


@triton.jit
def _fwd_kernel_token_att1(
    Q, K, sm_scale, B_Loc, B_Start_Loc, B_Seqlen, max_input_len,
    Att_Out,
    stride_b_loc_b, stride_b_loc_s,
    stride_qbs, stride_qh, stride_qd,
    stride_kbs, stride_kh, stride_kd,
    att_stride_h, att_stride_bs,
    kv_group_num,
    BLOCK_DMODEL: tl.constexpr,
    BLOCK_N: tl.constexpr
):
    cur_batch = tl.program_id(0)
    cur_head = tl.program_id(1)
    start_n = tl.program_id(2)
    
    cur_kv_head = cur_head // kv_group_num

    offs_d = tl.arange(0, BLOCK_DMODEL)
    cur_batch_seq_len = tl.load(B_Seqlen + cur_batch)
    cur_batch_in_all_start_index = tl.load(B_Start_Loc + cur_batch)

    cur_batch_start_index = max_input_len - cur_batch_seq_len
    cur_batch_end_index = max_input_len

    off_q = cur_batch * stride_qbs + cur_head * stride_qh + offs_d * stride_qd

    offs_n = start_n * BLOCK_N + tl.arange(0, BLOCK_N)

    block_stard_index = start_n * BLOCK_N
    block_mask = tl.where(block_stard_index < cur_batch_seq_len, 1, 0)

    for start_mark in range(0, block_mask, 1):
        q = tl.load(Q + off_q + start_mark)
        offs_n_new = cur_batch_start_index + offs_n
        k_loc = tl.load(B_Loc + stride_b_loc_b * cur_batch + stride_b_loc_s * offs_n_new, mask=offs_n_new < cur_batch_end_index, other=0)
        off_k = k_loc[:, None] * stride_kbs + cur_kv_head * stride_kh + offs_d[None, :] * stride_kd
        k = tl.load(K + off_k, mask=offs_n_new[:, None] < cur_batch_end_index, other=0.0)
        att_value = tl.sum(q[None, :] * k, 1)
        att_value *= sm_scale
        off_o = cur_head * att_stride_h + (cur_batch_in_all_start_index + offs_n) * att_stride_bs
        tl.store(Att_Out + off_o, att_value, mask=offs_n_new < cur_batch_end_index)
    return


@torch.no_grad()
def token_att_fwd(q, k, att_out, B_Loc, B_Start_Loc, B_Seqlen, max_input_len):
    BLOCK = 32
    # shape constraints
    Lq, Lk = q.shape[-1], k.shape[-1]
    assert Lq == Lk
    assert Lk in {16, 32, 64, 128}
    sm_scale = 1.0 / (Lk ** 0.5)

    batch, head_num = B_Loc.shape[0], q.shape[1]

    grid = (batch, head_num, triton.cdiv(max_input_len, BLOCK))
    kv_group_num = q.shape[1] // k.shape[1]

    num_warps = 4 if Lk <= 64 else 8
    num_warps = 2

    _fwd_kernel_token_att1[grid](
        q, k, sm_scale, B_Loc, B_Start_Loc, B_Seqlen, max_input_len,
        att_out,
        B_Loc.stride(0), B_Loc.stride(1),
        q.stride(0), q.stride(1), q.stride(2),
        k.stride(0), k.stride(1), k.stride(2),
        att_out.stride(0), att_out.stride(1),
        kv_group_num=kv_group_num,
        BLOCK_DMODEL=Lk,
        BLOCK_N=BLOCK,
        num_warps=num_warps,
        num_stages=1,
    )
    return
