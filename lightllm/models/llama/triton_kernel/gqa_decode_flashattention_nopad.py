import torch

import triton
import triton.language as tl
import math
import torch.nn.functional as F


@triton.jit
def _fwd_kernel(
    Q,
    K,
    V,
    sm_scale,
    Req_to_tokens,
    B_req_idx,
    B_seqlen,
    Out,
    stride_qbs,
    stride_qh,
    stride_qd,
    stride_kbs,
    stride_kh,
    stride_kd,
    stride_vbs,
    stride_vh,
    stride_vd,
    stride_obs,
    stride_oh,
    stride_od,
    stride_req_to_tokens_b,
    stride_req_to_tokens_s,
    kv_group_num,
    Q_HEAD_NUM: tl.constexpr,
    BLOCK_DMODEL: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    cur_batch = tl.program_id(0)
    cur_kv_head = tl.program_id(1)

    cur_q_head_offs = tl.arange(0, Q_HEAD_NUM)

    cur_batch_req_idx = tl.load(B_req_idx + cur_batch)
    cur_batch_seq_len = tl.load(B_seqlen + cur_batch)

    # initialize offsets
    offs_n = tl.arange(0, BLOCK_N)
    offs_d = tl.arange(0, BLOCK_DMODEL)

    cur_q_head_range = cur_kv_head * kv_group_num + cur_q_head_offs

    off_q = cur_batch * stride_qbs + cur_q_head_range[:, None] * stride_qh + offs_d[None, :]
    off_k = cur_kv_head * stride_kh + offs_d[:, None]
    off_v = cur_kv_head * stride_vh + offs_d[None, :]

    q = tl.load(Q + off_q, mask=cur_q_head_range[:, None] < (cur_kv_head + 1) * kv_group_num, other=0.0)

    k_ptrs = K + off_k
    v_ptrs = V + off_v

    # initialize pointer to m and l
    m_i = tl.zeros([Q_HEAD_NUM], dtype=tl.float32) - float("inf")
    l_i = tl.zeros([Q_HEAD_NUM], dtype=tl.float32)
    acc = tl.zeros([Q_HEAD_NUM, BLOCK_DMODEL], dtype=tl.float32)

    for start_n in range(0, cur_batch_seq_len, BLOCK_N):
        start_n = tl.multiple_of(start_n, BLOCK_N)
        # -- compute qk ----
        kv_loc = tl.load(
            Req_to_tokens + cur_batch_req_idx * stride_req_to_tokens_b + start_n + offs_n,
            mask=(start_n + offs_n) < cur_batch_seq_len,
            other=0,
        )
        k = tl.load(
            k_ptrs + kv_loc[None, :] * stride_kbs, mask=(start_n + offs_n[None, :]) < cur_batch_seq_len, other=0.0
        )

        qk = tl.zeros([Q_HEAD_NUM, BLOCK_N], dtype=tl.float32)
        qk += tl.dot(q, k)
        qk *= sm_scale
        qk = tl.where(cur_batch_seq_len - 1 >= (start_n + offs_n[None, :]), qk, float("-inf"))

        # -- compute m_ij, p, l_ij
        m_ij = tl.max(qk, 1)
        p = tl.exp(qk - m_ij[:, None])
        l_ij = tl.sum(p, 1)
        # -- update m_i and l_i
        m_i_new = tl.maximum(m_i, m_ij)
        alpha = tl.exp(m_i - m_i_new)
        beta = tl.exp(m_ij - m_i_new)
        l_i_new = alpha * l_i + beta * l_ij
        # -- update output accumulator --
        # scale p
        p_scale = beta / l_i_new
        p = p * p_scale[:, None]
        # scale acc
        acc_scale = l_i / l_i_new * alpha
        acc = acc * acc_scale[:, None]
        # update acc
        v = tl.load(
            v_ptrs + kv_loc[:, None] * stride_vbs, mask=(start_n + offs_n[:, None]) < cur_batch_seq_len, other=0.0
        )

        p = p.to(v.dtype)
        acc += tl.dot(p, v)
        # update m_i and l_i
        l_i = l_i_new
        m_i = m_i_new
    # initialize pointers to output
    off_o = cur_batch * stride_obs + cur_q_head_range[:, None] * stride_oh + offs_d[None, :]
    out_ptrs = Out + off_o
    tl.store(out_ptrs, acc, mask=cur_q_head_range[:, None] < (cur_kv_head + 1) * kv_group_num)
    return


@torch.no_grad()
def gqa_decode_attention_fwd(q, k, v, o, req_to_tokens, b_req_idx, b_seq_len):

    BLOCK = 32
    # shape constraints
    Lq, Lk, Lv = q.shape[-1], k.shape[-1], v.shape[-1]
    assert Lq == Lk and Lk == Lv
    assert Lk in {16, 32, 64, 128}

    sm_scale = 1.0 / (Lq ** 0.5)  # 计算scale系数
    batch = b_req_idx.shape[0]
    kv_group_num = q.shape[1] // k.shape[1]
    kv_head_num = k.shape[1]

    grid = (batch, kv_head_num)

    num_warps = 4  # if Lk <= 64 else 8
    _fwd_kernel[grid](
        q,
        k,
        v,
        sm_scale,
        req_to_tokens,
        b_req_idx,
        b_seq_len,
        o,
        q.stride(0),
        q.stride(1),
        q.stride(2),
        k.stride(0),
        k.stride(1),
        k.stride(2),
        v.stride(0),
        v.stride(1),
        v.stride(2),
        o.stride(0),
        o.stride(1),
        o.stride(2),
        req_to_tokens.stride(0),
        req_to_tokens.stride(1),
        kv_group_num=kv_group_num,
        Q_HEAD_NUM=max(16, triton.next_power_of_2(kv_group_num)),
        BLOCK_DMODEL=Lk,
        BLOCK_N=BLOCK,
        num_warps=num_warps,
        num_stages=2,
    )
    return
