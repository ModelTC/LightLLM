import torch
import triton
import triton.language as tl

from lightllm.utils.device_utils import is_tesla


@triton.jit
def _fwd_kernel(
    Q,
    K,
    V,
    sm_scale,
    Out,
    B_Start_Loc,
    B_Seqlen,
    Req_to_tokens,
    B_req_idx,
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
    b_prompt_cache_len,
    b_image_token_end,
    H: tl.constexpr,
    QK_HEAD_DIM: tl.constexpr,
    V_HEAD_DIM: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    start_m = tl.program_id(0)
    cur_bh = tl.program_id(1)
    cur_batch = cur_bh // H
    cur_head = cur_bh % H

    cur_kv_head = cur_head // kv_group_num

    cur_batch_in_all_start_index = tl.load(B_Start_Loc + cur_batch)
    prompt_cache_len = tl.load(b_prompt_cache_len + cur_batch)
    total_len = tl.load(B_Seqlen + cur_batch)
    cur_batch_seq_len = total_len - prompt_cache_len  # NEW len
    cur_batch_req_idx = tl.load(B_req_idx + cur_batch)

    block_start_loc = BLOCK_M * start_m
    if block_start_loc >= cur_batch_seq_len:
        return

    offs_n = tl.arange(0, BLOCK_N)
    offs_d_qk = tl.arange(0, QK_HEAD_DIM)
    offs_d_v = tl.arange(0, V_HEAD_DIM)
    offs_m = block_start_loc + tl.arange(0, BLOCK_M)

    # Q pointers
    off_q = (
        (cur_batch_in_all_start_index + offs_m[:, None]) * stride_qbs
        + cur_head * stride_qh
        + offs_d_qk[None, :] * stride_qd
    )

    q_valid = offs_m < cur_batch_seq_len
    q = tl.load(Q + off_q, mask=q_valid[:, None], other=0.0)

    # online softmax state
    m_i = tl.zeros([BLOCK_M], dtype=tl.float32) - float("inf")
    l_i = tl.zeros([BLOCK_M], dtype=tl.float32)
    acc = tl.zeros([BLOCK_M, V_HEAD_DIM], dtype=tl.float32)
    # absolute q positions in the request
    q_pos = prompt_cache_len + offs_m  # [M]
    q_image_end = tl.load(
        b_image_token_end + cur_batch_in_all_start_index + offs_m, mask=q_valid, other=0
    )

    causal_end = tl.minimum(prompt_cache_len + block_start_loc + BLOCK_M, total_len)
    block_image_end = tl.minimum(tl.max(q_image_end, axis=0), total_len)
    block_end_loc = tl.maximum(causal_end, block_image_end)

    for start_n in range(0, block_end_loc, BLOCK_N):
        start_n = tl.multiple_of(start_n, BLOCK_N)

        k_pos = start_n + offs_n  # [N]
        k_valid = k_pos < block_end_loc

        # map logical pos -> mem_index (for K/V)
        kv_loc = tl.load(
            Req_to_tokens
            + stride_req_to_tokens_b * cur_batch_req_idx
            + stride_req_to_tokens_s * k_pos,
            mask=k_valid,
            other=0,
        ).to(tl.int64)

        # load K
        off_k = (
            kv_loc[None, :] * stride_kbs
            + cur_kv_head * stride_kh
            + offs_d_qk[:, None] * stride_kd
        )
        k = tl.load(K + off_k, mask=k_valid[None, :], other=0.0)
        qk = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32)
        qk += tl.dot(q, k)

        causal_mask = q_pos[:, None] >= k_pos[None, :]
        image_mask = k_pos[None, :] < q_image_end[:, None]
        mask = (causal_mask | image_mask) & k_valid[None, :]
        qk = tl.where(mask, qk * sm_scale, -1.0e8)

        # online softmax
        m_ij = tl.maximum(m_i, tl.max(qk, 1))
        qk -= m_ij[:, None]
        p = tl.math.exp2(qk)
        l_ij = tl.sum(p, 1)

        alpha = tl.math.exp2(m_i - m_ij)
        l_i = l_i * alpha + l_ij
        acc = acc * alpha[:, None]

        # load V
        off_v = (
            kv_loc[:, None] * stride_vbs
            + cur_kv_head * stride_vh
            + offs_d_v[None, :] * stride_vd
        )
        v = tl.load(V + off_v, mask=k_valid[:, None], other=0.0)

        p = p.to(v.dtype)
        acc = tl.dot(p, v, acc)

        m_i = m_ij

    acc = acc / l_i[:, None]

    off_o = (
        (cur_batch_in_all_start_index + offs_m[:, None]) * stride_obs
        + cur_head * stride_oh
        + offs_d_v[None, :] * stride_od
    )
    tl.store(Out + off_o, acc, mask=q_valid[:, None])


@torch.no_grad()
def context_attention_fwd_neo(
    q,
    k,
    v,
    o,
    b_req_idx,
    b_start_loc,
    b_seq_len,
    b_prompt_cache_len,
    max_input_len,
    req_to_token_indexs,
    b_image_token_end,
):
    BLOCK_M = 128 if not is_tesla() else 64

    Lq, Lk, Lv = q.shape[-1], k.shape[-1], v.shape[-1]
    assert Lq == Lk and Lk == Lv
    assert Lk in {16, 32, 64, 128, 256}
    sm_scale = 1.0 / (Lq**0.5) * 1.4426950408889634

    batch, head = b_seq_len.shape[0], q.shape[1]
    kv_group_num = q.shape[1] // k.shape[1]

    grid = lambda meta: (triton.cdiv(max_input_len, meta["BLOCK_M"]), batch * head, 1)

    BLOCK_N = BLOCK_M
    num_warps = 4 if Lk <= 64 else 8
    num_stages = 1

    _fwd_kernel[grid](
        q,
        k,
        v,
        sm_scale,
        o,
        b_start_loc,
        b_seq_len,
        req_to_token_indexs,
        b_req_idx,
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
        req_to_token_indexs.stride(0),
        req_to_token_indexs.stride(1),
        kv_group_num=kv_group_num,
        b_prompt_cache_len=b_prompt_cache_len,
        b_image_token_end=b_image_token_end,
        H=head,
        QK_HEAD_DIM=Lk,
        V_HEAD_DIM=Lk,
        BLOCK_M=BLOCK_M,
        BLOCK_N=BLOCK_N,
        num_warps=num_warps,
        num_stages=num_stages,
    )
