# From : https://triton-lang.org/main/getting-started/tutorials/06-fused-attention.html

import torch

import triton
import triton.language as tl
import math
import torch.nn.functional as F

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
    H: tl.constexpr,
    BLOCK_DMODEL: tl.constexpr,
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
    cur_batch_seq_len = tl.load(B_Seqlen + cur_batch) - prompt_cache_len
    cur_batch_req_idx = tl.load(B_req_idx + cur_batch)

    block_start_loc = BLOCK_M * start_m

    # initialize offsets
    offs_n = tl.arange(0, BLOCK_N)
    offs_d = tl.arange(0, BLOCK_DMODEL)
    offs_m = block_start_loc + tl.arange(0, BLOCK_M)
    off_q = (
        (cur_batch_in_all_start_index + offs_m[:, None]) * stride_qbs
        + cur_head * stride_qh
        + offs_d[None, :] * stride_qd
    )

    q = tl.load(Q + off_q, mask=offs_m[:, None] < cur_batch_seq_len, other=0.0)

    # initialize pointer to m and l
    m_i = tl.zeros([BLOCK_M], dtype=tl.float32) - float("inf")
    l_i = tl.zeros([BLOCK_M], dtype=tl.float32)
    acc = tl.zeros([BLOCK_M, BLOCK_DMODEL], dtype=tl.float32)

    block_mask = tl.where(block_start_loc < cur_batch_seq_len, 1, 0)
    block_end_loc = tl.minimum(block_start_loc + BLOCK_M + prompt_cache_len, cur_batch_seq_len + prompt_cache_len)

    # causal mask
    for start_n in range(0, block_mask * block_end_loc, BLOCK_N):
        start_n = tl.multiple_of(start_n, BLOCK_N)
        # -- compute qk ----
        kv_loc = tl.load(
            Req_to_tokens + stride_req_to_tokens_b * cur_batch_req_idx + stride_req_to_tokens_s * (start_n + offs_n),
            mask=(start_n + offs_n) < block_end_loc,
            other=0,
        ).to(tl.int64)
        off_k = kv_loc[None, :] * stride_kbs + cur_kv_head * stride_kh + offs_d[:, None] * stride_kd
        k = tl.load(K + off_k, mask=(start_n + offs_n[None, :]) < block_end_loc, other=0.0)
        qk = tl.dot(q, k)

        mask = offs_m[:, None] + prompt_cache_len >= (start_n + offs_n[None, :])
        qk = tl.where(mask, qk * sm_scale, -1.0e8)
        m_ij = tl.maximum(m_i, tl.max(qk, 1))
        qk -= m_ij[:, None]
        p = tl.math.exp2(qk)
        l_ij = tl.sum(p, 1)

        # -- update m_i and l_i
        alpha = tl.math.exp2(m_i - m_ij)
        l_i = l_i * alpha + l_ij
        # -- update output accumulator --
        acc = acc * alpha[:, None]
        # update acc
        off_v = kv_loc[:, None] * stride_vbs + cur_kv_head * stride_vh + offs_d[None, :] * stride_vd
        v = tl.load(V + off_v, mask=(start_n + offs_n[:, None]) < block_end_loc, other=0.0)
        p = p.to(v.dtype)
        acc = tl.dot(p, v, acc)
        # update m_i and l_i
        m_i = m_ij

    acc = acc / l_i[:, None]
    off_o = (
        (cur_batch_in_all_start_index + offs_m[:, None]) * stride_obs
        + cur_head * stride_oh
        + offs_d[None, :] * stride_od
    )
    out_ptrs = Out + off_o
    tl.store(out_ptrs, acc, mask=offs_m[:, None] < cur_batch_seq_len)


@torch.no_grad()
def context_attention_fwd(
    q, k, v, o, b_req_idx, b_start_loc, b_seq_len, b_prompt_cache_len, max_input_len, req_to_token_indexs
):
    BLOCK_M = 128 if not is_tesla() else 64
    # shape constraints
    Lq, Lk, Lv = q.shape[-1], k.shape[-1], v.shape[-1]
    assert Lq == Lk and Lk == Lv
    assert Lk in {16, 32, 64, 128, 256}

    # 计算scale系数, 并乘以 1/log(2) = 1.4426950408889634,
    # 算子内部使用 tl.math.exp2 来使计算与标准attention等价。
    sm_scale = 1.0 / (Lq ** 0.5) * 1.4426950408889634
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
        H=head,
        BLOCK_DMODEL=Lk,
        BLOCK_M=BLOCK_M,
        BLOCK_N=BLOCK_N,
        num_warps=num_warps,
        num_stages=num_stages,
    )


@triton.jit
def _fwd_kernel_no_prompt_cache(
    Q,
    K,
    V,
    sm_scale,
    Out,
    B_Start_Loc,
    B_Seqlen,
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
    kv_group_num,
    H,
    BLOCK_DMODEL: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    start_m = tl.program_id(0)
    cur_bh = tl.program_id(1)
    cur_batch = cur_bh // H
    cur_head = cur_bh % H

    cur_kv_head = cur_head // kv_group_num

    cur_batch_in_all_start_index = tl.load(B_Start_Loc + cur_batch)
    cur_batch_seq_len = tl.load(B_Seqlen + cur_batch)

    block_start_loc = BLOCK_M * start_m

    # initialize offsets
    offs_n = tl.arange(0, BLOCK_N)
    offs_d = tl.arange(0, BLOCK_DMODEL)
    offs_m = block_start_loc + tl.arange(0, BLOCK_M)
    off_q = (
        (cur_batch_in_all_start_index + offs_m[:, None]) * stride_qbs
        + cur_head * stride_qh
        + offs_d[None, :] * stride_qd
    )
    off_k = offs_n[None, :] * stride_kbs + cur_kv_head * stride_kh + offs_d[:, None] * stride_kd
    off_v = offs_n[:, None] * stride_vbs + cur_kv_head * stride_vh + offs_d[None, :] * stride_vd

    q = tl.load(Q + off_q, mask=offs_m[:, None] < cur_batch_seq_len, other=0.0)

    k_ptrs = K + off_k
    v_ptrs = V + off_v

    # initialize pointer to m and l
    m_i = tl.zeros([BLOCK_M], dtype=tl.float32) - float("inf")
    l_i = tl.zeros([BLOCK_M], dtype=tl.float32)
    acc = tl.zeros([BLOCK_M, BLOCK_DMODEL], dtype=tl.float32)

    block_mask = tl.where(block_start_loc < cur_batch_seq_len, 1, 0)
    block_end_loc = tl.minimum(block_start_loc + BLOCK_M, cur_batch_seq_len)

    # causal mask
    for start_n in range(0, block_mask * block_end_loc, BLOCK_N):
        start_n = tl.multiple_of(start_n, BLOCK_N)
        # -- compute qk ----
        k = tl.load(
            k_ptrs + (cur_batch_in_all_start_index + start_n) * stride_kbs,
            mask=(start_n + offs_n[None, :]) < block_end_loc,
            other=0,
        )
        qk = tl.dot(q, k)

        mask = offs_m[:, None] >= (start_n + offs_n[None, :])
        qk = tl.where(mask, qk * sm_scale, -1.0e8)
        m_ij = tl.maximum(m_i, tl.max(qk, 1))
        qk -= m_ij[:, None]
        p = tl.math.exp2(qk)
        l_ij = tl.sum(p, 1)

        # -- update m_i and l_i
        alpha = tl.math.exp2(m_i - m_ij)
        l_i = l_i * alpha + l_ij
        # -- update output accumulator --
        acc = acc * alpha[:, None]
        # update acc
        v = tl.load(
            v_ptrs + (cur_batch_in_all_start_index + start_n) * stride_vbs,
            mask=(start_n + offs_n[:, None]) < block_end_loc,
            other=0.0,
        )
        p = p.to(v.dtype)
        acc = tl.dot(p, v, acc)
        # update m_i
        m_i = m_ij

    acc = acc / l_i[:, None]
    off_o = (
        (cur_batch_in_all_start_index + offs_m[:, None]) * stride_obs
        + cur_head * stride_oh
        + offs_d[None, :] * stride_od
    )
    out_ptrs = Out + off_o
    tl.store(out_ptrs, acc, mask=offs_m[:, None] < cur_batch_seq_len)


@torch.no_grad()
def context_attention_fwd_no_prompt_cache(q, k, v, o, b_start_loc, b_seq_len, max_input_len):

    BLOCK_M = 128 if not is_tesla() else 64
    # shape constraints
    Lq, Lk, Lv = q.shape[-1], k.shape[-1], v.shape[-1]
    assert Lq == Lk and Lk == Lv
    assert Lk in {16, 32, 64, 128, 256}

    # 计算scale系数, 并乘以 1/log(2) = 1.4426950408889634,
    # 算子内部使用 tl.math.exp2 来使计算与标准attention等价。
    sm_scale = 1.0 / (Lq ** 0.5) * 1.4426950408889634
    batch, head = b_seq_len.shape[0], q.shape[1]
    kv_group_num = q.shape[1] // k.shape[1]

    grid = (triton.cdiv(max_input_len, BLOCK_M), batch * head, 1)
    BLOCK_N = BLOCK_M
    num_warps = 4 if Lk <= 64 else 8
    num_stages = 1

    _fwd_kernel_no_prompt_cache[grid](
        q,
        k,
        v,
        sm_scale,
        o,
        b_start_loc,
        b_seq_len,
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
        kv_group_num=kv_group_num,
        H=head,
        BLOCK_DMODEL=Lk,
        BLOCK_M=BLOCK_M,
        BLOCK_N=BLOCK_N,
        num_warps=num_warps,
        num_stages=num_stages,
    )
    return


@triton.jit
def _fwd_kernel_contiguous_kv(
    Q,
    K,
    V,
    sm_scale,
    Out,
    B_Start_Loc,
    B_kv_start_loc,
    B_Seqlen,
    b_prompt_cache_len,
    stride_qbs,
    stride_qh,
    stride_qd,
    stride_ks,
    stride_kh,
    stride_kd,
    stride_vs,
    stride_vh,
    stride_vd,
    stride_obs,
    stride_oh,
    stride_od,
    kv_group_num,
    H: tl.constexpr,
    BLOCK_DMODEL: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    start_m = tl.program_id(0)
    cur_bh = tl.program_id(1)
    cur_batch = cur_bh // H
    cur_head = cur_bh % H

    cur_kv_head = cur_head // kv_group_num
    prompt_cache_len = tl.load(b_prompt_cache_len + cur_batch)
    cur_batch_in_all_start_index = tl.load(B_Start_Loc + cur_batch)
    cur_batch_seq_len = tl.load(B_Seqlen + cur_batch) - prompt_cache_len
    kv_start_loc = tl.load(B_kv_start_loc + cur_batch)

    block_start_loc = BLOCK_M * start_m

    # initialize offsets
    offs_n = tl.arange(0, BLOCK_N)
    offs_d = tl.arange(0, BLOCK_DMODEL)
    offs_m = block_start_loc + tl.arange(0, BLOCK_M)
    off_q = (
        (cur_batch_in_all_start_index + offs_m[:, None]) * stride_qbs
        + cur_head * stride_qh
        + offs_d[None, :] * stride_qd
    )
    q = tl.load(Q + off_q, mask=offs_m[:, None] < cur_batch_seq_len, other=0.0)

    # initialize pointer to m and l
    m_i = tl.zeros([BLOCK_M], dtype=tl.float32) - float("inf")
    l_i = tl.zeros([BLOCK_M], dtype=tl.float32)
    acc = tl.zeros([BLOCK_M, BLOCK_DMODEL], dtype=tl.float32)

    stride_ks = tl.cast(stride_ks, tl.int64)
    stride_vs = tl.cast(stride_vs, tl.int64)

    block_mask = tl.where(block_start_loc < cur_batch_seq_len, 1, 0)
    block_end_loc = tl.minimum(block_start_loc + BLOCK_M + prompt_cache_len, cur_batch_seq_len + prompt_cache_len)
    # causal mask
    for start_n in range(0, block_mask * block_end_loc, BLOCK_N):
        start_n = tl.multiple_of(start_n, BLOCK_N)
        # -- compute qk ----
        # k = tl.load(
        #     k_ptrs + (start_n + offs_n[None, :]) * stride_ks,
        #     mask=(start_n + offs_n[None, :]) < block_end_loc,
        #     other=0,
        # )
        off_k = (
            (kv_start_loc + start_n + offs_n[None, :]) * stride_ks
            + cur_kv_head * stride_kh
            + offs_d[:, None] * stride_kd
        )
        k = tl.load(K + off_k, mask=(start_n + offs_n[None, :]) < block_end_loc, other=0.0)

        qk = tl.dot(q, k)
        mask = (offs_m[:, None] + prompt_cache_len) >= (start_n + offs_n[None, :])
        qk = tl.where(mask, qk * sm_scale, -1.0e8)
        m_ij = tl.maximum(m_i, tl.max(qk, 1))
        qk -= m_ij[:, None]
        p = tl.math.exp2(qk)
        l_ij = tl.sum(p, 1)

        # -- update m_i and l_i
        alpha = tl.math.exp2(m_i - m_ij)
        l_i = l_i * alpha + l_ij
        # -- update output accumulator --
        acc = acc * alpha[:, None]
        # update acc
        # v = tl.load(
        #     v_ptrs + (start_n + offs_n[:, None]) * stride_vs,
        #     mask=(start_n + offs_n[:, None]) < block_end_loc,
        #     other=0.0,
        # )
        off_v = (
            (kv_start_loc + start_n + offs_n[:, None]) * stride_vs
            + cur_kv_head * stride_vh
            + offs_d[None, :] * stride_vd
        )
        v = tl.load(V + off_v, mask=(start_n + offs_n[:, None]) < block_end_loc, other=0.0)

        p = p.to(v.dtype)
        acc = tl.dot(p, v, acc)
        # update m_i
        m_i = m_ij

    acc = acc / l_i[:, None]
    off_o = (
        (cur_batch_in_all_start_index + offs_m[:, None]) * stride_obs
        + cur_head * stride_oh
        + offs_d[None, :] * stride_od
    )
    out_ptrs = Out + off_o
    tl.store(out_ptrs, acc, mask=offs_m[:, None] < cur_batch_seq_len)


@torch.no_grad()
def context_attention_fwd_contiguous_kv(
    q, k, v, o, b_start_loc, b_kv_start_loc, b_seq_len, max_q_input_len, b_prompt_cache_len
):
    BLOCK_M = 128 if not is_tesla() else 64
    # shape constraints
    Lq, Lk, Lv = q.shape[-1], k.shape[-1], v.shape[-1]
    assert Lq == Lk and Lk == Lv
    assert Lk in {16, 32, 64, 128, 256}

    # 计算scale系数, 并乘以 1/log(2) = 1.4426950408889634,
    # 算子内部使用 tl.math.exp2 来使计算与标准attention等价。
    sm_scale = 1.0 / (Lq ** 0.5) * 1.4426950408889634
    batch, head = b_seq_len.shape[0], q.shape[1]
    kv_group_num = q.shape[1] // k.shape[1]

    grid = lambda meta: (triton.cdiv(max_q_input_len, meta["BLOCK_M"]), batch * head, 1)
    BLOCK_N = BLOCK_M
    num_warps = 4 if Lk <= 64 else 8
    num_stages = 1

    _fwd_kernel_contiguous_kv[grid](
        Q=q,
        K=k,
        V=v,
        sm_scale=sm_scale,
        Out=o,
        B_Start_Loc=b_start_loc,
        B_kv_start_loc=b_kv_start_loc,
        B_Seqlen=b_seq_len,
        b_prompt_cache_len=b_prompt_cache_len,
        stride_qbs=q.stride(0),
        stride_qh=q.stride(1),
        stride_qd=q.stride(2),
        stride_ks=k.stride(0),
        stride_kh=k.stride(1),
        stride_kd=k.stride(2),
        stride_vs=v.stride(0),
        stride_vh=v.stride(1),
        stride_vd=v.stride(2),
        stride_obs=o.stride(0),
        stride_oh=o.stride(1),
        stride_od=o.stride(2),
        kv_group_num=kv_group_num,
        H=head,
        BLOCK_DMODEL=Lk,
        BLOCK_M=BLOCK_M,
        BLOCK_N=BLOCK_N,
        num_warps=num_warps,
        num_stages=num_stages,
    )


def torch_context_attention_fwd(q, k, v, o, b_req_idx, b_start_loc, b_seq_len, b_prompt_cache_len, req_to_token_indexs):

    batch = b_start_loc.shape[0]
    print(q.shape)
    for i in range(batch):
        req_idx = b_req_idx[i]
        start_loc = b_start_loc[i]
        seq_len = b_seq_len[i]
        prompt_cache_len = b_prompt_cache_len[i]
        cur_q = q[start_loc : start_loc + seq_len - prompt_cache_len, :, :]
        cur_q = cur_q.clone().to(torch.float32)
        print(cur_q.shape)

        kv_loc = req_to_token_indexs[req_idx, :seq_len]
        cur_k = k[kv_loc, :, :]
        cur_k = cur_k.clone().to(torch.float32)

        cur_v = v[kv_loc, :, :]
        cur_v = cur_v.clone().to(torch.float32)

        cur_q = cur_q.transpose(0, 1)
        cur_k = cur_k.transpose(0, 1)
        cur_v = cur_v.transpose(0, 1)

        dk = cur_q.shape[-1]

        p = torch.matmul(cur_q, cur_k.transpose(-2, -1)) / torch.sqrt(torch.tensor(dk, dtype=torch.float32))

        q_index = torch.arange(cur_q.shape[1]).unsqueeze(-1).to(p.device)
        k_index = torch.arange(cur_k.shape[1]).unsqueeze(0).to(p.device)
        mask = (q_index + prompt_cache_len >= k_index).int()
        mask = mask.unsqueeze(0).expand(cur_q.shape[0], -1, -1)

        p = p.masked_fill(mask == 0, float("-inf"))

        s = F.softmax(p, dim=-1)

        o[start_loc : start_loc + seq_len - prompt_cache_len, :, :] = torch.matmul(s, cur_v).transpose(0, 1)


def test():
    import torch
    import numpy as np

    Z, H, N_CTX, D_HEAD = 16, 16, 2048, 128
    dtype = torch.float16
    prompt_cache_len = 128
    q = torch.empty((Z * (N_CTX - prompt_cache_len), H, D_HEAD), dtype=dtype, device="cuda").normal_(mean=0.1, std=0.2)
    k = torch.empty((Z * N_CTX, H, D_HEAD), dtype=dtype, device="cuda").normal_(mean=0.4, std=0.2)
    v = torch.empty((Z * N_CTX, H, D_HEAD), dtype=dtype, device="cuda").normal_(mean=0.3, std=0.2)
    o = torch.empty((Z * (N_CTX - prompt_cache_len), H, D_HEAD), dtype=dtype, device="cuda").normal_(mean=0.3, std=0.2)
    torch_o = torch.empty((Z * (N_CTX - prompt_cache_len), H, D_HEAD), dtype=dtype, device="cuda").normal_(
        mean=0.3, std=0.2
    )
    req_to_token_indexs = torch.empty((1000, N_CTX + 7000), dtype=torch.int32, device="cuda")
    max_input_len = N_CTX
    b_start_loc = torch.zeros((Z,), dtype=torch.int32, device="cuda")
    b_seq_len = torch.ones((Z,), dtype=torch.int32, device="cuda")
    b_req_idx = torch.ones((Z,), dtype=torch.int32, device="cuda")
    b_prompt_cache_len = torch.zeros(Z, dtype=torch.int32, device="cuda")

    for i in range(Z):
        b_seq_len[i] = N_CTX
        b_req_idx[i] = i
        req_to_token_indexs[i][:N_CTX] = torch.tensor(np.arange(N_CTX), dtype=torch.int32).cuda() + N_CTX * i
        if i != 0:
            b_start_loc[i] = b_start_loc[i - 1] + N_CTX - prompt_cache_len
        b_prompt_cache_len[i] = prompt_cache_len
    torch_context_attention_fwd(
        q, k, v, torch_o, b_req_idx, b_start_loc, b_seq_len, b_prompt_cache_len, req_to_token_indexs
    )

    import time

    torch.cuda.synchronize()
    a = time.time()
    for i in range(1000):
        context_attention_fwd(
            q, k, v, o, b_req_idx, b_start_loc, b_seq_len, b_prompt_cache_len, max_input_len, req_to_token_indexs
        )
    torch.cuda.synchronize()
    b = time.time()
    # print(o.shape, torch_out.shape)
    print((b - a))

    print("max ", torch.max(torch.abs(torch_o - o)))
    print("mean ", torch.mean(torch.abs(torch_o - o)))
    assert torch.allclose(torch_o, o, atol=1e-2, rtol=0)


if __name__ == "__main__":
    test()
