import torch

import triton
import triton.language as tl
import torch.nn.functional as F
from .token_attention_nopad_att1 import token_att_fwd
from .token_attention_nopad_softmax import token_softmax_fwd
from .token_attention_nopad_reduceV import token_att_fwd2


@triton.jit
def _fwd_kernel(
    Q, K, V, sm_scale, Alibi, B_Loc, B_Seqlen, max_input_len,
    Out,
    stride_qbs, stride_qh, stride_qd,
    stride_kbs, stride_kh, stride_kd,
    stride_vbs, stride_vh, stride_vd,
    stride_obs, stride_oh, stride_od,
    stride_b_loc_b, stride_b_loc_s,
    BLOCK_DMODEL: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    cur_batch = tl.program_id(0)
    cur_head = tl.program_id(1)

    cur_batch_seq_len = tl.load(B_Seqlen + cur_batch)

    # initialize offsets
    offs_n = tl.arange(0, BLOCK_N)
    offs_d = tl.arange(0, BLOCK_DMODEL)

    off_q = cur_batch * stride_qbs + cur_head * stride_qh + offs_d * stride_qd
    off_k = cur_head * stride_kh + offs_d[None, :] * stride_kd
    off_v = cur_head * stride_vh + offs_d[None, :] * stride_vd
    off_b_loc = cur_batch * stride_b_loc_b + (max_input_len - cur_batch_seq_len) * stride_b_loc_s

    q = tl.load(Q + off_q)

    k_ptrs = K + off_k
    v_ptrs = V + off_v

    m_i = -float("inf")
    l_i = 0.0
    acc = tl.zeros([BLOCK_DMODEL], dtype=tl.float32)

    alibi_m = tl.load(Alibi + cur_head)

    for start_n in range(0, cur_batch_seq_len, BLOCK_N):
        start_n = tl.multiple_of(start_n, BLOCK_N)
        # -- compute qk ----
        k_index = tl.load(B_Loc + off_b_loc + (start_n + offs_n) * stride_b_loc_s, mask=(start_n + offs_n) < cur_batch_seq_len, other=0)
        k = tl.load(k_ptrs + k_index[:, None] * stride_kbs, mask=(start_n + offs_n[:, None]) < cur_batch_seq_len, other=0.0)

        qk = tl.zeros([BLOCK_N,], dtype=tl.float32)
        qk += tl.sum(q[None, :] * k, 1)
        qk *= sm_scale

        alibi_loc = cur_batch_seq_len - 1 - (start_n + offs_n)
        qk -= alibi_loc * alibi_m

        qk = tl.where(cur_batch_seq_len > (start_n + offs_n), qk, float("-inf"))

        m_ij = tl.max(qk, 0)
        p = tl.exp(qk - m_ij)
        l_ij = tl.sum(p, 0)
        # -- update m_i and l_i
        m_i_new = tl.maximum(m_i, m_ij)
        alpha = tl.exp(m_i - m_i_new)
        beta = tl.exp(m_ij - m_i_new)
        l_i_new = alpha * l_i + beta * l_ij
        # -- update output accumulator --
        # scale p
        p_scale = beta / l_i_new
        p = p * p_scale
        # scale acc
        acc_scale = l_i / l_i_new * alpha
        acc = acc * acc_scale
        # update acc
        v_index = k_index
        v = tl.load(v_ptrs + v_index[:, None] * stride_vbs, mask=(start_n + offs_n[:, None]) < cur_batch_seq_len, other=0.0)
        # print(p)
        acc += tl.sum(p[:, None] * v, 0)

        # update m_i and l_i
        l_i = l_i_new
        m_i = m_i_new
    # initialize pointers to output

    off_o = cur_batch * stride_obs + cur_head * stride_oh + offs_d * stride_od
    out_ptrs = Out + off_o
    tl.store(out_ptrs, acc)
    return


# @torch.no_grad()
# def token_attention_fwd(q, k, v, o, alibi, b_loc, b_start_loc, b_seq_len, max_input_len):
#     BLOCK = 128
#     Lq, Lk, Lv = q.shape[-1], k.shape[-1], v.shape[-1]
#     assert Lq == Lk and Lk == Lv
#     assert Lk in {16, 32, 64, 128}

#     sm_scale = 1.0 / (Lq**0.5) # 计算scale系数
#     batch, head = b_seq_len.shape[0], q.shape[1]

#     grid = (batch, head) # batch, head,

#     num_warps = 4 if Lk <= 64 else 8
#     num_warps = 4

#     _fwd_kernel[grid](
#         q, k, v, sm_scale, alibi, b_loc, b_seq_len, max_input_len,
#         o,
#         q.stride(0), q.stride(1), q.stride(2),
#         k.stride(0), k.stride(1), k.stride(2),
#         v.stride(0), v.stride(1), v.stride(2),
#         o.stride(0), o.stride(1), o.stride(2),
#         b_loc.stride(0), b_loc.stride(1),
#         BLOCK_DMODEL=Lk,
#         BLOCK_N=BLOCK,
#         num_warps=num_warps,
#         num_stages=1,
#     )
#     return

@torch.no_grad()
def token_attention_fwd(q, k, v, o, alibi, b_loc, b_start_loc, b_seq_len, max_len_in_batch):
    head_num = k.shape[1]
    batch_size = b_seq_len.shape[0]
    calcu_shape1 = (batch_size, head_num, k.shape[2])
    total_token_num = k.shape[0]

    att_m_tensor = torch.empty((head_num, total_token_num), dtype=q.dtype, device="cuda")

    token_att_fwd(q.view(calcu_shape1),
                  k,
                  att_m_tensor,
                  alibi,
                  b_loc,
                  b_start_loc,
                  b_seq_len,
                  max_len_in_batch)
    prob = torch.empty_like(att_m_tensor)
    token_softmax_fwd(att_m_tensor, b_start_loc, b_seq_len, prob, max_len_in_batch)
    att_m_tensor = None
    token_att_fwd2(prob,
                   v,
                   o.view(calcu_shape1),
                   b_loc,
                   b_start_loc,
                   b_seq_len,
                   max_len_in_batch)
    prob = None
    return


def torch_att(xq, xk, xv, bs, seqlen, num_head, head_dim):
    xq = xq.view(bs, 1, num_head, head_dim)
    xk = xk.view(bs, seqlen, num_head, head_dim)
    xv = xv.view(bs, seqlen, num_head, head_dim)

    logics = torch.sum(xq * xk, dim=3, keepdim=False) * 1 / (head_dim**0.5)
    prob = torch.softmax(logics, dim=1)
    prob = prob.view(bs, seqlen, num_head, 1)

    return torch.sum(prob * xv, dim=1, keepdim=False)


def test():
    import torch

    Z, H, N_CTX, D_HEAD = 22, 112 // 8, 2048, 128
    dtype = torch.float16
    q = torch.empty((Z, H, D_HEAD), dtype=dtype, device="cuda").normal_(mean=0.1, std=0.2)
    k = torch.empty((Z * N_CTX, H, D_HEAD), dtype=dtype, device="cuda").normal_(mean=0.4, std=0.2)
    v = torch.empty((Z * N_CTX, H, D_HEAD), dtype=dtype, device="cuda").normal_(mean=0.3, std=0.2)
    o = torch.empty((Z, H, D_HEAD), dtype=dtype, device="cuda").normal_(mean=0.3, std=0.2)
    alibi = torch.zeros((H,), dtype=torch.float32, device="cuda")

    max_input_len = N_CTX
    b_start_loc = torch.zeros((Z,), dtype=torch.int32, device="cuda")
    b_loc = torch.zeros((Z, N_CTX), dtype=torch.int32, device="cuda")
    b_seq_len = torch.ones((Z,), dtype=torch.int32, device="cuda")

    b_seq_len[:] = N_CTX
    b_start_loc[0] = 0
    b_start_loc[1] = N_CTX
    b_start_loc[2] = 2 * N_CTX
    b_start_loc[3] = 3 * N_CTX

    for i in range(Z):
        b_loc[i, :] = torch.arange(i * N_CTX, (i + 1) * N_CTX, dtype=torch.int32, device="cuda")

    token_attention_fwd(q, k, v, o, alibi, b_loc, b_seq_len, max_input_len)
    import time
    torch.cuda.synchronize()
    start = time.time()
    token_attention_fwd(q, k, v, o, alibi, b_loc, b_seq_len, max_input_len)
    torch.cuda.synchronize()
    print("cost time:", (time.time() - start) * 1000)

    torch_att(q, k, v, Z, N_CTX, H, D_HEAD)
    torch.cuda.synchronize()
    start = time.time()
    torch_out = torch_att(q, k, v, Z, N_CTX, H, D_HEAD)
    torch.cuda.synchronize()
    print("cost time:", (time.time() - start) * 1000)

    print("max ", torch.max(torch.abs(torch_out - o)))
    print("mean ", torch.mean(torch.abs(torch_out - o)))
    assert torch.allclose(torch_out, o, atol=1e-2, rtol=0)
