import torch

import triton
import triton.language as tl
import math
import torch.nn.functional as F

from lightllm.utils.device_utils import get_device_sm_count


@triton.jit
def _fwd_kernel(
    Q_nope,
    Q_rope,
    KV_nope,
    KV_rope,
    q_nope_dim: tl.constexpr,
    q_rope_dim: tl.constexpr,
    qhead: tl.constexpr,
    sm_scale,
    B_Start_Loc,
    B_Seqlen,
    Out,
    Req_to_tokens,
    B_req_idx,
    batch_size: tl.constexpr,
    mtp_size: tl.constexpr,
    stride_qnope_bs,
    stride_qnope_h,
    stride_qrope_bs,
    stride_qrope_h,
    stride_kvnope_bs,
    stride_kvrope_bs,
    stride_o_bs,
    stride_req_to_token_b,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    sm_count: tl.constexpr,
):
    pid = tl.program_id(0)
    offs_m = tl.arange(0, BLOCK_M)
    offs_n = tl.arange(0, BLOCK_N)

    # q
    offs_d = tl.arange(0, q_nope_dim)
    offs_dv = tl.arange(0, q_rope_dim)
    offs_md = offs_m[:, None] * stride_qnope_h + offs_d[None, :]
    offs_mdv = offs_m[:, None] * stride_qrope_h + offs_dv[None, :]

    for cur_batch in range(pid, batch_size, sm_count):
        cur_batch_in_all_start_index = tl.load(B_Start_Loc + cur_batch)
        cur_batch_seq_len = tl.load(B_Seqlen + cur_batch)
        cur_batch_req_idx = tl.load(B_req_idx + cur_batch)
        nblock_num = tl.cdiv(cur_batch_seq_len, BLOCK_N)

        offs_q = cur_batch_in_all_start_index * stride_qnope_bs + offs_md
        offs_qv = cur_batch_in_all_start_index * stride_qrope_bs + offs_mdv

        q_nope = tl.load(Q_nope + offs_q)
        q_rope = tl.load(Q_rope + offs_qv)
        req_to_tokens_ptr = Req_to_tokens + stride_req_to_token_b * cur_batch_req_idx

        m_i = tl.zeros([BLOCK_M], dtype=tl.float32) - float("inf")
        l_i = tl.zeros([BLOCK_M], dtype=tl.float32)
        acc = tl.zeros([BLOCK_M, q_nope_dim], dtype=tl.float32)

        # 从后往前遍历KV，只有第一个block需要causal mask
        first_iter = True
        for block_idx in tl.range(nblock_num - 1, -1, -1, flatten=False, warp_specialize=False):
            start_n = block_idx * BLOCK_N

            offs_n_new = start_n + offs_n
            seq_n_mask = offs_n_new < cur_batch_seq_len
            seq_n_mask_2d = seq_n_mask[None, :]
            kv_loc = tl.load(
                req_to_tokens_ptr + offs_n_new,
                mask=seq_n_mask,
                other=0,
            ).to(tl.int64)
            offs_kv = kv_loc[None, :] * stride_kvnope_bs + offs_d[:, None]
            kv_nope = tl.load(KV_nope + offs_kv, mask=seq_n_mask_2d, other=0.0)
            offs_kv_rope = kv_loc[None, :] * stride_kvrope_bs + offs_dv[:, None]
            kv_rope = tl.load(KV_rope + offs_kv_rope, mask=seq_n_mask_2d, other=0.0)

            qk = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32)

            qk = tl.dot(q_nope, kv_nope)
            qk += tl.dot(q_rope, kv_rope)
            qk *= sm_scale

            # MTP causal mask: 只在第一次循环（最后一个block）需要mask
            if first_iter:
                # token_idx: 第i个token在BLOCK_M中的索引 [0, 0, ..., 1, 1, ..., mtp_size-1]
                token_idx = offs_m // qhead  # [BLOCK_M]
                # 每个token的最大可见位置: cur_batch_seq_len - mtp_size + 1 + token_idx
                max_visible_pos = cur_batch_seq_len - mtp_size + 1 + token_idx  # [BLOCK_M]
                # 生成阶梯状causal mask
                mask = offs_n_new[None, :] < max_visible_pos[:, None]  # [BLOCK_M, BLOCK_N]
                qk = tl.where(mask, qk, float("-inf"))

            # max_get_scale
            m_ij = tl.max(qk, 1)
            if first_iter:
                m_i = m_ij
                # 处理 m_i = -inf：避免 exp(-inf - (-inf)) = NaN
                qk_safe = tl.where(m_i[:, None] == float("-inf"), float("-inf"), qk - m_i[:, None])
                qk = tl.exp(qk_safe)
                acc = tl.dot(qk.to(kv_nope.dtype), tl.trans(kv_nope))
                l_ij = tl.sum(qk, 1)
                l_i = l_ij
            else:
                m_i_new = tl.maximum(m_i, m_ij)
                alpha = tl.exp(m_i - m_i_new)
                m_i = m_i_new
                qk = tl.exp(qk - m_i[:, None])
                acc = acc * alpha[:, None] + tl.dot(qk.to(kv_nope.dtype), tl.trans(kv_nope))
                l_ij = tl.sum(qk, 1)
                l_i = alpha * l_i + l_ij

            first_iter = False

        inv_l_i = 1.0 / l_i
        acc = acc * inv_l_i[:, None]
        offs_o = cur_batch_in_all_start_index * stride_o_bs + offs_md
        tl.store(Out + offs_o, acc)

    return


@torch.no_grad()
def fa3_mla_mtp(
    q_nope,
    q_rope,
    kv_nope,
    kv_rope,
    o,
    b_req_idx,
    b_start_loc,
    b_seq_len,
    mtp_size,
    req_to_token_indexs,
):

    q_nope_dim = q_nope.shape[-1]
    q_rope_dim = q_rope.shape[-1]
    assert q_nope_dim == kv_nope.shape[-1]
    assert q_rope_dim == kv_rope.shape[-1]

    batch_size, qhead = b_seq_len.shape[0], q_nope.shape[1]
    BLOCK_M, BLOCK_N = mtp_size * qhead, 64

    sm_count = get_device_sm_count()
    num_blocks_m = triton.cdiv(batch_size * qhead * mtp_size, BLOCK_M)
    grid = (min(sm_count, num_blocks_m),)
    num_warps = 8
    softmax_scale = 1.0 / math.sqrt(q_nope_dim + q_rope_dim)
    num_stages = 1

    _fwd_kernel[grid](
        q_nope,
        q_rope,
        kv_nope,
        kv_rope,
        q_nope_dim,
        q_rope_dim,
        qhead,
        softmax_scale,
        b_start_loc,
        b_seq_len,
        o,
        req_to_token_indexs,
        b_req_idx,
        batch_size,
        mtp_size,
        q_nope.stride(0),  # stride_qnope_bs
        q_nope.stride(1),  # stride_qnope_h
        q_rope.stride(0),  # stride_qrope_bs
        q_rope.stride(1),  # stride_qrope_h
        kv_nope.stride(0),  # stride_kvnope_bs
        kv_rope.stride(0),  # stride_kvrope_bs
        o.stride(0),  # stride_o_bs
        req_to_token_indexs.stride(0),  # stride_req_to_token_b
        BLOCK_M=BLOCK_M,
        BLOCK_N=BLOCK_N,
        sm_count=sm_count,
        num_warps=num_warps,
        num_stages=num_stages,
    )
    return


def test():
    import torch
    import numpy as np

    Z, N_CTX, H, S_Q, D_HEAD, ROPE_HEAD = 128, 8192, 16, 2, 512, 64
    prompt_cache_len = N_CTX - S_Q
    dtype = torch.float16

    q = torch.empty((Z * S_Q, H, D_HEAD), dtype=dtype, device="cuda").normal_(mean=0.3, std=0.2)
    q_rope = torch.empty((Z * S_Q, H, ROPE_HEAD), dtype=dtype, device="cuda").normal_(mean=0.3, std=0.2)

    kv = torch.empty((Z * N_CTX, 1, D_HEAD), dtype=dtype, device="cuda").normal_(mean=0.3, std=0.2)
    kv_rope = torch.empty((Z * N_CTX, 1, ROPE_HEAD), dtype=dtype, device="cuda").normal_(mean=0.3, std=0.2)

    o = torch.empty((Z * S_Q, H, D_HEAD), dtype=dtype, device="cuda").normal_(mean=0.7, std=0.2)

    req_to_token_indexs = torch.zeros((1000, N_CTX + 10), dtype=torch.int32, device="cuda")
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

    fa3_mla_mtp(
        q,
        q_rope,
        kv,
        kv_rope,
        o,
        b_req_idx,
        b_start_loc,
        b_seq_len,
        S_Q,
        req_to_token_indexs,
    )


if __name__ == "__main__":
    test()
