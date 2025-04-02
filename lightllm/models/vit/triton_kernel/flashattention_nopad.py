import torch

import triton
import triton.language as tl
import math
import torch.nn.functional as F

TESLA = "Tesla" in torch.cuda.get_device_name(0)
HOPPER = (
    "H100" in torch.cuda.get_device_name(0)
    or "H200" in torch.cuda.get_device_name(0)
    or "H800" in torch.cuda.get_device_name(0)
    or "Hopper" in torch.cuda.get_device_name(0)
)


if triton.__version__ >= "2.1.0":

    @triton.jit
    def _fwd_kernel(
        Q,
        K,
        V,
        sm_scale,
        seq_len,
        Out,
        q_stride_b,
        q_stride_s,
        q_stride_h,
        q_stride_d,
        k_stride_b,
        k_stride_s,
        k_stride_h,
        k_stride_d,
        v_stride_b,
        v_stride_s,
        v_stride_h,
        v_stride_d,
        o_stride_b,
        o_stride_s,
        o_stride_h,
        o_stride_d,
        BLOCK_M: tl.constexpr,
        BLOCK_DMODEL: tl.constexpr,
        BLOCK_N: tl.constexpr,
    ):
        cur_batch = tl.program_id(2)
        cur_head = tl.program_id(1)
        start_m = tl.program_id(0)

        # initialize offsets
        offs_n = tl.arange(0, BLOCK_N)
        offs_d = tl.arange(0, BLOCK_DMODEL)
        offs_m = start_m * BLOCK_M + tl.arange(0, BLOCK_M)
        off_q = cur_batch * q_stride_b + cur_head * q_stride_h + offs_m[:, None] * q_stride_s + offs_d[None, :]
        q = tl.load(Q + off_q, mask=offs_m[:, None] < seq_len, other=0.0)
        # initialize pointer to m and l
        m_i = tl.zeros([BLOCK_M], dtype=tl.float32) - float("inf")
        l_i = tl.zeros([BLOCK_M], dtype=tl.float32)
        acc = tl.zeros([BLOCK_M, BLOCK_DMODEL], dtype=tl.float32)

        for start_n in range(0, seq_len, BLOCK_N):
            start_n = tl.multiple_of(start_n, BLOCK_N)
            # -- compute qk ----
            off_k = (
                cur_batch * k_stride_b
                + (start_n + offs_n[None, :]) * k_stride_s
                + cur_head * k_stride_h
                + offs_d[:, None]
            )
            k = tl.load(K + off_k, mask=(start_n + offs_n[None, :]) < seq_len, other=0.0)

            qk = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32)
            qk += tl.dot(q, k)
            qk *= sm_scale
            qk += tl.where((start_n + offs_n[None, :]) < seq_len, 0, float("-inf"))

            # -- compute m_ij, p, l_ij
            m_ij = tl.maximum(tl.max(qk, 1), l_i)
            p = tl.exp(qk - m_ij[:, None])
            l_ij = tl.sum(p, 1)

            acc_scale = tl.exp(m_i - m_ij)
            acc = acc * acc_scale[:, None]

            # update acc
            off_v = (
                cur_batch * v_stride_b
                + (start_n + offs_n[:, None]) * v_stride_s
                + cur_head * v_stride_h
                + offs_d[None, :]
            )
            v = tl.load(V + off_v, mask=(start_n + offs_n[:, None]) < seq_len, other=0.0)
            p = p.to(v.dtype)
            acc += tl.dot(p, v)
            # update m_i and l_i
            m_i = m_ij
            l_i_new = tl.exp(l_i - m_ij) + l_ij
            l_i = m_ij + tl.log(l_i_new)

        o_scale = tl.exp(m_i - l_i)
        acc = acc * o_scale[:, None]
        # initialize pointers to output
        off_o = cur_batch * o_stride_b + offs_m[:, None] * o_stride_s + cur_head * o_stride_h + offs_d[None, :]
        out_ptrs = Out + off_o
        tl.store(out_ptrs, acc, mask=offs_m[:, None] < seq_len)
        return

    @torch.no_grad()
    def _flash_attention_triton_fwd(
        q,
        k,
        v,
        o,
    ):
        BLOCK = 64
        # shape constraints
        batch_size, seq_len, head_num, head_dim = q.shape

        sm_scale = 1.0 / (head_dim ** 0.5)  # 计算scale系数
        # grid = (batch_size, head_num, triton.cdiv(seq_len, BLOCK))  # batch, head,
        grid = (triton.cdiv(seq_len, BLOCK), head_num, batch_size)  # batch, head,
        num_warps = 4
        _fwd_kernel[grid](
            q,
            k,
            v,
            sm_scale,
            seq_len,
            o,
            q.stride(0),
            q.stride(1),
            q.stride(2),
            q.stride(3),
            k.stride(0),
            k.stride(1),
            k.stride(2),
            k.stride(3),
            v.stride(0),
            v.stride(1),
            v.stride(2),
            v.stride(3),
            o.stride(0),
            o.stride(1),
            o.stride(2),
            o.stride(3),
            BLOCK_M=BLOCK,
            BLOCK_DMODEL=head_dim,
            BLOCK_N=BLOCK,
            num_warps=num_warps,
            num_stages=2,
        )
        return

else:
    raise Exception("error triton version!")

_flash_attn_v3_available = False
try:
    from flash_attn_interface import _flash_attn_forward

    _flash_attn_v3_available = True

    def flash_attention_v3_fwd(
        q,
        k,
        v,
        o,
    ):
        head_dim = q.shape[-1]
        softmax_scale = head_dim ** -0.5
        _flash_attn_forward(
            q,
            k,
            v,
            None,
            None,  # k_new, v_new
            None,  # qv
            o,  # out
            None,
            None,
            None,  # cu_seqlens_q/k/k_new
            None,
            None,  # seqused_q/k
            None,
            None,  # max_seqlen_q/k
            None,
            None,
            None,  # page_table, kv_batch_idx, leftpad_k,
            None,
            None,  # rotary_cos/sin
            None,
            None,
            None,
            softmax_scale,
            causal=False,
            window_size=(-1, -1),
            softcap=0.0,
            num_splits=1,
            pack_gqa=None,
            sm_margin=0,
        )
        return

except ImportError:
    print("Failed to import _flash_attn_forward from hopper.flash_attn_interface.")
    _flash_attn_v3_available = False


def flash_attention_fwd(q, k, v, o):
    """
    统一的 Flash Attention 接口。如果 _flash_attn_forward 存在，
    则使用 flash_attention_v3_fwd，否则使用 Triton 版本。
    """
    if _flash_attn_v3_available and HOPPER:
        flash_attention_v3_fwd(q, k, v, o)
    else:
        _flash_attention_triton_fwd(q, k, v, o)


def torch_att(q, k, v):
    head_dim = q.shape[-1]
    q = q.transpose(1, 2)
    k = k.transpose(1, 2)
    v = v.transpose(1, 2)
    scale = head_dim ** -0.5
    attn = (q * scale) @ k.transpose(-2, -1)
    attn = attn.softmax(dim=-1)
    out = attn @ v
    out = out.transpose(1, 2).contiguous()
    return out


def test():
    import torch
    import numpy as np

    B, L, H, D = 4, 1025, 7, 128
    dtype = torch.float16
    q = torch.empty((B, L, H, D), dtype=dtype, device="cuda").normal_(mean=0.1, std=0.2)
    k = torch.empty((B, L, H, D), dtype=dtype, device="cuda").normal_(mean=0.1, std=0.2)
    v = torch.empty((B, L, H, D), dtype=dtype, device="cuda").normal_(mean=0.1, std=0.2)
    o = torch.empty((B, L, H, D), dtype=dtype, device="cuda").normal_(mean=0.1, std=0.2)
    torch_out = torch_att(q, k, v)
    import time

    torch.cuda.synchronize()
    a = time.time()
    for i in range(100):
        flash_attention_fwd(q, k, v, o)
        # o = torch_att(q, k, v)
    torch.cuda.synchronize()
    b = time.time()
    # print(o.shape, torch_out.shape)
    print((b - a) / 100 * 1000)

    print("max ", torch.max(torch.abs(torch_out - o)))
    print("mean ", torch.mean(torch.abs(torch_out - o)))
    assert torch.allclose(torch_out, o, atol=1e-2, rtol=0)
