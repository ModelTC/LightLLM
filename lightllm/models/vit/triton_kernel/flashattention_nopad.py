import torch
import triton
import triton.language as tl
import math
import time
import torch.nn.functional as F
from typing import Optional, Tuple
from lightllm.utils.device_utils import is_hopper

if triton.__version__ >= "2.1.0":

    @triton.jit
    def _fwd_kernel(
        Q,
        K,
        V,
        sm_scale,
        Out,
        q_stride_s,
        q_stride_h,
        q_stride_d,
        k_stride_s,
        k_stride_h,
        k_stride_d,
        v_stride_s,
        v_stride_h,
        v_stride_d,
        o_stride_s,
        o_stride_h,
        o_stride_d,
        head_dim_act,
        cu_seqlens,
        BLOCK_M: tl.constexpr,
        BLOCK_DMODEL: tl.constexpr,
        BLOCK_N: tl.constexpr,
    ):
        cur_batch = tl.program_id(2)
        cur_head = tl.program_id(1)
        start_m = tl.program_id(0)

        seq_start = tl.load(cu_seqlens + cur_batch).to(tl.int32)
        seq_end = tl.load(cu_seqlens + cur_batch + 1).to(tl.int32)
        seq_len = seq_end - seq_start

        # initialize offsets
        offs_n = tl.arange(0, BLOCK_N)
        offs_d = tl.arange(0, BLOCK_DMODEL)
        offs_m = start_m * BLOCK_M + tl.arange(0, BLOCK_M)

        mask_d = offs_d < head_dim_act
        off_q = cur_head * q_stride_h + (seq_start + offs_m[:, None]) * q_stride_s + offs_d[None, :] * q_stride_d
        q = tl.load(Q + off_q, mask=(offs_m[:, None] < seq_len) & mask_d[None, :], other=0.0)
        # initialize pointer to m and l
        m_i = tl.zeros([BLOCK_M], dtype=tl.float32) - float("inf")
        l_i = tl.zeros([BLOCK_M], dtype=tl.float32) - float("inf")
        acc = tl.zeros([BLOCK_M, BLOCK_DMODEL], dtype=tl.float32)

        for start_n in range(0, seq_len, BLOCK_N):
            start_n = tl.multiple_of(start_n, BLOCK_N)
            # -- compute qk ----
            off_k = (
                (seq_start + start_n + offs_n[None, :]) * k_stride_s
                + cur_head * k_stride_h
                + offs_d[:, None] * k_stride_d
            )
            k = tl.load(K + off_k, mask=((start_n + offs_n[None, :]) < seq_len) & mask_d[:, None], other=0.0)

            qk = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32)
            qk += tl.dot(q, k)
            qk *= sm_scale
            qk += tl.where((start_n + offs_n[None, :]) < seq_len, 0, float("-inf"))

            # -- compute m_ij, p, l_ij
            m_ij = tl.maximum(tl.max(qk, 1), m_i)
            p = tl.exp(qk - m_ij[:, None])
            l_ij = tl.sum(p, 1)

            acc_scale = tl.exp(m_i - m_ij)
            acc = acc * acc_scale[:, None]

            # update acc
            off_v = (
                (seq_start + start_n + offs_n[:, None]) * v_stride_s
                + cur_head * v_stride_h
                + offs_d[None, :] * v_stride_d
            )
            v = tl.load(V + off_v, mask=((start_n + offs_n[:, None]) < seq_len) & mask_d[None, :], other=0.0)
            p = p.to(v.dtype)
            acc += tl.dot(p, v)
            # update m_i and l_i
            m_i = m_ij
            l_i_new = tl.exp(l_i - m_ij) + l_ij
            l_i = m_ij + tl.log(l_i_new)

        o_scale = tl.exp(m_i - l_i)
        acc = acc * o_scale[:, None]
        # initialize pointers to output
        off_o = (seq_start + offs_m[:, None]) * o_stride_s + cur_head * o_stride_h + offs_d[None, :] * o_stride_d
        out_ptrs = Out + off_o
        tl.store(out_ptrs, acc, mask=(offs_m[:, None] < seq_len) & mask_d[None, :])
        return

    @torch.no_grad()
    def _flash_attention_triton_fwd(
        q,
        k,
        v,
        o,
        cu_seqlens,  # q k v cu_seqlens,
        max_seqlen,
    ):
        BLOCK = 64
        # shape constraints
        assert q.ndim == k.ndim == v.ndim == o.ndim == 3, "q, k, v, o must be 3D tensors"
        _, head_num, head_dim = q.shape
        batch_size = cu_seqlens.numel() - 1

        sm_scale = 1.0 / (head_dim ** 0.5)  # 计算scale系数
        d_pad = triton.next_power_of_2(head_dim)
        grid = (triton.cdiv(max_seqlen, BLOCK), head_num, batch_size)  # batch, head,
        num_warps = 4
        _fwd_kernel[grid](
            q,
            k,
            v,
            sm_scale,
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
            head_dim,
            cu_seqlens,
            BLOCK_M=BLOCK,
            BLOCK_DMODEL=d_pad,
            BLOCK_N=BLOCK,
            num_warps=num_warps,
            num_stages=2,
        )
        return

else:
    raise Exception("error triton version!")

_flash_attn_v3_available = False
try:
    from sgl_kernel.flash_attn import flash_attn_varlen_func

    _flash_attn_v3_available = True

    def flash_attention_v3_fwd(
        q,
        k,
        v,
        o,
        cu_seqlens,
        max_seqlen,
    ):
        head_dim = q.shape[-1]
        softmax_scale = head_dim ** -0.5
        window_size = (-1, -1)
        torch.ops.sgl_kernel.fwd.default(
            q,
            k,
            v,
            None,  # k_new
            None,  # v_new
            None,  # qv
            o,  # out
            cu_seqlens,
            cu_seqlens,
            None,  # cu_seqlens_k_new
            None,
            None,
            max_seqlen,
            max_seqlen,
            None,  # page_table,
            None,  # kv_batch_idx
            None,  # leftpad_k
            None,  # rotary cos
            None,  # rotary sin
            None,  # seqlens_rotary
            None,
            None,
            None,
            softmax_scale,
            False,
            window_size[0],
            window_size[1],
            0.0,
            is_rotary_interleaved=False,
            scheduler_metadata=None,
            num_splits=1,
            pack_gqa=None,
            sm_margin=0,
            sinks=None,
        )

        return

except ImportError:
    print("Failed to import _flash_attn_forward from hopper.flash_attn_interface.")
    _flash_attn_v3_available = False


def flash_attention_fwd(q, k, v, o, cu_seqlens, max_seqlen):
    """
    统一的 Flash Attention 接口。如果 sgl_kernel 存在，
    则使用 sgl_kernel里的接口，否则使用 Triton 版本。
    """
    global _flash_attn_v3_available
    if _flash_attn_v3_available and is_hopper():
        try:
            flash_attention_v3_fwd(q, k, v, o, cu_seqlens, max_seqlen)
        except Exception as e:
            print(f"Failed to use sgl_kernel: {e}")
            _flash_attn_v3_available = False
    else:
        _flash_attention_triton_fwd(q, k, v, o, cu_seqlens, max_seqlen)
