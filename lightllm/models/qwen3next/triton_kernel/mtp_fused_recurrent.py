# SPDX-License-Identifier: Apache-2.0
#                            MIT
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# SPDX-FileCopyrightText: Songlin Yang, Yu Zhang
#
# Extracted from fused_recurrent.py — directly launches the triton kernel
# without a torch.autograd.Function wrapper.  Used by the MTP spec-decode
# verify path of the GDN (Gated DeltaNet) layer in Qwen3Next.
#
# Upstream source: flash-linear-attention / fused-recurrent gated delta rule.
# https://github.com/fla-org/flash-linear-attention
# ruff: noqa: E501

import torch
import triton
import triton.language as tl


# ---------------------------------------------------------------------------
# Triton kernel
# ---------------------------------------------------------------------------


@triton.jit(do_not_specialize=["N", "T"])
def _fused_recurrent_gated_delta_rule_fwd_kernel(
    q,
    k,
    v,
    o,
    h0,
    ht,
    cu_seqlens,
    ssm_state_indices,
    ssm_state_write_indices,
    num_accepted_tokens,
    A_log,
    dt_bias,
    a_raw,
    b_raw,
    scale,
    N: tl.int64,
    T: tl.int64,
    H: tl.constexpr,
    HV: tl.constexpr,
    K: tl.constexpr,
    V: tl.constexpr,
    BK: tl.constexpr,
    BV: tl.constexpr,
    stride_q_tok: tl.constexpr,
    stride_k_tok: tl.constexpr,
    stride_v_tok: tl.constexpr,
    stride_a_tok: tl.constexpr,
    stride_b_tok: tl.constexpr,
    stride_init_state_token: tl.constexpr,
    stride_final_state_token: tl.constexpr,
    stride_indices_seq: tl.constexpr,
    stride_indices_tok: tl.constexpr,
    stride_write_indices_seq: tl.constexpr,
    stride_write_indices_tok: tl.constexpr,
    SOFTPLUS_BETA: tl.constexpr,
    SOFTPLUS_THRESHOLD: tl.constexpr,
):
    i_k, i_v, i_nh = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    i_n, i_hv = i_nh // HV, i_nh % HV
    i_h = i_hv // (HV // H)
    bos, eos = (
        tl.load(cu_seqlens + i_n).to(tl.int64),
        tl.load(cu_seqlens + i_n + 1).to(tl.int64),
    )
    all = T
    T = eos - bos

    if T == 0:
        return

    o_k = i_k * BK + tl.arange(0, BK)
    o_v = i_v * BV + tl.arange(0, BV)

    p_q = q + bos * stride_q_tok + i_h * K + o_k
    p_k = k + bos * stride_k_tok + i_h * K + o_k
    p_v = v + bos * stride_v_tok + i_hv * V + o_v
    b_A_log = tl.load(A_log + i_hv).to(tl.float32)
    b_dt_bias = tl.load(dt_bias + i_hv).to(tl.float32)
    p_a_raw = a_raw + bos * stride_a_tok + i_hv
    p_b_raw = b_raw + bos * stride_b_tok + i_hv

    p_o = o + ((i_k * all + bos) * HV + i_hv) * V + o_v

    mask_k = o_k < K
    mask_v = o_v < V
    mask_h = mask_k[:, None] & mask_v[None, :]

    b_h = tl.zeros([BK, BV], dtype=tl.float32)
    i_t = tl.load(num_accepted_tokens + i_n).to(tl.int64) - 1
    p_h0 = h0 + tl.load(ssm_state_indices + i_n * stride_indices_seq + i_t).to(tl.int64) * stride_init_state_token
    p_h0 = p_h0 + i_hv * K * V + o_k[:, None] * V + o_v[None, :]
    b_h += tl.load(p_h0, mask=mask_h, other=0).to(tl.float32)

    for i_t in range(0, T):
        b_q = tl.load(p_q, mask=mask_k, other=0).to(tl.float32)
        b_k = tl.load(p_k, mask=mask_k, other=0).to(tl.float32)
        b_v = tl.load(p_v, mask=mask_v, other=0).to(tl.float32)

        b_q = b_q / tl.sqrt(tl.sum(b_q * b_q) + 1e-6)
        b_k = b_k / tl.sqrt(tl.sum(b_k * b_k) + 1e-6)
        b_q = b_q * scale
        b_a = tl.load(p_a_raw).to(tl.float32)
        x = b_a + b_dt_bias
        softplus_x = tl.where(
            SOFTPLUS_BETA * x <= SOFTPLUS_THRESHOLD,
            (1.0 / SOFTPLUS_BETA) * tl.log(1.0 + tl.exp(SOFTPLUS_BETA * x)),
            x,
        )
        b_g = -tl.exp(b_A_log) * softplus_x
        b_h *= tl.exp(b_g)
        b_b = tl.load(p_b_raw).to(tl.float32)
        b_beta = tl.sigmoid(b_b)
        b_v -= tl.sum(b_h * b_k[:, None], 0)
        b_v *= b_beta
        b_h += b_k[:, None] * b_v[None, :]
        b_o = tl.sum(b_h * b_q[:, None], 0)
        tl.store(p_o, b_o.to(p_o.dtype.element_ty), mask=mask_v)

        write_idx = tl.load(ssm_state_write_indices + i_n * stride_write_indices_seq + i_t).to(tl.int64)
        p_ht = ht + write_idx * stride_final_state_token
        p_ht = p_ht + i_hv * K * V + o_k[:, None] * V + o_v[None, :]
        tl.store(p_ht, b_h.to(p_ht.dtype.element_ty), mask=mask_h)

        p_q += stride_q_tok
        p_k += stride_k_tok
        p_o += HV * V
        p_v += stride_v_tok
        p_a_raw += stride_a_tok
        p_b_raw += stride_b_tok


# ---------------------------------------------------------------------------
# Public API — directly launches the triton kernel (no autograd.Function)
# ---------------------------------------------------------------------------


def mtp_fused_recurrent_gated_delta_rule(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    initial_state: torch.Tensor,
    cu_seqlens: torch.Tensor,
    ssm_state_indices: torch.Tensor,
    ssm_state_write_indices: torch.Tensor,
    num_accepted_tokens: torch.Tensor,
    A_log: torch.Tensor,
    dt_bias: torch.Tensor,
    a_raw: torch.Tensor,
    b_raw: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Fused recurrent gated delta rule with fused gating (GDN layer).

    Directly launches the triton kernel — no ``torch.autograd.Function``.

    Args:
        q:  ``[1, T, H, K]`` queries.
        k:  ``[1, T, H, K]`` keys.
        v:  ``[1, T, HV, V]`` values (GVA when HV > H).
        initial_state: ``[N, HV, K, V]`` initial SSM state.
        cu_seqlens: ``[N+1]`` int64 cumulative sequence lengths for the
            varlen (MTP verify) path.
        ssm_state_indices: ``[N, S+1]`` int32 slot indices (2D).
        ssm_state_write_indices: ``[N, S+1]`` int32 write slot indices.
        num_accepted_tokens: ``[N]`` int32.  The read offset for each
            sequence is ``num_accepted_tokens[i] - 1``.
        A_log: ``[HV]`` per-head log decay.
        dt_bias: ``[HV]`` per-head dt bias.
        a_raw: ``[T, HV]`` raw alpha.
        b_raw: ``[T, HV]`` raw beta.

    Returns:
        ``(o, final_state)`` where ``o`` is ``[1, T, HV, V]`` and
        ``final_state`` is ``[N, HV, K, V]``.
    """
    scale = k.shape[-1] ** -0.5

    assert q.dim() == 4 and q.shape[0] == 1, "q must be [1, T, H, K]"
    assert k.dim() == 4 and k.shape[0] == 1, "k must be [1, T, H, K]"
    assert v.dim() == 4 and v.shape[0] == 1, "v must be [1, T, HV, V]"
    T, H, K = k.shape[1], k.shape[2], k.shape[3]
    V = v.shape[-1]
    HV = v.shape[2]
    N = len(cu_seqlens) - 1
    q, stride_q_tok = _ensure_qkv_token_strided(q)
    k, stride_k_tok = _ensure_qkv_token_strided(k)
    v, stride_v_tok = _ensure_qkv_token_strided(v)
    a_raw, stride_a_tok = _ensure_gate_token_strided(a_raw)
    b_raw, stride_b_tok = _ensure_gate_token_strided(b_raw)
    BK = triton.next_power_of_2(K)
    BV = min(triton.next_power_of_2(V), 8)
    num_warps = 1
    num_stages = 3
    NK, NV = triton.cdiv(K, BK), triton.cdiv(V, BV)
    assert NK == 1, "NK > 1 is not supported yet"

    o = q.new_empty(NK, *v.shape)
    final_state = initial_state

    stride_init_state_token = initial_state.stride(0)
    stride_final_state_token = final_state.stride(0)

    assert ssm_state_indices.stride(-1) == 1, "2D ssm_state_indices must have contiguous rows"
    stride_indices_seq, stride_indices_tok = ssm_state_indices.stride()

    assert ssm_state_write_indices.stride(-1) == 1, "2D ssm_state_write_indices must have contiguous rows"
    stride_write_indices_seq, stride_write_indices_tok = ssm_state_write_indices.stride()

    grid = (NK, NV, N * HV)
    _fused_recurrent_gated_delta_rule_fwd_kernel[grid](
        q=q,
        k=k,
        v=v,
        o=o,
        h0=initial_state,
        ht=final_state,
        cu_seqlens=cu_seqlens,
        ssm_state_indices=ssm_state_indices,
        ssm_state_write_indices=ssm_state_write_indices,
        num_accepted_tokens=num_accepted_tokens,
        A_log=A_log,
        dt_bias=dt_bias,
        a_raw=a_raw,
        b_raw=b_raw,
        scale=scale,
        N=N,
        T=T,
        H=H,
        HV=HV,
        K=K,
        V=V,
        BK=BK,
        BV=BV,
        stride_q_tok=stride_q_tok,
        stride_k_tok=stride_k_tok,
        stride_v_tok=stride_v_tok,
        stride_a_tok=stride_a_tok,
        stride_b_tok=stride_b_tok,
        stride_init_state_token=stride_init_state_token,
        stride_final_state_token=stride_final_state_token,
        stride_indices_seq=stride_indices_seq,
        stride_indices_tok=stride_indices_tok,
        stride_write_indices_seq=stride_write_indices_seq,
        stride_write_indices_tok=stride_write_indices_tok,
        SOFTPLUS_BETA=1.0,
        SOFTPLUS_THRESHOLD=20.0,
        num_warps=num_warps,
        num_stages=num_stages,
    )
    o = o.squeeze(0)
    return o, final_state


# ---------------------------------------------------------------------------
# Stride helpers
# ---------------------------------------------------------------------------


def _ensure_qkv_token_strided(x: torch.Tensor):
    if x.stride()[-2:] != (x.shape[-1], 1):
        x = x.contiguous()
    return x, x.stride(1)


def _ensure_gate_token_strided(x: torch.Tensor):
    if x.stride(1) != 1:
        x = x.contiguous()
    return x, x.stride(0)
