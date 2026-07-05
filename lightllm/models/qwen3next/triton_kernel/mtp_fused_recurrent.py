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
# Helpers
# ---------------------------------------------------------------------------


def _ensure_qkv_token_strided(x: torch.Tensor, inner_numel: int):
    if x is None:
        return None, 0

    assert x.shape[0] == 1 or x.shape[1] == 1, "q/k/v must use layout [tokens, 1, head, dim] or [1, tokens, head, dim]"

    tail_contiguous = x.stride()[-2:] == (x.shape[-1], 1)
    if not tail_contiguous:
        x = x.contiguous()
        return x, inner_numel
    tok_dim = 0 if x.shape[1] == 1 else 1
    return x, x.stride(tok_dim)


def _ensure_gate_token_strided(x: torch.Tensor, inner_numel: int):
    if x is None:
        return None, 0
    if x.stride(1) != 1:
        x = x.contiguous()
        return x, inner_numel
    return x, x.stride(0)


# ---------------------------------------------------------------------------
# Triton kernel
# ---------------------------------------------------------------------------


@triton.heuristics(
    {
        "USE_INITIAL_STATE": lambda args: args["h0"] is not None,
        "IS_VARLEN": lambda args: args["cu_seqlens"] is not None,
        "IS_CONTINUOUS_BATCHING": lambda args: args["ssm_state_indices"] is not None,
        "IS_SPEC_DECODING": lambda args: args["num_accepted_tokens"] is not None,
        "HAS_SEPARATE_WRITE_INDICES": lambda args: args["ssm_state_write_indices"] is not None,
    }
)
@triton.jit(do_not_specialize=["N", "T"])
def _fused_recurrent_gated_delta_rule_fwd_kernel(
    q,
    k,
    v,
    beta,
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
    B: tl.constexpr,
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
    USE_INITIAL_STATE: tl.constexpr,
    IS_BETA_HEADWISE: tl.constexpr,
    USE_QK_L2NORM_IN_KERNEL: tl.constexpr,
    IS_VARLEN: tl.constexpr,
    IS_CONTINUOUS_BATCHING: tl.constexpr,
    IS_SPEC_DECODING: tl.constexpr,
    IS_KDA: tl.constexpr,
    HAS_SEPARATE_WRITE_INDICES: tl.constexpr,
    FUSE_GATING: tl.constexpr,
):
    i_k, i_v, i_nh = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    i_n, i_hv = i_nh // HV, i_nh % HV
    i_h = i_hv // (HV // H)
    if IS_VARLEN:
        bos, eos = (
            tl.load(cu_seqlens + i_n).to(tl.int64),
            tl.load(cu_seqlens + i_n + 1).to(tl.int64),
        )
        all = T
        T = eos - bos
    else:
        bos, eos = i_n * T, i_n * T + T
        all = B * T

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
    if USE_INITIAL_STATE:
        if IS_CONTINUOUS_BATCHING:
            if IS_SPEC_DECODING:
                i_t = tl.load(num_accepted_tokens + i_n).to(tl.int64) - 1
            else:
                i_t = 0
            p_h0 = (
                h0 + tl.load(ssm_state_indices + i_n * stride_indices_seq + i_t).to(tl.int64) * stride_init_state_token
            )
        else:
            p_h0 = h0 + bos * HV * K * V
        p_h0 = p_h0 + i_hv * K * V + o_k[:, None] * V + o_v[None, :]
        b_h += tl.load(p_h0, mask=mask_h, other=0).to(tl.float32)

    for i_t in range(0, T):
        b_q = tl.load(p_q, mask=mask_k, other=0).to(tl.float32)
        b_k = tl.load(p_k, mask=mask_k, other=0).to(tl.float32)
        b_v = tl.load(p_v, mask=mask_v, other=0).to(tl.float32)

        if USE_QK_L2NORM_IN_KERNEL:
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

        if HAS_SEPARATE_WRITE_INDICES:
            write_idx = tl.load(ssm_state_write_indices + i_n * stride_write_indices_seq + i_t).to(tl.int64)
        else:
            write_idx = tl.load(ssm_state_indices + i_n * stride_indices_seq + i_t).to(tl.int64)
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
    beta: torch.Tensor | None = None,
    scale: float | None = None,
    initial_state: torch.Tensor | None = None,
    cu_seqlens: torch.Tensor | None = None,
    ssm_state_indices: torch.Tensor | None = None,
    ssm_state_write_indices: torch.Tensor | None = None,
    num_accepted_tokens: torch.Tensor | None = None,
    use_qk_l2norm_in_kernel: bool = False,
    A_log: torch.Tensor | None = None,
    dt_bias: torch.Tensor | None = None,
    a_raw: torch.Tensor | None = None,
    b_raw: torch.Tensor | None = None,
    out: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Fused recurrent gated delta rule with fused gating (GDN layer).

    Directly launches the triton kernel — no ``torch.autograd.Function``.

    Args:
        q:  ``[B, T, H, K]`` or ``[1, T, H, K]`` queries.
        k:  ``[B, T, H, K]`` or ``[1, T, H, K]`` keys.
        v:  ``[B, T, HV, V]`` or ``[1, T, HV, V]`` values (GVA when HV > H).
        beta: ``[B, T, HV]`` betas (unused when ``FUSE_GATING=True``).
        scale: sqrt(d_head) ** -0.5.  Defaults to ``K ** -0.5`` when None.
        initial_state: ``[N, HV, K, V]`` initial SSM state.
        cu_seqlens: ``[N+1]`` int64 cumulative sequence lengths for the
            varlen (MTP verify) path.  None for equal-length decode.
        ssm_state_indices: ``[N,]`` or ``[N, S+1]`` int32 slot indices.
        ssm_state_write_indices: separate write indices for the state
            propagation optimisation.
        num_accepted_tokens: ``[N,]`` int32.  When not None the read offset
            for each sequence is ``num_accepted_tokens[i] - 1``.
        A_log: ``[HV]`` per-head log decay (fused-gating mode).
        dt_bias: ``[HV]`` per-head dt bias (fused-gating mode).
        a_raw: ``[B*T, HV]`` raw alpha (fused-gating mode).
        b_raw: ``[B*T, HV]`` raw beta (fused-gating mode).
        out: optional pre-allocated output tensor.

    Returns:
        ``(o, final_state)`` where ``o`` is ``[B, T, HV, V]`` and
        ``final_state`` is ``[N, HV, K, V]``.
    """
    fuse_gating = A_log is not None

    if scale is None:
        scale = k.shape[-1] ** -0.5
    else:
        assert scale > 0, "scale must be positive"
    if not fuse_gating and beta is None:
        beta = torch.ones_like(q[..., 0])

    B, T, H, K, V = *k.shape, v.shape[-1]
    HV = v.shape[2]
    N = B if cu_seqlens is None else len(cu_seqlens) - 1
    q, stride_q_tok = _ensure_qkv_token_strided(q, H * K)
    k, stride_k_tok = _ensure_qkv_token_strided(k, H * K)
    v, stride_v_tok = _ensure_qkv_token_strided(v, HV * V)
    a_raw, stride_a_tok = _ensure_gate_token_strided(a_raw, HV)
    b_raw, stride_b_tok = _ensure_gate_token_strided(b_raw, HV)
    BK = triton.next_power_of_2(K)
    if T == 1:
        BV = min(triton.next_power_of_2(V), 32)
        num_warps = 4
        num_stages = 1
    else:
        BV = min(triton.next_power_of_2(V), 8)
        num_warps = 1
        num_stages = 3
    NK, NV = triton.cdiv(K, BK), triton.cdiv(V, BV)
    assert NK == 1, "NK > 1 is not supported yet"

    if out is not None:
        o = out.unsqueeze(0) if out.ndim == v.ndim else out
    else:
        o = q.new_empty(NK, *v.shape)
    final_state = initial_state

    stride_init_state_token = initial_state.stride(0)
    stride_final_state_token = final_state.stride(0)

    if ssm_state_indices is None:
        stride_indices_seq, stride_indices_tok = 1, 1
    elif ssm_state_indices.ndim == 1:
        stride_indices_seq, stride_indices_tok = ssm_state_indices.stride(0), 1
    else:
        assert ssm_state_indices.stride(-1) == 1, "2D ssm_state_indices must have contiguous rows"
        stride_indices_seq, stride_indices_tok = ssm_state_indices.stride()

    if ssm_state_write_indices is None:
        stride_write_indices_seq, stride_write_indices_tok = 1, 1
    elif ssm_state_write_indices.ndim == 1:
        stride_write_indices_seq, stride_write_indices_tok = ssm_state_write_indices.stride(0), 1
    else:
        assert ssm_state_write_indices.stride(-1) == 1, "2D ssm_state_write_indices must have contiguous rows"
        stride_write_indices_seq, stride_write_indices_tok = ssm_state_write_indices.stride()

    grid = (NK, NV, N * HV)
    _fused_recurrent_gated_delta_rule_fwd_kernel[grid](
        q=q,
        k=k,
        v=v,
        beta=beta,
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
        B=B,
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
        IS_BETA_HEADWISE=False if fuse_gating else (beta.ndim == v.ndim),
        USE_QK_L2NORM_IN_KERNEL=use_qk_l2norm_in_kernel,
        IS_KDA=False,
        FUSE_GATING=fuse_gating,
        num_warps=num_warps,
        num_stages=num_stages,
    )
    o = o.squeeze(0)
    return o, final_state
