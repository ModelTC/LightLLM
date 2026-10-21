# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# SPDX-FileCopyrightText: Songlin Yang, Yu Zhang
#
# This file contains code copied from the flash-linear-attention project.
# The original source code was licensed under the MIT license and included
# the following copyright notice:
# Copyright (c) 2023-2025, Songlin Yang, Yu Zhang
# ruff: noqa: E501

import torch

import triton
import triton.language as tl

from .op import exp


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
def fused_recurrent_gated_delta_rule_fwd_kernel(
    q,
    k,
    v,
    g,
    beta,
    o,
    h0,
    ht,
    cu_seqlens,
    ssm_state_indices,
    ssm_state_write_indices,  # NEW: separate write indices for state propagation optimization
    num_accepted_tokens,
    # Fused gating parameters (only used when FUSE_GATING=True)
    A_log,  # [HV] per-head log decay
    dt_bias,  # [HV] per-head dt bias
    a_raw,  # [B*T, HV] raw alpha values (before softplus)
    b_raw,  # [B*T, HV] raw beta values (before sigmoid)
    scale,
    N: tl.int64,  # num of sequences
    T: tl.int64,  # num of tokens
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
    stride_write_indices_seq: tl.constexpr,  # NEW: stride for write indices
    stride_write_indices_tok: tl.constexpr,  # NEW: stride for write indices
    SOFTPLUS_BETA: tl.constexpr,  # softplus beta parameter (default 1.0)
    SOFTPLUS_THRESHOLD: tl.constexpr,  # softplus threshold (default 20.0)
    USE_INITIAL_STATE: tl.constexpr,  # whether to use initial state
    INPLACE_FINAL_STATE: tl.constexpr,  # whether to store final state inplace
    IS_BETA_HEADWISE: tl.constexpr,  # whether beta is headwise vector or scalar,
    USE_QK_L2NORM_IN_KERNEL: tl.constexpr,
    IS_VARLEN: tl.constexpr,
    IS_CONTINUOUS_BATCHING: tl.constexpr,
    IS_SPEC_DECODING: tl.constexpr,
    IS_KDA: tl.constexpr,
    HAS_SEPARATE_WRITE_INDICES: tl.constexpr,  # NEW: whether to use separate write indices
    FUSE_GATING: tl.constexpr,  # whether to compute g/beta inline from raw values
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
        # no tokens to process for this sequence
        return

    o_k = i_k * BK + tl.arange(0, BK)
    o_v = i_v * BV + tl.arange(0, BV)

    p_q = q + bos * stride_q_tok + i_h * K + o_k
    p_k = k + bos * stride_k_tok + i_h * K + o_k
    p_v = v + bos * stride_v_tok + i_hv * V + o_v
    if FUSE_GATING:
        # Fused gating: load per-head constants once, compute g/beta inline per token
        b_A_log = tl.load(A_log + i_hv).to(tl.float32)
        b_dt_bias = tl.load(dt_bias + i_hv).to(tl.float32)
        p_a_raw = a_raw + bos * stride_a_tok + i_hv
        p_b_raw = b_raw + bos * stride_b_tok + i_hv
    else:
        if IS_BETA_HEADWISE:
            p_beta = beta + (bos * HV + i_hv) * V + o_v
        else:
            p_beta = beta + bos * HV + i_hv

        if not IS_KDA:
            p_g = g + bos * HV + i_hv
        else:
            p_gk = g + (bos * HV + i_hv) * K + o_k

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
        # [BK, BV]
        if FUSE_GATING:
            # Compute g = -exp(A_log) * softplus(a_raw + dt_bias) inline
            b_a = tl.load(p_a_raw).to(tl.float32)
            x = b_a + b_dt_bias
            softplus_x = tl.where(
                SOFTPLUS_BETA * x <= SOFTPLUS_THRESHOLD,
                (1.0 / SOFTPLUS_BETA) * tl.log(1.0 + tl.exp(SOFTPLUS_BETA * x)),
                x,
            )
            b_g = -tl.exp(b_A_log) * softplus_x
            b_h *= exp(b_g)
            # Compute beta = sigmoid(b_raw) inline
            b_b = tl.load(p_b_raw).to(tl.float32)
            b_beta = tl.sigmoid(b_b)
        else:
            if not IS_KDA:
                b_g = tl.load(p_g).to(tl.float32)
                b_h *= exp(b_g)
            else:
                b_gk = tl.load(p_gk).to(tl.float32)
                b_h *= exp(b_gk[:, None])
            if IS_BETA_HEADWISE:
                b_beta = tl.load(p_beta, mask=mask_v, other=0).to(tl.float32)
            else:
                b_beta = tl.load(p_beta).to(tl.float32)
        # [BV]
        b_v -= tl.sum(b_h * b_k[:, None], 0)
        b_v *= b_beta
        # [BK, BV]
        b_h += b_k[:, None] * b_v[None, :]
        # [BV]
        b_o = tl.sum(b_h * b_q[:, None], 0)
        tl.store(p_o, b_o.to(p_o.dtype.element_ty), mask=mask_v)

        # keep the states for multi-query tokens
        if INPLACE_FINAL_STATE:
            # Use separate write indices if provided (for state propagation optimization)
            # Otherwise fall back to read indices
            if HAS_SEPARATE_WRITE_INDICES:
                write_idx = tl.load(ssm_state_write_indices + i_n * stride_write_indices_seq + i_t).to(tl.int64)
            else:
                write_idx = tl.load(ssm_state_indices + i_n * stride_indices_seq + i_t).to(tl.int64)
            p_ht = ht + write_idx * stride_final_state_token
        else:
            p_ht = ht + (bos + i_t) * stride_final_state_token
        p_ht = p_ht + i_hv * K * V + o_k[:, None] * V + o_v[None, :]
        tl.store(p_ht, b_h.to(p_ht.dtype.element_ty), mask=mask_h)

        p_q += stride_q_tok
        p_k += stride_k_tok
        p_o += HV * V
        p_v += stride_v_tok
        if FUSE_GATING:
            p_a_raw += stride_a_tok
            p_b_raw += stride_b_tok
        else:
            if not IS_KDA:
                p_g += HV
            else:
                p_gk += HV * K
            p_beta += HV * (V if IS_BETA_HEADWISE else 1)


def _ensure_qkv_token_strided(x: torch.Tensor, inner_numel: int):
    """Return q/k/v and per-token stride, copying only when needed.

    Supports the decode layout [tokens, 1, head, dim] and the MTP verify /
    varlen layout [1, tokens, head, dim]; the token dimension is the non-unit
    leading dim. Both are column views of a packed projection output, so the
    tail [head, dim] is contiguous and no copy is needed.
    """
    if x is None:
        return None, 0

    assert x.shape[0] == 1 or x.shape[1] == 1, "q/k/v must use layout [tokens, 1, head, dim] or [1, tokens, head, dim]"

    # Packed tail [head, dim] means the last two strides are [dim, 1].
    tail_contiguous = x.stride()[-2:] == (x.shape[-1], 1)
    if not tail_contiguous:
        x = x.contiguous()
        return x, inner_numel
    # Token dim is the non-unit leading dim (dim 0 for decode, dim 1 for verify).
    tok_dim = 0 if x.shape[1] == 1 else 1
    return x, x.stride(tok_dim)


def _ensure_gate_token_strided(x: torch.Tensor, inner_numel: int):
    """Return a_raw/b_raw and token stride, copying only when needed."""
    if x is None:
        return None, 0
    # a_raw/b_raw are 2D [tokens, HV]; the tail HV dimension must be packed.
    if x.stride(1) != 1:
        x = x.contiguous()
        return x, inner_numel
    return x, x.stride(0)


def fused_recurrent_gated_delta_rule_fwd(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    scale: float,
    initial_state: torch.Tensor,
    inplace_final_state: bool = True,
    cu_seqlens: torch.LongTensor | None = None,
    ssm_state_indices: torch.Tensor | None = None,
    ssm_state_write_indices: torch.Tensor | None = None,  # NEW: separate write indices
    num_accepted_tokens: torch.Tensor | None = None,
    use_qk_l2norm_in_kernel: bool = False,
    # Fused gating parameters
    A_log: torch.Tensor | None = None,
    dt_bias: torch.Tensor | None = None,
    a_raw: torch.Tensor | None = None,
    b_raw: torch.Tensor | None = None,
    out: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    B, T, H, K, V = *k.shape, v.shape[-1]
    HV = v.shape[2]
    # Decode passes cu_seqlens=None (equal-length one-token sequences); the
    # Qwen3Next MTP verify path passes cu_seqlens for variable-length verify
    # chunks. Both flow through the per-token strided-view path below.
    N = B if cu_seqlens is None else len(cu_seqlens) - 1
    q, stride_q_tok = _ensure_qkv_token_strided(q, H * K)
    k, stride_k_tok = _ensure_qkv_token_strided(k, H * K)
    v, stride_v_tok = _ensure_qkv_token_strided(v, HV * V)
    a_raw, stride_a_tok = _ensure_gate_token_strided(a_raw, HV)
    b_raw, stride_b_tok = _ensure_gate_token_strided(b_raw, HV)
    BK = triton.next_power_of_2(K)
    if T == 1:
        # Decode path: use larger BV to reduce kernel instances (4 blocks instead of 16)
        # and more warps for better SM utilization at T=1 where there's no pipelining benefit
        BV = min(triton.next_power_of_2(V), 32)
        num_warps = 4
        num_stages = 1
    else:
        # Prefill path: small BV for better pipelining across sequence length
        BV = min(triton.next_power_of_2(V), 8)
        num_warps = 1
        num_stages = 3
    NK, NV = triton.cdiv(K, BK), triton.cdiv(V, BV)
    assert NK == 1, "NK > 1 is not supported yet"

    fuse_gating = A_log is not None

    if out is not None:
        o = out.unsqueeze(0) if out.ndim == v.ndim else out
    else:
        o = q.new_empty(NK, *v.shape)
    if inplace_final_state:
        final_state = initial_state
    else:
        final_state = q.new_empty(T, HV, K, V, dtype=initial_state.dtype)

    stride_init_state_token = initial_state.stride(0)
    stride_final_state_token = final_state.stride(0)

    # Strides for read indices. The kernel advances along a row with `+ i_t`
    # (token stride 1), so 2D index tensors must have contiguous rows.
    if ssm_state_indices is None:
        stride_indices_seq, stride_indices_tok = 1, 1
    elif ssm_state_indices.ndim == 1:
        stride_indices_seq, stride_indices_tok = ssm_state_indices.stride(0), 1
    else:
        assert ssm_state_indices.stride(-1) == 1, "2D ssm_state_indices must have contiguous rows"
        stride_indices_seq, stride_indices_tok = ssm_state_indices.stride()

    # Strides for write indices (if provided); same contiguous-row requirement
    if ssm_state_write_indices is None:
        stride_write_indices_seq, stride_write_indices_tok = 1, 1
    elif ssm_state_write_indices.ndim == 1:
        stride_write_indices_seq, stride_write_indices_tok = ssm_state_write_indices.stride(0), 1
    else:
        assert ssm_state_write_indices.stride(-1) == 1, "2D ssm_state_write_indices must have contiguous rows"
        stride_write_indices_seq, stride_write_indices_tok = ssm_state_write_indices.stride()

    grid = (NK, NV, N * HV)
    fused_recurrent_gated_delta_rule_fwd_kernel[grid](
        q=q,
        k=k,
        v=v,
        g=g,
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
        INPLACE_FINAL_STATE=inplace_final_state,
        IS_KDA=False,
        FUSE_GATING=fuse_gating,
        num_warps=num_warps,
        num_stages=num_stages,
    )
    o = o.squeeze(0)
    return o, final_state


class FusedRecurrentFunction(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        g: torch.Tensor,
        beta: torch.Tensor,
        scale: float,
        initial_state: torch.Tensor,
        inplace_final_state: bool = True,
        cu_seqlens: torch.LongTensor | None = None,
        ssm_state_indices: torch.Tensor | None = None,
        ssm_state_write_indices: torch.Tensor | None = None,
        num_accepted_tokens: torch.Tensor | None = None,
        use_qk_l2norm_in_kernel: bool = False,
        A_log: torch.Tensor | None = None,
        dt_bias: torch.Tensor | None = None,
        a_raw: torch.Tensor | None = None,
        b_raw: torch.Tensor | None = None,
        out: torch.Tensor | None = None,
    ):
        # q/k/v/a_raw/b_raw may be non-contiguous column views of one projection
        # output; the kernel handles them via per-token strides (no copies).
        o, final_state = fused_recurrent_gated_delta_rule_fwd(
            q=q,
            k=k,
            v=v,
            g=g.contiguous() if g is not None else None,
            beta=beta.contiguous() if beta is not None else None,
            scale=scale,
            initial_state=initial_state,
            inplace_final_state=inplace_final_state,
            cu_seqlens=cu_seqlens,
            ssm_state_indices=ssm_state_indices,
            ssm_state_write_indices=ssm_state_write_indices,
            num_accepted_tokens=num_accepted_tokens,
            use_qk_l2norm_in_kernel=use_qk_l2norm_in_kernel,
            A_log=A_log,
            dt_bias=dt_bias,
            a_raw=a_raw,
            b_raw=b_raw,
            out=out,
        )

        return o, final_state


def fused_recurrent_gated_delta_rule(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor = None,
    beta: torch.Tensor = None,
    scale: float = None,
    initial_state: torch.Tensor = None,
    inplace_final_state: bool = True,
    cu_seqlens: torch.LongTensor | None = None,
    ssm_state_indices: torch.Tensor | None = None,
    ssm_state_write_indices: torch.Tensor | None = None,  # NEW: separate write indices for state propagation
    num_accepted_tokens: torch.Tensor | None = None,
    use_qk_l2norm_in_kernel: bool = False,
    # Fused gating: pass raw values to compute g/beta inline in the kernel
    A_log: torch.Tensor | None = None,
    dt_bias: torch.Tensor | None = None,
    a_raw: torch.Tensor | None = None,
    b_raw: torch.Tensor | None = None,
    out: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    r"""
    Args:
        q (torch.Tensor):
            queries of shape `[B, T, H, K]`.
        k (torch.Tensor):
            keys of shape `[B, T, H, K]`.
        v (torch.Tensor):
            values of shape `[B, T, HV, V]`.
            GVA is applied if `HV > H`.
        g (torch.Tensor):
            g (decays) of shape `[B, T, HV]`.
        beta (torch.Tensor):
            betas of shape `[B, T, HV]`.
        scale (Optional[int]):
            Scale factor for the RetNet attention scores.
            If not provided, it will default to `1 / sqrt(K)`. Default: `None`.
        initial_state (Optional[torch.Tensor]):
            Initial state of shape `[N, HV, K, V]` for `N` input sequences.
            For equal-length input sequences, `N` equals the batch size `B`.
            Default: `None`.
        inplace_final_state: bool:
            Whether to store the final state in-place to save memory.
            Default: `True`.
        cu_seqlens (Optional[torch.LongTensor]):
            Cumulative sequence lengths of shape `[N+1]` for variable-length
            inputs (the Qwen3Next MTP verify path). `None` for plain decode,
            where sequences are treated as equal-length (one token each).
        ssm_state_indices (Optional[torch.Tensor]):
            Indices to map the input sequences to the initial/final states.
        num_accepted_tokens (Optional[torch.Tensor]):
            Number of accepted tokens for each sequence during decoding.

    Returns:
        o (torch.Tensor):
            Outputs of shape `[B, T, HV, V]`.
        final_state (torch.Tensor):
            Final state of shape `[N, HV, K, V]`.

    Examples::
        >>> import torch
        >>> import torch.nn.functional as F
        >>> from fla.ops.gated_delta_rule import fused_recurrent_gated_delta_rule
        # decode inputs
        >>> B, T, H, HV, K, V = 4, 1, 4, 8, 512, 512
        >>> q = torch.randn(B, T, H, K, device='cuda')
        >>> k = F.normalize(torch.randn(B, T, H, K, device='cuda'), p=2, dim=-1)
        >>> v = torch.randn(B, T, HV, V, device='cuda')
        >>> g = F.logsigmoid(torch.rand(B, T, HV, device='cuda'))
        >>> beta = torch.rand(B, T, HV, device='cuda').sigmoid()
        >>> h0 = torch.randn(B, HV, K, V, device='cuda')
        >>> o, ht = fused_gated_recurrent_delta_rule(
            q, k, v, g, beta,
            initial_state=h0,
        )
    """
    if scale is None:
        scale = k.shape[-1] ** -0.5
    else:
        assert scale > 0, "scale must be positive"
    fuse_gating = A_log is not None
    if not fuse_gating and beta is None:
        beta = torch.ones_like(q[..., 0])
    o, final_state = FusedRecurrentFunction.apply(
        q,
        k,
        v,
        g,
        beta,
        scale,
        initial_state,
        inplace_final_state,
        cu_seqlens,
        ssm_state_indices,
        ssm_state_write_indices,
        num_accepted_tokens,
        use_qk_l2norm_in_kernel,
        A_log,
        dt_bias,
        a_raw,
        b_raw,
        out,
    )
    return o, final_state
