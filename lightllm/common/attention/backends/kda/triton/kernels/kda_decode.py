import torch
import triton
import triton.language as tl


@triton.jit
def _kda_decode(
    Q,
    K,
    V,
    G,
    B,
    A,
    Bias,
    State,
    Idx,
    CuSeqLens,
    Accepted,
    O,
    SQ: tl.constexpr,
    SK: tl.constexpr,
    SV: tl.constexpr,
    SG: tl.constexpr,
    SB: tl.constexpr,
    H: tl.constexpr,
    D: tl.constexpr,
    LOWER: tl.constexpr,
    BV: tl.constexpr,
    MTP_SIZE: tl.constexpr,
):
    row_head = tl.program_id(1)
    row, head = row_head // H, row_head % H
    if MTP_SIZE > 1:
        start = tl.load(CuSeqLens + row)
        end = tl.load(CuSeqLens + row + 1)
        if start == end:
            return
        accepted = tl.load(Accepted + row) - 1
        state_idx = tl.load(Idx + row * MTP_SIZE + accepted)
    else:
        start, end = row, row + 1
        state_idx = tl.load(Idx + row)
    ki = tl.arange(0, D)
    vi = tl.program_id(0) * BV + tl.arange(0, BV)
    bias = tl.load(Bias + head * D + ki)
    amplitude = tl.exp(tl.load(A + head))
    state_offset = head * D * D + ki[:, None] * D + vi[None, :]
    state = tl.load(State + state_idx * H * D * D + state_offset).to(tl.float32)
    for token in range(start, end):
        q = tl.load(Q + token * SQ + head * D + ki).to(tl.float32)
        k = tl.load(K + token * SK + head * D + ki).to(tl.float32)
        v = tl.load(V + token * SV + head * D + vi).to(tl.float32)
        q *= tl.rsqrt(tl.sum(q * q, 0) + 1e-6) * (D ** -0.5)
        k *= tl.rsqrt(tl.sum(k * k, 0) + 1e-6)
        gate = tl.load(G + token * SG + head * D + ki).to(tl.float32)
        decay = tl.exp(LOWER * tl.sigmoid(amplitude * (gate + bias)))
        beta = tl.sigmoid(tl.load(B + token * SB + head).to(tl.float32))
        state *= decay[:, None]
        delta = (v - tl.sum(state * k[:, None], 0)) * beta
        state += k[:, None] * delta[None, :]
        if MTP_SIZE > 1:
            state_idx = tl.load(Idx + row * MTP_SIZE + token - start)
        tl.store(State + state_idx * H * D * D + state_offset, state)
        out = tl.sum(state * q[:, None], 0)
        tl.store(O + (token * H + head) * D + vi, out)
        # Match successive single-token calls when the state cache is BF16.
        state = state.to(State.dtype.element_ty).to(tl.float32)


def fused_recurrent_kda(
    q,
    k,
    v,
    raw_gate,
    raw_beta,
    a_log,
    gate_bias,
    initial_state,
    ssm_state_indices,
    lower_bound=-5.0,
    inplace_final_state=True,
    cu_seqlens=None,
    num_accepted_tokens=None,
):
    assert inplace_final_state
    batch, _, heads, dim = q.shape
    assert dim == 128
    mtp_size = 1
    token_axis = 0
    if cu_seqlens is not None:
        assert q.shape[0] == 1 and ssm_state_indices.ndim == 2
        batch, mtp_size = ssm_state_indices.shape
        assert mtp_size > 1 and cu_seqlens.numel() == batch + 1
        assert num_accepted_tokens is not None and num_accepted_tokens.numel() == batch
        token_axis = 1
    else:
        assert q.shape[1] == 1 and ssm_state_indices.ndim == 1
    out = torch.empty_like(v, memory_format=torch.contiguous_format)
    _kda_decode[(triton.cdiv(dim, 32), batch * heads)](
        q,
        k,
        v,
        raw_gate,
        raw_beta,
        a_log,
        gate_bias,
        initial_state,
        ssm_state_indices,
        cu_seqlens,
        num_accepted_tokens,
        out,
        q.stride(token_axis),
        k.stride(token_axis),
        v.stride(token_axis),
        raw_gate.stride(token_axis),
        raw_beta.stride(token_axis),
        heads,
        dim,
        lower_bound,
        32,
        mtp_size,
        num_warps=4,
    )
    return out, initial_state
