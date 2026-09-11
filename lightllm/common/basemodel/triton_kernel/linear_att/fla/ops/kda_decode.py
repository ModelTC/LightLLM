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
):
    row_head = tl.program_id(1)
    row, head = row_head // H, row_head % H
    ki = tl.arange(0, D)
    vi = tl.program_id(0) * BV + tl.arange(0, BV)
    q = tl.load(Q + row * SQ + head * D + ki).to(tl.float32)
    k = tl.load(K + row * SK + head * D + ki).to(tl.float32)
    v = tl.load(V + row * SV + head * D + vi).to(tl.float32)
    q *= tl.rsqrt(tl.sum(q * q, 0) + 1e-6) * (D ** -0.5)
    k *= tl.rsqrt(tl.sum(k * k, 0) + 1e-6)
    gate = tl.load(G + row * SG + head * D + ki).to(tl.float32)
    bias = tl.load(Bias + head * D + ki)
    amplitude = tl.exp(tl.load(A + head))
    decay = tl.exp(LOWER * tl.sigmoid(amplitude * (gate + bias)))
    beta = tl.sigmoid(tl.load(B + row * SB + head).to(tl.float32))
    req = tl.load(Idx + row)
    ptr = State + (req * H + head) * D * D + ki[:, None] * D + vi[None, :]
    state = tl.load(ptr).to(tl.float32) * decay[:, None]
    delta = (v - tl.sum(state * k[:, None], 0)) * beta
    state += k[:, None] * delta[None, :]
    tl.store(ptr, state)
    out = tl.sum(state * q[:, None], 0)
    tl.store(O + row_head * D + vi, out)


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
):
    assert inplace_final_state and q.shape[1] == 1
    batch, _, heads, dim = q.shape
    assert dim == 128
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
        out,
        q.stride(0),
        k.stride(0),
        v.stride(0),
        raw_gate.stride(0),
        raw_beta.stride(0),
        heads,
        dim,
        lower_bound,
        32,
        num_warps=4,
    )
    return out, initial_state
