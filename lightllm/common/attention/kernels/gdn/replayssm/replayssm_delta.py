# SPDX-License-Identifier: Apache-2.0
"""One gated-delta step shared by prefix folding."""

import triton
import triton.language as tl


@triton.jit
def _delta_rule_step(state, raw_k, raw_v, g, beta, KDA: tl.constexpr):
    # Folding applies this same update once per accepted history token.
    raw_k /= tl.sqrt(tl.sum(raw_k * raw_k) + 1.0e-6)
    decay = tl.exp(g)
    if KDA:
        state *= decay[:, None]
        d = beta * (raw_v - tl.sum(state * raw_k[:, None], 0))
        state += raw_k[:, None] * d[None, :]
    else:
        sk = tl.sum(state * raw_k[:, None], 0)
        d = beta * (raw_v - decay * sk)
        state = state * decay + raw_k[:, None] * d[None, :]
    return state


@triton.jit
def _apply_accepted_prefix(
    state,
    RawKeys,
    RawValues,
    Gates,
    Betas,
    key_slot,
    slot,
    kk,
    vv,
    prefix_len,
    K: tl.constexpr,
    V: tl.constexpr,
    L: tl.constexpr,
    KDA: tl.constexpr,
):
    # The cursor can temporarily exceed the stored history length.
    for j in range(tl.minimum(prefix_len, L)):
        raw_k = tl.load(RawKeys + (key_slot * L + j) * K + kk, kk < K, 0).to(tl.float32)
        raw_v = tl.load(RawValues + (slot * L + j) * V + vv, vv < V, 0).to(tl.float32)
        if KDA:
            g = tl.load(Gates + (slot * L + j) * K + kk, kk < K, 0).to(tl.float32)
        else:
            g = tl.load(Gates + slot * L + j)
        beta = tl.load(Betas + slot * L + j).to(tl.float32)
        state = _delta_rule_step(state, raw_k, raw_v, g, beta, KDA)
    return state
