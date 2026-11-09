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
