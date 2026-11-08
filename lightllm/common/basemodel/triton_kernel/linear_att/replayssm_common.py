# SPDX-License-Identifier: Apache-2.0
"""Shared ReplaySSM recurrence primitives."""

import triton
import triton.language as tl


@triton.jit
def _replay_update(state, raw_k, raw_v, g, beta, KDA: tl.constexpr):
    # Replay folding and materialization use the same update order.
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
