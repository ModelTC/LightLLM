# SPDX-License-Identifier: Apache-2.0
"""Shared ReplaySSM recurrence primitives."""

import triton
import triton.language as tl


@triton.jit
def _replay_update(state, raw_k, raw_v, g, beta, KDA: tl.constexpr):
    # 所有 replay fold 路径共用这一条递推，保证 raw/derived/history
    # materialize 使用相同的数值顺序。
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
