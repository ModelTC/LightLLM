# SPDX-License-Identifier: Apache-2.0
"""ReplaySSM accepted-prefix length updates."""

import triton
import triton.language as tl


@triton.jit(do_not_specialize=["N"])
def _accept_updates(
    cursors,
    reqs,
    accepted_state_indices,
    N,
    HOLD: tl.constexpr,
    WIDTH: tl.constexpr,
    BLOCK: tl.constexpr,
):
    # N 是运行时请求数；BLOCK/WIDTH 仍是编译期参数，避免 eager batch 变化触发重新编译。
    i = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    r = tl.load(reqs + i, i < N, HOLD)
    valid = (i < N) & (r != HOLD)
    cursor = tl.load(cursors + r, valid, 0)
    count = tl.load(accepted_state_indices + r, valid, 0) + 1 if WIDTH > 1 else 1
    tl.store(cursors + r, cursor + count, valid)
