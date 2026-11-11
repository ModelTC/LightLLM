# SPDX-License-Identifier: Apache-2.0
"""Advance the accepted-prefix cursor. This kernel does not fold state."""

import triton
import triton.language as tl


@triton.jit(do_not_specialize=["N"])
def _advance_cursor(
    cursors,
    write_bases,
    reqs,
    accepted_state_indices,
    N,
    HOLD: tl.constexpr,
    WIDTH: tl.constexpr,
    BLOCK: tl.constexpr,
):
    # 写起点由 replay kernel 记下。半区已经折回时起点是 0，否则是原 cursor。
    i = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    r = tl.load(reqs + i, i < N, HOLD)
    valid = (i < N) & (r != HOLD)
    base = tl.load(write_bases + r, valid, 0)
    count = tl.load(accepted_state_indices + r, valid, 0) + 1 if WIDTH > 1 else 1
    tl.store(cursors + r, base + count, valid)
