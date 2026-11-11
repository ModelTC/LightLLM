# SPDX-License-Identifier: Apache-2.0
"""Update the accepted-prefix cursor after replay has returned.

Replay itself does not touch the cursor: every layer must still see the length
from the start of this step. After the last layer returns, advance the cursor
over the replayed window. MTP verify then trims the rejected tail.
"""

import triton
import triton.language as tl


@triton.jit(do_not_specialize=["N"])
def _advance_cursor(
    cursors,
    reqs,
    cu,
    N,
    HOLD: tl.constexpr,
    CAPACITY: tl.constexpr,
    BLOCK: tl.constexpr,
):
    # 先按整窗全接受。放不下时新 token 占 [0, window)，cursor 记为 window；否则加上 window。
    i = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    r = tl.load(reqs + i, i < N, HOLD)
    valid = (i < N) & (r != HOLD)
    cursor = tl.load(cursors + r, valid, 0)
    window = tl.load(cu + i + 1, valid, 0) - tl.load(cu + i, valid, 0)
    updated = tl.where(cursor + window > CAPACITY, window, cursor + window)
    tl.store(cursors + r, updated, valid)


def advance_cursor(history, reqs, cu_seqlens) -> None:
    """最后一层返回后，把 cursor 推进到整窗都被接受的位置。

    cu_seqlens 与 reqs 逐行对齐，长度是请求数加一。窗口长度为 0 时 cursor 保持不变。HOLD 请求跳过。
    """

    if reqs.numel() == 0:
        return
    assert reqs.is_cuda and cu_seqlens.is_cuda, "advance_cursor requires CUDA inputs"
    assert cu_seqlens.numel() == reqs.numel() + 1, "cu_seqlens must cover every request"
    _advance_cursor[(triton.cdiv(reqs.numel(), 256),)](
        history.cursors,
        reqs,
        cu_seqlens,
        reqs.numel(),
        history.hold_req_id,
        history.capacity,
        256,
        num_warps=4,
    )
