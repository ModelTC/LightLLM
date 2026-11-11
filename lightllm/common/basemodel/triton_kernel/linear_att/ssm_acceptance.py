from __future__ import annotations

import torch
import triton
import triton.language as tl


def commit_ssm_prefix(
    indices: torch.Tensor,
    reqs: torch.Tensor,
    lengths: torch.Tensor,
    replay_cursors: torch.Tensor | None = None,
    write_bases: torch.Tensor | None = None,
) -> None:
    """把每个请求的 accepted length 写入 MTP 索引。

    传入 write_bases 时，cursor 设为这次 replay 的写起点加接受长度。层内已经折回时写起点是 0。
    """
    if not reqs.is_cuda:
        indices[reqs] = (lengths - 1).to(indices.dtype)
        if replay_cursors is not None:
            valid = reqs != replay_cursors.numel() - 1
            accepted = lengths[valid].to(replay_cursors.dtype)
            if write_bases is not None:
                replay_cursors[reqs[valid]] = write_bases[reqs[valid]] + accepted
            else:
                replay_cursors[reqs[valid]] += accepted
        return
    if reqs.numel():
        _commit_prefix[(triton.cdiv(reqs.numel(), 256),)](
            indices,
            reqs,
            lengths,
            replay_cursors,
            write_bases if write_bases is not None else (replay_cursors if replay_cursors is not None else indices),
            reqs.numel(),
            indices.numel() - 1,
            reqs.stride(0),
            lengths.stride(0),
            256,
            WRITE_BASE=write_bases is not None,
            num_warps=4,
        )


@triton.jit(do_not_specialize=["N"])
def _commit_prefix(
    Indices,
    Reqs,
    Lengths,
    Cursors,
    WriteBase,
    N,
    HOLD: tl.constexpr,
    REQ_STRIDE: tl.constexpr,
    LENGTH_STRIDE: tl.constexpr,
    BLOCK: tl.constexpr,
    WRITE_BASE: tl.constexpr,
):
    row = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    req = tl.load(Reqs + row * REQ_STRIDE, row < N, 0)
    length = tl.load(Lengths + row * LENGTH_STRIDE, row < N, 0)
    tl.store(Indices + req, length - 1, row < N)
    if Cursors is not None:
        valid = (row < N) & (req != HOLD)
        base = tl.load(WriteBase + req, valid, 0) if WRITE_BASE else tl.load(Cursors + req, valid, 0)
        tl.store(Cursors + req, base + length, valid)
