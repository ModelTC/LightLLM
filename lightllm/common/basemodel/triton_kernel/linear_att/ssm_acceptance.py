from __future__ import annotations

import torch
import triton
import triton.language as tl


def commit_ssm_prefix(
    mtp_state_index: torch.Tensor,
    reqs: torch.Tensor,
    accepted_lengths: torch.Tensor,
    req_verify_width: torch.Tensor,
    replay_cursors: torch.Tensor,
) -> None:
    """写入 MTP 索引，并从已按验证宽度全接受推进的 cursor 上剪掉未接受尾部。

    accepted_lengths 含基准 token。req_verify_width 与 reqs 逐行对齐，是每个请求这次送去验证的 token 数。
    """

    assert reqs.is_cuda, "commit_ssm_prefix requires CUDA requests"
    if reqs.numel():
        _commit_prefix[(triton.cdiv(reqs.numel(), 256),)](
            mtp_state_index,
            reqs,
            accepted_lengths,
            replay_cursors,
            req_verify_width,
            reqs.numel(),
            mtp_state_index.numel() - 1,
            reqs.stride(0),
            accepted_lengths.stride(0),
            req_verify_width.stride(0),
            256,
            num_warps=4,
        )


@triton.jit(do_not_specialize=["N"])
def _commit_prefix(
    MtpIndex,
    Reqs,
    Accepted,
    Cursors,
    ReqVerifyWidth,
    N,
    HOLD: tl.constexpr,
    REQ_STRIDE: tl.constexpr,
    ACCEPTED_STRIDE: tl.constexpr,
    VERIFY_STRIDE: tl.constexpr,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    req = tl.load(Reqs + row * REQ_STRIDE, row < N, 0)
    accepted = tl.load(Accepted + row * ACCEPTED_STRIDE, row < N, 0)
    tl.store(MtpIndex + req, accepted - 1, row < N)
    valid = (row < N) & (req != HOLD)
    verify_tokens = tl.load(ReqVerifyWidth + row * VERIFY_STRIDE, valid, 0)
    cursor = tl.load(Cursors + req, valid, 0)
    tl.store(Cursors + req, cursor - (verify_tokens - accepted), valid)
