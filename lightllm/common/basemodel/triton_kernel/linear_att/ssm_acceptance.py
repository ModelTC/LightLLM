import triton
import triton.language as tl


@triton.jit
def _commit_prefix(
    Indices,
    Reqs,
    Lengths,
    Cursors,
    N: tl.constexpr,
    HOLD: tl.constexpr,
    REQ_STRIDE: tl.constexpr,
    LENGTH_STRIDE: tl.constexpr,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    req = tl.load(Reqs + row * REQ_STRIDE, row < N, 0)
    length = tl.load(Lengths + row * LENGTH_STRIDE, row < N, 0)
    tl.store(Indices + req, length - 1, row < N)
    if Cursors is not None:
        valid = (row < N) & (req != HOLD)
        cursor = tl.load(Cursors + req, valid, 0)
        tl.store(Cursors + req, cursor + length, valid)


def commit_ssm_prefix(indices, reqs, lengths, replay_cursors=None):
    if not reqs.is_cuda:
        indices[reqs] = (lengths - 1).to(indices.dtype)
        if replay_cursors is not None:
            valid = reqs != replay_cursors.numel() - 1
            replay_cursors[reqs[valid]] += lengths[valid].to(replay_cursors.dtype)
        return
    if reqs.numel():
        _commit_prefix[(triton.cdiv(reqs.numel(), 256),)](
            indices,
            reqs,
            lengths,
            replay_cursors,
            reqs.numel(),
            indices.numel() - 1,
            reqs.stride(0),
            lengths.stride(0),
            256,
            num_warps=4,
        )
