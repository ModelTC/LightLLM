"""Greedy token selection and log-probability without a full softmax buffer."""

import torch
import triton
import triton.language as tl


@triton.jit
def _partials(X, T, M, S, INDICES, NAN, V: tl.constexpr, STRIDE: tl.constexpr, P: tl.constexpr, B: tl.constexpr):
    row = tl.program_id(0)
    part = tl.program_id(1)
    i = part * B + tl.arange(0, B)
    valid = i < V
    x = tl.load(X + row * STRIDE + i, valid, other=-float("inf")).to(tl.float32)
    temperature = tl.load(T + row)
    x = tl.div_rn(x, temperature)
    tl.store(X + row * STRIDE + i, x, valid)
    is_nan = x != x
    nan = tl.min(tl.where(valid & is_nan, i, 2147483647), 0)
    maximum = tl.max(tl.where(is_nan, -float("inf"), x), 0)
    winner = tl.min(tl.where(valid & ~is_nan & (x == maximum), i, 2147483647), 0)
    denom = tl.sum(tl.where(valid & (maximum != -float("inf")), tl.exp(x - maximum), 0.0), 0)
    offset = row * P + part
    tl.store(M + offset, maximum)
    tl.store(S + offset, denom)
    tl.store(INDICES + offset, winner)
    tl.store(NAN + offset, nan)


@triton.jit
def _finish(M, S, INDICES, NAN, OUT, LOGP, P: tl.constexpr, B: tl.constexpr):
    row = tl.program_id(0)
    i = tl.arange(0, B)
    valid = i < P
    m = tl.load(M + row * P + i, valid, other=-float("inf"))
    s = tl.load(S + row * P + i, valid, other=0.0)
    ids = tl.load(INDICES + row * P + i, valid, other=2147483647)
    nans = tl.load(NAN + row * P + i, valid, other=2147483647)
    maximum = tl.max(m, 0)
    winner = tl.min(tl.where(valid & (m == maximum), ids, 2147483647), 0)
    first_nan = tl.min(nans, 0)
    winner = tl.where(first_nan < 2147483647, first_nan, winner)
    denom = tl.sum(tl.where(valid, s * tl.exp(m - maximum), 0.0), 0)
    logp = -tl.log(denom)
    logp = tl.where(first_nan < 2147483647, float("nan"), logp)
    tl.store(OUT + row, winner.to(tl.int64))
    tl.store(LOGP + row, logp)


def greedy_sample(logits, temperatures):
    """Scale logits in place, preserving torch.div rounding and first-index argmax ties."""
    assert logits.dtype == torch.float32 and logits.stride(1) == 1
    rows, vocab = logits.shape
    block = 4096
    parts = triton.cdiv(vocab, block)
    maximum = torch.empty((rows, parts), device=logits.device, dtype=torch.float32)
    sums = torch.empty_like(maximum)
    indices = torch.empty((rows, parts), device=logits.device, dtype=torch.int32)
    nans = torch.empty_like(indices)
    ids = torch.empty(rows, device=logits.device, dtype=torch.int64)
    logp = torch.empty(rows, device=logits.device, dtype=torch.float32)
    _partials[(rows, parts)](
        logits, temperatures, maximum, sums, indices, nans, vocab, logits.stride(0), parts, block, num_warps=4
    )
    _finish[(rows,)](maximum, sums, indices, nans, ids, logp, parts, triton.next_power_of_2(parts), num_warps=4)
    return ids, logp
