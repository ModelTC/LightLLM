"""Two-stage vocabulary argmax with PyTorch tie and NaN semantics."""

import torch
import triton
import triton.language as tl


@triton.jit
def _argmax_partials(X, M, INDICES, NAN, V: tl.constexpr, S: tl.constexpr, P: tl.constexpr, B: tl.constexpr):
    r = tl.program_id(0)
    p = tl.program_id(1)
    i = p * B + tl.arange(0, B)
    x = tl.load(X + r * S + i, i < V, -float("inf")).to(tl.float32)
    isnan = x != x
    n = tl.min(tl.where((i < V) & isnan, i, 2147483647), 0)
    m = tl.max(tl.where(isnan, -float("inf"), x), 0)
    idx = tl.min(tl.where((i < V) & ~isnan & (x == m), i, 2147483647), 0)
    tl.store(M + r * P + p, m)
    tl.store(INDICES + r * P + p, idx)
    tl.store(NAN + r * P + p, n)


@triton.jit
def _argmax_reduce(M, INDICES, NAN, OUTPUT, P: tl.constexpr, B: tl.constexpr):
    r = tl.program_id(0)
    i = tl.arange(0, B)
    m = tl.load(M + r * P + i, i < P, -float("inf"))
    idx = tl.load(INDICES + r * P + i, i < P, 2147483647)
    nan = tl.min(tl.load(NAN + r * P + i, i < P, 2147483647), 0)
    maximum = tl.max(m, 0)
    winner = tl.min(tl.where(m == maximum, idx, 2147483647), 0)
    tl.store(OUTPUT + r, tl.where(nan < 2147483647, nan, winner).to(tl.int64))


def argmax(logits):
    """Select the first maximum (or first NaN) from contiguous FP32 vocabulary rows."""
    assert logits.dtype == torch.float32 and logits.stride(1) == 1
    rows, vocab = logits.shape
    block = 4096
    parts = triton.cdiv(vocab, block)
    maxima = torch.empty((rows, parts), device=logits.device, dtype=torch.float32)
    indices = torch.empty((rows, parts), device=logits.device, dtype=torch.int32)
    nans = torch.empty_like(indices)
    output = torch.empty(rows, device=logits.device, dtype=torch.int64)
    _argmax_partials[(rows, parts)](logits, maxima, indices, nans, vocab, logits.stride(0), parts, block, num_warps=4)
    _argmax_reduce[(rows,)](maxima, indices, nans, output, parts, triton.next_power_of_2(parts), num_warps=4)
    return output
