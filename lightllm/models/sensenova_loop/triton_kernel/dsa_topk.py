"""Exact canonical top-k sets stored in increasing key order for sparse gathers."""
import torch
import triton
import triton.language as tl


@triton.jit
def _count(Scores, Cutoff, Counts, NK: tl.constexpr, NB: tl.constexpr, B: tl.constexpr):
    row = tl.program_id(0)
    block = tl.program_id(1)
    col = block * B + tl.arange(0, B)
    cutoff = tl.load(Cutoff + row)
    count = tl.full((), 0, tl.int32)
    if cutoff > float("-inf"):
        values = tl.load(Scores + row * NK + col, col < NK, other=float("inf"))
        count = tl.sum(((values == cutoff) & (col < NK)).to(tl.int32), 0)
    tl.store(Counts + row * NB + block, count)


@triton.jit
def _repair(
    Scores,
    Cutoff,
    Strict,
    Counts,
    Indices,
    NK: tl.constexpr,
    K: tl.constexpr,
    NB: tl.constexpr,
    BC: tl.constexpr,
    B: tl.constexpr,
):
    row = tl.program_id(0)
    block = tl.program_id(1)
    ids = tl.arange(0, BC)
    counts = tl.load(Counts + row * NB + ids, ids < NB, other=0)
    total = tl.sum(counts, 0)
    prefix = tl.sum(tl.where(ids < block, counts, 0), 0)
    strict = tl.load(Strict + row)
    needed = K - strict
    if total > needed and prefix < needed:
        col = block * B + tl.arange(0, B)
        cutoff = tl.load(Cutoff + row)
        values = tl.load(Scores + row * NK + col, col < NK, other=float("inf"))
        match = (values == cutoff) & (col < NK)
        slot = prefix + tl.cumsum(match.to(tl.int32), 0) - 1
        tl.store(Indices + row * K + strict + slot, col, match & (slot < needed))


@triton.jit
def _compact(Indices, Values, Output, Cutoff, Strict, K: tl.constexpr, BLOCK: tl.constexpr):
    row = tl.program_id(0)
    col = tl.arange(0, BLOCK)
    value = tl.load(Values + row * K + col, col < K, other=float("inf"))
    index = tl.load(Indices + row * K + col, col < K, other=-1)
    cutoff = tl.min(value, 0)
    finite = (value > float("-inf")) & (value < float("inf")) & (index >= 0) & (col < K)
    strict = finite & (value > cutoff)
    count = tl.sum(strict.to(tl.int32), 0)
    strict_slot = tl.cumsum(strict.to(tl.int32), 0) - 1
    other_slot = count + tl.cumsum((~strict).to(tl.int32), 0) - 1
    slot = tl.where(strict, strict_slot, other_slot)
    tl.store(Output + row * K + slot, tl.where(finite, index, -1), col < K)
    tl.store(Cutoff + row, cutoff)
    tl.store(Strict + row, count)


@triton.jit
def _sort_indices(Scores, Indices, Output, Values, NK: tl.constexpr, K: tl.constexpr, BLOCK: tl.constexpr):
    row = tl.program_id(0)
    col = tl.arange(0, BLOCK)
    index = tl.load(Indices + row * K + col, col < K, other=-1)
    index = tl.sort(tl.where(index >= 0, index, 2147483647), descending=False)
    valid = (col < K) & (index < NK)
    value = tl.load(Scores + row * NK + index, valid, other=float("-inf"))
    value = tl.where(value == 0, 0.0, value)
    tl.store(Output + row * K + col, tl.where(valid, index, -1), col < K)
    tl.store(Values + row * K + col, value, col < K)


def canonical_topk(scores, indices, values):
    count, keys = scores.shape
    slots = indices.shape[-1]
    blocks = triton.cdiv(keys, 4096)
    cutoff = values.new_empty(count)
    strict = indices.new_empty(count)
    counts = indices.new_empty((count, blocks))
    compact = torch.empty_like(indices)
    output = torch.empty_like(indices)
    selected = torch.empty_like(values)
    if scores.device.type == "cpu":
        indices = torch.argsort(scores, dim=-1, descending=True, stable=True)[:, :slots]
        indices = indices.masked_fill(~torch.isfinite(scores.gather(1, indices)), -1)
        indices = torch.sort(indices.masked_fill(indices < 0, keys), dim=-1).values
        valid = indices < keys
        indices = indices.masked_fill(~valid, -1)
        values = scores.gather(1, indices.clamp_min(0)).masked_fill(~valid, -torch.inf)
        return indices.to(torch.int32), values
    with torch.cuda.device(scores.device):
        _compact[(count,)](indices, values, compact, cutoff, strict, slots, triton.next_power_of_2(slots), num_warps=4)
        _count[(count, blocks)](scores, cutoff, counts, keys, blocks, 4096, num_warps=4)
        _repair[(count, blocks)](
            scores,
            cutoff,
            strict,
            counts,
            compact,
            keys,
            slots,
            blocks,
            triton.next_power_of_2(blocks),
            4096,
            num_warps=4,
        )
        _sort_indices[(count,)](
            scores, compact, output, selected, keys, slots, triton.next_power_of_2(slots), num_warps=4
        )
    return output, selected
