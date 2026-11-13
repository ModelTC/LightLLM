"""Exact FP32 radix selection, parallel over key blocks and query rows.

Four byte histograms locate the exact kth FP32 value. A prefix scan selects
lower token IDs at the cutoff and emits increasing IDs directly. No score
quantization, approximate threshold, int64 score matrix, or per-row sort.
"""

import torch
import triton
import triton.language as tl


@triton.jit
def _ordered(score):
    # Normalize signed zero before the monotone IEEE-754 bit transform.
    raw = score.to(tl.uint32, bitcast=True)
    bits = tl.where((raw & 0x7FFFFFFF) == 0, 0, raw).to(tl.uint32)
    order = tl.where((bits & 0x80000000) != 0, ~bits, bits ^ 0x80000000)
    finite = (raw & 0x7F800000) != 0x7F800000
    return tl.where(finite, order, 0).to(tl.uint32)


@triton.jit
def _histogram(
    Scores,
    Lengths,
    Prefix,
    Hist,
    NK: tl.constexpr,
    NB: tl.constexpr,
    SHIFT: tl.constexpr,
    B: tl.constexpr,
    USE_PDL: tl.constexpr = False,
):
    row, block = tl.program_id(0), tl.program_id(1)
    if USE_PDL:
        tl.extra.cuda.gdc_wait()
    col = block * B + tl.arange(0, B)
    length = tl.load(Lengths + row)
    value = tl.load(Scores + row * NK + col, (col < NK) & (col < length), float("-inf"))
    order = _ordered(value)
    valid = order != 0
    if SHIFT < 24:
        prefix = tl.load(Prefix + row)
        valid = valid & ((order >> (SHIFT + 8)) == (prefix >> (SHIFT + 8)))
    bins = ((order >> SHIFT) & 255).to(tl.int32)
    histogram = tl.histogram(bins, 256, mask=valid)
    tl.store(Hist + (row * NB + block) * 256 + tl.arange(0, 256), histogram)
    if USE_PDL:
        tl.extra.cuda.gdc_launch_dependents()


@triton.jit
def _choose_byte(
    Hist,
    Prefix,
    Remaining,
    Strict,
    Offsets,
    Allowances,
    Output,
    NB: tl.constexpr,
    SHIFT: tl.constexpr,
    K: tl.constexpr,
    BB: tl.constexpr,
    BK: tl.constexpr,
    USE_PDL: tl.constexpr = False,
):
    row = tl.program_id(0)
    if USE_PDL:
        tl.extra.cuda.gdc_wait()
    blocks = tl.arange(0, BB)
    bins = tl.arange(0, 256)
    block_hist = tl.load(
        Hist + (row * NB + blocks[:, None]) * 256 + bins[None, :],
        blocks[:, None] < NB,
        0,
    )
    histogram = tl.sum(block_hist, 0)
    total = tl.sum(histogram, 0)
    if SHIFT == 24:
        remaining = tl.minimum(K, total)
        prefix = tl.full((), 0, tl.uint32)
    else:
        remaining = tl.minimum(tl.load(Remaining + row), total)
        prefix = tl.load(Prefix + row)
    cumulative = tl.cumsum(histogram, 0, reverse=True)
    chosen = tl.max(tl.where((remaining > 0) & (cumulative >= remaining), bins, 0), 0)
    higher = tl.sum(tl.where(bins > chosen, histogram, 0), 0)
    tl.store(Prefix + row, prefix | (chosen.to(tl.uint32) << SHIFT))
    tl.store(Remaining + row, remaining - higher)
    # Each pass knows which items became strictly greater than the final
    # cutoff. Accumulate these block counts without rescanning the scores.
    strict = tl.sum(tl.where(bins[None, :] > chosen, block_hist, 0), 1)
    if SHIFT < 24:
        strict += tl.load(Strict + row * NB + blocks, blocks < NB, 0)
    if SHIFT > 0:
        tl.store(Strict + row * NB + blocks, strict, blocks < NB)
    else:
        equal = tl.sum(tl.where(bins[None, :] == chosen, block_hist, 0), 1)
        need = tl.maximum(K - tl.sum(strict, 0), 0)
        equal_before = tl.cumsum(equal, 0) - equal
        allowance = tl.minimum(equal, tl.maximum(need - equal_before, 0))
        selected = strict + allowance
        offset = tl.cumsum(selected, 0) - selected
        tl.store(Offsets + row * NB + blocks, offset, blocks < NB)
        tl.store(Allowances + row * NB + blocks, allowance, blocks < NB)
        slots = tl.arange(0, BK)
        tl.store(Output + row * K + slots, -1, slots < K)
    if USE_PDL:
        tl.extra.cuda.gdc_launch_dependents()


@triton.jit
def _scatter(
    Scores,
    Lengths,
    Cutoff,
    Offsets,
    Allowances,
    Output,
    Table,
    Requests,
    TS: tl.constexpr,
    PHYSICAL: tl.constexpr,
    NK: tl.constexpr,
    NB: tl.constexpr,
    K: tl.constexpr,
    B: tl.constexpr,
    USE_PDL: tl.constexpr = False,
):
    row, block = tl.program_id(0), tl.program_id(1)
    if USE_PDL:
        tl.extra.cuda.gdc_wait()
    col = block * B + tl.arange(0, B)
    length, cutoff = tl.load(Lengths + row), tl.load(Cutoff + row)
    value = tl.load(Scores + row * NK + col, (col < NK) & (col < length), float("-inf"))
    order = _ordered(value)
    equal = (order != 0) & (order == cutoff)
    allowance = tl.load(Allowances + row * NB + block)
    selected = ((order != 0) & (order > cutoff)) | (equal & (tl.cumsum(equal.to(tl.int32), 0) <= allowance))
    slot = tl.load(Offsets + row * NB + block) + tl.cumsum(selected.to(tl.int32), 0) - 1
    write = selected & (slot < K)
    if PHYSICAL:
        request = tl.load(Requests + row)
        mapped = tl.load(Table + request * TS + col, write, other=0)
    else:
        mapped = col.to(tl.int32)
    tl.store(Output + row * K + slot, mapped, write)


def _pdl_launch_kwargs(tensor, enabled):
    if not enabled or tensor.device.type != "cuda":
        return {"USE_PDL": False}
    if torch.cuda.get_device_capability(tensor.device)[0] < 9:
        return {"USE_PDL": False}
    return {"USE_PDL": True, "launch_pdl": True}


def exact_radix_topk(scores, lengths, topk, *, block_size=4096, table=None, requests=None, use_pdl=False):
    """Select exact finite topk values, optionally mapped to physical pages."""
    rows, keys = scores.shape
    count = min(topk, keys)
    physical = table is not None or requests is not None
    if physical:
        if table is None or requests is None:
            raise ValueError("table and requests must be provided together")
        if table.ndim != 2 or requests.shape != (rows,):
            raise ValueError("invalid page-table mapping shapes")
        if table.device != scores.device or requests.device != scores.device:
            raise ValueError("page-table mapping must use the score device")
        table = table.contiguous()
        requests = requests.contiguous()
    if block_size not in (1024, 2048, 4096, 8192):
        raise ValueError("Unsupported DSA radix key tile")
    blocks = triton.cdiv(keys, block_size)
    output = torch.empty((rows, count), dtype=torch.int32, device=scores.device)
    if rows == 0:
        return output
    hist = torch.empty((rows, blocks, 256), dtype=torch.int32, device=scores.device)
    prefix = torch.empty(rows, dtype=torch.uint32, device=scores.device)
    remaining = torch.empty(rows, dtype=torch.int32, device=scores.device)
    strict = torch.empty((rows, blocks), dtype=torch.int32, device=scores.device)
    offsets = torch.empty_like(strict)
    allowances = torch.empty_like(strict)
    reduce_warps = 8 if blocks >= 64 else 4
    pdl_kwargs = _pdl_launch_kwargs(scores, use_pdl)
    for shift in (24, 16, 8, 0):
        _histogram[(rows, blocks)](
            scores,
            lengths,
            prefix,
            hist,
            keys,
            blocks,
            shift,
            block_size,
            num_warps=4,
            **pdl_kwargs,
        )
        _choose_byte[(rows,)](
            hist,
            prefix,
            remaining,
            strict,
            offsets,
            allowances,
            output,
            blocks,
            shift,
            count,
            triton.next_power_of_2(blocks),
            triton.next_power_of_2(count),
            num_warps=reduce_warps,
            **pdl_kwargs,
        )
    _scatter[(rows, blocks)](
        scores,
        lengths,
        prefix,
        offsets,
        allowances,
        output,
        table if physical else scores,
        requests if physical else lengths,
        table.stride(0) if physical else 1,
        physical,
        keys,
        blocks,
        count,
        block_size,
        num_warps=4,
        **pdl_kwargs,
    )
    return output
