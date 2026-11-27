import torch
import triton
import triton.language as tl

from lightllm.utils.sgl_utils import sgl_ops


@triton.jit
def _ordered_score_keys(values):
    bits = values.to(tl.uint32, bitcast=True)
    bits = tl.where((bits & 0x7FFFFFFF) == 0, 0, bits).to(tl.uint32)
    return bits ^ tl.where((bits >> 31) != 0, 0xFFFFFFFF, 0x80000000).to(tl.uint32)


@triton.jit
def _score_keys(scores, offsets, valid):
    return _ordered_score_keys(tl.load(scores + offsets, valid, 0))


@triton.jit
def _radix_row(
    scores,
    lengths,
    out,
    stride,
    out_stride,
    width: tl.constexpr,
    count: tl.constexpr,
    block: tl.constexpr,
    out_block: tl.constexpr,
    out_col_stride: tl.constexpr = 1,
):
    row = tl.program_id(0)
    length = tl.minimum(tl.load(lengths + row), width)
    if width <= count or length <= count:
        destinations = tl.arange(0, out_block)
        tl.store(
            out + row * out_stride + destinations * out_col_stride,
            tl.where(destinations < length, destinations, -1),
            destinations < count,
        )
    else:
        offsets = tl.arange(0, block)
        active = offsets < length
        keys = _score_keys(scores + row * stride, offsets, active)
        remaining = count
        equal = active
        greater = tl.full((block,), False, tl.int1)
        bins = tl.arange(0, 256)
        step = 0
        refine = True
        while step < 4 and refine:
            digit = ((keys >> (24 - step * 8)) & 255).to(tl.int32)
            histogram = tl.histogram(digit, 256, mask=equal)
            inclusive = tl.cumsum(histogram, 0, reverse=True)
            chosen = tl.max(tl.where(inclusive >= remaining, bins, 0), 0)
            greater = greater | (equal & (digit > chosen))
            equal = equal & (digit == chosen)
            refine = tl.max(tl.where(bins == chosen, inclusive, 0), 0) != remaining
            remaining = remaining - tl.max(tl.where(bins == chosen + 1, inclusive, 0), 0)
            step += 1
        ties = tl.cumsum(equal.to(tl.int32), 0)
        selected = greater | (equal & (ties <= remaining))
        ranks = tl.cumsum(selected.to(tl.int32), 0) - 1
        tl.store(out + row * out_stride + ranks * out_col_stride, offsets, selected & (ranks < count))


@triton.jit
def _radix_hist(
    scores,
    lengths,
    prefixes,
    histograms,
    stride,
    width: tl.constexpr,
    tiles: tl.constexpr,
    step: tl.constexpr,
    tile_width: tl.constexpr,
):
    row = tl.program_id(0)
    tile = tl.program_id(1)
    offsets = tile * tile_width + tl.arange(0, tile_width)
    length = tl.minimum(tl.load(lengths + row), width)
    valid = offsets < length
    keys = _score_keys(scores + row * stride, offsets, valid)
    if step:
        prefix = tl.load(prefixes + row)
        valid &= (keys >> (32 - step * 8)) == prefix
    digit = ((keys >> (24 - step * 8)) & 255).to(tl.int32)
    histogram = tl.histogram(digit, 256, mask=valid)
    tl.store(histograms + (row * tiles + tile) * 256 + tl.arange(0, 256), histogram)


@triton.jit
def _radix_next(
    histograms,
    prefixes,
    ranks,
    tiles: tl.constexpr,
    padded_tiles: tl.constexpr,
    step: tl.constexpr,
    count: tl.constexpr,
):
    row = tl.program_id(0)
    tile = tl.arange(0, padded_tiles)
    bins = tl.arange(0, 256)
    counts = tl.sum(
        tl.load(histograms + (row * tiles + tile[:, None]) * 256 + bins[None, :], tile[:, None] < tiles, 0),
        axis=0,
    )
    remaining = tl.load(ranks + row) if step else count
    descending = tl.cumsum(counts, axis=0, reverse=True)
    chosen = tl.max(tl.where(descending >= remaining, bins, 0), 0)
    higher = tl.sum(tl.where(bins > chosen, counts, 0), 0)
    prefix = tl.load(prefixes + row) if step else tl.full((), 0, tl.uint32)
    tl.store(prefixes + row, (prefix << 8) | chosen.to(tl.uint32))
    tl.store(ranks + row, remaining - higher)


@triton.jit
def _radix_counts(
    scores, lengths, prefixes, tile_counts, stride, width: tl.constexpr, tiles: tl.constexpr, tile_width: tl.constexpr
):
    row = tl.program_id(0)
    tile = tl.program_id(1)
    offsets = tile * tile_width + tl.arange(0, tile_width)
    valid = offsets < tl.minimum(tl.load(lengths + row), width)
    keys = _score_keys(scores + row * stride, offsets, valid)
    cutoff = tl.load(prefixes + row)
    greater = tl.sum((valid & (keys > cutoff)).to(tl.int32), 0)
    equal = tl.sum((valid & (keys == cutoff)).to(tl.int32), 0)
    tl.store(tile_counts + row * 2 * tiles + tile, greater)
    tl.store(tile_counts + row * 2 * tiles + tiles + tile, equal)


@triton.jit
def _radix_output(
    scores,
    lengths,
    prefixes,
    ranks,
    tile_counts,
    out,
    stride,
    out_stride,
    width: tl.constexpr,
    count: tl.constexpr,
    tiles: tl.constexpr,
    padded_tiles: tl.constexpr,
    tile_width: tl.constexpr,
    out_block: tl.constexpr,
):
    row = tl.program_id(0)
    tile = tl.program_id(1)
    length = tl.minimum(tl.load(lengths + row), width)
    if length <= count:
        if tile == 0:
            indices = tl.arange(0, out_block)
            tl.store(out + row * out_stride + indices, tl.where(indices < length, indices, -1), indices < count)
    else:
        blocks = tl.arange(0, padded_tiles)
        higher_counts = tl.load(tile_counts + row * 2 * tiles + blocks, blocks < tiles, 0)
        equal_before = tl.sum(tl.load(tile_counts + row * 2 * tiles + tiles + blocks, blocks < tile, 0), 0)
        higher_before = tl.sum(tl.where(blocks < tile, higher_counts, 0), 0)
        higher_total = tl.sum(higher_counts, 0)
        remaining = tl.load(ranks + row)
        positions = tile * tile_width + tl.arange(0, tile_width)
        valid = positions < length
        keys = _score_keys(scores + row * stride, positions, valid)
        cutoff = tl.load(prefixes + row)
        greater = valid & (keys > cutoff)
        equal = valid & (keys == cutoff)
        greater_index = higher_before + tl.cumsum(greater.to(tl.int32), 0) - 1
        equal_index = equal_before + tl.cumsum(equal.to(tl.int32), 0) - 1
        tl.store(out + row * out_stride + greater_index, positions, greater & (greater_index < count))
        tl.store(
            out + row * out_stride + higher_total + equal_index,
            positions,
            equal & (equal_index < remaining) & (higher_total + equal_index < count),
        )


def _partitioned_radix_topk(scores, lengths, out):
    rows, width = scores.shape
    count = out.shape[1]
    tile_width = 2048
    tiles = triton.cdiv(width, tile_width)
    padded_tiles = triton.next_power_of_2(tiles)
    prefixes = torch.empty((rows,), device=scores.device, dtype=torch.uint32)
    ranks = torch.empty((rows,), device=scores.device, dtype=torch.int32)
    histograms = torch.empty((rows, tiles, 256), device=scores.device, dtype=torch.int32)
    counts = torch.empty((rows, 2, tiles), device=scores.device, dtype=torch.int32)
    lengths = lengths.contiguous()
    for step in range(4):
        _radix_hist[(rows, tiles)](
            scores,
            lengths,
            prefixes,
            histograms,
            scores.stride(0),
            width,
            tiles,
            step,
            tile_width,
        )
        _radix_next[(rows,)](histograms, prefixes, ranks, tiles, padded_tiles, step, count, num_warps=8)
    _radix_counts[(rows, tiles)](
        scores,
        lengths,
        prefixes,
        counts,
        scores.stride(0),
        width,
        tiles,
        tile_width,
    )
    _radix_output[(rows, tiles)](
        scores,
        lengths,
        prefixes,
        ranks,
        counts,
        out,
        scores.stride(0),
        out.stride(0),
        width,
        count,
        tiles,
        padded_tiles,
        tile_width,
        triton.next_power_of_2(count),
    )


def _single_block_radix_topk(scores, lengths, out):
    rows, width = scores.shape
    count = out.shape[1]
    _radix_row[(rows,)](
        scores,
        lengths.contiguous(),
        out,
        scores.stride(0),
        out.stride(0),
        width,
        count,
        triton.next_power_of_2(max(1, width)),
        triton.next_power_of_2(count),
        out_col_stride=out.stride(1),
        num_warps=8,
    )


def use_page_table_topk(scores, count):
    width = scores.shape[1]
    return scores.stride(1) == 1 and (
        (2048 < width < 65536 and count <= 512) or (8192 <= width <= 16384 and count == 2048)
    )


def ragged_topk(scores, lengths, out):
    rows, width = scores.shape
    count = out.shape[1]
    if rows == 0 or count == 0:
        return
    if width <= count and scores.is_cuda:
        _single_block_radix_topk(scores, lengths, out)
        return
    if (
        width >= 65536
        and width <= 262144
        and scores.is_cuda
        and scores.stride(1) == 1
        and scores.dtype == torch.float32
        and count <= 2048
    ):
        _partitioned_radix_topk(scores, lengths, out)
        return
    if (
        count == 2048
        and width <= 4096
        and scores.is_contiguous()
        and out.is_contiguous()
        and scores.dtype == torch.float32
        and hasattr(sgl_ops, "fast_topk_v2")
    ):
        torch.ops.sgl_kernel.fast_topk(scores, out, lengths.contiguous(), None)
        return
    try:
        from flashinfer import top_k_page_table_transform, top_k_ragged_transform
    except ImportError:
        if scores.is_cuda and scores.dtype == torch.float32 and scores.stride(1) == 1 and width <= 8192:
            _single_block_radix_topk(scores, lengths, out)
            return
        if scores.is_cuda and scores.dtype == torch.float32 and scores.stride(1) == 1 and width <= 262144:
            _partitioned_radix_topk(scores, lengths, out)
            return
        positions = torch.arange(width, device=scores.device)
        valid = positions[None, :] < lengths[:, None]
        selected = torch.topk(scores.masked_fill(~valid, -torch.inf), min(count, width), dim=-1, sorted=False).indices
        short_result = positions[: selected.shape[1]][None, :].masked_fill(
            positions[: selected.shape[1]][None, :] >= lengths[:, None], -1
        )
        out[:, : selected.shape[1]].copy_(torch.where(lengths[:, None] <= count, short_result, selected))
        if selected.shape[1] < count:
            out[:, selected.shape[1] :].fill_(-1)
    else:
        if use_page_table_topk(scores, count) and out.is_contiguous():
            identity_page = torch.zeros((rows, 1), dtype=torch.int32, device=scores.device)
            top_k_page_table_transform(
                scores, identity_page, lengths.contiguous(), count, page_size=triton.next_power_of_2(width), out=out
            )
        else:
            out.copy_(
                top_k_ragged_transform(scores.contiguous(), torch.zeros_like(lengths), lengths.contiguous(), count)
            )
