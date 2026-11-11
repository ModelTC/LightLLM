import torch
import triton
import triton.language as tl


@triton.jit
def _scatter_prompt_logprobs_to_cpu(
    mem_indexes,
    top_token_ids,
    top_logprobs,
    top_token_ids_buffer_ptr,
    top_logprobs_buffer_ptr,
    token_count,
    kv_cache_size,
    TOPK: tl.constexpr,
    MAX_TOPK: tl.constexpr,
    BLOCK: tl.constexpr,
):
    offsets = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    rows = offsets // TOPK
    columns = offsets - rows * TOPK
    mask = rows < token_count
    mem_index = tl.load(mem_indexes + rows, mask=mask, other=-1).to(tl.int64)
    write_mask = mask & (mem_index >= 0) & (mem_index < kv_cache_size)
    source_offset = rows * TOPK + columns
    destination_offset = mem_index * MAX_TOPK + columns

    token_ids = tl.load(top_token_ids + source_offset, mask=write_mask, other=-1)
    token_ids_dst = tl.load(top_token_ids_buffer_ptr).to(tl.pointer_type(tl.int32))
    tl.store(token_ids_dst + destination_offset, token_ids, mask=write_mask)

    logprobs = tl.load(top_logprobs + source_offset, mask=write_mask, other=0.0)
    logprobs_dst = tl.load(top_logprobs_buffer_ptr).to(tl.pointer_type(tl.float32))
    tl.store(logprobs_dst + destination_offset, logprobs, mask=write_mask)


def scatter_prompt_logprobs_to_cpu(
    mem_indexes: torch.Tensor,
    top_token_ids: torch.Tensor,
    top_logprobs: torch.Tensor,
    top_token_ids_buffer_ptr: torch.Tensor,
    top_logprobs_buffer_ptr: torch.Tensor,
    kv_cache_size: int,
    max_topk: int,
) -> None:
    token_count, topk = top_token_ids.shape
    assert mem_indexes.is_cuda and mem_indexes.is_contiguous()
    assert mem_indexes.dtype in (torch.int32, torch.int64)
    assert mem_indexes.numel() == token_count
    assert top_token_ids.is_cuda and top_token_ids.is_contiguous()
    assert top_token_ids.dtype == torch.int32
    assert top_logprobs.is_cuda and top_logprobs.is_contiguous()
    assert top_logprobs.dtype == torch.float32
    assert top_logprobs.shape == top_token_ids.shape
    assert 0 < topk <= max_topk

    block = 1024
    _scatter_prompt_logprobs_to_cpu[(triton.cdiv(token_count * topk, block),)](
        mem_indexes=mem_indexes,
        top_token_ids=top_token_ids,
        top_logprobs=top_logprobs,
        top_token_ids_buffer_ptr=top_token_ids_buffer_ptr,
        top_logprobs_buffer_ptr=top_logprobs_buffer_ptr,
        token_count=token_count,
        kv_cache_size=kv_cache_size,
        TOPK=topk,
        MAX_TOPK=max_topk,
        BLOCK=block,
    )


@triton.jit
def _output_logprobs_partial(
    logits,
    sampled_ids,
    rows,
    counts,
    candidates,
    candidate_rows,
    stride_row: tl.constexpr,
    vocab_size: tl.constexpr,
    tiles: tl.constexpr,
    CHECK_RANK: tl.constexpr,
    TOPK: tl.constexpr,
    BLOCK: tl.constexpr,
):
    index = tl.program_id(0)
    tile = tl.program_id(1)
    row = tl.load(rows + index * 3).to(tl.int64)
    row_logits = logits + row * stride_row
    ids = tile * BLOCK + tl.arange(0, BLOCK)
    scores = tl.load(row_logits + ids, ids < vocab_size, -float("inf")).to(tl.float32)
    if CHECK_RANK:
        sampled = tl.load(row_logits + tl.load(sampled_ids + row))
        # A tie at the 20th score still counts as being in the top 20.
        above = (scores > sampled) | (scores != scores)
        tl.store(counts + index * tiles + tile, tl.sum((above & (ids < vocab_size)).to(tl.int32), 0))
    if TOPK > 0:
        if index < candidate_rows:
            # Sort by float score, using the smaller token ID to break ties.
            bits = scores.to(tl.int32, bitcast=True)
            ordered = bits ^ tl.where(bits < 0, 0x7FFFFFFF, 0)
            ordered = tl.where(scores != scores, 0x7FFFFFFF, ordered)
            keys = (ordered.to(tl.int64) << 32) | (0xFFFFFFFF - ids.to(tl.int64))
            keys = tl.where(ids < vocab_size, keys, -0x8000000000000000)
            keys = tl.topk(keys, TOPK)
            tl.store(candidates + (index * tiles + tile) * TOPK + tl.arange(0, TOPK), keys)


@triton.jit
def _output_logprobs_finish(
    logits,
    sampled_ids,
    sampled_logprobs,
    rows,
    counts,
    candidates,
    output,
    dense_top_ids,
    candidate_rows,
    stride_row: tl.constexpr,
    vocab_size: tl.constexpr,
    tiles: tl.constexpr,
    CHECK_RANK: tl.constexpr,
    TOPK: tl.constexpr,
    TILE_BLOCK: tl.constexpr,
    DENSE_TOPK: tl.constexpr,
):
    index = tl.program_id(0)
    row = tl.load(rows + index * 3).to(tl.int64)
    row_logits = logits + row * stride_row
    sampled = tl.load(row_logits + tl.load(sampled_ids + row))
    sampled_logprob = tl.load(sampled_logprobs + row)
    if CHECK_RANK:
        if DENSE_TOPK:
            cutoff_id = tl.load(dense_top_ids + index * DENSE_TOPK + DENSE_TOPK - 1)
            in_top20 = sampled >= tl.load(row_logits + cutoff_id)
        else:
            tile = tl.arange(0, TILE_BLOCK)
            above = tl.load(counts + index * tiles + tile, tile < tiles, 0)
            in_top20 = tl.sum(above, 0) < 20
        in_top20 = in_top20 & (tl.abs(sampled) < float("inf"))
        tl.store(sampled_logprobs + row, -9999.0, mask=~in_top20)
    if TOPK > 0:
        if index < candidate_rows:
            width = tl.load(rows + index * 3 + 1)
            offset = tl.load(rows + index * 3 + 2)
            columns = tl.arange(0, TOPK)
            if DENSE_TOPK:
                ids = tl.load(dense_top_ids + index * DENSE_TOPK + columns, columns < DENSE_TOPK, vocab_size)
            else:
                positions = tl.arange(0, TILE_BLOCK * TOPK)
                keys = tl.load(
                    candidates + index * tiles * TOPK + positions, positions < tiles * TOPK, -0x8000000000000000
                )
                best = tl.topk(keys, TOPK)
                ids = 0xFFFFFFFF - (best & 0xFFFFFFFF)
            scores = tl.load(row_logits + ids, (columns < width) & (ids < vocab_size), -float("inf"))
            tl.store(output + offset + columns, tl.where(tl.abs(scores) < float("inf"), ids, -1), columns < width)
            tl.store(output + offset + width + columns, scores - sampled + sampled_logprob, columns < width)


def capture_output_logprobs(
    logits: torch.Tensor,
    sampled_ids: torch.Tensor,
    sampled_logprobs: torch.Tensor,
    rows: torch.Tensor,
    candidate_rows: int,
    max_topk: int,
    output_size: int,
    check_rank: bool,
):
    """Apply the sampled-token cutoff and write each row's requested IDs and scores.

    Rows contain (logit row, requested K, output offset), with positive K first.
    Each such row occupies exactly 2*K FP32 entries in the output.
    """
    block = 1024
    tiles = triton.cdiv(logits.shape[1], block)
    topk = triton.next_power_of_2(max_topk) if max_topk else 0
    dense_top_ids = None
    dense_topk = 0
    counts = None
    candidates = None
    if candidate_rows == logits.shape[0] and topk > 8:
        # Large dense selections use torch.topk directly on the original rows.
        # This avoids allocating the tile candidates when every row needs them.
        dense_topk = min(20 if check_rank else max_topk, logits.shape[1])
        dense_top_ids = logits.topk(dense_topk, dim=-1).indices
    else:
        counts = rows.new_empty((rows.shape[0], tiles)) if check_rank else None
        candidates = logits.new_empty((candidate_rows, tiles, topk), dtype=torch.int64) if topk else None
        _output_logprobs_partial[(rows.shape[0], tiles)](
            logits,
            sampled_ids,
            rows,
            counts,
            candidates,
            candidate_rows,
            logits.stride(0),
            logits.shape[1],
            tiles,
            check_rank,
            topk,
            block,
            num_warps=4,
        )
    output = logits.new_empty((output_size,), dtype=torch.float32) if topk else None
    _output_logprobs_finish[(rows.shape[0],)](
        logits,
        sampled_ids,
        sampled_logprobs,
        rows,
        counts,
        candidates,
        output,
        dense_top_ids,
        candidate_rows,
        logits.stride(0),
        logits.shape[1],
        tiles,
        check_rank,
        topk,
        triton.next_power_of_2(tiles),
        dense_topk,
        num_warps=4,
    )
    return output
