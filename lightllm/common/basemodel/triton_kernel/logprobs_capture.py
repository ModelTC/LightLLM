from typing import Tuple

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
    rows,
    candidates,
    sampled_ids,
    ranks,
    partial_maxes,
    partial_sums,
    partial_posinf_counts,
    stride_row: tl.constexpr,
    vocab_size: tl.constexpr,
    tiles: tl.constexpr,
    TOPK: tl.constexpr,
    BLOCK: tl.constexpr,
):
    # 词表按 1024 一段切开。每个程序处理一行中的一段，只保留这段里分数最高的 TOPK 个键。
    index = tl.program_id(0)
    tile = tl.program_id(1)
    row = tl.load(rows + index * 3).to(tl.int64)
    row_logits = logits + row * stride_row
    ids = tile * BLOCK + tl.arange(0, BLOCK)
    scores = tl.load(row_logits + ids, ids < vocab_size, -float("inf")).to(tl.float32)
    sampled = tl.load(row_logits + tl.load(sampled_ids + row))
    rank = tl.sum(((ids < vocab_size) & (scores > sampled)).to(tl.int32), 0)
    tl.store(ranks + index * tiles + tile, rank)
    posinf_count = tl.sum(((ids < vocab_size) & (scores == float("inf"))).to(tl.int32), 0)
    finite_scores = tl.where((scores != float("inf")) & (scores != -float("inf")), scores, -float("inf"))
    tile_max = tl.max(finite_scores, 0)
    tile_sum = tl.sum(tl.exp(finite_scores - tile_max), 0)
    tile_sum = tl.where(tile_max > -float("inf"), tile_sum, 0.0)
    tl.store(partial_maxes + index * tiles + tile, tile_max)
    tl.store(partial_sums + index * tiles + tile, tile_sum)
    tl.store(partial_posinf_counts + index * tiles + tile, posinf_count)
    # tl.topk 只能比较整数。高 32 位放可排序的分数，低 32 位放反转后的 token id。
    # 取最大的键时，分数高的在前；分数相同则 id 更小的在前。词表外的位置写成最小整数。
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
    candidates,
    token_ids,
    logprobs,
    partial_ranks,
    sampled_ranks,
    partial_maxes,
    partial_sums,
    partial_posinf_counts,
    stride_row: tl.constexpr,
    vocab_size: tl.constexpr,
    tiles: tl.constexpr,
    TOPK: tl.constexpr,
    TILE_BLOCK: tl.constexpr,
):
    # 每个程序处理一行：合并各段的局部 top-k，按该行请求的 K 写出结果。
    index = tl.program_id(0)
    row = tl.load(rows + index * 3).to(tl.int64)
    row_logits = logits + row * stride_row
    sampled = tl.load(row_logits + tl.load(sampled_ids + row))
    width = tl.load(rows + index * 3 + 1)
    offset = tl.load(rows + index * 3 + 2)
    columns = tl.arange(0, TOPK)
    positions = tl.arange(0, TILE_BLOCK * TOPK)
    keys = tl.load(candidates + index * tiles * TOPK + positions, positions < tiles * TOPK, -0x8000000000000000)
    best = tl.topk(keys, TOPK)
    # 键的低 32 位是反转后的 token id，这里还原。结果仍按分数从高到低排列。
    ids = 0xFFFFFFFF - (best & 0xFFFFFFFF)
    scores = tl.load(row_logits + ids, (columns < width) & (ids < vocab_size), -float("inf"))
    stored_ids = tl.where(scores == scores, ids, -1).to(tl.int64)
    rank_tiles = tl.arange(0, TILE_BLOCK)
    tile_maxes = tl.load(partial_maxes + index * tiles + rank_tiles, rank_tiles < tiles, -float("inf"))
    tile_sums = tl.load(partial_sums + index * tiles + rank_tiles, rank_tiles < tiles, 0.0)
    posinf_counts = tl.load(partial_posinf_counts + index * tiles + rank_tiles, rank_tiles < tiles, 0)
    row_max = tl.max(tile_maxes, 0)
    row_sum = tl.sum(tl.where(tile_maxes > -float("inf"), tile_sums * tl.exp(tile_maxes - row_max), 0.0), 0)
    posinf_count = tl.sum(posinf_counts, 0)
    finite_logprobs = scores - (row_max + tl.log(row_sum))
    uniform_logprob = -tl.log(float(vocab_size))
    row_logprobs = tl.where(
        posinf_count > 0,
        tl.where(scores == float("inf"), -tl.log(posinf_count.to(tl.float32)), -float("inf")),
        tl.where(row_max > -float("inf"), finite_logprobs, uniform_logprob),
    )
    tl.store(token_ids + offset + columns, stored_ids, columns < width)
    tl.store(logprobs + offset + columns, row_logprobs, columns < width)
    rank = 1 + tl.sum(tl.load(partial_ranks + index * tiles + rank_tiles, rank_tiles < tiles, 0), 0)
    rank = tl.where(sampled > -float("inf"), rank, vocab_size + 1)
    tl.store(sampled_ranks + index, rank)


def capture_output_logprobs(
    logits: torch.Tensor,
    sampled_ids: torch.Tensor,
    sampled_logprobs: torch.Tensor,
    rows: torch.Tensor,
    max_topk: int,
    output_count: int,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """返回每行的 top-k token id 和 logprob。

    rows 每项是 (logits 行号, K, 写入起点)。写入起点按 token 个数累加，
    第 r 行的候选是两个数组的 [起点, 起点 + K)。另返回每行采样 token 的
    1-based rank（同分共享排名），不修改内部采样 logprob。K=0 仍计算排名。
    """
    block = 1024
    # 先按词表分段留下局部 top-k，再逐行合并成最终候选。
    tiles = triton.cdiv(logits.shape[1], block)
    # Triton 的 topk 块长必须是 2 的幂。请求的 K 最大 20，这里向上取整；真正写入仍按每行的 K 截断。
    topk = triton.next_power_of_2(max(2, max_topk))
    candidates = logits.new_empty((rows.shape[0], tiles, topk), dtype=torch.int64)
    partial_ranks = logits.new_empty((rows.shape[0], tiles), dtype=torch.int32)
    sampled_ranks = logits.new_empty((rows.shape[0],), dtype=torch.int32)
    partial_maxes = logits.new_empty((rows.shape[0], tiles), dtype=torch.float32)
    partial_sums = logits.new_empty((rows.shape[0], tiles), dtype=torch.float32)
    partial_posinf_counts = logits.new_empty((rows.shape[0], tiles), dtype=torch.int32)
    _output_logprobs_partial[(rows.shape[0], tiles)](
        logits=logits,
        rows=rows,
        candidates=candidates,
        sampled_ids=sampled_ids,
        ranks=partial_ranks,
        partial_maxes=partial_maxes,
        partial_sums=partial_sums,
        partial_posinf_counts=partial_posinf_counts,
        stride_row=logits.stride(0),
        vocab_size=logits.shape[1],
        tiles=tiles,
        TOPK=topk,
        BLOCK=block,
        num_warps=4,
    )
    token_ids = torch.empty((output_count,), dtype=torch.int64, device=logits.device)
    logprobs = torch.empty((output_count,), dtype=torch.float32, device=logits.device)
    _output_logprobs_finish[(rows.shape[0],)](
        logits=logits,
        sampled_ids=sampled_ids,
        sampled_logprobs=sampled_logprobs,
        rows=rows,
        candidates=candidates,
        token_ids=token_ids,
        logprobs=logprobs,
        partial_ranks=partial_ranks,
        sampled_ranks=sampled_ranks,
        partial_maxes=partial_maxes,
        partial_sums=partial_sums,
        partial_posinf_counts=partial_posinf_counts,
        stride_row=logits.stride(0),
        vocab_size=logits.shape[1],
        tiles=tiles,
        TOPK=topk,
        TILE_BLOCK=triton.next_power_of_2(tiles),
        num_warps=4,
    )
    return token_ids, logprobs, sampled_ranks
