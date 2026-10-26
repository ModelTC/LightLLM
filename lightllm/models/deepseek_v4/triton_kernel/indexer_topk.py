# Copyright 2023-2024 SGLang Team
# SPDX-License-Identifier: Apache-2.0

import torch
import triton
import triton.language as tl

from lightllm.utils.vllm_utils import vllm_ops
from lightllm.utils.custom_kernel_utis import get_pdl_launch_kwargs


@triton.jit
def _indexer_topk_rank_kernel(
    scores,
    valid_lengths,
    page_table,
    compressed_kv_indices,
    out_logical_indices,
    score_stride,
    page_table_stride,
    SCORE_WIDTH: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_R: tl.constexpr,
    PAGE_SIZE: tl.constexpr,
    TOPK: tl.constexpr,
    TOPK_BLOCK: tl.constexpr,
    RAW_OUTPUT: tl.constexpr,
    USE_PDL: tl.constexpr,
):
    chunk = tl.program_id(0)
    row = tl.program_id(1)
    if USE_PDL:
        tl.extra.cuda.gdc_wait()
    length = tl.load(valid_lengths + row)
    if length <= TOPK:
        if chunk == 0:
            col = tl.arange(0, TOPK_BLOCK)
            page = tl.load(page_table + row * page_table_stride + col // PAGE_SIZE, mask=col < length, other=0)
            if USE_PDL:
                tl.extra.cuda.gdc_launch_dependents()
            tl.store(
                compressed_kv_indices + row * TOPK + col,
                tl.where(col < length, page * PAGE_SIZE + col % PAGE_SIZE, -1),
                col < TOPK,
            )
            if RAW_OUTPUT:
                tl.store(out_logical_indices + row * TOPK + col, tl.where(col < length, col, -1), col < TOPK)
        elif USE_PDL:
            tl.extra.cuda.gdc_launch_dependents()
    else:
        target = chunk * BLOCK_R + tl.arange(0, BLOCK_R)
        col = tl.arange(0, BLOCK_N)
        x = tl.load(scores + row * score_stride + col, mask=(col < length) & (col < SCORE_WIDTH), other=0)
        value = tl.load(scores + row * score_stride + target, mask=target < length, other=0)
        # A tiny query batch can spread exact pairwise ranks across the GPU.
        # Equal scores prefer the smaller logical index, giving unique ranks.
        greater = (col[None, :] < length) & (
            (x[None, :] > value[:, None]) | ((x[None, :] == value[:, None]) & (col[None, :] < target[:, None]))
        )
        rank = tl.sum(greater.to(tl.int32), axis=1)
        chosen = (target < length) & (rank < TOPK)
        page = tl.load(page_table + row * page_table_stride + target // PAGE_SIZE, mask=chosen, other=0)
        if USE_PDL:
            tl.extra.cuda.gdc_launch_dependents()
        tl.store(compressed_kv_indices + row * TOPK + rank, page * PAGE_SIZE + target % PAGE_SIZE, mask=chosen)
        if RAW_OUTPUT:
            tl.store(out_logical_indices + row * TOPK + rank, target, mask=chosen)


@triton.jit
def _indexer_topk_to_kv_indices_kernel(
    logical_indices,
    page_table,
    compressed_kv_indices,
    valid_lengths,
    page_table_stride,
    PAGE_SIZE: tl.constexpr,
    DENSE: tl.constexpr,
    TOPK: tl.constexpr,
    TOPK_BLOCK: tl.constexpr,
    RAW_OUTPUT: tl.constexpr,
    USE_PDL: tl.constexpr,
):
    row = tl.program_id(0)
    col = tl.arange(0, TOPK_BLOCK)
    if USE_PDL:
        tl.extra.cuda.gdc_wait()
    if DENSE:
        idx = tl.where(col < tl.load(valid_lengths + row), col, -1)
    else:
        idx = tl.load(logical_indices + row * TOPK + col, col < TOPK, -1)
    page = tl.load(page_table + row * page_table_stride + idx // PAGE_SIZE, mask=idx >= 0, other=0)
    if USE_PDL:
        tl.extra.cuda.gdc_launch_dependents()
    tl.store(
        compressed_kv_indices + row * TOPK + col, tl.where(idx >= 0, page * PAGE_SIZE + idx % PAGE_SIZE, -1), col < TOPK
    )
    if DENSE and RAW_OUTPUT:
        tl.store(logical_indices + row * TOPK + col, idx, col < TOPK)


@torch.no_grad()
def indexer_topk(scores, valid_lengths, page_table, compressed_kv_indices, page_size, out_logical_indices=None):
    """Select C4 entries and map row-relative indices to physical cache slots.

    scores: FP32 [T, width] with contiguous columns; valid_lengths: int32 [T].
    page_table and outputs use int32, with contiguous output entries per query.
    Dense rows use a Triton page transform. Tiny short-row batches fuse
    pairwise rank selection and page mapping; larger short batches use
    FlashInfer. For long rows vLLM selects each valid prefix and Triton
    translates its int32 output in place. Output ordering is unspecified and padding is -1.
    """
    if scores.shape[0] == 0:
        return
    topk = compressed_kv_indices.shape[1]
    assert topk > 0 and compressed_kv_indices.stride(0) == topk
    logical_indices = compressed_kv_indices if out_logical_indices is None else out_logical_indices
    assert logical_indices.shape == compressed_kv_indices.shape and logical_indices.stride(0) == topk
    if scores.shape[1] <= topk:
        _indexer_topk_to_kv_indices_kernel[(scores.shape[0],)](
            logical_indices,
            page_table,
            compressed_kv_indices,
            valid_lengths,
            page_table.stride(0),
            PAGE_SIZE=page_size,
            DENSE=True,
            TOPK=topk,
            TOPK_BLOCK=triton.next_power_of_2(topk),
            RAW_OUTPUT=out_logical_indices is not None,
            num_warps=4,
            **get_pdl_launch_kwargs(scores.device),
        )
    elif (scores.shape[1] <= 1024 and scores.shape[0] <= 4) or (scores.shape[1] <= 2048 and scores.shape[0] <= 2):
        # Bound the quadratic work; this launch is only for small decode batches.
        block_r = 4 if scores.shape[1] <= 1024 else 8
        _indexer_topk_rank_kernel[(triton.cdiv(scores.shape[1], block_r), scores.shape[0])](
            scores,
            valid_lengths,
            page_table,
            compressed_kv_indices,
            logical_indices,
            scores.stride(0),
            page_table.stride(0),
            SCORE_WIDTH=scores.shape[1],
            BLOCK_N=triton.next_power_of_2(scores.shape[1]),
            BLOCK_R=block_r,
            PAGE_SIZE=page_size,
            TOPK=topk,
            TOPK_BLOCK=triton.next_power_of_2(topk),
            RAW_OUTPUT=out_logical_indices is not None,
            num_warps=4,
            num_stages=1,
            **get_pdl_launch_kwargs(scores.device),
        )
    elif scores.shape[1] > 2048 or topk != 512:
        vllm_ops.top_k_per_row_decode(
            scores,
            1,
            valid_lengths,
            logical_indices,
            scores.shape[0],
            scores.stride(0),
            scores.stride(1),
            topk,
        )
        _indexer_topk_to_kv_indices_kernel[(scores.shape[0],)](
            logical_indices,
            page_table,
            compressed_kv_indices,
            valid_lengths,
            page_table.stride(0),
            PAGE_SIZE=page_size,
            DENSE=False,
            TOPK=topk,
            TOPK_BLOCK=triton.next_power_of_2(topk),
            RAW_OUTPUT=False,
            num_warps=4,
            **get_pdl_launch_kwargs(scores.device),
        )
    else:
        # FlashInfer's DSA-specialized path requires a 512-entry selection.
        from flashinfer import top_k_page_table_transform

        top_k_page_table_transform(
            scores,
            page_table,
            valid_lengths,
            topk,
            page_size=page_size,
            out=compressed_kv_indices,
            out_raw_indices=out_logical_indices,
            dsa_graph_safe=True,
        )
