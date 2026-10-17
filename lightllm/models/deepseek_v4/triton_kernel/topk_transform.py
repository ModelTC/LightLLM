# Copyright 2023-2024 SGLang Team
# SPDX-License-Identifier: Apache-2.0

import torch
import triton
import triton.language as tl

from lightllm.utils.vllm_utils import vllm_ops
from .launch import pdl_kwargs


@triton.jit
def _short_topk_rank_kernel(
    scores,
    seq_lens,
    page_tables,
    out_page_indices,
    out_raw_indices,
    score_stride,
    page_table_stride,
    SCORE_WIDTH: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_R: tl.constexpr,
    PAGE_SIZE: tl.constexpr,
    RAW_OUTPUT: tl.constexpr,
    USE_PDL: tl.constexpr,
):
    chunk = tl.program_id(0)
    row = tl.program_id(1)
    if USE_PDL:
        tl.extra.cuda.gdc_wait()
    length = tl.load(seq_lens + row)
    if length <= 512:
        if chunk == 0:
            col = tl.arange(0, 512)
            page = tl.load(page_tables + row * page_table_stride + col // PAGE_SIZE, mask=col < length, other=0)
            if USE_PDL:
                tl.extra.cuda.gdc_launch_dependents()
            tl.store(out_page_indices + row * 512 + col, tl.where(col < length, page * PAGE_SIZE + col % PAGE_SIZE, -1))
            if RAW_OUTPUT:
                tl.store(out_raw_indices + row * 512 + col, tl.where(col < length, col, -1))
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
        chosen = (target < length) & (rank < 512)
        page = tl.load(page_tables + row * page_table_stride + target // PAGE_SIZE, mask=chosen, other=0)
        if USE_PDL:
            tl.extra.cuda.gdc_launch_dependents()
        tl.store(out_page_indices + row * 512 + rank, page * PAGE_SIZE + target % PAGE_SIZE, mask=chosen)
        if RAW_OUTPUT:
            tl.store(out_raw_indices + row * 512 + rank, target, mask=chosen)


@triton.jit
def _page_transform_kernel(
    raw_indices,
    page_tables,
    out_page_indices,
    seq_lens,
    page_table_stride,
    PAGE_SIZE: tl.constexpr,
    DENSE: tl.constexpr,
    RAW_OUTPUT: tl.constexpr,
    USE_PDL: tl.constexpr,
):
    row = tl.program_id(0)
    col = tl.arange(0, 512)
    if USE_PDL:
        tl.extra.cuda.gdc_wait()
    if DENSE:
        idx = tl.where(col < tl.load(seq_lens + row), col, -1)
    else:
        idx = tl.load(raw_indices + row * 512 + col)
    page = tl.load(page_tables + row * page_table_stride + idx // PAGE_SIZE, mask=idx >= 0, other=0)
    if USE_PDL:
        tl.extra.cuda.gdc_launch_dependents()
    tl.store(out_page_indices + row * 512 + col, tl.where(idx >= 0, page * PAGE_SIZE + idx % PAGE_SIZE, -1))
    if DENSE and RAW_OUTPUT:
        tl.store(raw_indices + row * 512 + col, idx)


@torch.no_grad()
def topk_transform_512(scores, seq_lens, page_tables, out_page_indices, page_size, out_raw_indices=None):
    """Select C4 entries and map row-relative indices to physical cache slots.

    scores: FP32 [T, width] with contiguous columns; seq_lens: int32 [T].
    page_tables and outputs use int32, with 512 contiguous output entries per row.
    Dense rows use a Triton page transform. Tiny short-row batches fuse
    pairwise rank selection and page mapping; larger short batches use
    FlashInfer. For long rows vLLM selects each valid prefix and Triton
    translates its int32 output in place. Output ordering is unspecified and padding is -1.
    """
    if scores.shape[0] == 0:
        return
    raw = out_page_indices if out_raw_indices is None else out_raw_indices
    if scores.shape[1] <= 512:
        _page_transform_kernel[(scores.shape[0],)](
            raw,
            page_tables,
            out_page_indices,
            seq_lens,
            page_tables.stride(0),
            PAGE_SIZE=page_size,
            DENSE=True,
            RAW_OUTPUT=out_raw_indices is not None,
            num_warps=4,
            **pdl_kwargs(scores.device),
        )
    elif (scores.shape[1] <= 1024 and scores.shape[0] <= 4) or (scores.shape[1] <= 2048 and scores.shape[0] <= 2):
        # Bound the quadratic work; this launch is only for small decode batches.
        block_r = 4 if scores.shape[1] <= 1024 else 8
        _short_topk_rank_kernel[(triton.cdiv(scores.shape[1], block_r), scores.shape[0])](
            scores,
            seq_lens,
            page_tables,
            out_page_indices,
            raw,
            scores.stride(0),
            page_tables.stride(0),
            SCORE_WIDTH=scores.shape[1],
            BLOCK_N=triton.next_power_of_2(scores.shape[1]),
            BLOCK_R=block_r,
            PAGE_SIZE=page_size,
            RAW_OUTPUT=out_raw_indices is not None,
            num_warps=4,
            num_stages=1,
            **pdl_kwargs(scores.device),
        )
    elif scores.shape[1] > 2048:
        vllm_ops.top_k_per_row_decode(
            scores,
            1,
            seq_lens,
            raw,
            scores.shape[0],
            scores.stride(0),
            scores.stride(1),
            512,
        )
        _page_transform_kernel[(scores.shape[0],)](
            raw,
            page_tables,
            out_page_indices,
            seq_lens,
            page_tables.stride(0),
            PAGE_SIZE=page_size,
            DENSE=False,
            RAW_OUTPUT=False,
            num_warps=4,
            **pdl_kwargs(scores.device),
        )
    else:
        from flashinfer import top_k_page_table_transform

        top_k_page_table_transform(
            scores,
            page_tables,
            seq_lens,
            512,
            page_size=page_size,
            out=out_page_indices,
            out_raw_indices=out_raw_indices,
            dsa_graph_safe=True,
        )
