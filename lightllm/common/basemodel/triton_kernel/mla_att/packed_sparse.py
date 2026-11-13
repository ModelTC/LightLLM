"""Bounded selected-KV staging for packed NVFP4 shared-KV attention.

The two independently paged sources may have different page sizes/strides.
Sparse indices and lengths describe visibility; an optional attention sink
adds a zero-valued entry to the same softmax denominator.
"""

import torch
import triton
import triton.language as tl

from ..kv_copy.nvfp4 import decode_e2m1

STAGED_KV_BYTES = 64 * 1024 * 1024
QUERY_TILE_SIZE = 32


@triton.jit(do_not_specialize=["begin", "index_row_stride", "extra_row_stride", "width", "extra_width"])
def _compact_two_sources(
    Indices,
    ExtraIndices,
    Lengths,
    ExtraLengths,
    Selected,
    Counts,
    begin,
    index_row_stride,
    extra_row_stride,
    INDEX_COL_STRIDE: tl.constexpr,
    EXTRA_COL_STRIDE: tl.constexpr,
    width,
    extra_width,
    HAS_EXTRA: tl.constexpr,
    FIRST_TOKENS: tl.constexpr,
    EXTRA_TOKENS: tl.constexpr,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    lane = tl.arange(0, BLOCK)
    first = tl.load(Indices + (row + begin).to(tl.int64) * index_row_stride + lane * INDEX_COL_STRIDE, lane < width, -1)
    length = tl.load(Lengths + row + begin)
    valid = (lane < width) & (lane < length) & (first >= 0) & (first < FIRST_TOKENS)
    index = first
    if HAS_EXTRA:
        extra_lane = lane - width
        extra = tl.load(
            ExtraIndices + (row + begin).to(tl.int64) * extra_row_stride + extra_lane * EXTRA_COL_STRIDE,
            (extra_lane >= 0) & (extra_lane < extra_width),
            -1,
        )
        extra_length = tl.load(ExtraLengths + row + begin)
        extra_valid = (
            (extra_lane >= 0)
            & (extra_lane < extra_width)
            & (extra_lane < extra_length)
            & (extra >= 0)
            & (extra < EXTRA_TOKENS)
        )
        valid |= extra_valid
        index = tl.where(lane < width, first, extra + FIRST_TOKENS)
    compact = tl.cumsum(valid.to(tl.int32)) - 1
    tl.store(Selected + row * (width + extra_width) + compact, index, valid)
    tl.store(Counts + row, tl.sum(valid.to(tl.int32)))


@triton.jit(do_not_specialize=["width"])
def _gather_packed_nvfp4(
    First,
    Extra,
    Selected,
    Counts,
    Out,
    width,
    FIRST_TOKENS: tl.constexpr,
    FIRST_PAGE: tl.constexpr,
    EXTRA_PAGE: tl.constexpr,
    FIRST_STRIDE: tl.constexpr,
    EXTRA_STRIDE: tl.constexpr,
    DIM: tl.constexpr,
    ROPE: tl.constexpr,
    HAS_EXTRA: tl.constexpr,
    BLOCK: tl.constexpr,
):
    query = tl.program_id(0).to(tl.int64)
    token = tl.program_id(1) * BLOCK + tl.arange(0, BLOCK)
    dim = tl.arange(0, triton.next_power_of_2(DIM))
    NOPE: tl.constexpr = DIM - ROPE
    DATA: tl.constexpr = NOPE // 2 + ROPE * 2
    SCALES: tl.constexpr = triton.cdiv(NOPE // 16, 8) * 8
    length = tl.load(Counts + query)
    valid = token < length
    index = tl.load(Selected + query * width + token, valid, 0).to(tl.int64)
    data = First + index // FIRST_PAGE * FIRST_STRIDE + index % FIRST_PAGE * DATA
    scales = First + index // FIRST_PAGE * FIRST_STRIDE + FIRST_PAGE * DATA + index % FIRST_PAGE * SCALES
    if HAS_EXTRA:
        extra = index - FIRST_TOKENS
        extra_data = Extra + extra // EXTRA_PAGE * EXTRA_STRIDE + extra % EXTRA_PAGE * DATA
        extra_scales = Extra + extra // EXTRA_PAGE * EXTRA_STRIDE + EXTRA_PAGE * DATA + extra % EXTRA_PAGE * SCALES
        data = tl.where(index >= FIRST_TOKENS, extra_data, data)
        scales = tl.where(index >= FIRST_TOKENS, extra_scales, scales)
    packed = tl.load(data[:, None] + dim[None, :] // 2, valid[:, None] & (dim[None, :] < NOPE), 0)
    scale = tl.load(scales[:, None] + dim[None, :] // 16, valid[:, None] & (dim[None, :] < NOPE), 0)
    values = decode_e2m1(packed, dim[None, :]).to(tl.float32) * scale.to(tl.float8e4nv, bitcast=True).to(tl.float32)
    tail = (data + NOPE // 2).to(tl.pointer_type(tl.bfloat16))
    rope = tl.load(
        tail[:, None] + dim[None, :] - NOPE, valid[:, None] & (dim[None, :] >= NOPE) & (dim[None, :] < DIM), 0
    )
    values = tl.where(dim[None, :] < NOPE, values, rope)
    tl.store(
        Out + (query * width + token[:, None]) * DIM + dim[None, :],
        values,
        (token[:, None] < width) & (dim[None, :] < DIM),
    )


@triton.jit(do_not_specialize=["begin", "lse_head_stride", "lse_row_stride"])
def _apply_sink(
    Native,
    Lse,
    Sink,
    Counts,
    Out,
    begin,
    NS: tl.constexpr,
    lse_head_stride,
    lse_row_stride,
    OS: tl.constexpr,
    HEADS: tl.constexpr,
    DIM: tl.constexpr,
    HAS_SINK: tl.constexpr,
    BLOCK: tl.constexpr,
):
    row, head = tl.program_id(0), tl.program_id(1)
    d = tl.arange(0, BLOCK)
    value = tl.load(Native + row * NS[0] + head * NS[1] + d * NS[2], d < DIM, 0).to(tl.float32)
    if HAS_SINK:
        lse = tl.load(Lse + head * lse_head_stride + row * lse_row_stride)
        sink = tl.load(Sink + head).to(tl.float32)
        value *= tl.sigmoid(lse - sink)
    nonempty = tl.load(Counts + row) > 0
    tl.store(
        Out + (row + begin).to(tl.int64) * OS[0] + head * OS[1] + d * OS[2], tl.where(nonempty, value, 0.0), d < DIM
    )


def packed_sparse_attention(
    q,
    first,
    extra,
    indices,
    lengths,
    extra_indices,
    extra_lengths,
    first_layout,
    extra_layout,
    first_page,
    extra_page,
    scale,
    sink,
    workspace,
    out,
):
    """Attend to the selected entries without a persistent dequantized cache."""
    from sgl_kernel.flash_attn import flash_attn_with_kvcache

    rows, heads, dim = q.shape
    assert q.dtype == torch.bfloat16 and first_layout.format == "nvfp4_block16"
    assert first_layout.kv_dim == dim and out.shape == q.shape
    assert first.dtype == torch.uint8 and first.ndim == 2 and first.stride(1) == 1
    indices = indices[:, 0] if indices.ndim == 3 else indices
    width = indices.shape[1]
    extra_width = 0
    if extra is not None:
        assert extra_layout == first_layout
        assert extra.dtype == torch.uint8 and extra.ndim == 2 and extra.stride(1) == 1
        extra_indices = extra_indices[:, 0] if extra_indices.ndim == 3 else extra_indices
        extra_width = extra_indices.shape[1]
    total_width = width + extra_width
    capacity = min(QUERY_TILE_SIZE, workspace["kv"].shape[0] // total_width)
    assert capacity > 0, "Reserved selected-KV workspace must fit at least one query"
    staged = workspace["kv"][: capacity * total_width]
    selected = workspace["selected"][: capacity * total_width]
    page_table = workspace["pages"][: capacity * total_width].view(capacity, total_width)
    for begin in range(0, rows, capacity):
        count = min(capacity, rows - begin)
        _compact_two_sources[(count,)](
            indices,
            extra_indices,
            lengths,
            extra_lengths,
            selected,
            workspace["lengths"],
            begin,
            indices.stride(0),
            extra_indices.stride(0) if extra is not None else 0,
            indices.stride(1),
            extra_indices.stride(1) if extra is not None else 0,
            width,
            extra_width,
            extra is not None,
            first.shape[0] * first_page,
            extra.shape[0] * extra_page if extra is not None else 0,
            triton.next_power_of_2(total_width),
        )
        _gather_packed_nvfp4[(count, triton.cdiv(total_width, 4))](
            first,
            extra,
            selected,
            workspace["lengths"],
            staged,
            total_width,
            first.shape[0] * first_page,
            first_page,
            extra_page,
            first.stride(0),
            extra.stride(0) if extra is not None else 0,
            dim,
            first_layout.rope_dim,
            extra is not None,
            4,
            num_warps=4,
        )
        result, lse, *_ = flash_attn_with_kvcache(
            q=None,
            qv=q[begin : begin + count],
            k_cache=None,
            v_cache=staged.view(-1, 1, 1, dim),
            page_table=page_table[:count],
            cache_seqlens=workspace["lengths"][:count],
            cu_seqlens_q=workspace["cu_q"][: count + 1],
            max_seqlen_q=1,
            softmax_scale=scale,
            causal=False,
            only_qv=True,
            num_splits=1,
            return_softmax_lse=True,
        )
        _apply_sink[(count, heads)](
            result,
            lse,
            sink,
            workspace["lengths"],
            out,
            begin,
            result.stride(),
            lse.stride(0),
            lse.stride(1),
            out.stride(),
            heads,
            dim,
            sink is not None,
            triton.next_power_of_2(dim),
            num_warps=4,
        )
    return out
