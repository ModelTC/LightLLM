"""Stage only selected latents, with workspace independent of context length."""

import torch
import triton
import triton.language as tl

from lightllm.common.basemodel.triton_kernel.kv_copy.nvfp4 import decode_e2m1
from .nvfp4_prefill import _compact_topk_indices_kernel


def _query_tile_size(rows):
    return min(rows, 32 if rows <= 128 else 512)


def prepare_selected_metadata(lengths, heads, width, cu_q):
    """Plan once per step, preserving the validated FA3 query grouping.

    Treat each native scheduler tensor as opaque: its layout differs between
    sgl-kernel versions. A flat tensor lets attention-state graph copy update
    all plans without retaining a nested collection of GPU tensors.
    """
    from sgl_kernel.flash_attn import get_scheduler_metadata

    rows = lengths.numel()
    if rows == 0:
        return lengths.new_empty((0,)), ()
    capacity = _query_tile_size(rows)
    plans, offsets, offset = [], [], 0
    for begin in range(0, rows, capacity):
        count = min(capacity, rows - begin)
        plan = get_scheduler_metadata(
            count,
            1,
            width,
            heads,
            1,
            64,
            lengths[begin : begin + count],
            qkv_dtype=torch.bfloat16,
            headdim_v=512,
            cu_seqlens_q=cu_q[: count + 1],
            page_size=1,
        )
        plans.append(plan)
        offsets.append((offset, plan.numel()))
        offset += plan.numel()
    return (torch.cat(plans) if len(plans) > 1 else plans[0]), tuple(offsets)


@triton.jit
def _gather_selected(
    Packed,
    Indices,
    Lengths,
    Current,
    RowTable,
    Output,
    KS: tl.constexpr,
    CS: tl.constexpr,
    WIDTH: tl.constexpr,
    HAS_CURRENT: tl.constexpr,
    BLOCK: tl.constexpr,
):
    query = tl.program_id(0)
    row = tl.program_id(1) * BLOCK + tl.arange(0, BLOCK)
    pair = tl.arange(0, 256)
    length = tl.load(Lengths + query)
    valid = row < length
    slot = tl.load(Indices + query * WIDTH + row, valid, 0).to(tl.int64)
    current = tl.load(RowTable + slot, valid, -1).to(tl.int64) if HAS_CURRENT else -1
    cached = valid & (current < 0)
    base = Packed + slot[:, None] * KS
    packed = tl.load(base + pair[None, :], cached[:, None], 0).to(tl.uint32)
    scale = tl.load(base + 256 + pair[None, :] // 8, cached[:, None], 0).to(tl.float8e4nv, bitcast=True)
    scale_bits = scale.to(tl.bfloat16).to(tl.uint16, bitcast=True).to(tl.uint32)
    low = decode_e2m1(packed, 0).to(tl.uint16, bitcast=True).to(tl.uint32)
    high = decode_e2m1(packed, 1).to(tl.uint16, bitcast=True).to(tl.uint32)
    # E2M1 x E4M3 has at most six significant bits and fits BF16 exactly.
    # Decode/store two elements per lane without expanding them to FP32.
    value = tl.inline_asm_elementwise(
        "mul.rn.bf16x2 $0, $1, $2;",
        constraints="=r,r,r",
        args=[low | (high << 16), scale_bits | (scale_bits << 16)],
        dtype=tl.uint32,
        is_pure=True,
        pack=1,
    )
    if HAS_CURRENT:
        raw_base = Current + current[:, None] * CS[0] + 2 * pair[None, :] * CS[2]
        raw_mask = valid[:, None] & (current >= 0)[:, None]
        raw_low = tl.load(raw_base, raw_mask, 0).to(tl.bfloat16).to(tl.uint16, bitcast=True).to(tl.uint32)
        raw_high = tl.load(raw_base + CS[2], raw_mask, 0).to(tl.bfloat16).to(tl.uint16, bitcast=True).to(tl.uint32)
        value = tl.where((current >= 0)[:, None], raw_low | (raw_high << 16), value)
    words = Output.to(tl.pointer_type(tl.uint32))
    tl.store(words + (query.to(tl.int64) * WIDTH + row[:, None]) * 256 + pair[None, :], value, valid[:, None])


def nvfp4_selected_attention(
    q,
    packed_kv,
    physical_indices,
    softmax_scale,
    current_kv=None,
    row_table=None,
    cache_lengths=None,
    page_table=None,
    cu_q=None,
    scheduler_metadata=None,
    scheduler_offsets=(),
):
    """Decode each selected KV once, then use FA3 for the original query heads.

    Batches of up to 128 rows stage at most 32 queries together (68 MiB at
    topk width 2176). Larger batches use up to 512 queries per tile (1088 MiB).
    Both work and workspace are bounded by selected tokens, never context size.
    """
    if physical_indices.ndim == 3:
        physical_indices = physical_indices[:, 0]
    rows, heads, dim = q.shape
    if dim != 512 or q.dtype != torch.bfloat16:
        from lightllm.common.basemodel.triton_kernel.mla_att.nvfp4 import nvfp4_mla_attention

        return nvfp4_mla_attention(
            q, packed_kv, physical_indices, softmax_scale, current_kv=current_kv, row_table=row_table
        )
    if rows == 0:
        return torch.empty_like(q)
    from sgl_kernel.flash_attn import flash_attn_with_kvcache

    assert physical_indices.is_contiguous()
    width = physical_indices.shape[1]
    capacity = _query_tile_size(rows)
    staged = torch.empty((capacity * width, 1, dim), dtype=q.dtype, device=q.device)
    if cache_lengths is None:
        compact = torch.empty((capacity, width), dtype=physical_indices.dtype, device=q.device)
        lengths = torch.empty((capacity,), dtype=torch.int32, device=q.device)
        starts = torch.zeros((capacity,), dtype=torch.int32, device=q.device)
        page_table = torch.arange(capacity * width, dtype=torch.int32, device=q.device).view(capacity, width)
        cu_q = torch.arange(capacity + 1, dtype=torch.int32, device=q.device)
    else:
        # GLM expand_topk guarantees a valid prefix followed by -1 padding.
        # Reuse its causal lengths rather than compacting that prefix again.
        assert cache_lengths.shape == (rows,) and cache_lengths.is_contiguous()
        assert page_table.shape[0] >= capacity and page_table.shape[1] == width
        assert cu_q.numel() >= capacity + 1
    output = torch.empty(q.shape, dtype=q.dtype, device=q.device)
    for tile, begin in enumerate(range(0, rows, capacity)):
        count = min(capacity, rows - begin)
        if cache_lengths is None:
            _compact_topk_indices_kernel[(count,)](
                physical_indices[begin : begin + count],
                starts,
                compact,
                lengths,
                0,
                packed_kv.shape[0],
                TOPK=width,
                BLOCK=triton.next_power_of_2(width),
            )
            selected, selected_lengths = compact, lengths[:count]
        else:
            selected = physical_indices[begin : begin + count]
            selected_lengths = cache_lengths[begin : begin + count]
        _gather_selected[(count, triton.cdiv(width, 4))](
            packed_kv,
            selected,
            selected_lengths,
            current_kv,
            row_table,
            staged,
            packed_kv.stride(0),
            current_kv.stride() if current_kv is not None else (0, 0, 0),
            width,
            current_kv is not None,
            4,
            num_warps=4,
        )
        flash_attn_with_kvcache(
            q=None,
            qv=q[begin : begin + count],
            k_cache=None,
            v_cache=staged.view(-1, 1, 1, dim),
            page_table=page_table[:count],
            cache_seqlens=selected_lengths,
            cu_seqlens_q=cu_q[: count + 1],
            max_seqlen_q=1,
            softmax_scale=softmax_scale,
            causal=False,
            only_qv=True,
            scheduler_metadata=(
                scheduler_metadata.narrow(0, *scheduler_offsets[tile]) if scheduler_metadata is not None else None
            ),
            out=output[begin : begin + count],
        )
    return output
