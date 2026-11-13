"""Bounded NVFP4 staging followed by the native BF16 sparse MLA kernels."""

import torch
import triton
import triton.language as tl

from lightllm.common.basemodel.triton_kernel.kv_copy.nvfp4 import decode_e2m1
from .fp8_prefill import _merge_att_output_kernel
from .kpool import _get_query_block

MAX_STAGED_TOKENS = 2 * 1024 * 1024


def prefill_uses_selected(rows, tokens, width):
    return rows <= 8 or tokens > min(MAX_STAGED_TOKENS, rows * width)


@triton.jit
def _prefill_slots(Table, Requests, Bounds, Slots, STRIDE: tl.constexpr, batch_size, BLOCK: tl.constexpr):
    batch, start, end, block = _get_query_block(Bounds, batch_size, BLOCK)
    row = start + block * BLOCK + tl.arange(0, BLOCK)
    req = tl.load(Requests + batch).to(tl.int64)
    slot = tl.load(Table + req * STRIDE + row - start, row < end, 0)
    tl.store(Slots + row, slot, row < end)


def prefill_slots(req_to_token_indexs, b_req_idx, b1_cu_kv_seq_len, total_tokens):
    """Materialize physical slots only for bounded historical-KV staging."""
    slots = torch.empty(total_tokens, dtype=torch.int32, device=b_req_idx.device)
    _prefill_slots[(total_tokens // 256 + b_req_idx.numel(),)](
        req_to_token_indexs,
        b_req_idx,
        b1_cu_kv_seq_len,
        slots,
        req_to_token_indexs.stride(0),
        b_req_idx.numel(),
        256,
    )
    return slots


@triton.jit(do_not_specialize=["chunk_end"])
def _compact_topk_indices_kernel(
    TopkIndices,
    KVStarts,
    ChunkIndices,
    ChunkLengths,
    chunk_start,
    chunk_end,
    TOPK: tl.constexpr,
    BLOCK: tl.constexpr,
    COMPACT: tl.constexpr = True,
):
    row = tl.program_id(0)
    lane = tl.arange(0, BLOCK)
    index = tl.load(TopkIndices + row * TOPK + lane, lane < TOPK, -1)
    ragged_index = index + tl.load(KVStarts + row)
    valid = (lane < TOPK) & (index >= 0) & (ragged_index >= chunk_start) & (ragged_index < chunk_end)
    if COMPACT:
        compact_index = tl.cumsum(valid.to(tl.int32)) - 1
        tl.store(ChunkIndices + row * TOPK + compact_index, ragged_index - chunk_start, valid)
        tl.store(ChunkLengths + row, tl.sum(valid.to(tl.int32)))
    else:
        # FlashMLA requires -1 padding and the original sparse order.
        tl.store(ChunkIndices + row * TOPK + lane, tl.where(valid, ragged_index - chunk_start, -1), lane < TOPK)


def nvfp4_cached_prefill(
    q, packed_kv, ragged, starts, topk_indices, physical_indices, scale, current_kv, row_table, workspace=None
):
    """Select the bounded cached-prefill implementation from its work size."""
    if q.dtype == torch.bfloat16 and q.shape[0] and q.shape[-1] == 512:
        # Bound history decoding by both workspace and the selected work.
        if ragged is None or prefill_uses_selected(q.shape[0], ragged.numel(), topk_indices.shape[1]):
            from .nvfp4_selected import nvfp4_selected_attention

            return nvfp4_selected_attention(
                q,
                packed_kv,
                physical_indices,
                scale,
                current_kv,
                row_table,
                staging_buffer=workspace["gathered_kv"] if workspace is not None else None,
            )
        return nvfp4_sparse_prefill(
            q, packed_kv, ragged, starts, topk_indices, scale, current_kv, row_table, workspace=workspace
        )
    from lightllm.common.basemodel.triton_kernel.mla_att.nvfp4 import nvfp4_mla_attention

    return nvfp4_mla_attention(q, packed_kv, physical_indices, scale, current_kv=current_kv, row_table=row_table)


def current_chunk_indices(relative, starts, tokens):
    indices = torch.empty_like(relative)
    _compact_topk_indices_kernel[(relative.shape[0],)](
        relative,
        starts,
        indices,
        None,
        0,
        tokens,
        TOPK=relative.shape[1],
        BLOCK=triton.next_power_of_2(relative.shape[1]),
        COMPACT=False,
    )
    return indices


@triton.jit(do_not_specialize=["TOKENS"])
def _gather(
    Packed,
    Slots,
    RowTable,
    Current,
    Out,
    TOKENS,
    KS: tl.constexpr,
    SS: tl.constexpr,
    CS: tl.constexpr,
    HAS_CURRENT: tl.constexpr,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0).to(tl.int64) * BLOCK + tl.arange(0, BLOCK)
    dim = tl.arange(0, 512)
    slot = tl.load(Slots + row * SS, row < TOKENS, 0).to(tl.int64)
    current = tl.load(RowTable + slot, row < TOKENS, -1).to(tl.int64) if HAS_CURRENT else -1
    cached = (row < TOKENS) & (current < 0)
    base = Packed + slot[:, None] * KS
    packed = tl.load(base + dim[None, :] // 2, cached[:, None], 0)
    scales = tl.load(base + 256 + dim[None, :] // 16, cached[:, None], 0).to(tl.float8e4nv, bitcast=True)
    value = decode_e2m1(packed, dim[None, :]).to(tl.float32) * scales.to(tl.float32)
    if HAS_CURRENT:
        raw = tl.load(
            Current + current[:, None] * CS[0] + dim[None, :] * CS[2],
            (row < TOKENS)[:, None] & (current >= 0)[:, None],
            0,
        )
        value = tl.where((current >= 0)[:, None], raw, value)
    tl.store(Out + row[:, None] * 512 + dim[None, :], value, (row < TOKENS)[:, None])


def nvfp4_sparse_prefill(
    q,
    packed_kv,
    ragged,
    starts,
    topk_indices,
    softmax_scale,
    current_kv,
    row_table,
    workspace_bytes=MAX_STAGED_TOKENS * 1024,
    workspace=None,
):
    """Stage each visible latent once per chunk, preserving current BF16 values.

    Workspace is bounded independently of the persistent KV capacity. Large
    ragged batches use an FP32 online merge over workspace-sized KV chunks.
    """
    from sgl_kernel.flash_mla import flash_mla_sparse_fwd
    from sgl_kernel.flash_attn import flash_attn_with_kvcache

    rows, heads, dim = q.shape
    assert dim == 512 and q.dtype == torch.bfloat16
    if not rows or not ragged.numel():
        return torch.zeros_like(q)
    assert topk_indices.shape[0] == rows and topk_indices.is_contiguous()
    capacity = max(4, workspace_bytes // (dim * q.element_size()) // 4 * 4)
    if workspace is not None:
        capacity = min(capacity, workspace["gathered_kv"].shape[0])
    chunk_size = min(capacity, ragged.numel())
    if workspace is None:
        staged = torch.empty((chunk_size, 1, dim), dtype=q.dtype, device=q.device)
        indices = torch.empty_like(topk_indices)
        lengths = torch.empty((rows,), dtype=torch.int32, device=q.device)
        cu_q = torch.arange(rows + 1, dtype=torch.int32, device=q.device)
    else:
        staged = workspace["gathered_kv"][:chunk_size]
        indices = workspace["chunk_topk_indices"]
        lengths = workspace["chunk_cache_seqlens"]
        cu_q = workspace["cu_seqlens_q"]
    padded_q = q
    if ragged.numel() <= chunk_size:
        padded_heads = triton.cdiv(heads, 64) * 64
        if heads != padded_heads:
            padded_q = q.new_zeros((rows, padded_heads, dim))
            padded_q[:, :heads] = q
    output = lse = None
    for begin in range(0, ragged.numel(), chunk_size):
        count = min(chunk_size, ragged.numel() - begin)
        _gather[(triton.cdiv(count, 4),)](
            packed_kv,
            ragged[begin : begin + count],
            row_table,
            current_kv,
            staged,
            count,
            packed_kv.stride(0),
            ragged.stride(0),
            current_kv.stride(),
            True,
            4,
            num_warps=4,
        )
        single_chunk = ragged.numel() <= chunk_size
        _compact_topk_indices_kernel[(rows,)](
            topk_indices,
            starts,
            indices,
            lengths,
            begin,
            begin + count,
            TOPK=topk_indices.shape[1],
            BLOCK=triton.next_power_of_2(topk_indices.shape[1]),
            COMPACT=not single_chunk,
        )
        if single_chunk:
            return flash_mla_sparse_fwd(
                q=padded_q,
                kv=staged,
                indices=indices.unsqueeze(1),
                sm_scale=softmax_scale,
                d_v=512,
            )[0][:, :heads]
        chunk_out, chunk_lse, *_ = flash_attn_with_kvcache(
            q=None,
            qv=q,
            k_cache=None,
            v_cache=staged[:count].view(-1, 1, 1, dim),
            page_table=indices,
            cache_seqlens=lengths,
            cu_seqlens_q=cu_q,
            max_seqlen_q=1,
            softmax_scale=softmax_scale,
            causal=False,
            only_qv=True,
            num_splits=1,
            return_softmax_lse=True,
        )
        if begin == 0:
            output, lse = chunk_out.float(), chunk_lse
        else:
            _merge_att_output_kernel[(rows, triton.cdiv(heads, 4))](
                output,
                lse,
                chunk_out,
                chunk_lse,
                chunk_lse.stride(1),
                chunk_lse.stride(0),
                q_head_num=heads,
                q_token_num=rows,
                BLOCK_HEADS=4,
            )
    return output.to(q.dtype)
