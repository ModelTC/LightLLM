import torch
import triton
import triton.language as tl


KV_GATHER_CHUNK_SIZE = 65536


@triton.jit
def _build_prefill_row_table_kernel(PrefillMemIndex, RowTable, num_tokens, BLOCK: tl.constexpr):
    row = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mem_index = tl.load(PrefillMemIndex + row, row < num_tokens, 0)
    tl.store(RowTable + mem_index, row, row < num_tokens)


@triton.jit
def _gather_prefill_kv_kernel(
    PackedKV,
    MemIndex,
    PrefillRowTable,
    PrefillKV,
    Output,
    num_tokens,
    packed_stride_s: tl.constexpr,
    prefill_stride_s: tl.constexpr,
    prefill_stride_d: tl.constexpr,
    BLOCK_ROWS: tl.constexpr,
):
    row = tl.program_id(0) * BLOCK_ROWS + tl.arange(0, BLOCK_ROWS)
    dim = tl.arange(0, 512)
    mem_index = tl.load(MemIndex + row, row < num_tokens, 0).to(tl.int64)
    prefill_row = tl.load(PrefillRowTable + mem_index, row < num_tokens, -1)
    cached = (row < num_tokens) & (prefill_row < 0)
    current = (row < num_tokens) & (prefill_row >= 0)
    packed_row = PackedKV + mem_index[:, None] * packed_stride_s

    quantized = tl.load(
        packed_row.to(tl.pointer_type(tl.float8e4nv)) + dim[None, :],
        cached[:, None] & (dim[None, :] < 448),
        0.0,
    ).to(tl.float32)
    scales = tl.load(packed_row + 576 + dim[None, :] // 64, cached[:, None] & (dim[None, :] < 448), 127)
    values = quantized * (scales.to(tl.uint32) << 23).to(tl.float32, bitcast=True)
    tail = tl.load(
        (packed_row + 448).to(tl.pointer_type(tl.bfloat16)) + dim[None, :] - 448,
        cached[:, None] & (dim[None, :] >= 448),
        0,
    ).to(tl.float32)
    values = tl.where(dim[None, :] < 448, values, tail)
    current_values = tl.load(
        PrefillKV + prefill_row[:, None] * prefill_stride_s + dim[None, :] * prefill_stride_d,
        current[:, None],
        0,
    ).to(tl.float32)
    tl.store(
        Output + row[:, None] * 512 + dim[None, :],
        tl.where(current[:, None], current_values, values),
        row[:, None] < num_tokens,
    )


@triton.jit
def _compact_topk_indices_kernel(
    TopkIndices,
    KVStarts,
    ChunkIndices,
    ChunkLengths,
    chunk_start,
    chunk_end,
    TOPK: tl.constexpr,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    lane = tl.arange(0, BLOCK)
    index = tl.load(TopkIndices + row * TOPK + lane, lane < TOPK, -1)
    ragged_index = index + tl.load(KVStarts + row)
    valid = (lane < TOPK) & (index >= 0) & (ragged_index >= chunk_start) & (ragged_index < chunk_end)
    compact_index = tl.cumsum(valid.to(tl.int32)) - 1
    tl.store(ChunkIndices + row * TOPK + compact_index, ragged_index - chunk_start, valid)
    tl.store(ChunkLengths + row, tl.sum(valid.to(tl.int32)))


@triton.jit
def _merge_att_output_kernel(
    Output,
    Lse,
    ChunkOutput,
    ChunkLse,
    chunk_lse_stride_s: tl.constexpr,
    chunk_lse_stride_h: tl.constexpr,
    NUM_HEADS: tl.constexpr,
    NUM_Q_TOKENS: tl.constexpr,
    BLOCK_HEADS: tl.constexpr,
):
    row = tl.program_id(0)
    head = tl.program_id(1) * BLOCK_HEADS + tl.arange(0, BLOCK_HEADS)
    dim = tl.arange(0, 512)
    previous_lse = tl.load(Lse + head * NUM_Q_TOKENS + row, head < NUM_HEADS, -float("inf"))
    chunk_lse = tl.load(
        ChunkLse + row * chunk_lse_stride_s + head * chunk_lse_stride_h, head < NUM_HEADS, -float("inf")
    )
    # FA3 uses natural-log LSE and returns +inf for an empty row.
    previous_lse = tl.where(previous_lse == float("inf"), -float("inf"), previous_lse)
    chunk_lse = tl.where(chunk_lse == float("inf"), -float("inf"), chunk_lse)
    maximum = tl.maximum(previous_lse, chunk_lse)
    maximum = tl.where(maximum == -float("inf"), 0, maximum)
    previous_weight = tl.exp(previous_lse - maximum)
    chunk_weight = tl.exp(chunk_lse - maximum)
    denominator = previous_weight + chunk_weight
    offset = (row * NUM_HEADS + head[:, None]) * 512 + dim[None, :]
    previous = tl.load(Output + offset, (head[:, None] < NUM_HEADS) & (previous_weight[:, None] > 0), 0)
    current = tl.load(ChunkOutput + offset, (head[:, None] < NUM_HEADS) & (chunk_weight[:, None] > 0), 0).to(tl.float32)
    result = (previous * previous_weight[:, None] + current * chunk_weight[:, None]) / tl.where(
        denominator > 0, denominator, 1
    )[:, None]
    tl.store(Output + offset, result, head[:, None] < NUM_HEADS)
    tl.store(
        Lse + head * NUM_Q_TOKENS + row,
        tl.where(denominator > 0, maximum + tl.log(denominator), -float("inf")),
        head < NUM_HEADS,
    )


@torch.no_grad()
def context_attention_fwd_fp8(
    q: torch.Tensor,
    packed_kv: torch.Tensor,
    ragged_mem_index: torch.Tensor,
    prefill_mem_index: torch.Tensor,
    prefill_kv: torch.Tensor,
    kv_starts: torch.Tensor,
    topk_indices: torch.Tensor,
    softmax_scale: float,
):
    """Run NoPE sparse prefill with a bounded gather buffer and BF16 current KV."""
    from sgl_kernel.flash_attn import flash_attn_with_kvcache

    num_q_tokens, num_heads, _ = q.shape
    prefill_row_table = torch.full((packed_kv.shape[0],), -1, dtype=torch.int32, device=q.device)
    _build_prefill_row_table_kernel[(triton.cdiv(prefill_mem_index.numel(), 256),)](
        prefill_mem_index, prefill_row_table, prefill_mem_index.numel(), BLOCK=256
    )
    gather_chunk_size = min(KV_GATHER_CHUNK_SIZE, ragged_mem_index.numel())
    gathered_kv = q.new_empty((gather_chunk_size, 1, 512))
    chunk_indices = torch.empty_like(topk_indices)
    chunk_lengths = torch.empty(num_q_tokens, dtype=torch.int32, device=q.device)
    cu_seqlens_q = torch.arange(num_q_tokens + 1, dtype=torch.int32, device=q.device)
    output = lse = None
    for chunk_start in range(0, ragged_mem_index.numel(), gather_chunk_size):
        num_chunk_tokens = min(gather_chunk_size, ragged_mem_index.numel() - chunk_start)
        _gather_prefill_kv_kernel[(triton.cdiv(num_chunk_tokens, 4),)](
            packed_kv,
            ragged_mem_index[chunk_start : chunk_start + num_chunk_tokens],
            prefill_row_table,
            prefill_kv,
            gathered_kv,
            num_chunk_tokens,
            packed_kv.stride(0),
            prefill_kv.stride(0),
            prefill_kv.stride(2),
            BLOCK_ROWS=4,
        )
        _compact_topk_indices_kernel[(num_q_tokens,)](
            topk_indices,
            kv_starts,
            chunk_indices,
            chunk_lengths,
            chunk_start,
            chunk_start + num_chunk_tokens,
            TOPK=topk_indices.shape[1],
            BLOCK=triton.next_power_of_2(topk_indices.shape[1]),
        )
        chunk_kv = gathered_kv[:num_chunk_tokens].view(-1, 1, 1, 512)
        chunk_output, chunk_lse, *_ = flash_attn_with_kvcache(
            # only_qv ignores Q/K; these views avoid allocating placeholder tensors.
            q=q[..., :64],
            k_cache=chunk_kv[..., :64],
            qv=q,
            v_cache=chunk_kv,
            page_table=chunk_indices,
            cache_seqlens=chunk_lengths,
            cu_seqlens_q=cu_seqlens_q,
            max_seqlen_q=1,
            softmax_scale=softmax_scale,
            causal=False,
            only_qv=True,
            num_splits=1,
            return_softmax_lse=True,
        )
        if chunk_start == 0:
            output, lse = chunk_output, chunk_lse
        else:
            # Keep the online reduction in FP32 across all remaining chunks.
            output = output.float()
            _merge_att_output_kernel[(num_q_tokens, triton.cdiv(num_heads, 4))](
                output,
                lse,
                chunk_output,
                chunk_lse,
                chunk_lse.stride(1),
                chunk_lse.stride(0),
                NUM_HEADS=num_heads,
                NUM_Q_TOKENS=num_q_tokens,
                BLOCK_HEADS=4,
            )
        del chunk_output, chunk_lse
    return output.to(q.dtype)
