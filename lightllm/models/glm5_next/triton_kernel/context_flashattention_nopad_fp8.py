import torch
import triton
import triton.language as tl


# Bound the temporary BF16 gather buffer to 2 GiB (2097152 × 1 × 512 × 2 bytes).
KV_GATHER_CHUNK_SIZE = 2097152


@triton.jit
def _gather_prefill_kv_kernel(
    packed_kv,
    prefill_cache_kv,
    output,
    req_to_token_indexs,
    b_req_idx,
    b1_cu_kv_seq_len,
    b1_cu_q_seq_len,
    chunk_start,
    chunk_token_num,
    packed_kv_stride_s: tl.constexpr,
    prefill_cache_kv_stride_s: tl.constexpr,
    prefill_cache_kv_stride_d: tl.constexpr,
    req_to_token_indexs_stride_b: tl.constexpr,
    req_to_token_indexs_stride_s: tl.constexpr,
    batch_size: tl.constexpr,
    BLOCK_ROWS: tl.constexpr,
):
    block = tl.program_id(0)
    lanes = tl.arange(0, triton.next_power_of_2(batch_size))
    ends = tl.load(b1_cu_kv_seq_len + lanes + 1, lanes < batch_size, 0)
    ends = tl.minimum(tl.maximum(ends - chunk_start, 0), chunk_token_num)
    block_ends = ends // BLOCK_ROWS + lanes + 1
    batch = tl.sum(((block >= block_ends) & (lanes < batch_size)).to(tl.int32), 0)
    start = tl.load(b1_cu_kv_seq_len + batch)
    copy_start = tl.minimum(tl.maximum(start - chunk_start, 0), chunk_token_num)
    copy_end = tl.minimum(tl.maximum(tl.load(b1_cu_kv_seq_len + batch + 1) - chunk_start, 0), chunk_token_num)
    local_block = block - (copy_start // BLOCK_ROWS + batch)
    row = copy_start + local_block * BLOCK_ROWS + tl.arange(0, BLOCK_ROWS)
    live = row < copy_end
    req = tl.load(b_req_idx + batch).to(tl.int64)
    position = chunk_start + row - start
    q_start = tl.load(b1_cu_q_seq_len + batch)
    q_len = tl.load(b1_cu_q_seq_len + batch + 1) - q_start
    cache_len = tl.load(b1_cu_kv_seq_len + batch + 1) - start - q_len
    prefill_row = tl.where(position >= cache_len, q_start + position - cache_len, -1)
    kv_mem_index = tl.load(
        req_to_token_indexs + req * req_to_token_indexs_stride_b + position * req_to_token_indexs_stride_s,
        live & (prefill_row < 0),
        0,
    ).to(tl.int64)
    dim = tl.arange(0, 512)
    cached = live & (prefill_row < 0)
    current = live & (prefill_row >= 0)
    packed_row = packed_kv + kv_mem_index[:, None] * packed_kv_stride_s

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
        prefill_cache_kv + prefill_row[:, None] * prefill_cache_kv_stride_s + dim[None, :] * prefill_cache_kv_stride_d,
        current[:, None],
        0,
    ).to(tl.float32)
    tl.store(
        output + row[:, None] * 512 + dim[None, :],
        tl.where(current[:, None], current_values, values),
        live[:, None],
    )


@triton.jit
def _compact_topk_indices_kernel(
    topk_indices,
    q_kv_start_loc,
    chunk_topk_indices,
    chunk_cache_seqlens,
    chunk_start,
    chunk_end,
    topk_indices_stride_s: tl.constexpr,
    topk_indices_stride_d: tl.constexpr,
    TOPK: tl.constexpr,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    lane = tl.arange(0, BLOCK)
    index = tl.load(topk_indices + row * topk_indices_stride_s + lane * topk_indices_stride_d, lane < TOPK, -1)
    kv_index = index + tl.load(q_kv_start_loc + row)
    valid = (lane < TOPK) & (index >= 0) & (kv_index >= chunk_start) & (kv_index < chunk_end)
    compact_index = tl.cumsum(valid.to(tl.int32)) - 1
    tl.store(chunk_topk_indices + row * TOPK + compact_index, kv_index - chunk_start, valid)
    tl.store(chunk_cache_seqlens + row, tl.sum(valid.to(tl.int32)))


@triton.jit
def _merge_att_output_kernel(
    output,
    softmax_lse,
    chunk_output,
    chunk_softmax_lse,
    chunk_lse_stride_s: tl.constexpr,
    chunk_lse_stride_h,
    q_head_num: tl.constexpr,
    q_token_num,
    BLOCK_HEADS: tl.constexpr,
):
    row = tl.program_id(0)
    head = tl.program_id(1) * BLOCK_HEADS + tl.arange(0, BLOCK_HEADS)
    dim = tl.arange(0, 512)
    previous_lse = tl.load(softmax_lse + head * q_token_num + row, head < q_head_num, -float("inf"))
    chunk_lse_block = tl.load(
        chunk_softmax_lse + row * chunk_lse_stride_s + head * chunk_lse_stride_h, head < q_head_num, -float("inf")
    )
    # FA3 uses natural-log LSE and returns +inf for an empty row.
    previous_lse = tl.where(previous_lse == float("inf"), -float("inf"), previous_lse)
    chunk_lse_block = tl.where(chunk_lse_block == float("inf"), -float("inf"), chunk_lse_block)
    maximum = tl.maximum(previous_lse, chunk_lse_block)
    maximum = tl.where(maximum == -float("inf"), 0, maximum)
    previous_weight = tl.exp(previous_lse - maximum)
    chunk_weight = tl.exp(chunk_lse_block - maximum)
    denominator = previous_weight + chunk_weight
    offset = (row * q_head_num + head[:, None]) * 512 + dim[None, :]
    previous = tl.load(output + offset, (head[:, None] < q_head_num) & (previous_weight[:, None] > 0), 0)
    current = tl.load(chunk_output + offset, (head[:, None] < q_head_num) & (chunk_weight[:, None] > 0), 0).to(
        tl.float32
    )
    result = (previous * previous_weight[:, None] + current * chunk_weight[:, None]) / tl.where(
        denominator > 0, denominator, 1
    )[:, None]
    tl.store(output + offset, result, head[:, None] < q_head_num)
    tl.store(
        softmax_lse + head * q_token_num + row,
        tl.where(denominator > 0, maximum + tl.log(denominator), -float("inf")),
        head < q_head_num,
    )


def allocate_prefill_workspace(q, topk_indices, q_token_capacity=None):
    capacity = q.shape[0] if q_token_capacity is None else q_token_capacity
    return {
        "chunk_topk_indices": torch.empty((capacity, topk_indices.shape[1]), dtype=topk_indices.dtype, device=q.device),
        "chunk_cache_seqlens": torch.empty(capacity, dtype=torch.int32, device=q.device),
        "cu_seqlens_q": torch.arange(capacity + 1, dtype=torch.int32, device=q.device),
    }


@torch.no_grad()
def context_attention_fwd_fp8(
    q,
    packed_kv,
    prefill_cache_kv,
    q_kv_start_loc,
    topk_indices,
    req_to_token_indexs,
    b_req_idx,
    b1_cu_kv_seq_len,
    b1_cu_q_seq_len,
    total_token_num,
    softmax_scale,
    workspace=None,
):
    """Run GLM sparse prefill with FlashAttention and request-table KV gathering."""
    q_token_num, q_head_num, _ = q.shape
    from sgl_kernel.flash_attn import flash_attn_with_kvcache

    gather_chunk_size = min(KV_GATHER_CHUNK_SIZE, total_token_num)
    workspace = workspace or allocate_prefill_workspace(q, topk_indices)
    chunk_topk_indices = workspace["chunk_topk_indices"]
    chunk_cache_seqlens = workspace["chunk_cache_seqlens"]
    cu_seqlens_q = workspace["cu_seqlens_q"]
    # Keep this potentially 2 GiB buffer local to a DSA layer.
    has_history = total_token_num > q_token_num
    gathered_kv = q.new_empty((gather_chunk_size, 1, 512)) if has_history else None
    if not has_history and prefill_cache_kv.stride(-1) != 1:
        prefill_cache_kv = prefill_cache_kv.contiguous()
    output = softmax_lse = None
    for chunk_start in range(0, total_token_num, gather_chunk_size):
        chunk_token_num = min(gather_chunk_size, total_token_num - chunk_start)
        if has_history:
            _gather_prefill_kv_kernel[(chunk_token_num // 4 + b_req_idx.numel(),)](
                packed_kv,
                prefill_cache_kv,
                gathered_kv,
                req_to_token_indexs,
                b_req_idx,
                b1_cu_kv_seq_len,
                b1_cu_q_seq_len,
                chunk_start,
                chunk_token_num,
                packed_kv.stride(0),
                prefill_cache_kv.stride(0),
                prefill_cache_kv.stride(2),
                req_to_token_indexs_stride_b=req_to_token_indexs.stride(0),
                req_to_token_indexs_stride_s=req_to_token_indexs.stride(1),
                batch_size=b_req_idx.numel(),
                BLOCK_ROWS=4,
            )
            chunk_kv = gathered_kv[:chunk_token_num].view(-1, 1, 1, 512)
        else:
            # Current BF16 KV already has the complete request layout.
            chunk_kv = prefill_cache_kv[chunk_start : chunk_start + chunk_token_num].view(-1, 1, 1, 512)
        _compact_topk_indices_kernel[(q_token_num,)](
            topk_indices,
            q_kv_start_loc,
            chunk_topk_indices,
            chunk_cache_seqlens,
            chunk_start,
            chunk_start + chunk_token_num,
            topk_indices.stride(0),
            topk_indices.stride(1),
            TOPK=topk_indices.shape[1],
            BLOCK=triton.next_power_of_2(topk_indices.shape[1]),
        )
        chunk_output, chunk_softmax_lse, *_ = flash_attn_with_kvcache(
            # only_qv ignores Q/K; these views avoid allocating placeholder tensors.
            q=q[..., :64],
            k_cache=chunk_kv[..., :64],
            qv=q,
            v_cache=chunk_kv,
            page_table=chunk_topk_indices,
            cache_seqlens=chunk_cache_seqlens,
            cu_seqlens_q=cu_seqlens_q,
            max_seqlen_q=1,
            softmax_scale=softmax_scale,
            causal=False,
            only_qv=True,
            num_splits=1,
            return_softmax_lse=True,
        )
        if chunk_start == 0:
            output, softmax_lse = chunk_output, chunk_softmax_lse
        else:
            # Keep the online reduction in FP32 across all remaining chunks.
            output = output.float()
            _merge_att_output_kernel[(q_token_num, triton.cdiv(q_head_num, 4))](
                output,
                softmax_lse,
                chunk_output,
                chunk_softmax_lse,
                chunk_softmax_lse.stride(1),
                chunk_softmax_lse.stride(0),
                q_head_num=q_head_num,
                q_token_num=q_token_num,
                BLOCK_HEADS=4,
            )
        del chunk_output, chunk_softmax_lse
    return output.to(q.dtype)
