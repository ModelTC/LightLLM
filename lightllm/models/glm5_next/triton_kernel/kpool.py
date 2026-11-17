import torch
import triton
import triton.language as tl

from lightllm.models.deepseek3_2.triton_kernel.hadamard_transform import _butterfly_stage
from lightllm.utils.device_utils import get_device_sm_count


@triton.jit
def _get_query_block(b1_cu_q_seq_len, batch_size, BLOCK: tl.constexpr):
    block = tl.program_id(0)
    # Request b owns [cu[b] // BLOCK + b, cu[b + 1] // BLOCK + b + 1).
    # The extra block covers unaligned boundaries without padding to max_q_seq_len.
    if batch_size == 1:
        batch = 0
    else:
        batch = tl.full((), 0, tl.int32)
        # Keep the search tile independent of the scheduler's request count.
        lanes = tl.arange(0, 128)
        for first in range(0, batch_size, 128):
            batches = first + lanes
            q_ends = tl.load(b1_cu_q_seq_len + batches + 1, batches < batch_size, 0)
            block_ends = q_ends // BLOCK + batches + 1
            batch += tl.sum(((block >= block_ends) & (batches < batch_size)).to(tl.int32), 0)
    q_start = tl.load(b1_cu_q_seq_len + batch)
    q_end = tl.load(b1_cu_q_seq_len + batch + 1)
    local_block = block - (q_start // BLOCK + batch)
    return batch, q_start, q_end, local_block


@triton.jit
def _prefill_nsa_params(
    position_ids,
    b_req_idx,
    b1_cu_q_seq_len,
    b1_cu_kv_seq_len,
    q_kv_seq_len,
    q_req_idx,
    q_kv_start_loc,
    batch_size,
    BLOCK: tl.constexpr,
):
    batch, q_start, q_end, block = _get_query_block(b1_cu_q_seq_len, batch_size, BLOCK)
    rows = q_start + block * BLOCK + tl.arange(0, BLOCK)
    valid = rows < q_end
    tl.store(q_kv_seq_len + rows, tl.load(position_ids + rows, valid, 0) + 1, valid)
    tl.store(q_req_idx + rows, tl.load(b_req_idx + batch), valid)
    tl.store(q_kv_start_loc + rows, tl.load(b1_cu_kv_seq_len + batch), valid)


def prefill_nsa_params(position_ids, b_req_idx, b1_cu_q_seq_len, b1_cu_kv_seq_len):
    """Expand existing request metadata to query rows without visiting historical KV."""
    q_kv_seq_len, q_req_idx, q_kv_start_loc = [torch.empty_like(position_ids) for _ in range(3)]
    _prefill_nsa_params[(position_ids.numel() // 256 + b_req_idx.numel(),)](
        position_ids,
        b_req_idx,
        b1_cu_q_seq_len,
        b1_cu_kv_seq_len,
        q_kv_seq_len,
        q_req_idx,
        q_kv_start_loc,
        b_req_idx.numel(),
        BLOCK=256,
    )
    return q_kv_seq_len, q_req_idx, q_kv_start_loc


@triton.jit(do_not_specialize=["batch_size"])
def _compress_pools(
    k_and_gate,
    indexer_tail,
    indexer_k,
    index_kpool_compress_ape,
    q_kv_seq_len,
    mem_index,
    b_req_idx,
    b1_cu_q_seq_len,
    b_mtp_index,
    k_and_gate_stride: tl.constexpr,
    indexer_tail_stride_b: tl.constexpr,
    indexer_tail_stride_s: tl.constexpr,
    indexer_k_stride: tl.constexpr,
    HOLD_REQUEST_ID: tl.constexpr,
    batch_size,
    SINGLE_QUERY: tl.constexpr,
    TAIL_SIZE: tl.constexpr,
    HAS_MTP_INDEX: tl.constexpr,
):
    if SINGLE_QUERY:
        batch = tl.program_id(0)
        row = batch
        length = tl.load(q_kv_seq_len + row)
        valid = length > 0 and length % 4 == 0
        q_start = row
        if HAS_MTP_INDEX:
            q_start -= tl.load(b_mtp_index + row)
    else:
        batch, q_start, q_end, pool_index = _get_query_block(b1_cu_q_seq_len, batch_size, 4)
        prefix = tl.load(q_kv_seq_len + q_start, q_start < q_end, 1) - 1
        # Only process pool-closing rows, including a pool spanning the old tail.
        length = (prefix // 4 + pool_index + 1) * 4
        row = q_start + length - prefix - 1
        valid = row < q_end
    req = tl.load(b_req_idx + batch)
    if valid and req != HOLD_REQUEST_ID:
        pool = tl.arange(0, 4)
        cols = tl.arange(0, 128)
        raw_rows = row - 3 + pool
        from_chunk = raw_rows >= q_start
        chunk_ptr = k_and_gate + raw_rows[:, None] * k_and_gate_stride + cols[None, :]
        tail_slots = (length - 4 + pool) % TAIL_SIZE
        tail_ptr = (
            indexer_tail + req * indexer_tail_stride_b + tail_slots[:, None] * indexer_tail_stride_s + cols[None, :]
        )
        pool_k = tl.where(
            from_chunk[:, None],
            tl.load(chunk_ptr, from_chunk[:, None], 0),
            tl.load(tail_ptr, ~from_chunk[:, None], 0),
        ).to(tl.float32)
        pool_scores = tl.where(
            from_chunk[:, None],
            tl.load(chunk_ptr + 128, from_chunk[:, None], 0),
            tl.load(tail_ptr + 128, ~from_chunk[:, None], 0),
        ).to(tl.float32)
        pool_scores += tl.load(index_kpool_compress_ape + pool[:, None] * 128 + cols[None, :])
        pool_scores = tl.exp(pool_scores - tl.max(pool_scores, 0)[None, :])
        pool_weights = pool_scores / tl.sum(pool_scores, 0)[None, :]
        pooled_k = tl.sum(pool_k * pool_weights, 0).to(tl.bfloat16).to(tl.float32).reshape(1, 128)
        for step in tl.static_range(7):
            pooled_k = _butterfly_stage(pooled_k, 64 >> step, 1 << step, 1, 128)
        pooled_k = (pooled_k * (128 ** -0.5)).to(tl.bfloat16).to(tl.float32)
        k_scale = tl.exp2(tl.ceil(tl.log2(tl.maximum(tl.max(tl.abs(pooled_k), 1), 1e-4) / 448.0)))
        pooled_k = tl.minimum(tl.maximum(pooled_k / k_scale[:, None], -448.0), 448.0).to(tl.float8e4nv)
        # A completed pool lives at its closing token, which is a current query row.
        loc = tl.load(mem_index + row).to(tl.int64)
        dest = indexer_k + loc * indexer_k_stride
        tl.store(dest + cols, pooled_k.reshape(128).to(tl.uint8, bitcast=True))
        tl.store((dest + 128).to(tl.pointer_type(tl.float32)), tl.sum(k_scale, 0))


@triton.jit
def _save_pool_tails(
    k_and_gate,
    indexer_tail,
    b_req_idx,
    b1_cu_q_seq_len,
    b_seq_len,
    k_and_gate_stride: tl.constexpr,
    indexer_tail_stride_b: tl.constexpr,
    indexer_tail_stride_s: tl.constexpr,
    HOLD_REQUEST_ID: tl.constexpr,
    TAIL_SIZE: tl.constexpr,
):
    batch = tl.program_id(0)
    req = tl.load(b_req_idx + batch)
    if req != HOLD_REQUEST_ID:
        length = tl.load(b_seq_len + batch)
        q_start = tl.load(b1_cu_q_seq_len + batch)
        q_end = tl.load(b1_cu_q_seq_len + batch + 1)
        slots = tl.arange(0, triton.next_power_of_2(TAIL_SIZE))
        cols = tl.arange(0, 256)
        positions = length - TAIL_SIZE + slots
        raw_rows = q_end - length + positions
        # Absolute positions address a short ring, preserving the raw history
        # needed after rejecting candidates or revisiting draft positions.
        mask = ((slots < TAIL_SIZE) & (positions >= 0) & (raw_rows >= q_start))[:, None]
        k_and_gate_block = tl.load(k_and_gate + raw_rows[:, None] * k_and_gate_stride + cols[None, :], mask, 0)
        tl.store(
            indexer_tail
            + req * indexer_tail_stride_b
            + (positions % TAIL_SIZE)[:, None] * indexer_tail_stride_s
            + cols[None, :],
            k_and_gate_block,
            mask,
        )


def compress_pools(
    k_and_gate,
    indexer_tail,
    indexer_k,
    index_kpool_compress_ape,
    q_kv_seq_len,
    mem_index,
    b_req_idx,
    b1_cu_q_seq_len,
    b_seq_len,
    max_q_seq_len,
    b_mtp_index=None,
):
    batch_size = b_req_idx.numel()
    single_query = max_q_seq_len == 1 and q_kv_seq_len.numel() == batch_size
    blocks = batch_size if single_query else q_kv_seq_len.numel() // 4 + batch_size
    _compress_pools[(blocks,)](
        k_and_gate,
        indexer_tail,
        indexer_k,
        index_kpool_compress_ape,
        q_kv_seq_len,
        mem_index,
        b_req_idx,
        b1_cu_q_seq_len,
        b_mtp_index,
        k_and_gate.stride(0),
        indexer_tail.stride(0),
        indexer_tail.stride(1),
        indexer_k.stride(0),
        indexer_tail.shape[0] - 1,
        batch_size,
        single_query,
        indexer_tail.shape[1],
        b_mtp_index is not None,
        num_warps=4,
    )
    # Complete all boundary pools before replacing the previous chunk's tail.
    _save_pool_tails[(b_req_idx.numel(),)](
        k_and_gate,
        indexer_tail,
        b_req_idx,
        b1_cu_q_seq_len,
        b_seq_len,
        k_and_gate.stride(0),
        indexer_tail.stride(0),
        indexer_tail.stride(1),
        indexer_tail.shape[0] - 1,
        indexer_tail.shape[1],
        num_warps=4,
    )


@triton.jit(do_not_specialize=["batch_size", "q_token_num"])
def _get_pool_ranges(
    q_kv_seq_len,
    b1_cu_q_seq_len,
    ks,
    ke,
    q_pool_seq_len,
    batch_size,
    q_token_num,
    max_pools,
    SINGLE_QUERY: tl.constexpr,
    BLOCK: tl.constexpr,
):
    if SINGLE_QUERY:
        row = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
        batch = row
        valid = row < q_token_num
    else:
        batch, q_start, q_end, local_block = _get_query_block(b1_cu_q_seq_len, batch_size, BLOCK)
        row = q_start + local_block * BLOCK + tl.arange(0, BLOCK)
        valid = row < q_end
    length = tl.load(q_kv_seq_len + row, valid, 0) // 4
    start = batch * max_pools
    tl.store(ks + row, start, valid)
    tl.store(ke + row, start + length, valid)
    tl.store(q_pool_seq_len + row, length, valid)


def get_pool_ranges(q_kv_seq_len, b1_cu_q_seq_len, max_q_seq_len, max_pools):
    """Build per-query pool bounds directly from the packed request boundaries."""
    ks, ke, q_pool_seq_len = [torch.empty_like(q_kv_seq_len) for _ in range(3)]
    batch_size, q_token_num = b1_cu_q_seq_len.numel() - 1, q_kv_seq_len.numel()
    single_query = max_q_seq_len == 1 and q_token_num == batch_size
    block = 256
    blocks = triton.cdiv(q_token_num, block) if single_query else q_token_num // block + batch_size
    _get_pool_ranges[(blocks,)](
        q_kv_seq_len,
        b1_cu_q_seq_len,
        ks,
        ke,
        q_pool_seq_len,
        batch_size,
        q_token_num,
        max_pools,
        single_query,
        block,
        num_warps=4,
    )
    return ks, ke, q_pool_seq_len


@triton.jit(do_not_specialize=["max_pools"])
def _gather_pools(
    indexer_k,
    req_to_token_indexs,
    b_req_idx,
    b_seq_len,
    k_fp8,
    k_scale,
    indexer_k_stride: tl.constexpr,
    req_to_token_indexs_stride_b: tl.constexpr,
    max_pools,
    POOLS_PER_BLOCK: tl.constexpr,
):
    batch_row = tl.program_id(1)
    pool_indices = tl.program_id(0) * POOLS_PER_BLOCK + tl.arange(0, POOLS_PER_BLOCK)
    sequence_length = tl.load(b_seq_len + batch_row)
    cur_req_idx = tl.load(b_req_idx + batch_row)
    valid_pools = (pool_indices < max_pools) & (pool_indices < sequence_length // 4)
    # Four tokens form one pool; its key is stored at the last token.
    token_slots = tl.load(
        req_to_token_indexs + cur_req_idx * req_to_token_indexs_stride_b + pool_indices * 4 + 3, valid_pools, 0
    ).to(tl.int64)
    key_offsets = tl.arange(0, 128)
    key_bytes = tl.load(
        indexer_k + token_slots[:, None] * indexer_k_stride + key_offsets[None, :], valid_pools[:, None], 0
    )
    pool_scales = tl.load(
        (indexer_k + token_slots * indexer_k_stride + 128).to(tl.pointer_type(tl.float32)), valid_pools, 1.0
    )
    output_rows = batch_row.to(tl.int64) * max_pools + pool_indices
    tl.store(
        k_fp8 + output_rows[:, None] * 128 + key_offsets[None, :],
        key_bytes.to(tl.float8e4nv, bitcast=True),
        (pool_indices < max_pools)[:, None],
    )
    tl.store(k_scale + output_rows, pool_scales, pool_indices < max_pools)


def gather_pools(indexer_k, req_to_token_indexs, b_req_idx, b_seq_len, max_pools):
    """Gather FP8 keys and FP32 scales for prefill.

    b_seq_len counts tokens; max_pools is the pool capacity per row.
    """
    k_fp8 = torch.empty(
        (b_req_idx.numel() * max_pools, 128),
        device=indexer_k.device,
        dtype=torch.float8_e4m3fn,
    )
    k_scale = torch.empty((k_fp8.shape[0],), device=k_fp8.device, dtype=torch.float32)
    # Use grid X for pools because grid Y is limited to 65535 blocks.
    _gather_pools[(triton.cdiv(max_pools, 32), b_req_idx.numel())](
        indexer_k,
        req_to_token_indexs,
        b_req_idx,
        b_seq_len,
        k_fp8,
        k_scale,
        indexer_k.stride(0),
        req_to_token_indexs.stride(0),
        max_pools,
        POOLS_PER_BLOCK=32,
        num_warps=4,
    )
    return k_fp8, k_scale


@triton.jit(do_not_specialize=["max_pages_per_row"])
def _gather_paged_pools(
    indexer_k,
    req_to_token_indexs,
    b_req_idx,
    q_pool_seq_len,
    indexer_k_pages,
    block_table,
    indexer_k_stride: tl.constexpr,
    req_to_token_indexs_stride_b: tl.constexpr,
    max_pages_per_row,
    REUSE_GROUP_SIZE: tl.constexpr,
):
    batch_row = tl.program_id(1)
    # Avoid compiling a separate kernel for each batch size.
    num_batch_rows = tl.num_programs(1)
    cur_req_idx = tl.load(b_req_idx + batch_row)
    gather_pool_count = tl.load(q_pool_seq_len + batch_row)
    # The first row copies shared pages; other rows only update the page table.
    owner_row = batch_row
    if REUSE_GROUP_SIZE > 1:
        group_start_row = batch_row // REUSE_GROUP_SIZE * REUSE_GROUP_SIZE
        group_rows = group_start_row + tl.arange(0, REUSE_GROUP_SIZE)
        group_req_idx = tl.load(b_req_idx + group_rows, group_rows < num_batch_rows, -1)
        group_pool_counts = tl.load(q_pool_seq_len + group_rows, group_rows < num_batch_rows, 0)
        can_share_pages = (
            (group_start_row + REUSE_GROUP_SIZE <= num_batch_rows)
            & (tl.min(group_req_idx, 0) == tl.max(group_req_idx, 0))
            & (tl.max(group_pool_counts, 0) >= 4096)
        )
        owner_row = tl.where(can_share_pages, group_start_row, batch_row)
        gather_pool_count = tl.where(can_share_pages, tl.max(group_pool_counts, 0), gather_pool_count)
    pool_offsets = tl.arange(0, 64)
    key_offsets = tl.arange(0, 128)
    # Empty rows need one initialized page.
    for logical_page in range(tl.program_id(0), tl.cdiv(tl.maximum(gather_pool_count, 1), 64), tl.num_programs(0)):
        if batch_row != owner_row:
            tl.store(
                block_table + batch_row * max_pages_per_row + logical_page,
                owner_row * max_pages_per_row + logical_page,
            )
        else:
            pool_indices = logical_page * 64 + pool_offsets
            valid_pools = pool_indices < gather_pool_count
            token_slots = tl.load(
                req_to_token_indexs + cur_req_idx * req_to_token_indexs_stride_b + pool_indices * 4 + 3, valid_pools, 0
            ).to(tl.int64)
            key_bytes = tl.load(
                indexer_k + token_slots[:, None] * indexer_k_stride + key_offsets[None, :],
                valid_pools[:, None],
                0,
            )
            pool_scales = tl.load(
                (indexer_k + token_slots * indexer_k_stride + 128).to(tl.pointer_type(tl.float32)),
                valid_pools,
                1.0,
            )
            output_page_ptr = indexer_k_pages + (batch_row.to(tl.int64) * max_pages_per_row + logical_page) * (64 * 132)
            # Page layout: [64 x 128 key bytes][64 FP32 scales].
            tl.store(output_page_ptr + pool_offsets[:, None] * 128 + key_offsets[None, :], key_bytes)
            tl.store((output_page_ptr + 64 * 128).to(tl.pointer_type(tl.float32)) + pool_offsets, pool_scales)


def gather_paged_pools(indexer_k, req_to_token_indexs, b_req_idx, q_pool_seq_len, max_pools, reuse_group=4):
    """Gather decode pages and the page table.

    q_pool_seq_len counts valid pools; max_pools is the pool capacity per row.
    """
    if reuse_group not in (1, 2, 4, 8):
        raise ValueError("reuse_group must be 1, 2, 4 or 8")
    batch_size = b_req_idx.numel()
    max_pages_per_row = triton.cdiv(max_pools, 64)
    # Reserve each row's capacity so sharing can change on graph replay.
    indexer_k_pages = torch.empty(
        (batch_size * max_pages_per_row, 64, 1, 132), device=indexer_k.device, dtype=torch.uint8
    )
    block_table = torch.arange(batch_size * max_pages_per_row, device=indexer_k_pages.device, dtype=torch.int32).view(
        batch_size, max_pages_per_row
    )
    estimated_copy_groups = max(1, triton.cdiv(batch_size, reuse_group)) if max_pools >= 4096 else batch_size
    page_blocks_per_row = min(max_pages_per_row, triton.cdiv(get_device_sm_count() * 4, estimated_copy_groups))
    _gather_paged_pools[(page_blocks_per_row, batch_size)](
        indexer_k,
        req_to_token_indexs,
        b_req_idx,
        q_pool_seq_len,
        indexer_k_pages,
        block_table,
        indexer_k.stride(0),
        req_to_token_indexs.stride(0),
        max_pages_per_row,
        REUSE_GROUP_SIZE=reuse_group if batch_size >= reuse_group else 1,
        num_warps=4,
    )
    return indexer_k_pages, block_table


@triton.jit
def _expand_topk(
    topk_pool_indices,
    q_kv_seq_len,
    q_req_idx,
    req_to_token_indexs,
    topk_mem_indices,
    topk_indices,
    req_to_token_indexs_stride_b: tl.constexpr,
    req_to_token_indexs_stride_s: tl.constexpr,
    index_topk: tl.constexpr,
    WIDTH: tl.constexpr,
    DENSE: tl.constexpr,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    lane = tl.arange(0, BLOCK)
    length = tl.load(q_kv_seq_len + row)
    req = tl.load(q_req_idx + row).to(tl.int64)
    if DENSE:
        token = lane
        valid = lane < length
    else:
        closed_tokens = tl.minimum(length // 4 * 4, index_topk)
        group = tl.load(topk_pool_indices + row * (index_topk // 4) + lane // 4, lane < closed_tokens, -1)
        token = tl.where(lane < closed_tokens, group * 4 + lane % 4, length // 4 * 4 + lane - closed_tokens)
        valid = (lane < closed_tokens + length % 4) & (token >= 0)
    mem = tl.load(
        req_to_token_indexs + req * req_to_token_indexs_stride_b + token * req_to_token_indexs_stride_s,
        valid & (lane < WIDTH),
        -1,
    )
    tl.store(topk_mem_indices + row * WIDTH + lane, mem, lane < WIDTH)
    tl.store(topk_indices + row * WIDTH + lane, tl.where(valid, token, -1), lane < WIDTH)


def expand_topk(topk_pool_indices, q_kv_seq_len, q_req_idx, req_to_token_indexs, index_topk, dense=False):
    width = triton.cdiv(index_topk + 3, 128) * 128
    topk_mem_indices = torch.empty((q_kv_seq_len.numel(), width), dtype=torch.int32, device=q_kv_seq_len.device)
    topk_indices = torch.empty_like(topk_mem_indices)
    _expand_topk[(q_kv_seq_len.numel(),)](
        topk_pool_indices,
        q_kv_seq_len,
        q_req_idx,
        req_to_token_indexs,
        topk_mem_indices,
        topk_indices,
        req_to_token_indexs.stride(0),
        req_to_token_indexs.stride(1),
        index_topk,
        width,
        dense,
        triton.next_power_of_2(width),
        num_warps=4,
    )
    return topk_mem_indices, topk_indices
