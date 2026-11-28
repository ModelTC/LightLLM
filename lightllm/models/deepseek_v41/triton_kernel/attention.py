"""V4.1 KV pair compression, packed KV cache writes, and sparse indexer selection."""

import torch
import triton
import triton.language as tl
from triton.language.extra import libdevice

from lightllm.common.basemodel.triton_kernel.norm.rmsnorm import rmsnorm_num_warps
from lightllm.common.basemodel.triton_kernel.ragged_topk import ragged_topk


@triton.jit
def _compress_and_normalize_kv_pairs_kernel(
    kv,
    pooling_logits,
    request_ids,
    position_ids,
    pending_pair_state,
    compressed_kv,
    norm_weight,
    eps,
    KV_DIM: tl.constexpr,
    PENDING_STATE_ROW_STRIDE: tl.constexpr,
    KV_ROW_STRIDE: tl.constexpr,
    POOLING_LOGITS_ROW_STRIDE: tl.constexpr,
    SKIP_OPEN_PAIRS: tl.constexpr,
):
    token_row = tl.program_id(0)
    feature_indices = tl.arange(0, KV_DIM)
    position_id = tl.load(position_ids + token_row)
    completes_pair = position_id % 2 == 1
    if SKIP_OPEN_PAIRS:
        if not completes_pair:
            # The dense latent still feeds the indexer projection. Match the
            # signed zero produced by the normal RMSNorm weight multiply.
            rms_weight = tl.load(norm_weight + feature_indices).to(tl.float32)
            tl.store(compressed_kv + token_row * KV_DIM + feature_indices, 0.0 * rms_weight)
            return
    request_id = tl.load(request_ids + token_row)
    previous_request_id = tl.load(request_ids + token_row - 1, completes_pair & (token_row > 0), -1)
    previous_in_batch = completes_pair & (token_row > 0) & (previous_request_id == request_id)
    previous_kv = tl.load(kv + (token_row - 1) * KV_ROW_STRIDE + feature_indices, previous_in_batch, 0.0)
    previous_pooling_logit = tl.load(
        pooling_logits + (token_row - 1) * POOLING_LOGITS_ROW_STRIDE + feature_indices, previous_in_batch, 0.0
    )
    request_pending_state = pending_pair_state + request_id * PENDING_STATE_ROW_STRIDE
    previous_in_state = completes_pair & ~previous_in_batch
    previous_kv += tl.load(request_pending_state + feature_indices, previous_in_state, 0.0)
    previous_pooling_logit += tl.load(request_pending_state + KV_DIM + feature_indices, previous_in_state, 0.0)
    current_kv = tl.load(kv + token_row * KV_ROW_STRIDE + feature_indices, completes_pair, 0.0)
    current_pooling_logit = tl.load(
        pooling_logits + token_row * POOLING_LOGITS_ROW_STRIDE + feature_indices, completes_pair, 0.0
    )
    current_weight = tl.sigmoid(current_pooling_logit - previous_pooling_logit)
    pooled_kv = previous_kv + (current_kv - previous_kv) * current_weight
    # Match the BF16 intermediate and separately rounded FP32 squares of RMSNorm.
    pooled_kv = pooled_kv.to(tl.bfloat16).to(tl.float32)
    mean_square = tl.sum(libdevice.mul_rn(pooled_kv, pooled_kv), axis=0) / KV_DIM
    inverse_rms = 1 / tl.sqrt(mean_square + eps)
    rms_weight = tl.load(norm_weight + feature_indices).to(tl.float32)
    tl.store(compressed_kv + token_row * KV_DIM + feature_indices, (pooled_kv * inverse_rms) * rms_weight)


@triton.jit
def _save_pending_kv_pair_kernel(
    kv,
    pooling_logits,
    request_ids,
    position_ids,
    pending_pair_state,
    num_tokens,
    KV_DIM: tl.constexpr,
    HOLD_REQUEST_ID: tl.constexpr,
    PENDING_STATE_ROW_STRIDE: tl.constexpr,
    KV_ROW_STRIDE: tl.constexpr,
    POOLING_LOGITS_ROW_STRIDE: tl.constexpr,
):
    token_row = tl.program_id(0)
    request_id = tl.load(request_ids + token_row)
    position_id = tl.load(position_ids + token_row)
    # Only a request ending mid-pair needs to retain its final token for the next forward.
    next_request_id = tl.load(request_ids + token_row + 1, token_row + 1 < num_tokens, -1)
    save_pending_token = (request_id != HOLD_REQUEST_ID) & (next_request_id != request_id) & (position_id % 2 == 0)
    feature_indices = tl.arange(0, KV_DIM)
    request_pending_state = pending_pair_state + request_id * PENDING_STATE_ROW_STRIDE
    tl.store(
        request_pending_state + feature_indices,
        tl.load(kv + token_row * KV_ROW_STRIDE + feature_indices),
        save_pending_token,
    )
    tl.store(
        request_pending_state + KV_DIM + feature_indices,
        tl.load(pooling_logits + token_row * POOLING_LOGITS_ROW_STRIDE + feature_indices),
        save_pending_token,
    )


def compress_and_normalize_kv_pairs(
    kv, pooling_logits, request_ids, position_ids, pending_pair_state, hold_request_id, norm_weight, eps
):
    """Pool completed KV pairs, apply RMSNorm, and save the pending token for each unfinished pair."""
    compressed_kv = torch.empty_like(kv, dtype=torch.bfloat16)
    _compress_and_normalize_kv_pairs_kernel[(kv.shape[0],)](
        kv,
        pooling_logits,
        request_ids,
        position_ids,
        pending_pair_state,
        compressed_kv,
        norm_weight,
        eps,
        kv.shape[1],
        pending_pair_state.stride(0),
        kv.stride(0),
        pooling_logits.stride(0),
        SKIP_OPEN_PAIRS=kv.shape[0] >= 1024,
        num_warps=rmsnorm_num_warps,
    )
    # Save pending tokens after pooling, which can still read the preceding forward's state.
    _save_pending_kv_pair_kernel[(kv.shape[0],)](
        kv,
        pooling_logits,
        request_ids,
        position_ids,
        pending_pair_state,
        kv.shape[0],
        kv.shape[1],
        hold_request_id,
        pending_pair_state.stride(0),
        kv.stride(0),
        pooling_logits.stride(0),
    )
    return compressed_kv


@triton.jit
def _rotate_and_cache_compressed_kv_kernel(
    kv,
    rope_cos_sin,
    full_kv_slots,
    position_ids,
    compressed_kv_cache,
    COMPRESSION_RATIO: tl.constexpr,
    COMPRESSED_PAGE_SIZE: tl.constexpr,
    CACHE_PAGE_STRIDE_BYTES: tl.constexpr,
):
    KV_DIM: tl.constexpr = 512
    NOPE_DIM: tl.constexpr = 448
    ROPE_DIM: tl.constexpr = 64
    FP8_GROUP_SIZE: tl.constexpr = 64
    FP8_MAX: tl.constexpr = 448
    KV_SLOT_BYTES: tl.constexpr = 576
    SCALE_SLOT_BYTES: tl.constexpr = 8

    token_row = tl.program_id(0)
    position_id = tl.load(position_ids + token_row)
    if (position_id + 1) % COMPRESSION_RATIO != 0:
        return
    compressed_slot = tl.load(full_kv_slots + token_row).to(tl.int64) // COMPRESSION_RATIO
    page_id, page_slot = compressed_slot // COMPRESSED_PAGE_SIZE, compressed_slot % COMPRESSED_PAGE_SIZE
    feature_indices = tl.arange(0, KV_DIM)
    complex_pair_indices = tl.arange(0, KV_DIM // 2)
    kv_real = tl.load(kv + token_row * KV_DIM + complex_pair_indices * 2).to(tl.float32)
    kv_imag = tl.load(kv + token_row * KV_DIM + complex_pair_indices * 2 + 1).to(tl.float32)
    rope_frequency = (
        rope_cos_sin + (position_id - COMPRESSION_RATIO + 1) * ROPE_DIM + (complex_pair_indices - NOPE_DIM // 2) * 2
    )
    cos = tl.load(rope_frequency, complex_pair_indices >= NOPE_DIM // 2, 1.0)
    sin = tl.load(rope_frequency + 1, complex_pair_indices >= NOPE_DIM // 2, 0.0)
    rotated_kv = tl.reshape(tl.join(kv_real * cos - kv_imag * sin, kv_real * sin + kv_imag * cos), (KV_DIM,))
    rotated_kv = rotated_kv.to(tl.bfloat16).to(tl.float32)
    grouped_kv = tl.reshape(rotated_kv, (KV_DIM // FP8_GROUP_SIZE, FP8_GROUP_SIZE))
    fp8_scales = tl.maximum(tl.max(tl.abs(grouped_kv), 1), 1e-4) / FP8_MAX
    scale_bits = fp8_scales.to(tl.uint32, bitcast=True)
    scale_exponents = ((scale_bits >> 23) & 255) + ((scale_bits & 0x7FFFFF) != 0).to(tl.uint32)
    inverse_scales = ((254 - scale_exponents) << 23).to(tl.float32, bitcast=True)
    quantized_kv = tl.reshape(grouped_kv * inverse_scales[:, None], (KV_DIM,))
    rope_feature_indices = tl.arange(0, ROPE_DIM)
    rope_tail = tl.gather(rotated_kv, NOPE_DIM + rope_feature_indices, axis=0)
    scale_group_indices = tl.arange(0, KV_DIM // FP8_GROUP_SIZE)
    cache_page = compressed_kv_cache + page_id * CACHE_PAGE_STRIDE_BYTES
    # Within each page, packed KV rows come first and the per-slot scale exponents follow.
    tl.store(
        (cache_page + page_slot * KV_SLOT_BYTES).to(tl.pointer_type(tl.float8e4nv)) + feature_indices,
        quantized_kv,
        feature_indices < NOPE_DIM,
    )
    tl.store(
        (cache_page + page_slot * KV_SLOT_BYTES + NOPE_DIM).to(tl.pointer_type(tl.bfloat16)) + rope_feature_indices,
        rope_tail,
    )
    tl.store(
        cache_page + COMPRESSED_PAGE_SIZE * KV_SLOT_BYTES + page_slot * SCALE_SLOT_BYTES + scale_group_indices,
        scale_exponents,
        scale_group_indices < NOPE_DIM // FP8_GROUP_SIZE,
    )


def rotate_and_cache_compressed_kv(
    kv, freqs_cis, full_kv_slots, position_ids, compressed_kv_cache, compression_ratio, compressed_page_size
):
    """Apply RoPE and write FP8 NoPE and BF16 RoPE fields to the Hopper FlashMLA cache."""
    _rotate_and_cache_compressed_kv_kernel[(kv.shape[0],)](
        kv,
        torch.view_as_real(freqs_cis).flatten(-2),
        full_kv_slots,
        position_ids,
        compressed_kv_cache,
        compression_ratio,
        compressed_page_size,
        compressed_kv_cache.stride(0),
        enable_fp_fusion=False,
    )


@triton.jit
def _prepare_indexer_logits_kernel(
    logits,
    valid_lengths,
    candidate_block_mask,
    candidate_block_scores,
    logits_row_stride,
    logits_width,
    BLOCKS_PER_PROGRAM: tl.constexpr,
):
    query_row = tl.program_id(0)
    block_ids = tl.program_id(1) * BLOCKS_PER_PROGRAM + tl.arange(0, BLOCKS_PER_PROGRAM)
    key_indices = block_ids[:, None] * 8 + tl.arange(0, 8)[None, :]
    valid_length = tl.load(valid_lengths + query_row)
    valid_keys = key_indices < valid_length
    if candidate_block_mask is not None:
        keep_blocks = tl.load(
            candidate_block_mask + query_row * (logits_width // 8) + block_ids, block_ids < logits_width // 8, False
        )
        valid_keys &= keep_blocks[:, None]
    masked_logits = tl.load(logits + query_row * logits_row_stride + key_indices, valid_keys, -float("inf"))
    tl.store(logits + query_row * logits_row_stride + key_indices, masked_logits, key_indices < logits_width)
    if candidate_block_scores is not None:
        latest_block_id = tl.maximum(valid_length - 1, 0) // 8
        block_scores = tl.max(masked_logits, 1)
        block_scores = tl.where(tl.max(libdevice.isnan(masked_logits).to(tl.int32), 1) != 0, float("nan"), block_scores)
        # Keep the latest block among the candidates offered to later layers.
        block_scores = tl.where(block_ids == latest_block_id, float("inf"), block_scores)
        tl.store(
            candidate_block_scores + query_row * (logits_width // 8) + block_ids,
            block_scores,
            block_ids < logits_width // 8,
        )


@triton.jit
def _map_indexer_topk_to_slots_kernel(
    logits,
    topk_key_indices,
    page_table,
    selected_slots,
    valid_lengths,
    selected_lengths,
    logits_row_stride,
    page_table_row_stride,
    PAGE_SIZE: tl.constexpr,
    TOPK: tl.constexpr,
    TOPK_BLOCK_SIZE: tl.constexpr,
):
    query_row = tl.program_id(0)
    topk_column = tl.arange(0, TOPK_BLOCK_SIZE)
    valid_length = tl.load(valid_lengths + query_row)
    key_index = topk_column
    if topk_key_indices is not None:
        if valid_length > TOPK:
            key_index = tl.load(topk_key_indices + query_row * TOPK + topk_column, topk_column < TOPK, 0).to(tl.int32)
    valid_key = (topk_column < TOPK) & (key_index < valid_length)
    key_score = tl.load(logits + query_row * logits_row_stride + key_index, valid_key, -float("inf"))
    valid_key &= tl.abs(key_score) < float("inf")
    if topk_key_indices is not None:
        if valid_length > TOPK:
            invalid_key_index = 2147483647
            key_index = tl.sort(tl.where(valid_key, key_index, invalid_key_index))
            valid_key = key_index != invalid_key_index
    packed_column = tl.cumsum(valid_key.to(tl.int32)) - 1
    selected_count = tl.sum(valid_key.to(tl.int32))
    physical_page_id = tl.load(page_table + query_row * page_table_row_stride + key_index // PAGE_SIZE, valid_key, 0)
    tl.store(
        selected_slots + query_row * TOPK + topk_column,
        -1,
        (topk_column < TOPK) & (topk_column >= selected_count),
    )
    tl.store(
        selected_slots + query_row * TOPK + packed_column,
        physical_page_id * PAGE_SIZE + key_index % PAGE_SIZE,
        valid_key,
    )
    tl.store(selected_lengths + query_row, selected_count)


def select_indexer_topk_slots(
    logits,
    valid_lengths,
    page_table,
    selected_slots,
    selected_lengths,
    page_size,
    candidate_block_mask,
    candidate_block_topk,
):
    """Keep the latest candidate block and pack finite top-k keys into cache slots in logical order."""
    num_queries, num_keys = logits.shape
    topk = selected_slots.shape[1]
    # A candidate budget covering the entire row offers every block to subsequent layers.
    if candidate_block_topk is not None and candidate_block_topk >= num_keys // 8:
        candidate_block_mask.fill_(True)
        candidate_block_mask = None
        candidate_block_topk = None
    if candidate_block_mask is not None or num_keys > topk:
        block_scores = logits.new_empty((num_queries, num_keys // 8)) if candidate_block_topk is not None else None
        _prepare_indexer_logits_kernel[(num_queries, triton.cdiv(num_keys, 1024))](
            logits,
            valid_lengths,
            candidate_block_mask if block_scores is None else None,
            block_scores,
            logits.stride(0),
            num_keys,
            128,
        )
        if block_scores is not None:
            selected_block_ids = block_scores.topk(candidate_block_topk, sorted=False).indices
            candidate_block_mask.zero_().scatter_(1, selected_block_ids, True)
    topk_key_indices = logits.topk(topk, sorted=False).indices if num_keys > topk else None
    _map_indexer_topk_to_slots_kernel[(num_queries,)](
        logits,
        topk_key_indices,
        page_table,
        selected_slots,
        valid_lengths,
        selected_lengths,
        logits.stride(0),
        page_table.stride(0),
        page_size,
        topk,
        triton.next_power_of_2(topk),
    )


@triton.jit
def _candidate_scores(LOGITS, LENGTHS, OUT, TOPK_LENGTHS, STRIDE, WIDTH, TOPK: tl.constexpr, BLOCK: tl.constexpr):
    row, tile = tl.program_id(0), tl.program_id(1)
    length = tl.load(LENGTHS + row)
    block_length = tl.cdiv(length, 8)
    needs_topk = block_length > TOPK
    tl.store(TOPK_LENGTHS + row, tl.where(needs_topk, block_length, 0), tile == 0)
    if needs_topk & (tile * BLOCK < block_length):
        block = tile * BLOCK + tl.arange(0, BLOCK)
        col = block[:, None] * 8 + tl.arange(0, 8)[None, :]
        value = tl.load(LOGITS + row * STRIDE + col, col < length, -float("inf"))
        score = tl.max(value, 1)
        score = tl.where(block == block_length - 1, float("inf"), score)
        tl.store(OUT + row * WIDTH + block, score, (block < WIDTH) & (block < block_length))


@triton.jit
def _sort_candidates(INDICES, LENGTHS, TOPK: tl.constexpr, BLOCK: tl.constexpr):
    row = tl.program_id(0)
    col = tl.arange(0, BLOCK)
    block_length = tl.cdiv(tl.load(LENGTHS + row), 8)
    if block_length <= TOPK:
        tl.store(INDICES + row * TOPK + col, tl.where(col < block_length, col, -1), col < TOPK)
    else:
        index = tl.load(INDICES + row * TOPK + col, col < TOPK, -1)
        valid = (index >= 0) & (index < block_length)
        index = tl.sort(tl.where(valid, index, 2147483647))
        valid = (index != 2147483647) & ((col == 0) | (index != tl.gather(index, tl.maximum(col - 1, 0), axis=0)))
        rank = tl.cumsum(valid.to(tl.int32)) - 1
        count = tl.sum(valid.to(tl.int32))
        tl.store(INDICES + row * TOPK + col, -1, (col < TOPK) & (col >= count))
        tl.store(INDICES + row * TOPK + rank, index, valid)


# Prefill changes the page-table stride as the context grows.
@triton.jit(do_not_specialize=["PAGE_ROW"])
def _candidate_indexer_logits(
    Q,
    CACHE,
    WEIGHTS,
    PAGES,
    PAGE_ROWS,
    CANDIDATES,
    LENGTHS,
    LOGITS,
    Q_ROW: tl.constexpr,
    Q_HEAD: tl.constexpr,
    CACHE_PAGE: tl.constexpr,
    WEIGHT_ROW: tl.constexpr,
    PAGE_ROW,
    CANDIDATE_ROW: tl.constexpr,
    HEADS: tl.constexpr,
    BLOCK_H: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    row, tile = tl.program_id(0), tl.program_id(1)
    col = tile * BLOCK_N + tl.arange(0, BLOCK_N)
    # The producer sorts valid blocks and pads their tail with -1. Test one
    # scalar before materializing the dot's vector inputs and live ranges.
    first = tl.load(CANDIDATES + row * CANDIDATE_ROW + tile * (BLOCK_N // 8), tile * (BLOCK_N // 8) < CANDIDATE_ROW, -1)
    length = tl.load(LENGTHS + row)
    if (first >= 0) & (first * 8 < length):
        block = tl.load(CANDIDATES + row * CANDIDATE_ROW + col // 8, col < CANDIDATE_ROW * 8, -1)
        position = block * 8 + col % 8
        valid = (block >= 0) & (position < length)
        page_row = tl.load(PAGE_ROWS + row) if PAGE_ROWS is not None else row
        page = tl.load(PAGES + page_row * PAGE_ROW + position // 64, valid, 0).to(tl.int64)
        base = CACHE + page * CACHE_PAGE
        d = tl.arange(0, 128)
        h = tl.arange(0, BLOCK_H)
        q = tl.load(Q + row * Q_ROW + h[:, None] * Q_HEAD + d[None, :], h[:, None] < HEADS, 0.0)
        k = tl.load(
            (base[None, :] + (position % 64)[None, :] * 128 + d[:, None]).to(tl.pointer_type(tl.float8e4nv)),
            valid[None, :],
            0.0,
        )
        scale = tl.load(
            (base + 64 * 128 + position % 64 * 4).to(tl.pointer_type(tl.float32)),
            valid,
            0,
        )
        weights = tl.load(WEIGHTS + row * WEIGHT_ROW + h, h < HEADS, 0)
        scores = tl.sum(tl.maximum(tl.dot(q, k), 0.0) * weights[:, None], axis=0) * scale
        tl.store(
            LOGITS + row * CANDIDATE_ROW * 8 + col, tl.where(valid, scores, -float("inf")), col < CANDIDATE_ROW * 8
        )
    else:
        tl.store(LOGITS + row * CANDIDATE_ROW * 8 + col, -float("inf"), col < CANDIDATE_ROW * 8)


def candidate_indexer_logits(query, cache, weights, pages, candidates, lengths, *, page_rows=None):
    """Score the producer's sorted, -1-padded eight-key blocks in FP8 slabs."""
    assert query.shape[-1] == 128
    block_n = 256 if query.shape[0] >= 32 else 64
    logits = torch.empty((query.shape[0], candidates.shape[1] * 8), dtype=torch.float32, device=query.device)
    _candidate_indexer_logits[(query.shape[0], triton.cdiv(logits.shape[1], block_n))](
        query,
        cache,
        weights,
        pages,
        page_rows,
        candidates,
        lengths,
        logits,
        query.stride(0),
        query.stride(1),
        cache.stride(0),
        weights.stride(0),
        pages.stride(0),
        candidates.stride(0),
        query.shape[1],
        max(16, triton.next_power_of_2(query.shape[1])),
        block_n,
        num_warps=4,
    )
    return logits


@triton.jit
def _paged_topk_transform(
    LOGITS,
    RAW,
    PAGES,
    PAGE_ROWS,
    OUT,
    VALID_LENGTHS,
    LENGTHS,
    LOGIT_STRIDE,
    PAGE_STRIDE,
    PAGE_SIZE: tl.constexpr,
    TOPK: tl.constexpr,
    BLOCK: tl.constexpr,
    CANDIDATES,
    CANDIDATE_STRIDE: tl.constexpr,
    SPARSE: tl.constexpr,
):
    row = tl.program_id(0)
    col = tl.arange(0, BLOCK)
    length = tl.load(VALID_LENGTHS + row)
    width = tl.minimum(length, CANDIDATE_STRIDE * 8) if SPARSE else length
    fits = width <= TOPK
    if fits:
        index = col
    else:
        index = tl.load(RAW + row * TOPK + col, col < TOPK, -1)
    valid = (col < TOPK) & (index >= 0) & (index < width)
    score = tl.load(LOGITS + row * LOGIT_STRIDE + index, valid, -float("inf"))
    valid &= tl.abs(score) < float("inf")
    if SPARSE:
        block = tl.load(CANDIDATES + row * CANDIDATE_STRIDE + index // 8, valid, -1)
        index = block * 8 + index % 8
        valid &= (block >= 0) & (index < length)
    # Dense keys and the sorted, unique candidate blocks are already in logical
    # order when every score fits. The selected top-k can contain dummy repeats.
    if not fits:
        index = tl.sort(tl.where(valid, index, 2147483647))
        previous = tl.gather(index, tl.maximum(col - 1, 0), axis=0)
        valid = (index != 2147483647) & ((col == 0) | (index != previous))
    rank = tl.cumsum(valid.to(tl.int32)) - 1
    count = tl.sum(valid.to(tl.int32))
    page_row = tl.load(PAGE_ROWS + row) if PAGE_ROWS is not None else row
    page = tl.load(PAGES + page_row * PAGE_STRIDE + index // PAGE_SIZE, valid, 0)
    tl.store(OUT + row * TOPK + col, -1, (col < TOPK) & (col >= count))
    tl.store(OUT + row * TOPK + rank, page * PAGE_SIZE + index % PAGE_SIZE, valid)
    tl.store(LENGTHS + row, count)


def select_indexer_candidates(logits, lengths, indices):
    topk = indices.shape[1]
    scores = torch.empty((logits.shape[0], triton.cdiv(logits.shape[1], 8)), dtype=torch.float32, device=logits.device)
    # Rows that fit are written directly by _sort_candidates. Report zero valid
    # scores to the top-k operator for those rows.
    topk_lengths = torch.empty_like(lengths)
    _candidate_scores[(logits.shape[0], triton.cdiv(scores.shape[1], 128))](
        logits,
        lengths,
        scores,
        topk_lengths,
        logits.stride(0),
        scores.shape[1],
        topk,
        128,
    )
    if scores.shape[1] > topk:
        ragged_topk(scores, topk_lengths, indices)
    _sort_candidates[(logits.shape[0],)](
        indices,
        lengths,
        topk,
        triton.next_power_of_2(topk),
    )


def select_paged_topk(logits, lengths, pages, slots, selected_lengths, candidate_blocks=None, *, page_rows=None):
    raw = torch.empty_like(slots)
    topk = slots.shape[1]
    if logits.shape[1] > topk:
        valid_lengths = lengths if candidate_blocks is None else lengths.clamp_max(logits.shape[1])
        ragged_topk(logits, valid_lengths, raw)
    _paged_topk_transform[(logits.shape[0],)](
        logits,
        raw,
        pages,
        page_rows,
        slots,
        lengths,
        selected_lengths,
        logits.stride(0),
        pages.stride(0),
        64,
        topk,
        triton.next_power_of_2(topk),
        candidate_blocks,
        candidate_blocks.stride(0) if candidate_blocks is not None else 0,
        candidate_blocks is not None,
    )


@triton.jit
def _gather_indexer_k(
    CACHE,
    REQS,
    LENGTHS,
    STARTS,
    TABLE,
    VALUES,
    SCALES,
    CACHE_STRIDE: tl.constexpr,
    TABLE_STRIDE: tl.constexpr,
    ENTRIES: tl.constexpr,
    RATIO: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    PAGE: tl.constexpr,
):
    batch, block = tl.program_id(0), tl.program_id(1)
    length = tl.load(LENGTHS + batch)
    if block * PAGE < length:
        key = block * PAGE + tl.arange(0, PAGE)
        req = tl.load(REQS + batch).to(tl.int64)
        full = key * RATIO + RATIO - 1
        slot = tl.load(TABLE + req * TABLE_STRIDE + full, key < length, 0).to(tl.int64) // RATIO
        page = slot // ENTRIES
        within = slot % ENTRIES
        base = page * CACHE_STRIDE + (within // PAGE) * PAGE * (HEAD_DIM + 4)
        offset = within % PAGE
        dim = tl.arange(0, HEAD_DIM)
        value = tl.load(
            CACHE.to(tl.pointer_type(tl.float8e4nv)) + base[:, None] + offset[:, None] * HEAD_DIM + dim[None, :],
            key[:, None] < length,
            0.0,
        )
        scale = tl.load(
            CACHE.to(tl.pointer_type(tl.float32)) + (base + PAGE * HEAD_DIM) // 4 + offset,
            key < length,
            0.0,
        )
        out = tl.load(STARTS + batch) + key
        tl.store(VALUES + out[:, None] * HEAD_DIM + dim[None, :], value, key[:, None] < length)
        tl.store(SCALES + out, scale, key < length)


def gather_indexer_k(packed, ratio, reqs, lengths, starts, table, token_capacity, capacity):
    """Gather each prefill request's keys once for DeepGEMM's dense MQA scorer."""
    import torch

    values = torch.empty((token_capacity, 128), dtype=torch.float8_e4m3fn, device=packed.device)
    scales = torch.empty((token_capacity,), dtype=torch.float32, device=packed.device)
    _gather_indexer_k[(reqs.shape[0], triton.cdiv(capacity, 64))](
        packed,
        reqs,
        lengths,
        starts,
        table,
        values,
        scales,
        packed.stride(0),
        table.stride(0),
        64,
        ratio,
        128,
        64,
        num_warps=4,
    )
    return values, scales
