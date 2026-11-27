# Copyright 2023-2024 SGLang Team
# SPDX-License-Identifier: Apache-2.0

import torch
import triton
import triton.language as tl

from lightllm.utils.custom_kernel_utis import get_pdl_launch_kwargs


@triton.jit
def _rotate_and_quantize_indexer_queries_kernel(
    queries,
    head_weights,
    rope_coefficients,
    position_ids,
    query_fp8_out,
    scaled_head_weights_out,
    num_query_rows,
    NUM_QUERY_HEADS: tl.constexpr,
    HEAD_WEIGHT_SCALE: tl.constexpr,
    QUERY_ROWS_PER_BLOCK: tl.constexpr,
    USE_PDL: tl.constexpr,
):
    QUERY_HEAD_DIM: tl.constexpr = 128
    ROPE_DIM: tl.constexpr = 64
    NUM_COMPLEX_PAIRS: tl.constexpr = QUERY_HEAD_DIM // 2
    FIRST_ROPE_PAIR: tl.constexpr = (QUERY_HEAD_DIM - ROPE_DIM) // 2

    query_row = tl.program_id(0) * QUERY_ROWS_PER_BLOCK + tl.arange(0, QUERY_ROWS_PER_BLOCK)
    valid_query_row = query_row < num_query_rows
    head_column = tl.arange(0, QUERY_HEAD_DIM)
    if USE_PDL:
        tl.extra.cuda.gdc_wait()
    query_values = tl.load(
        queries + query_row[:, None] * QUERY_HEAD_DIM + head_column[None, :], mask=valid_query_row[:, None], other=0
    ).to(tl.float32)
    query_real, query_imag = tl.split(tl.reshape(query_values, (QUERY_ROWS_PER_BLOCK, NUM_COMPLEX_PAIRS, 2)))
    pair_index = tl.arange(0, NUM_COMPLEX_PAIRS)
    token_position = tl.load(position_ids + query_row // NUM_QUERY_HEADS, mask=valid_query_row, other=0)
    rope_frequency_ptrs = (
        rope_coefficients + token_position[:, None] * ROPE_DIM + (pair_index[None, :] - FIRST_ROPE_PAIR) * 2
    )
    rope_mask = valid_query_row[:, None] & (pair_index[None, :] >= FIRST_ROPE_PAIR)
    rope_cos = tl.load(rope_frequency_ptrs, mask=rope_mask, other=1)
    rope_sin = tl.load(rope_frequency_ptrs + 1, mask=rope_mask, other=0)
    rotated_query = tl.reshape(
        tl.join(query_real * rope_cos - query_imag * rope_sin, query_real * rope_sin + query_imag * rope_cos),
        (QUERY_ROWS_PER_BLOCK, QUERY_HEAD_DIM),
    )
    if USE_PDL:
        tl.extra.cuda.gdc_launch_dependents()
    rotated_query = rotated_query.to(tl.bfloat16).to(tl.float32)
    query_fp8_scale = tl.maximum(tl.max(tl.abs(rotated_query), axis=1), 1e-4) / 448
    query_fp8_scale = tl.exp2(tl.ceil(tl.log2(query_fp8_scale)))
    tl.store(
        query_fp8_out + query_row[:, None] * QUERY_HEAD_DIM + head_column[None, :],
        tl.clamp(rotated_query / query_fp8_scale[:, None], -448, 448),
        mask=valid_query_row[:, None],
    )
    head_weight = tl.load(head_weights + query_row, mask=valid_query_row, other=0).to(tl.float32)
    tl.store(
        scaled_head_weights_out + query_row,
        head_weight * HEAD_WEIGHT_SCALE * query_fp8_scale,
        mask=valid_query_row,
    )


@torch.no_grad()
def rotate_and_quantize_indexer_queries(
    queries, head_weights, head_weight_scale, freqs_cis, position_ids, *, query_fp8_out, scaled_head_weights_out
):
    """Apply RoPE and FP8 quantization to contiguous BF16 queries with shape [tokens, heads, 128].

    Fold each query head's FP8 scale and head_weight_scale into FP32 weights shaped [tokens, heads, 1].
    """
    num_tokens, num_heads = queries.shape[:2]
    num_query_rows = num_tokens * num_heads
    if num_query_rows:
        query_rows_per_block = min(32, triton.next_power_of_2(max(1, num_query_rows // 1024)))
        _rotate_and_quantize_indexer_queries_kernel[(triton.cdiv(num_query_rows, query_rows_per_block),)](
            queries,
            head_weights,
            torch.view_as_real(freqs_cis).flatten(-2),
            position_ids,
            query_fp8_out,
            scaled_head_weights_out,
            num_query_rows,
            NUM_QUERY_HEADS=num_heads,
            HEAD_WEIGHT_SCALE=float(head_weight_scale),
            QUERY_ROWS_PER_BLOCK=query_rows_per_block,
            num_warps=1,
            num_stages=1,
            **get_pdl_launch_kwargs(queries.device),
        )
    return query_fp8_out, scaled_head_weights_out


@triton.jit
def _rotate_and_cache_indexer_keys_kernel(
    keys,
    full_kv_slots,
    position_ids,
    cached_fp8_keys,
    cached_key_scales,
    key_row_stride,
    key_column_stride,
    rope_coefficients,
    FP8_MIN: tl.constexpr,
    FP8_MAX: tl.constexpr,
    AMAX_FLOOR: tl.constexpr,
    KEY_HEAD_DIM: tl.constexpr,
    COMPRESSION_RATIO: tl.constexpr,
    KEYS_PER_PAGE: tl.constexpr,
    KEY_CACHE_PAGE_STRIDE_BYTES: tl.constexpr,
):
    ROPE_DIM: tl.constexpr = 64
    FIRST_ROPE_PAIR: tl.constexpr = (KEY_HEAD_DIM - ROPE_DIM) // 2

    token_index = tl.program_id(0)
    token_position = tl.load(position_ids + token_index)
    if (token_position + 1) % COMPRESSION_RATIO != 0:
        return

    full_kv_slot = tl.load(full_kv_slots + token_index).to(tl.int64)
    compressed_key_slot = full_kv_slot // COMPRESSION_RATIO
    cache_page = compressed_key_slot // KEYS_PER_PAGE
    key_offset_in_page = compressed_key_slot % KEYS_PER_PAGE

    head_column = tl.arange(0, KEY_HEAD_DIM)
    key_values = tl.load(keys + token_index * key_row_stride + head_column * key_column_stride).to(tl.float32)
    pair_index = tl.arange(0, KEY_HEAD_DIM // 2)
    key_real, key_imag = tl.split(tl.reshape(key_values, (KEY_HEAD_DIM // 2, 2)))
    compression_group_start = token_position - COMPRESSION_RATIO + 1
    rope_frequency_ptrs = rope_coefficients + compression_group_start * ROPE_DIM + (pair_index - FIRST_ROPE_PAIR) * 2
    rope_cos = tl.load(rope_frequency_ptrs, pair_index >= FIRST_ROPE_PAIR, 1.0)
    rope_sin = tl.load(rope_frequency_ptrs + 1, pair_index >= FIRST_ROPE_PAIR, 0.0)
    rotated_key = tl.reshape(
        tl.join(key_real * rope_cos - key_imag * rope_sin, key_real * rope_sin + key_imag * rope_cos),
        (KEY_HEAD_DIM,),
    )
    rotated_key = rotated_key.to(tl.bfloat16).to(tl.float32)
    key_amax = tl.maximum(tl.max(tl.abs(rotated_key), axis=0), AMAX_FLOOR)
    key_fp8_scale = key_amax / FP8_MAX
    key_fp8_scale = tl.exp2(tl.ceil(tl.log2(key_fp8_scale)))
    quantized_key = tl.clamp(rotated_key / key_fp8_scale, min=FP8_MIN, max=FP8_MAX).to(tl.float8e4nv)

    key_byte_offset = cache_page * KEY_CACHE_PAGE_STRIDE_BYTES + key_offset_in_page * KEY_HEAD_DIM
    tl.store(cached_fp8_keys + key_byte_offset + head_column, quantized_key)
    scale_element_offset = (
        cache_page * KEY_CACHE_PAGE_STRIDE_BYTES + KEYS_PER_PAGE * KEY_HEAD_DIM
    ) // 4 + key_offset_in_page
    tl.store(cached_key_scales + scale_element_offset, key_fp8_scale)


@torch.no_grad()
def rotate_and_cache_indexer_keys(
    keys: torch.Tensor,
    freqs_cis: torch.Tensor,
    full_kv_slots: torch.Tensor,
    position_ids: torch.Tensor,
    packed_key_cache: torch.Tensor,
    compression_ratio: int,
    keys_per_page: int,
):
    """Write FP8 keys for completed compression groups into the indexer cache.

    The input slots address uncompressed KV tokens. RoPE uses the start position of each group.
    Each page stores FP8 key bytes followed by one FP32 power-of-two scale per key.
    """
    num_tokens = full_kv_slots.shape[0]
    if num_tokens == 0:
        return
    key_head_dim = keys.shape[-1]
    keys = keys.reshape(-1, key_head_dim)
    _rotate_and_cache_indexer_keys_kernel[(num_tokens,)](
        keys,
        full_kv_slots,
        position_ids,
        packed_key_cache.view(torch.float8_e4m3fn),
        packed_key_cache.view(torch.float32),
        keys.stride(0),
        keys.stride(1),
        torch.view_as_real(freqs_cis).flatten(-2),
        FP8_MIN=torch.finfo(torch.float8_e4m3fn).min,
        FP8_MAX=torch.finfo(torch.float8_e4m3fn).max,
        AMAX_FLOOR=1e-4,
        KEY_HEAD_DIM=key_head_dim,
        COMPRESSION_RATIO=compression_ratio,
        KEYS_PER_PAGE=keys_per_page,
        KEY_CACHE_PAGE_STRIDE_BYTES=packed_key_cache.stride(0),
        num_warps=1,
        num_stages=1,
        enable_fp_fusion=False,
    )
