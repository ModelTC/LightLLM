"""Indexed MLA over native packed rows, including unquantized current prefill KV."""

import triton
import triton.language as tl


@triton.jit
def prepare_sparse_metadata(
    TABLE,
    REQS,
    LENGTHS,
    CU_Q,
    CU_K,
    KS,
    KE,
    LENGTH,
    FRESH_START,
    FRESH_OFFSET,
    RAGGED,
    TABLE_STRIDE: tl.constexpr,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    req = tl.load(REQS + row)
    start_q, end_q = tl.load(CU_Q + row), tl.load(CU_Q + row + 1)
    start_k, length = tl.load(CU_K + row), tl.load(LENGTHS + row)
    ready = length - (end_q - start_q)
    offsets = tl.arange(0, BLOCK)
    for start in range(0, end_q - start_q, BLOCK):
        pos = start + offsets
        mask = pos < end_q - start_q
        tl.store(KS + start_q + pos, start_k, mask)
        tl.store(KE + start_q + pos, start_k + ready + pos + 1, mask)
        tl.store(LENGTH + start_q + pos, tl.maximum(ready + pos + 1, 0), mask)
        tl.store(FRESH_START + start_q + pos, start_k + ready, mask)
        tl.store(FRESH_OFFSET + start_q + pos, start_q - start_k - ready, mask)
    for start in range(0, length, BLOCK):
        pos = start + offsets
        values = tl.load(TABLE + req * TABLE_STRIDE + pos, pos < length, other=-1)
        tl.store(RAGGED + start_k + pos, values, pos < length)


@triton.jit
def indexer_logits(
    Q,
    WEIGHTS,
    KEYS,
    SCALES,
    KS,
    KE,
    RAGGED,
    OUTPUT,
    KEY_STRIDE: tl.constexpr,
    SCALE_STRIDE: tl.constexpr,
    HEADS: tl.constexpr,
    MAX_SEQ: tl.constexpr,
    BLOCK_H: tl.constexpr,
    BLOCK_N: tl.constexpr = 64,
    POOL_SIZE: tl.constexpr = 1,
):
    query, block = tl.program_id(0), tl.program_id(1)
    heads = tl.arange(0, BLOCK_H)
    dims = tl.arange(0, 128)
    q = tl.load(Q + query * HEADS * 128 + heads[:, None] * 128 + dims[None, :], heads[:, None] < HEADS, other=0.0).to(
        tl.bfloat16
    )
    weights = tl.load(WEIGHTS + query * HEADS + heads, heads < HEADS, other=0.0)
    start, end = tl.load(KS + query), tl.load(KE + query)
    for chunk in range(block * BLOCK_N, (end - start) // POOL_SIZE, tl.num_programs(1) * BLOCK_N):
        positions = chunk + tl.arange(0, BLOCK_N)
        valid = positions < (end - start) // POOL_SIZE
        tokens = tl.load(RAGGED + start + positions * POOL_SIZE + POOL_SIZE - 1, valid, other=-1)
        keys = tl.load(KEYS + tokens[None, :] * KEY_STRIDE + dims[:, None], valid[None, :], other=0.0)
        scales = tl.load(SCALES + tokens * SCALE_STRIDE, valid, other=0.0)
        scores = tl.dot(q, keys.to(tl.bfloat16))
        result = tl.sum(tl.maximum(scores, 0) * weights[:, None], 0) * scales
        tl.store(OUTPUT + query * MAX_SEQ + positions, result, valid)


@triton.jit
def indexed_nope(
    Q,
    KV,
    INDICES,
    LOGICAL,
    CURRENT,
    FRESH_START,
    FRESH_OFFSET,
    OUT,
    Q_STRIDE: tl.constexpr,
    Q_HEAD_STRIDE: tl.constexpr,
    KV_STRIDE: tl.constexpr,
    CURRENT_STRIDE: tl.constexpr,
    HEADS: tl.constexpr,
    DIM: tl.constexpr,
    TOPK: tl.constexpr,
    SCALE: tl.constexpr,
    QUANTIZED: tl.constexpr,
    PREFILL: tl.constexpr,
    BLOCK_H: tl.constexpr = 16,
    BLOCK_N: tl.constexpr = 32,
):
    row, head_block = tl.program_id(0), tl.program_id(1)
    heads = head_block * BLOCK_H + tl.arange(0, BLOCK_H)
    dims = tl.arange(0, DIM)
    query = tl.load(Q + row * Q_STRIDE + heads[:, None] * Q_HEAD_STRIDE + dims[None, :], heads[:, None] < HEADS, 0)
    maximum = tl.full((BLOCK_H,), -float("inf"), tl.float32)
    denominator = tl.zeros((BLOCK_H,), tl.float32)
    accum = tl.zeros((BLOCK_H, DIM), tl.float32)
    if QUANTIZED and PREFILL:
        fresh_start = tl.load(FRESH_START + row)
        fresh_offset = tl.load(FRESH_OFFSET + row)
    for start in range(0, TOPK, BLOCK_N):
        slots = start + tl.arange(0, BLOCK_N)
        tokens = tl.load(INDICES + row * TOPK + slots, slots < TOPK, -1).to(tl.int64)
        valid = tokens >= 0
        pointers = KV + tokens[:, None] * KV_STRIDE
        if QUANTIZED:
            exponent = tl.load(pointers + 576 + dims[None, :] // 64, valid[:, None] & (dims[None, :] < 448), 127)
            scale = (exponent.to(tl.uint32) << 23).to(tl.float32, bitcast=True)
            payload = (
                tl.load(
                    pointers.to(tl.pointer_type(tl.float8e4nv)) + dims[None, :],
                    valid[:, None] & (dims[None, :] < 448),
                    0.0,
                ).to(tl.float32)
                * scale
            )
            tail = tl.load(
                (pointers + 448).to(tl.pointer_type(tl.bfloat16)) + dims[None, :] - 448,
                valid[:, None] & (dims[None, :] >= 448),
                0,
            ).to(tl.float32)
            values = tl.where(dims[None, :] < 448, payload, tail)
        else:
            values = tl.load(pointers + dims[None, :], valid[:, None], 0).to(tl.float32)
        if QUANTIZED and PREFILL:
            logical = tl.load(LOGICAL + row * TOPK + slots, slots < TOPK, -1)
            fresh = valid & (logical >= fresh_start)
            current = tl.load(
                CURRENT + (logical + fresh_offset)[:, None] * CURRENT_STRIDE + dims[None, :],
                fresh[:, None],
                0,
            ).to(tl.float32)
            values = tl.where(fresh[:, None], current, values)
        values = values.to(query.dtype)
        scores = tl.dot(query, tl.trans(values)) * (SCALE * 1.4426950408889634)
        scores = tl.where(valid[None, :], scores, -float("inf"))
        next_max = tl.maximum(maximum, tl.max(scores, 1))
        safe_max = tl.where(next_max == -float("inf"), 0.0, next_max)
        correction = tl.exp2(maximum - safe_max)
        probabilities = tl.where(valid[None, :], tl.exp2(scores - safe_max[:, None]), 0.0)
        denominator = denominator * correction + tl.sum(probabilities, 1)
        accum = accum * correction[:, None] + tl.dot(probabilities.to(values.dtype), values)
        maximum = next_max
    result = accum / tl.where(denominator > 0, denominator, 1.0)[:, None]
    tl.store(OUT + row * HEADS * DIM + heads[:, None] * DIM + dims[None, :], result, heads[:, None] < HEADS)


@triton.jit
def indexed_mla(
    Q,
    PAYLOAD,
    SCALE,
    ROPE,
    INDICES,
    RAGGED_INDICES,
    CURRENT_KV,
    FRESH_START,
    FRESH_OFFSET,
    OUT,
    Q_STRIDE: tl.constexpr,
    Q_HEAD_STRIDE: tl.constexpr,
    KV_STRIDE: tl.constexpr,
    SCALE_STRIDE: tl.constexpr,
    ROPE_STRIDE: tl.constexpr,
    CURRENT_STRIDE: tl.constexpr,
    HEADS: tl.constexpr,
    TOPK: tl.constexpr,
    SOFTMAX_SCALE: tl.constexpr,
    QUANTIZED: tl.constexpr,
    PREFILL: tl.constexpr,
    BLOCK_H: tl.constexpr = 16,
    BLOCK_N: tl.constexpr = 32,
):
    query, head_block = tl.program_id(0), tl.program_id(1)
    heads = head_block * BLOCK_H + tl.arange(0, BLOCK_H)
    dims = tl.arange(0, 512)
    rotary_dims = tl.arange(0, 64)
    q_nope = tl.load(
        Q + query * Q_STRIDE + heads[:, None] * Q_HEAD_STRIDE + dims[None, :], heads[:, None] < HEADS, other=0
    )
    q_rope = tl.load(
        Q + query * Q_STRIDE + heads[:, None] * Q_HEAD_STRIDE + 512 + rotary_dims[None, :],
        heads[:, None] < HEADS,
        other=0,
    )
    maximum = tl.full((BLOCK_H,), -float("inf"), tl.float32)
    denominator = tl.zeros((BLOCK_H,), tl.float32)
    accum = tl.zeros((BLOCK_H, 512), tl.float32)
    if QUANTIZED and PREFILL:
        fresh_start = tl.load(FRESH_START + query)
        fresh_offset = tl.load(FRESH_OFFSET + query)
    for start in range(0, TOPK, BLOCK_N):
        positions = start + tl.arange(0, BLOCK_N)
        tokens = tl.load(INDICES + query * TOPK + positions, positions < TOPK, other=-1)
        valid = tokens >= 0
        nope = tl.load(PAYLOAD + tokens[:, None] * KV_STRIDE + dims[None, :], valid[:, None], other=0.0).to(tl.float32)
        if QUANTIZED:
            scales = tl.load(SCALE + tokens[:, None] * SCALE_STRIDE + dims[None, :] // 128, valid[:, None], other=1)
            nope *= scales
        rope = tl.load(ROPE + tokens[:, None] * ROPE_STRIDE + rotary_dims[None, :], valid[:, None], other=0).to(
            tl.float32
        )
        if QUANTIZED and PREFILL:
            logical = tl.load(RAGGED_INDICES + query * TOPK + positions, positions < TOPK, other=-1)
            fresh = valid & (logical >= fresh_start)
            current_row = logical + fresh_offset
            current_nope = tl.load(
                CURRENT_KV + current_row[:, None] * CURRENT_STRIDE + dims[None, :], fresh[:, None], other=0
            ).to(tl.float32)
            current_rope = tl.load(
                CURRENT_KV + current_row[:, None] * CURRENT_STRIDE + 512 + rotary_dims[None, :], fresh[:, None], other=0
            ).to(tl.float32)
            nope = tl.where(fresh[:, None], current_nope, nope)
            rope = tl.where(fresh[:, None], current_rope, rope)
        value = nope.to(q_nope.dtype)
        scores = tl.dot(q_nope, tl.trans(value)) + tl.dot(q_rope, tl.trans(rope.to(q_rope.dtype)))
        scores = tl.where(valid[None, :], scores * (SOFTMAX_SCALE * 1.4426950408889634), -float("inf"))
        next_max = tl.maximum(maximum, tl.max(scores, 1))
        correction = tl.exp2(maximum - next_max)
        probabilities = tl.exp2(scores - next_max[:, None])
        denominator = denominator * correction + tl.sum(probabilities, 1)
        accum = accum * correction[:, None] + tl.dot(probabilities.to(value.dtype), value)
        maximum = next_max
    output = accum / denominator[:, None]
    tl.store(OUT + query * HEADS * 512 + heads[:, None] * 512 + dims[None, :], output, heads[:, None] < HEADS)
