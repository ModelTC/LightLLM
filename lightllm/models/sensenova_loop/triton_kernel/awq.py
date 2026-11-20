"""Packed INT4 embedding lookup for SenseNova Loop AWQ exports."""

import triton
import triton.language as tl


@triton.jit
def int4_embedding_kernel(
    Ids,
    Weight,
    Scales,
    Zeros,
    Out,
    DIM: tl.constexpr,
    VOCAB_START: tl.constexpr,
    VOCAB_END: tl.constexpr,
    VOCAB: tl.constexpr,
    GROUP: tl.constexpr,
    BLOCK: tl.constexpr,
):
    token = tl.program_id(0)
    global_row = tl.load(Ids + token)
    row = global_row - VOCAB_START
    dims = tl.program_id(1) * BLOCK + tl.arange(0, BLOCK)
    valid = (global_row >= VOCAB_START) & (global_row < VOCAB_END) & (row >= 0) & (row < VOCAB) & (dims < DIM)
    packed = tl.load(Weight + row * (DIM // 8) + dims // 8, valid, 0)
    quantized = (packed >> ((dims % 8) * 4)) & 15
    scale = tl.load(Scales + row * (DIM // GROUP) + dims // GROUP, valid, 0).to(tl.float32)
    zero = tl.load(Zeros + row * (DIM // GROUP) + dims // GROUP, valid, 0).to(tl.float32)
    value = (quantized.to(tl.float32) - zero) * scale
    tl.store(Out + token * DIM + dims, value, dims < DIM)
