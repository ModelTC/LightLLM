"""Kernels for the Loop AWQ format with runtime input scales and packed embeddings."""

import triton
import triton.language as tl


@triton.jit
def scale_input_kernel(X, Scale, Out, N: tl.constexpr, K: tl.constexpr, BLOCK: tl.constexpr):
    offsets = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    x = tl.load(X + offsets, offsets < N, 0).to(tl.float32)
    scale = tl.load(Scale + offsets % K).to(tl.float32)
    tl.store(Out + offsets, x / scale, offsets < N)


@triton.jit
def int4_embedding_kernel(
    Ids, Weight, Scales, Zeros, Out, DIM: tl.constexpr, VOCAB: tl.constexpr, GROUP: tl.constexpr, BLOCK: tl.constexpr
):
    token = tl.program_id(0)
    row = tl.load(Ids + token)
    dims = tl.program_id(1) * BLOCK + tl.arange(0, BLOCK)
    valid = (row >= 0) & (row < VOCAB) & (dims < DIM)
    packed = tl.load(Weight + row * (DIM // 8) + dims // 8, valid, 0)
    quantized = (packed >> ((dims % 8) * 4)) & 15
    scale = tl.load(Scales + row * (DIM // GROUP) + dims // GROUP, valid, 0).to(tl.float32)
    zero = tl.load(Zeros + row * (DIM // GROUP) + dims // GROUP, valid, 0).to(tl.float32)
    value = (quantized.to(tl.float32) - zero) * scale
    tl.store(Out + token * DIM + dims, value, dims < DIM)
