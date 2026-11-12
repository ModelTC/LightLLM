import triton
import triton.language as tl

from .kernels import request_for_token


@triton.jit
def _gate(Key, Query, Value, Mask, Output, D: tl.constexpr, C: tl.constexpr, HAS_MASK: tl.constexpr, B: tl.constexpr):
    token = tl.program_id(0).to(tl.int64)
    branch = tl.program_id(1)
    col = tl.arange(0, B)
    offset = token * C * D + branch * D + col
    key = tl.load(Key + offset, col < D, other=0).to(tl.float32)
    query = tl.load(Query + offset, col < D, other=0).to(tl.float32)

    product = (key * query).to(Key.dtype.element_ty).to(tl.float32)
    dot = tl.sum(product, 0).to(Key.dtype.element_ty).to(tl.float32)
    dot = (dot / tl.sqrt(D * 1.0)).to(Key.dtype.element_ty).to(tl.float32)
    magnitude = tl.maximum(tl.abs(dot), 1e-6).to(Key.dtype.element_ty).to(tl.float32)
    root = tl.sqrt(magnitude).to(Key.dtype.element_ty).to(tl.float32)
    signed = tl.where(dot > 0, root, tl.where(dot < 0, -root, 0.0))
    gate = tl.sigmoid(signed).to(Key.dtype.element_ty).to(tl.float32)
    value = tl.load(Value + token * D + col, col < D, other=0).to(tl.float32)
    result = value * gate
    if HAS_MASK:
        keep = tl.load(Mask + token)
        result = tl.where(keep, result, 0.0)
    tl.store(Output + offset, result, col < D)


@triton.jit
def _conv(
    Input,
    Gated,
    Weight,
    Starts,
    History,
    Output,
    WIDTH: tl.constexpr,
    KERNEL: tl.constexpr,
    DILATION: tl.constexpr,
    REQUESTS: tl.constexpr,
    SEARCH_STEPS: tl.constexpr,
    BLOCK: tl.constexpr,
):
    token = tl.program_id(0).to(tl.int64)
    col = tl.program_id(1) * BLOCK + tl.arange(0, BLOCK)
    request, start = request_for_token(token, Starts, True, REQUESTS, SEARCH_STEPS)
    context: tl.constexpr = (KERNEL - 1) * DILATION
    position = token - start
    acc = tl.full((BLOCK,), 0, tl.float32)
    for tap in tl.static_range(KERNEL):
        source = position - context + tap * DILATION
        current = tl.load(Input + (start + source) * WIDTH + col, (col < WIDTH) & (source >= 0), other=0)
        previous = tl.load(
            History + (request.to(tl.int64) * context + context + source) * WIDTH + col,
            (col < WIDTH) & (source < 0),
            other=0,
        )
        value = tl.where(source >= 0, current, previous).to(tl.float32)
        weight = tl.load(Weight + col * KERNEL + tap, col < WIDTH, other=0).to(tl.float32)
        acc = acc + value * weight

    conv = acc.to(Input.dtype.element_ty).to(tl.float32)
    activated = (conv * tl.sigmoid(conv)).to(Input.dtype.element_ty).to(tl.float32)
    gated = tl.load(Gated + token * WIDTH + col, col < WIDTH, other=0).to(tl.float32)
    tl.store(Output + token * WIDTH + col, gated + activated, col < WIDTH)


@triton.jit
def _conv_history(
    Input,
    Starts,
    Previous,
    Accepted,
    Output,
    WIDTH: tl.constexpr,
    CONTEXT: tl.constexpr,
    USE_ACCEPTED: tl.constexpr,
    BLOCK: tl.constexpr,
):
    request = tl.program_id(0).to(tl.int64)
    flat = tl.program_id(1) * BLOCK + tl.arange(0, BLOCK)
    time = flat // WIDTH
    col = flat % WIDTH
    start = tl.load(Starts + request)
    end = tl.load(Starts + request + 1)
    count = tl.load(Accepted + request) if USE_ACCEPTED else end - start
    source = count - CONTEXT + time
    old = tl.load(
        Previous + (request * CONTEXT + count + time) * WIDTH + col,
        (time < CONTEXT) & (source < 0),
        other=0,
    )
    new = tl.load(
        Input + (start + source.to(tl.int64)) * WIDTH + col,
        (time < CONTEXT) & (source >= 0),
        other=0,
    )
    tl.store(Output + request * CONTEXT * WIDTH + flat, tl.where(source >= 0, new, old), time < CONTEXT)
