import triton
import triton.language as tl


@triton.jit
def request_for_token(token, Starts, valid, REQUESTS: tl.constexpr, SEARCH_STEPS: tl.constexpr):
    lo = tl.full(token.shape, 0, tl.int32)
    hi = tl.full(token.shape, REQUESTS, tl.int32)
    for _ in tl.static_range(SEARCH_STEPS):
        mid = (lo + hi) // 2
        start = tl.load(Starts + mid, (mid < REQUESTS) & valid, other=2147483647)
        right = start <= token
        active = lo < hi
        lo = tl.where(active & right, mid + 1, lo)
        hi = tl.where(active & ~right, mid, hi)
    request = tl.maximum(lo - 1, 0)
    start = tl.load(Starts + request, valid, other=0)
    return request, start


@triton.jit
def _hash(
    Tokens,
    Starts,
    History,
    Multipliers,
    Sizes,
    Offsets,
    Output,
    TOKENS: tl.constexpr,
    REQUESTS: tl.constexpr,
    SEARCH_STEPS: tl.constexpr,
    CONTEXT: tl.constexpr,
    HEADS: tl.constexpr,
    HEADS_PER_NGRAM: tl.constexpr,
    EOS: tl.constexpr,
    BLOCK_T: tl.constexpr,
    BLOCK_H: tl.constexpr,
):
    token = tl.program_id(0) * BLOCK_T + tl.arange(0, BLOCK_T)
    request, start = request_for_token(token, Starts, token < TOKENS, REQUESTS, SEARCH_STEPS)
    position = token - start
    head = tl.arange(0, BLOCK_H)
    ngram_order = head // HEADS_PER_NGRAM + 2
    current = tl.load(Tokens + token, token < TOKENS, other=EOS).to(tl.int64)
    multiplier = tl.load(Multipliers)
    mixed = tl.broadcast_to((current * multiplier)[:, None], (BLOCK_T, BLOCK_H))
    alive = tl.full((BLOCK_T,), True, tl.int1)
    for shift in tl.static_range(1, CONTEXT + 1):
        in_chunk = position >= shift
        source = tl.load(Tokens + token - shift, (token < TOKENS) & in_chunk, other=EOS).to(tl.int64)
        from_history = tl.load(
            History + request * CONTEXT + CONTEXT + position - shift,
            (token < TOKENS) & ~in_chunk,
            other=EOS,
        )
        source = tl.where(in_chunk, source, from_history)
        shifted = tl.where(alive, source, EOS)
        multiplier = tl.load(Multipliers + shift)
        mixed = mixed ^ tl.where(ngram_order[None, :] > shift, (shifted * multiplier)[:, None], 0)

        alive = alive & (source != EOS)
    sizes = tl.load(Sizes + head, head < HEADS, other=1)
    offsets = tl.load(Offsets + head, head < HEADS, other=0)
    remainder = mixed % sizes[None, :]
    remainder = tl.where(remainder < 0, remainder + sizes[None, :], remainder)
    tl.store(
        Output + token[:, None] * HEADS + head[None, :],
        remainder + offsets[None, :],
        (token[:, None] < TOKENS) & (head[None, :] < HEADS),
    )


@triton.jit
def _history(
    Tokens,
    Starts,
    Previous,
    Accepted,
    Output,
    CONTEXT: tl.constexpr,
    USE_ACCEPTED: tl.constexpr,
    EOS: tl.constexpr,
    BLOCK: tl.constexpr,
):
    request = tl.program_id(0)
    col = tl.arange(0, BLOCK)
    start = tl.load(Starts + request)
    end = tl.load(Starts + request + 1)
    count = tl.load(Accepted + request) if USE_ACCEPTED else end - start
    source = count - CONTEXT + col
    old = tl.load(Previous + request * CONTEXT + count + col, (col < CONTEXT) & (source < 0), other=EOS)
    new = tl.load(Tokens + start + source, (col < CONTEXT) & (source >= 0), other=EOS).to(tl.int64)
    tl.store(Output + request * CONTEXT + col, tl.where(source >= 0, new, old), col < CONTEXT)
