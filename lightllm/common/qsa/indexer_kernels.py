import triton
import triton.language as tl


@triton.jit
def _rotate(X, Cos, Sin, Output, HEADS: tl.constexpr, D: tl.constexpr, R: tl.constexpr, BLOCK: tl.constexpr):
    row = tl.program_id(0).to(tl.int64)
    head = tl.program_id(1)
    col = tl.arange(0, BLOCK)
    x = tl.load(X + (row * HEADS + head) * D + col, col < D, other=0)
    if R > 0:
        partner = tl.where(col < R // 2, col + R // 2, col - R // 2)
        other = tl.load(X + (row * HEADS + head) * D + partner, col < R, other=0).to(tl.float32)
        other = tl.where(col < R // 2, -other, other)
        cos = tl.load(Cos + row * R + col, col < R, other=0).to(tl.float32)
        sin = tl.load(Sin + row * R + col, col < R, other=0).to(tl.float32)
        left = (x.to(tl.float32) * cos).to(X.dtype.element_ty).to(tl.float32)
        right = (other * sin).to(X.dtype.element_ty).to(tl.float32)
        x = tl.where(col < R, left + right, x.to(tl.float32))
    tl.store(Output + (row * HEADS + head) * D + col, x, col < D)


@triton.jit
def _pool(
    Raw,
    Indices,
    Lengths,
    First,
    Norm,
    Cos,
    Sin,
    Cache,
    RequestRows,
    MAPPED_ROWS: tl.constexpr,
    PHYSICAL_CACHE: tl.constexpr,
    CAPACITY: tl.constexpr,
    BLOCKS: tl.constexpr,
    D: tl.constexpr,
    R: tl.constexpr,
    RATIO: tl.constexpr,
    EPS: tl.constexpr,
    BD: tl.constexpr,
    BR: tl.constexpr,
):
    batch_row = tl.program_id(0)
    request = tl.load(RequestRows + batch_row).to(tl.int64) if MAPPED_ROWS else batch_row.to(tl.int64)
    block = tl.load(First + batch_row) + tl.program_id(1)
    count = tl.load(Lengths + batch_row)
    if (request >= 0) & (block < count // RATIO):
        token = tl.arange(0, BR)
        col = tl.arange(0, BD)
        slots = tl.load(Indices + request * CAPACITY + block * RATIO + token, token < RATIO, other=0).to(tl.int64)
        values = tl.load(
            Raw + slots[:, None] * D + col[None, :], (token[:, None] < RATIO) & (col[None, :] < D), other=0
        ).to(tl.float32)
        pooled = (tl.sum(values, 0) / RATIO).to(Raw.dtype.element_ty).to(tl.float32)
        inv = tl.rsqrt(tl.sum(pooled * pooled, 0) / D + EPS)
        weight = tl.load(Norm + col, col < D, other=0).to(tl.float32)
        normalized = ((pooled * inv) * (1 + weight)).to(Raw.dtype.element_ty).to(tl.float32)
        if R > 0:
            slot = tl.load(Indices + request * CAPACITY + block * RATIO).to(tl.int64)
            partner = tl.where(col < R // 2, col + R // 2, col - R // 2)

            partner = tl.minimum(tl.maximum(partner, 0), BD - 1)
            other = tl.gather(normalized, partner, 0)
            other = tl.where(col < R // 2, -other, other)
            cos = tl.load(Cos + slot * R + col, col < R, other=0).to(tl.float32)
            sin = tl.load(Sin + slot * R + col, col < R, other=0).to(tl.float32)
            left = (normalized * cos).to(Raw.dtype.element_ty).to(tl.float32)
            right = (other * sin).to(Raw.dtype.element_ty).to(tl.float32)
            normalized = tl.where(col < R, left + right, normalized)
        if PHYSICAL_CACHE:
            destination = tl.load(Indices + request * CAPACITY + block * RATIO + RATIO - 1).to(tl.int64)
        else:
            destination = request * BLOCKS + block
        tl.store(Cache + destination * D + col, normalized, col < D)


@triton.jit
def _scores(
    Query,
    Cache,
    Requests,
    Lengths,
    Indices,
    Output,
    QUERY_START: tl.constexpr,
    HEADS: tl.constexpr,
    D: tl.constexpr,
    BLOCKS: tl.constexpr,
    CAPACITY: tl.constexpr,
    PHYSICAL_CACHE: tl.constexpr,
    RATIO: tl.constexpr,
    BH: tl.constexpr,
    BD: tl.constexpr,
    BN: tl.constexpr,
    MIN_BLOCKS: tl.constexpr = 0,
):
    row = tl.program_id(0)
    query_row = row + QUERY_START
    request = tl.load(Requests + query_row).to(tl.int64)
    visible = tl.load(Lengths + query_row) // RATIO
    block = tl.program_id(1) * BN + tl.arange(0, BN)

    scores = tl.full((BN,), -float("inf"), tl.float32)
    if (tl.program_id(1) * BN < visible) & (visible >= MIN_BLOCKS):
        head = tl.arange(0, BH)
        col = tl.arange(0, BD)
        q = tl.load(
            Query + (query_row.to(tl.int64) * HEADS + head[:, None]) * D + col[None, :],
            (head[:, None] < HEADS) & (col[None, :] < D),
            other=0,
        )
        if PHYSICAL_CACHE:
            key_rows = tl.load(
                Indices + request * CAPACITY + block * RATIO + RATIO - 1,
                (block < visible) & (block < BLOCKS),
                other=0,
            ).to(tl.int64)
        else:
            key_rows = request * BLOCKS + block
        k = tl.load(
            Cache + key_rows[None, :] * D + col[:, None],
            (block[None, :] < visible) & (col[:, None] < D),
            other=0,
        )
        logits = tl.dot(q, k, input_precision="ieee")
        logits = tl.where(head[:, None] < HEADS, tl.maximum(logits, 0), 0)
        scores = tl.sum(logits, 0) * (D**-0.5)
        scores = tl.where(block < visible, scores, -float("inf"))
    tl.store(
        Output + row.to(tl.int64) * BLOCKS + block,
        scores,
        (block < BLOCKS) & ((MIN_BLOCKS == 0) | (visible >= MIN_BLOCKS)),
    )


@triton.jit
def _expand(
    Selected,
    Requests,
    Lengths,
    Indices,
    Output,
    QUERY_START: tl.constexpr,
    TOPK: tl.constexpr,
    BLOCK_BUDGET: tl.constexpr,
    CAPACITY: tl.constexpr,
    RATIO: tl.constexpr,
    WIDTH: tl.constexpr,
    B: tl.constexpr,
):
    row = tl.program_id(0)
    query_row = row + QUERY_START
    request = tl.load(Requests + query_row).to(tl.int64)
    visible = tl.load(Lengths + query_row)
    complete = visible // RATIO
    chosen = tl.minimum(complete, BLOCK_BUDGET)
    col = tl.arange(0, B)
    rank = col // RATIO
    block = tl.full((B,), 0, tl.int64)
    if TOPK > 0:
        block = tl.load(Selected + row.to(tl.int64) * TOPK + rank, (rank < chosen) & (col < WIDTH), other=0)
    full = col < chosen * RATIO
    tail_col = col - chosen * RATIO
    tail = (tail_col >= 0) & (tail_col < visible % RATIO)
    position = tl.where(full, block * RATIO + col % RATIO, complete * RATIO + tail_col)
    slot = tl.load(Indices + request * CAPACITY + position, (col < WIDTH) & (full | tail), other=-1)
    tl.store(Output + query_row.to(tl.int64) * WIDTH + col, slot, col < WIDTH)
