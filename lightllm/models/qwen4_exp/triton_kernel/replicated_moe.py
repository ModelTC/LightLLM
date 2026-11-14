import triton
import triton.language as tl


@triton.jit
def pack_routes(
    Ids,
    Mask,
    Masked,
    Routes,
    Counts,
    TOKENS: tl.constexpr,
    TOPK: tl.constexpr,
    FIRST: tl.constexpr,
    BLOCK: tl.constexpr,
):
    expert = tl.program_id(0)
    offset = tl.arange(0, BLOCK)
    valid = offset < TOKENS * TOPK
    live = tl.load(Mask + offset // TOPK, valid, other=0)
    selected = tl.load(Ids + offset, valid, other=-1)
    selected = tl.where(live, selected, -1)
    hit = valid & (selected == FIRST + expert)
    position = tl.cumsum(hit.to(tl.int32), 0) - 1
    tl.store(Routes + expert * TOKENS * TOPK + position, offset, hit)
    tl.store(Counts + expert, tl.sum(hit.to(tl.int32), 0))
    if expert == 0:
        tl.store(Masked + offset, selected, valid)


@triton.jit
def expert_up(
    Input,
    Weight,
    Output,
    Routes,
    Counts,
    K: tl.constexpr,
    INTER: tl.constexpr,
    TOKENS: tl.constexpr,
    TOPK: tl.constexpr,
    BM: tl.constexpr,
    BN: tl.constexpr,
    BK: tl.constexpr,
):
    expert = tl.program_id(0)
    end = tl.load(Counts + expert)
    start = 0
    offsets = tl.arange(0, 2 * BN)
    col = tl.program_id(1) * BN + offsets % BN
    weight_col = col + (offsets // BN) * INTER
    kk = tl.arange(0, BK)
    for base in range(start, end, BM):
        positions = base + tl.arange(0, BM)
        rows = tl.load(Routes + expert * TOKENS * TOPK + positions, positions < end, other=0).to(tl.int64)
        acc = tl.full((BM, 2 * BN), 0, tl.float32)
        for first in range(tl.cdiv(K, BK)):
            k = first * BK + kk
            x = tl.load(
                Input + (rows[:, None] // TOPK) * K + k[None, :],
                (positions[:, None] < end) & (k[None, :] < K),
                0,
            )
            w = tl.load(
                Weight + (expert * 2 * INTER + weight_col[None, :]).to(tl.int64) * K + k[:, None],
                (col[None, :] < INTER) & (k[:, None] < K),
                0,
            )
            acc = tl.dot(x, w, acc)
        gate, value = tl.split(acc.to(Input.dtype.element_ty).reshape((BM, 2, BN)).permute((0, 2, 1)))
        gate = gate.to(tl.float32)
        gate = (gate / (1.0 + tl.exp(-gate))).to(Input.dtype.element_ty)
        mid = value * gate
        cols = tl.program_id(1) * BN + tl.arange(0, BN)
        tl.store(
            Output + rows[:, None].to(tl.int64) * INTER + cols[None, :],
            mid,
            (positions[:, None] < end) & (cols[None, :] < INTER),
        )


@triton.jit
def expert_down(
    A,
    W,
    Y,
    Routes,
    Counts,
    K: tl.constexpr,
    N: tl.constexpr,
    T: tl.constexpr,
    TOPK: tl.constexpr,
    INPUT_DIV: tl.constexpr,
    BM: tl.constexpr = 16,
    BN: tl.constexpr = 64,
    BK: tl.constexpr = 128,
):
    e = tl.program_id(0)
    col = tl.program_id(1) * BN + tl.arange(0, BN)
    count = tl.load(Counts + e)
    kk = tl.arange(0, BK)
    for start in range(0, count, BM):
        pos = start + tl.arange(0, BM)
        route = tl.load(Routes + e * T * TOPK + pos, pos < count, 0).to(tl.int64)
        acc = tl.full((BM, BN), 0, tl.float32)
        for first in range(tl.cdiv(K, BK)):
            k = first * BK + kk
            a = tl.load(
                A + (route[:, None] // INPUT_DIV) * K + k[None, :], (pos[:, None] < count) & (k[None, :] < K), 0
            )
            w = tl.load(
                W + (e * N + col[None, :]).to(tl.int64) * K + k[:, None], (col[None, :] < N) & (k[:, None] < K), 0
            )
            acc = tl.dot(a, w, acc)
        tl.store(Y + route[:, None] * N + col[None, :], acc, (pos[:, None] < count) & (col[None, :] < N))


@triton.jit
def sum_routes(
    X,
    Weights,
    Ids,
    Y,
    H: tl.constexpr,
    TOPK: tl.constexpr,
    START: tl.constexpr,
    E: tl.constexpr,
    BK: tl.constexpr,
    BH: tl.constexpr = 256,
):
    row = tl.program_id(0)
    h = tl.program_id(1) * BH + tl.arange(0, BH)
    accumulator = tl.full((BH,), 0, tl.float32)
    for route in range(TOPK):
        expert = tl.load(Ids + row * TOPK + route)
        if expert >= START and expert < START + E:
            weight = tl.load(Weights + row * TOPK + route)
            value = tl.load(X + (row * TOPK + route).to(tl.int64) * H + h, h < H, 0).to(tl.float32)
            accumulator += value * weight
    tl.store(Y + row.to(tl.int64) * H + h, accumulator, h < H)


@triton.jit
def sum_partials(
    Input,
    Ids,
    Output,
    H: tl.constexpr,
    TOKENS: tl.constexpr,
    TOPK: tl.constexpr,
    EXPERTS_PER_RANK: tl.constexpr,
    BLOCK: tl.constexpr,
    UNROLL: tl.constexpr,
    Shared=None,
    Gate=None,
):
    row = tl.program_id(0)
    col = tl.program_id(1) * BLOCK + tl.arange(0, BLOCK)
    accumulator = tl.full((BLOCK,), 0, tl.float32)
    seen = tl.full((), 0, tl.int32)
    selected = tl.full((), 0, tl.int32)
    for reverse_route in tl.static_range(TOPK):
        route = TOPK - 1 - reverse_route
        expert = tl.load(Ids + row * TOPK + route)
        rank = tl.maximum(expert, 0) // EXPERTS_PER_RANK
        flag = 1 << rank
        last = (expert >= 0) & ((seen & flag) == 0)
        selected |= tl.where(last, 1 << route, 0)
        seen |= tl.where(last, flag, 0)
    for route in tl.range(TOPK, loop_unroll_factor=UNROLL):
        expert = tl.load(Ids + row * TOPK + route)
        rank = tl.maximum(expert, 0) // EXPERTS_PER_RANK
        last = (selected & (1 << route)) != 0
        value = tl.load(Input + (rank * TOKENS + row).to(tl.int64) * H + col, (col < H) & last, 0)
        accumulator += value.to(tl.float32)
    if Shared is not None:
        routed = accumulator.to(Output.dtype.element_ty).to(tl.float32)
        shared = tl.load(Shared + row.to(tl.int64) * H + col, col < H, 0).to(tl.float32)
        gate = tl.load(Gate + row).to(tl.float32)
        gate = tl.sigmoid(gate).to(Gate.dtype.element_ty).to(tl.float32)
        shared = (shared * gate).to(Shared.dtype.element_ty).to(tl.float32)
        tl.store(Output + row.to(tl.int64) * H + col, routed + shared, col < H)
    else:
        tl.store(Output + row.to(tl.int64) * H + col, accumulator, col < H)
