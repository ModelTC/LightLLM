import triton
import triton.language as tl


@triton.jit
def _attention(
    Q,
    K,
    V,
    Indices,
    Lengths,
    Partial,
    LSE,
    Output,
    Q_HEADS: tl.constexpr,
    KV_HEADS: tl.constexpr,
    KS: tl.constexpr,
    KH: tl.constexpr,
    KD: tl.constexpr,
    VS: tl.constexpr,
    VH: tl.constexpr,
    VD: tl.constexpr,
    GROUP: tl.constexpr,
    D: tl.constexpr,
    DV: tl.constexpr,
    SELECTED: tl.constexpr,
    SPLITS: tl.constexpr,
    SCALE: tl.constexpr,
    BG: tl.constexpr,
    BD: tl.constexpr,
    BV: tl.constexpr,
    BN: tl.constexpr,
    SHORT_DIRECT: tl.constexpr,
    SKIP_EMPTY: tl.constexpr,
):
    row = tl.program_id(0).to(tl.int64)
    kv_head = tl.program_id(1)
    split = tl.program_id(2)
    limit = SELECTED
    if Lengths is not None:
        limit = tl.minimum(SELECTED, tl.maximum(0, tl.load(Lengths + row)))
    if (not SHORT_DIRECT or limit > BN or split == 0) and (
        not SKIP_EMPTY or SPLITS == 1 or split * BN < limit or (SHORT_DIRECT and limit <= BN and split == 0)
    ):
        group = tl.arange(0, BG)
        head = kv_head * GROUP + group
        dim = tl.arange(0, BD)
        value_dim = tl.arange(0, BV)
        q = tl.load(
            Q + (row * Q_HEADS + head[:, None]) * D + dim[None, :],
            (group[:, None] < GROUP) & (dim[None, :] < D),
            other=0,
        )
        maximum = tl.full((BG,), -float("inf"), tl.float32)
        denominator = tl.full((BG,), 0, tl.float32)
        acc = tl.full((BG, BV), 0, tl.float32)
        for start in tl.range(split * BN, limit, SPLITS * BN):
            selected = start + tl.arange(0, BN)
            slot = tl.load(Indices + row * SELECTED + selected, selected < limit, other=-1).to(tl.int64)
            valid = (selected < limit) & (slot >= 0)
            key = tl.load(
                K + slot[None, :] * KS + kv_head * KH + dim[:, None] * KD, valid[None, :] & (dim[:, None] < D), other=0
            )
            score = tl.dot(q, key, input_precision="ieee") * SCALE
            score = tl.where(valid[None, :], score, -float("inf"))
            new_max = tl.maximum(maximum, tl.max(score, 1))

            safe_max = tl.where(new_max == -float("inf"), 0.0, new_max)
            alpha = tl.exp(maximum - safe_max)
            prob = tl.exp(score - safe_max[:, None])
            value = tl.load(
                V + slot[:, None] * VS + kv_head * VH + value_dim[None, :] * VD,
                valid[:, None] & (value_dim[None, :] < DV),
                other=0,
            )
            acc = acc * alpha[:, None]
            acc = tl.dot(prob.to(V.dtype.element_ty), value, acc, input_precision="ieee")
            denominator = denominator * alpha + tl.sum(prob, 1)
            maximum = new_max
        safe_denominator = tl.where(denominator > 0, denominator, 1.0)
        result = acc / safe_denominator[:, None]
        if SPLITS == 1 or (SHORT_DIRECT and limit <= BN):
            tl.store(
                Output + (row * Q_HEADS + head[:, None]) * DV + value_dim[None, :],
                result,
                (group[:, None] < GROUP) & (value_dim[None, :] < DV),
            )
        else:
            tl.store(
                Partial + ((row * Q_HEADS + head[:, None]) * SPLITS + split) * DV + value_dim[None, :],
                result,
                (group[:, None] < GROUP) & (value_dim[None, :] < DV),
            )
            lse = tl.where(denominator > 0, maximum + tl.log(safe_denominator), -float("inf"))
            tl.store(LSE + (row * Q_HEADS + head) * SPLITS + split, lse, group < GROUP)


@triton.jit
def _reduce(
    Lengths,
    Partial,
    LSE,
    Output,
    HEADS: tl.constexpr,
    DV: tl.constexpr,
    SPLITS: tl.constexpr,
    BS: tl.constexpr,
    BV: tl.constexpr,
    SELECTED: tl.constexpr,
    BN: tl.constexpr,
    SHORT_DIRECT: tl.constexpr,
    SKIP_EMPTY: tl.constexpr,
):
    row = tl.program_id(0).to(tl.int64)
    limit = SELECTED
    if Lengths is not None:
        limit = tl.minimum(SELECTED, tl.maximum(0, tl.load(Lengths + row)))
    if not SHORT_DIRECT or limit > BN:
        head = tl.program_id(1)
        split = tl.arange(0, BS)
        dim = tl.arange(0, BV)
        valid_split = split < SPLITS
        if SKIP_EMPTY:
            valid_split = valid_split & (split * BN < limit)
        lse = tl.load(LSE + (row * HEADS + head) * SPLITS + split, valid_split, other=-float("inf"))
        maximum = tl.max(lse, 0)
        maximum = tl.where(maximum == -float("inf"), 0.0, maximum)
        weights = tl.exp(lse - maximum)
        partial = tl.load(
            Partial + ((row * HEADS + head) * SPLITS + split[:, None]) * DV + dim[None, :],
            valid_split[:, None] & (dim[None, :] < DV),
            other=0,
        )
        total = tl.sum(weights, 0)
        result = tl.sum(partial * weights[:, None], 0) / tl.where(total > 0, total, 1.0)
        tl.store(Output + (row * HEADS + head) * DV + dim, result, dim < DV)
