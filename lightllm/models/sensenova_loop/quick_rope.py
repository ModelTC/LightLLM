"""Private experimental kernels; no architecture, selection or KV semantic changes."""
import torch
import triton
import triton.language as tl


@triton.jit
def _rope(
    Q,
    K,
    C,
    S,
    QN: tl.constexpr,
    KN: tl.constexpr,
    Q0: tl.constexpr,
    Q1: tl.constexpr,
    K0: tl.constexpr,
    K1: tl.constexpr,
    CA: tl.constexpr,
    CR: tl.constexpr,
    CD: tl.constexpr,
    H: tl.constexpr,
    SH: tl.constexpr,
    SW: tl.constexpr,
    B: tl.constexpr,
):
    row, head = tl.program_id(0), tl.program_id(1)
    d = tl.arange(0, B)
    axis = tl.where((d % 3 == 1) & (d < 3 * SH), 1, tl.where((d % 3 == 2) & (d < 3 * SW), 2, 0))
    c = tl.load(C + axis * CA + row * CR + d * CD, d < H, other=0).to(tl.float32)
    s = tl.load(S + axis * CA + row * CR + d * CD, d < H, other=0).to(tl.float32)
    if head < QN:
        pointer = Q + row * Q0 + head * Q1
    else:
        pointer = K + row * K0 + (head - QN) * K1
    left = tl.load(pointer + d, d < H, other=0).to(tl.float32)
    right = tl.load(pointer + H + d, d < H, other=0).to(tl.float32)
    tl.store(pointer + d, left * c - right * s, d < H)
    tl.store(pointer + H + d, right * c + left * s, d < H)


def apply_partial_mrope(query, key, cos, sin, rotary_dim, sections=(11, 11, 10)):
    q = query if query is not None else key
    qn = 0 if query is None else query.shape[1]
    if cos.ndim == 2:
        ca, cr, cd = 0, cos.stride(0), cos.stride(1)
    else:
        ca, cr, cd = cos.stride()
    _rope[(key.shape[0], qn + key.shape[1])](
        q,
        key,
        cos,
        sin,
        qn,
        key.shape[1],
        q.stride(0),
        q.stride(1),
        key.stride(0),
        key.stride(1),
        ca,
        cr,
        cd,
        rotary_dim // 2,
        sections[1],
        sections[2],
        triton.next_power_of_2(rotary_dim // 2),
        num_warps=4,
        enable_fp_fusion=False,
    )
    return query, key
