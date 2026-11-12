import torch
import torch.nn.functional as F
import triton
import triton.language as tl

from .kernels import _gate, injection_gates, mix_streams


@triton.jit
def _activate(value, DTYPE: tl.constexpr, C: tl.constexpr):
    value = value.to(DTYPE).to(tl.float32)
    value = (value / C).to(DTYPE).to(tl.float32)
    return value * tl.sigmoid(value)


@triton.jit(do_not_specialize=["M"])
def _down(
    X,
    W,
    P,
    Y,
    M,
    N: tl.constexpr,
    K: tl.constexpr,
    C: tl.constexpr,
    BM: tl.constexpr,
    BN: tl.constexpr,
    BK: tl.constexpr,
    S: tl.constexpr,
    Inject=None,
    Gates=None,
    GB: tl.constexpr = 0,
):
    if Inject is None:
        row_tile = tl.program_id(0)
        col_tile = tl.program_id(1)
        split = tl.program_id(2)
        is_gate: tl.constexpr = False
    else:
        block = tl.program_id(0)
        down_blocks = tl.cdiv(M, BM) * tl.cdiv(N, BN) * S
        is_gate = block >= down_blocks
        gate_block = block - down_blocks
        split = block % S
        tile = block // S
        row_tile = tile // tl.cdiv(N, BN)
        col_tile = tile % tl.cdiv(N, BN)

    if is_gate:
        _gate(X, Inject, Gates, gate_block // C, gate_block % C, K, C, GB)
    else:
        r = row_tile * BM + tl.arange(0, BM)
        c = col_tile * BN + tl.arange(0, BN)
        part = tl.cdiv(K, BK * S) * BK
        k = split * part + tl.arange(0, BK)
        acc = tl.full((BM, BN), 0, tl.float32)
        for _ in range(tl.cdiv(K, BK * S)):
            x = tl.load(X + r[:, None] * K + k[None, :], (r[:, None] < M) & (k[None, :] < K), other=0)
            w = tl.load(W + c[None, :] * K + k[:, None], (c[None, :] < N) & (k[:, None] < K), other=0)
            acc = tl.dot(x, w, acc, input_precision="ieee")
            k += BK
        if S == 1:
            tl.store(
                Y + r[:, None] * N + c[None, :],
                _activate(acc, Y.dtype.element_ty, C),
                (r[:, None] < M) & (c[None, :] < N),
            )
        else:
            tl.store(P + (split * M + r[:, None]) * N + c[None, :], acc, (r[:, None] < M) & (c[None, :] < N))


@triton.jit(do_not_specialize=["M"])
def _merge(P, Y, M, N: tl.constexpr, C: tl.constexpr, S: tl.constexpr, B: tl.constexpr):
    row = tl.program_id(0)
    col = tl.program_id(1) * B + tl.arange(0, B)
    split = tl.arange(0, S)
    value = tl.load(P + (split[:, None] * M + row) * N + col[None, :], col[None, :] < N, other=0)
    tl.store(Y + row * N + col, _activate(tl.sum(value, 0), Y.dtype.element_ty, C), col < N)


@triton.jit(do_not_specialize=["M"])
def _up_mix(
    X,
    W,
    Normalized,
    Y,
    M,
    D: tl.constexpr,
    K: tl.constexpr,
    C: tl.constexpr,
    BC: tl.constexpr,
    BM: tl.constexpr,
    BD: tl.constexpr,
    BK: tl.constexpr,
):
    row = tl.program_id(0) * BM + tl.arange(0, BM)
    n = tl.arange(0, BC * BD)
    branch = n // BD
    col = tl.program_id(1) * BD + n % BD
    weight_row = branch * D + col
    k = tl.arange(0, BK)
    acc = tl.full((BM, BC * BD), 0, tl.float32)
    for _ in range(tl.cdiv(K, BK)):
        x = tl.load(X + row[:, None] * K + k[None, :], (row[:, None] < M) & (k[None, :] < K), other=0)
        w = tl.load(
            W + weight_row[None, :] * K + k[:, None],
            (branch[None, :] < C) & (col[None, :] < D) & (k[:, None] < K),
            other=0,
        )
        acc = tl.dot(x, w, acc, input_precision="ieee")
        k += BK
    logits = acc.to(X.dtype.element_ty).to(tl.float32)
    gate = tl.sigmoid(logits).to(X.dtype.element_ty).to(tl.float32)
    normalized = tl.load(
        Normalized + row[:, None] * (C * D) + weight_row[None, :],
        (row[:, None] < M) & (branch[None, :] < C) & (col[None, :] < D),
        other=0,
    ).to(tl.float32)
    products = (normalized * gate).to(X.dtype.element_ty).to(tl.float32)
    mixed = tl.sum(products.reshape(BM, BC, BD), 1) / C
    out_col = tl.program_id(1) * BD + tl.arange(0, BD)
    tl.store(Y + row[:, None] * D + out_col[None, :], mixed, (row[:, None] < M) & (out_col[None, :] < D))


def _lowrank(x, w, branches, inject_weight=None, gates=None):
    m, k = x.shape
    n = w.shape[0]
    out = torch.empty((m, n), device=x.device, dtype=x.dtype)
    if not m:
        return out

    splits = 16 if k >= 8192 and n <= 512 else 1
    p = torch.empty((splits, m, n), device=x.device, dtype=torch.float32) if splits > 1 else out
    bm = 16 if m <= 16 else 32 if m <= 64 else 64 if m < 128 else 128
    grid = (triton.cdiv(m, bm), triton.cdiv(n, 64), splits)
    if inject_weight is not None:
        grid = (grid[0] * grid[1] * grid[2] + m * branches,)
    _down[grid](
        x,
        w,
        p,
        out,
        m,
        n,
        k,
        branches,
        bm,
        64,
        64,
        splits,
        inject_weight,
        gates,
        triton.next_power_of_2(k) if inject_weight is not None else 0,
        num_warps=8 if inject_weight is not None or bm == 128 else 4,
        enable_fp_fusion=False,
    )
    if splits > 1:
        _merge[(m, triton.cdiv(n, 128))](p, out, m, n, branches, splits, 128, num_warps=4, enable_fp_fusion=False)
    return out


def _mix(normalized, low, w, branches):
    m, k = low.shape
    dim = normalized.shape[1] // branches
    out = torch.empty((m, dim), device=low.device, dtype=low.dtype)
    if not m:
        return out
    bm = 16 if m <= 16 else 32 if m <= 64 else 64 if m < 128 else 128
    _up_mix[(triton.cdiv(m, bm), triton.cdiv(dim, 32))](
        low,
        w,
        normalized,
        out,
        m,
        dim,
        k,
        branches,
        triton.next_power_of_2(branches),
        bm,
        32,
        64,
        num_warps=8 if bm == 128 else 4,
        enable_fp_fusion=False,
    )
    return out


def project_and_mix(
    normalized: torch.Tensor, down_weight: torch.Tensor, up_weight: torch.Tensor, branches: int, inject_weight=None
):
    gates = None
    if normalized.dtype == torch.float32:
        low = F.silu(F.linear(normalized, down_weight) / branches)
        mixed = mix_streams(normalized, F.linear(low, up_weight), branches)
    else:
        rows = normalized.reshape(-1, normalized.shape[-1])
        fuse_gates = inject_weight is not None and 0 < rows.shape[0] <= 64
        fuse_gates = fuse_gates and rows.shape[1] >= 8192 and down_weight.shape[0] <= 512
        if fuse_gates:
            gates = torch.empty((*normalized.shape[:-1], branches), device=normalized.device, dtype=normalized.dtype)
        low = _lowrank(rows, down_weight, branches, inject_weight if fuse_gates else None, gates)
        mixed = _mix(rows, low, up_weight, branches)
        mixed = mixed.reshape((*normalized.shape[:-1], normalized.shape[-1] // branches))
    if inject_weight is None:
        return mixed
    if gates is None:
        gates = injection_gates(normalized, inject_weight, branches)
    return mixed, gates
