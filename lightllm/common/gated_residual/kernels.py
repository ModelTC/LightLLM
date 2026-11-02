import torch
import triton
import triton.language as tl


@triton.jit
def _norm(X, W, Y, D: tl.constexpr, C: tl.constexpr, EPS: tl.constexpr, B: tl.constexpr):
    row = tl.program_id(0)
    branch = tl.program_id(1)
    col = tl.arange(0, B)
    offset = row * C * D + branch * D + col
    x = tl.load(X + offset, col < D, other=0).to(tl.float32)
    weight = tl.load(W + branch * D + col, col < D, other=0).to(tl.float32)
    inv = tl.rsqrt(tl.sum(x * x, 0) / D + EPS)
    tl.store(Y + offset, (x * inv) * (1.0 + weight), col < D)


@triton.jit
def _mix(X, G, Y, D: tl.constexpr, C: tl.constexpr, BC: tl.constexpr, B: tl.constexpr):
    row = tl.program_id(0)
    col = tl.program_id(1) * B + tl.arange(0, B)
    branch = tl.arange(0, BC)
    offset = row * C * D + branch[:, None] * D + col[None, :]
    mask = (branch[:, None] < C) & (col[None, :] < D)
    x = tl.load(X + offset, mask, other=0).to(tl.float32)
    logits = tl.load(G + offset, mask, other=0).to(tl.float32)

    gate = tl.sigmoid(logits).to(G.dtype.element_ty).to(tl.float32)
    product = (x * gate).to(X.dtype.element_ty).to(tl.float32)
    product = tl.where(branch[:, None] < C, product, 0.0)
    result = tl.sum(product, 0) / C
    tl.store(Y + row * D + col, result, col < D)


@triton.jit
def _gates(X, W, Out, D: tl.constexpr, C: tl.constexpr, B: tl.constexpr):
    row = tl.program_id(0)
    branch = tl.program_id(1)
    col = tl.arange(0, B)
    x = tl.load(X + row * D + col, col < D, other=0).to(tl.float32)
    weight = tl.load(W + branch * D + col, col < D, other=0).to(tl.float32)

    logits = tl.sum(x * weight, 0).to(X.dtype.element_ty).to(tl.float32)
    scaled = (logits / C).to(X.dtype.element_ty).to(tl.float32)
    gate = tl.sigmoid(scaled).to(X.dtype.element_ty).to(tl.float32)
    tl.store(Out + row * C + branch, 2 * gate)


@triton.jit
def _inject(R, X, G, Y, D: tl.constexpr, C: tl.constexpr, B: tl.constexpr):
    row = tl.program_id(0)
    branch = tl.program_id(1)
    col = tl.program_id(2) * B + tl.arange(0, B)
    offset = row * C * D + branch * D + col
    x = tl.load(X + row * D + col, col < D, other=0).to(tl.float32)
    gate = tl.load(G + row * C + branch).to(tl.float32)
    residual = tl.load(R + offset, col < D, other=0).to(tl.float32)

    injection = (x * gate).to(R.dtype.element_ty).to(tl.float32)
    tl.store(Y + offset, residual + injection, col < D)


def grouped_norm(x: torch.Tensor, weight: torch.Tensor, branches: int, eps: float) -> torch.Tensor:
    output = torch.empty_like(x)
    rows = x.numel() // x.shape[-1]
    dim = x.shape[-1] // branches
    if rows:
        _norm[(rows, branches)](
            x, weight, output, dim, branches, eps, triton.next_power_of_2(dim), enable_fp_fusion=False
        )
    return output


def mix_streams(x: torch.Tensor, logits: torch.Tensor, branches: int) -> torch.Tensor:
    dim = x.shape[-1] // branches
    output = torch.empty((*x.shape[:-1], dim), dtype=x.dtype, device=x.device)
    rows = x.numel() // x.shape[-1]
    if rows:
        _mix[(rows, triton.cdiv(dim, 256))](x, logits, output, dim, branches, triton.next_power_of_2(branches), 256)
    return output


def inject_streams(residual: torch.Tensor, update: torch.Tensor, gates: torch.Tensor) -> torch.Tensor:
    output = torch.empty_like(residual)
    branches = gates.shape[-1]
    dim = update.shape[-1]
    rows = update.numel() // dim
    if rows:
        _inject[(rows, branches, triton.cdiv(dim, 256))](
            residual, update, gates, output, dim, branches, 256, enable_fp_fusion=False
        )
    return output


def injection_gates(x: torch.Tensor, weight: torch.Tensor, branches: int) -> torch.Tensor:
    output = torch.empty((*x.shape[:-1], branches), dtype=x.dtype, device=x.device)
    rows = x.numel() // x.shape[-1]
    if rows:
        block = triton.next_power_of_2(x.shape[-1])
        _gates[(rows, branches)](
            x, weight, output, x.shape[-1], branches, block, num_warps=8 if block >= 8192 else 4, enable_fp_fusion=False
        )
    return output
