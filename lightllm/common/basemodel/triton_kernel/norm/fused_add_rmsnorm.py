import torch

import triton
import triton.language as tl
import os

fused_add_rmsnorm_num_warps = int(os.getenv("RMSNORM_WARPS", "8"))


@triton.jit
def _fused_add_rmsnorm_fwd(
    X,  # pointer to the addend (e.g. attention / ffn output)
    RESIDUAL,  # pointer to the residual, updated in place to (residual + x)
    Y,  # pointer to the normalized output
    W,  # pointer to the weights
    x_stride0,
    residual_stride0,
    y_stride0,
    N,  # number of columns
    eps,  # epsilon to avoid division by zero
    HAS_WEIGHT: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    row = tl.program_id(0)
    X += row * x_stride0
    RESIDUAL += row * residual_stride0
    Y += row * y_stride0
    # pass 1: residual = residual + x (in place), accumulate variance of the updated residual
    _var = tl.zeros([BLOCK_SIZE], dtype=tl.float32)
    for off in range(0, N, BLOCK_SIZE):
        cols = off + tl.arange(0, BLOCK_SIZE)
        mask = cols < N
        x = tl.load(X + cols, mask=mask, other=0.0).to(tl.float32)
        r = tl.load(RESIDUAL + cols, mask=mask, other=0.0).to(tl.float32)
        # round the updated residual to the storage dtype first, then accumulate variance
        # from the rounded value so this matches the unfused (store; reload; rmsnorm) path
        # bit-for-bit instead of using the higher-precision fp32 sum.
        s = (r + x).to(RESIDUAL.dtype.element_ty)
        tl.store(RESIDUAL + cols, s, mask=mask)
        s = s.to(tl.float32)
        _var += s * s
    var = tl.sum(_var, axis=0) / N
    rstd = 1 / tl.sqrt(var + eps)
    # pass 2: normalize the (rounded) updated residual and optionally apply weight
    for off in range(0, N, BLOCK_SIZE):
        cols = off + tl.arange(0, BLOCK_SIZE)
        mask = cols < N
        if HAS_WEIGHT:
            w = tl.load(W + cols, mask=mask).to(tl.float32)
        s = tl.load(RESIDUAL + cols, mask=mask, other=0.0).to(tl.float32)
        y = s * rstd
        if HAS_WEIGHT:
            y = y * w
        tl.store(Y + cols * 1, y.to(Y.dtype.element_ty), mask=mask)


def fused_add_rmsnorm_forward(
    residual: torch.Tensor, x: torch.Tensor, weight: torch.Tensor, eps: float, out: torch.Tensor
) -> torch.Tensor:
    """Fused residual-add + RMSNorm.

    Computes ``residual <- residual + x`` (in place) and ``out <- rmsnorm(residual) * weight``
    in a single kernel, eliminating the separate elementwise-add launch and the extra HBM
    round-trip of the hidden state. ``residual`` and ``x`` share shape; ``residual`` is
    updated in place (so the running residual stream is preserved for the next layer).
    """
    assert residual.shape == x.shape
    r_arg = residual.view(-1, residual.shape[-1])
    x_arg = x.view(-1, x.shape[-1])
    y_arg = out.view(-1, out.shape[-1])
    assert r_arg.shape == x_arg.shape == y_arg.shape
    if weight is not None:
        assert r_arg.shape[-1] == weight.shape[0]
    # contiguous last dim (the kernel assumes unit stride within a row)
    assert r_arg.stride(1) == 1 and x_arg.stride(1) == 1 and y_arg.stride(1) == 1
    assert out.data_ptr() == y_arg.data_ptr()
    M, N = r_arg.shape
    MAX_FUSED_SIZE = 65536 // residual.element_size()
    BLOCK_SIZE = min(MAX_FUSED_SIZE, triton.next_power_of_2(N))
    if N > BLOCK_SIZE:
        raise RuntimeError("fused_add_rmsnorm doesn't support feature dim >= 64KB.")
    if BLOCK_SIZE > 16384:
        BLOCK_SIZE = 16384
    _fused_add_rmsnorm_fwd[(M,)](
        x_arg,
        r_arg,
        y_arg,
        weight,
        x_arg.stride(0),
        r_arg.stride(0),
        y_arg.stride(0),
        N,
        eps,
        HAS_WEIGHT=weight is not None,
        BLOCK_SIZE=BLOCK_SIZE,
        num_warps=fused_add_rmsnorm_num_warps,
    )
    return out
