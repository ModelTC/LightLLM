import torch

import triton
import triton.language as tl
import os

rmsnorm_num_warps = int(os.getenv("RMSNORM_WARPS", "8"))


@triton.jit
def _rms_norm_fwd_fused(
    X,  # pointer to the input
    Y,  # pointer to the output
    W,  # pointer to the weights
    x_stride0,  # how much to increase the pointer when moving by 1 row
    x_stride1,
    y_stride0,
    y_stride1,
    N,  # number of columns in X
    eps,  # epsilon to avoid division by zero
    BLOCK_SIZE: tl.constexpr,
):
    # Map the program id to the row of X and Y it should compute.
    row = tl.program_id(0)
    Y += row * y_stride0
    X += row * x_stride0
    # Compute variance
    _var = tl.zeros([BLOCK_SIZE], dtype=tl.float32)
    for off in range(0, N, BLOCK_SIZE):
        cols = off + tl.arange(0, BLOCK_SIZE)
        x = tl.load(X + cols * x_stride1, mask=cols < N, other=0.0).to(tl.float32)
        _var += x * x
    var = tl.sum(_var, axis=0) / N
    rstd = 1 / tl.sqrt(var + eps)
    # Normalize and apply linear transformation
    for off in range(0, N, BLOCK_SIZE):
        cols = off + tl.arange(0, BLOCK_SIZE)
        mask = cols < N
        w = tl.load(W + cols, mask=mask).to(tl.float32)
        x = tl.load(X + cols, mask=mask, other=0.0).to(tl.float32)
        x_hat = x * rstd
        y = x_hat * w
        # Write output
        tl.store(Y + cols * y_stride1, y.to(Y.dtype.element_ty), mask=mask)


def rmsnorm_forward(x: torch.Tensor, weight: torch.Tensor, eps: float, out=None):
    # allocate output
    y = torch.empty_like(x) if out is None else out
    # reshape input data into 2D tensor
    x_arg = x.view(-1, x.shape[-1])
    y_arg = y.view(-1, x.shape[-1])
    assert x_arg.shape[-1] == weight.shape[0] and x_arg.shape == y_arg.shape
    assert y.data_ptr() == y_arg.data_ptr()
    M, N = x_arg.shape
    # Less than 64KB per feature: enqueue fused kernel
    MAX_FUSED_SIZE = 65536 // x.element_size()
    BLOCK_SIZE = min(MAX_FUSED_SIZE, triton.next_power_of_2(N))
    # print("BLOCK_SIZE:", BLOCK_SIZE)
    if N > BLOCK_SIZE:
        raise RuntimeError("This layer norm doesn't support feature dim >= 64KB.")
    # heuristics for number of warps
    if BLOCK_SIZE > 16384:
        BLOCK_SIZE = 16384
    # enqueue kernel
    _rms_norm_fwd_fused[(M,)](
        x_arg,
        y_arg,
        weight,
        x_arg.stride(0),
        x_arg.stride(1),
        y_arg.stride(0),
        y_arg.stride(1),
        N,
        eps,
        BLOCK_SIZE=BLOCK_SIZE,
        num_warps=rmsnorm_num_warps,
    )
    return y


@triton.jit
def _add_rms_norm_fwd_fused(
    X,  # pointer to the input
    Y,  # pointer to the output
    R,  # pointer to the residual
    W,  # pointer to the weights
    x_stride0,  # how much to increase the pointer when moving by 1 row
    x_stride1,
    y_stride0,
    y_stride1,
    r_stride0,
    r_stride1,
    N: tl.constexpr,  # number of columns in X
    eps,  # epsilon to avoid division by zero
    BLOCK_SIZE: tl.constexpr,
):
    # Map the program id to the row of X and Y it should compute.
    row = tl.program_id(0)
    Y += row * y_stride0
    X += row * x_stride0
    R += row * r_stride0
    # Compute variance
    if BLOCK_SIZE >= N:
        cols = tl.arange(0, BLOCK_SIZE)
        mask = cols < N
        x = tl.load(X + cols, mask=mask, other=0.0).to(tl.float32)
        r = tl.load(R + cols, mask=mask, other=0.0).to(tl.float32)
        x += r
        _x = x.to(X.dtype.element_ty)
        tl.store(X + cols, _x, mask=mask)
        x = _x.to(tl.float32)
        _var = x * x
        var = tl.sum(_var, axis=0) / N
        rstd = 1 / tl.sqrt(var + eps)
        w = tl.load(W + cols, mask=mask).to(tl.float32)
        x = x * rstd * w
        tl.store(Y + cols, x.to(Y.dtype.element_ty), mask=mask)
    else:
        _var = tl.zeros([BLOCK_SIZE], dtype=tl.float32)
        for off in range(0, N, BLOCK_SIZE):
            cols = off + tl.arange(0, BLOCK_SIZE)
            mask = cols < N
            x = tl.load(X + cols, mask=mask, other=0.0).to(tl.float32)
            r = tl.load(R + cols, mask=mask, other=0.0).to(tl.float32)
            x += r
            _x = x.to(X.dtype.element_ty)
            tl.store(X + cols, _x, mask=mask)
            x = _x.to(tl.float32)
            _var += x * x
        var = tl.sum(_var, axis=0) / N
        rstd = 1 / tl.sqrt(var + eps)
        # Normalize and apply linear transformation
        for off in range(0, N, BLOCK_SIZE):
            cols = off + tl.arange(0, BLOCK_SIZE)
            mask = cols < N
            w = tl.load(W + cols, mask=mask).to(tl.float32)
            x = tl.load(X + cols, mask=mask, other=0.0).to(tl.float32)
            x = x * rstd * w
            # Write output
            tl.store(Y + cols, x.to(Y.dtype.element_ty), mask=mask)


def add_rmsnorm_fused_forward(x: torch.Tensor, residual: torch.Tensor, weight: torch.Tensor, eps: float, out=None):
    # allocate output
    y = torch.empty_like(x) if out is None else out
    # reshape input data into 2D tensor
    x_arg = x.view(-1, x.shape[-1])
    y_arg = y.view(-1, x.shape[-1])
    residual_arg = residual.view(-1, x.shape[-1])
    assert x_arg.shape[-1] == weight.shape[0] and x_arg.shape == y_arg.shape
    assert y.data_ptr() == y_arg.data_ptr()
    assert x_arg.shape == residual_arg.shape
    M, N = x_arg.shape
    # Less than 64KB per feature: enqueue fused kernel
    MAX_FUSED_SIZE = 65536 // x.element_size()
    BLOCK_SIZE = min(MAX_FUSED_SIZE, triton.next_power_of_2(N))
    # print("BLOCK_SIZE:", BLOCK_SIZE)
    if N > BLOCK_SIZE:
        raise RuntimeError("This layer norm doesn't support feature dim >= 64KB.")
    # heuristics for number of warps
    if BLOCK_SIZE > 16384:
        BLOCK_SIZE = 16384
    # enqueue kernel
    _add_rms_norm_fwd_fused[(M,)](
        x_arg,
        y_arg,
        residual_arg,
        weight,
        x_arg.stride(0),
        x_arg.stride(1),
        y_arg.stride(0),
        y_arg.stride(1),
        residual_arg.stride(0),
        residual_arg.stride(1),
        N,
        eps,
        BLOCK_SIZE=BLOCK_SIZE,
        num_warps=rmsnorm_num_warps,
    )
    return y
