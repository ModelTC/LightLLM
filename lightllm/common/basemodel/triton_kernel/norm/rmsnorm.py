import torch

import triton
import triton.language as tl
import os
from lightllm.common.triton_utils.autotuner import autotune

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
    HAS_WEIGHT: tl.constexpr,
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
    # Normalize and optionally apply linear transformation
    for off in range(0, N, BLOCK_SIZE):
        cols = off + tl.arange(0, BLOCK_SIZE)
        mask = cols < N
        if HAS_WEIGHT:
            w = tl.load(W + cols, mask=mask).to(tl.float32)
        x = tl.load(X + cols, mask=mask, other=0.0).to(tl.float32)
        x_hat = x * rstd
        y = x_hat
        if HAS_WEIGHT:
            y = x_hat * w
        # Write output
        tl.store(Y + cols * y_stride1, y.to(Y.dtype.element_ty), mask=mask)


@triton.jit
def _add_rms_norm_fwd_fused(
    X,
    R,
    Y,
    W,
    x_stride0,
    x_stride1,
    r_stride0,
    r_stride1,
    y_stride0,
    y_stride1,
    N,
    eps,
    HAS_WEIGHT: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    row = tl.program_id(0)
    X += row * x_stride0
    R += row * r_stride0
    Y += row * y_stride0

    _var = tl.zeros([BLOCK_SIZE], dtype=tl.float32)
    for off in range(0, N, BLOCK_SIZE):
        cols = off + tl.arange(0, BLOCK_SIZE)
        mask = cols < N
        x = tl.load(X + cols * x_stride1, mask=mask, other=0.0).to(tl.float32)
        r = tl.load(R + cols * r_stride1, mask=mask, other=0.0).to(tl.float32)
        x = x + r
        tl.store(X + cols * x_stride1, x.to(X.dtype.element_ty), mask=mask)
        _var += x * x

    var = tl.sum(_var, axis=0) / N
    rstd = 1 / tl.sqrt(var + eps)
    for off in range(0, N, BLOCK_SIZE):
        cols = off + tl.arange(0, BLOCK_SIZE)
        mask = cols < N
        x = tl.load(X + cols * x_stride1, mask=mask, other=0.0).to(tl.float32)
        y = x * rstd
        if HAS_WEIGHT:
            w = tl.load(W + cols, mask=mask).to(tl.float32)
            y *= w
        tl.store(Y + cols * y_stride1, y.to(Y.dtype.element_ty), mask=mask)


def rmsnorm_forward(x: torch.Tensor, weight: torch.Tensor, eps: float, out=None):
    # allocate output
    y = torch.empty_like(x) if out is None else out
    # reshape input data into 2D tensor
    x_arg = x.view(-1, x.shape[-1])
    y_arg = y.view(-1, x.shape[-1])
    assert x_arg.shape == y_arg.shape
    if weight is not None:
        assert x_arg.shape[-1] == weight.shape[0]
    assert y.data_ptr() == y_arg.data_ptr()
    M, N = x_arg.shape
    # Less than 64KB per feature: enqueue fused kernel
    MAX_FUSED_SIZE = 65536 // x_arg.element_size()
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
        HAS_WEIGHT=weight is not None,
        BLOCK_SIZE=BLOCK_SIZE,
        num_warps=rmsnorm_num_warps,
    )
    return y


def _get_add_rmsnorm_configs():
    return [{"num_warps": nw} for nw in [4, 8, 16]]


def _get_add_rmsnorm_static_key(x_arg: torch.Tensor, y_arg: torch.Tensor, weight: torch.Tensor):
    return {
        "x_dtype": str(x_arg.dtype),
        "out_dtype": str(y_arg.dtype),
        "weight_dtype": "none" if weight is None else str(weight.dtype),
        "N": x_arg.shape[1],
        "has_weight": weight is not None,
    }


@autotune(
    kernel_name="add_rmsnorm_forward:v1",
    configs_gen_func=_get_add_rmsnorm_configs,
    static_key_func=_get_add_rmsnorm_static_key,
    run_key_func=lambda x_arg: x_arg.shape[0],
    mutates_args=["x_arg", "y_arg"],
)
def _add_rmsnorm_forward(
    x_arg: torch.Tensor,
    residual_arg: torch.Tensor,
    y_arg: torch.Tensor,
    weight: torch.Tensor,
    eps: float,
    run_config: dict = None,
):
    M, N = x_arg.shape
    MAX_FUSED_SIZE = 65536 // x_arg.element_size()
    BLOCK_SIZE = min(MAX_FUSED_SIZE, triton.next_power_of_2(N))
    if N > BLOCK_SIZE:
        raise RuntimeError("This layer norm doesn't support feature dim >= 64KB.")
    if BLOCK_SIZE > 16384:
        BLOCK_SIZE = 16384
    if not run_config:
        run_config = {"num_warps": rmsnorm_num_warps}
    _add_rms_norm_fwd_fused[(M,)](
        x_arg,
        residual_arg,
        y_arg,
        weight,
        x_arg.stride(0),
        x_arg.stride(1),
        residual_arg.stride(0),
        residual_arg.stride(1),
        y_arg.stride(0),
        y_arg.stride(1),
        N,
        eps,
        HAS_WEIGHT=weight is not None,
        BLOCK_SIZE=BLOCK_SIZE,
        num_warps=run_config["num_warps"],
    )
    return y_arg


def add_rmsnorm_forward(x: torch.Tensor, residual: torch.Tensor, weight: torch.Tensor, eps: float, out=None):
    y = torch.empty_like(x) if out is None else out
    x_arg = x.view(-1, x.shape[-1])
    residual_arg = residual.view(-1, x.shape[-1])
    y_arg = y.view(-1, x.shape[-1])
    assert x_arg.shape == residual_arg.shape == y_arg.shape
    if weight is not None:
        assert x_arg.shape[-1] == weight.shape[0]
    assert y.data_ptr() == y_arg.data_ptr()
    _add_rmsnorm_forward(x_arg, residual_arg, y_arg, weight, eps)
    return y


def torch_rms_norm(x, weight, eps):
    return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + eps) * weight


def test_rms_norm(M, N, dtype, eps=1e-5, device="cuda"):
    # create data
    x_shape = (M, N)
    w_shape = (x_shape[-1],)
    weight = torch.rand(w_shape, dtype=dtype, device="cuda")
    x = -2.3 + 0.5 * torch.randn(x_shape, dtype=dtype, device="cuda")
    # forward pass
    y_tri = rmsnorm_forward(x, weight, eps)
    y_ref = torch_rms_norm(x.to(torch.float32), weight.to(torch.float32), eps).to(dtype)

    # compare
    print("type:", y_tri.dtype, y_ref.dtype)
    print("max delta:", torch.max(torch.abs(y_tri - y_ref)))
    assert torch.allclose(y_tri, y_ref, atol=1e-2, rtol=0)
    return
