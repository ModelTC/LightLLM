import triton
import triton.language as tl
import torch
from lightllm.common.triton_utils.autotuner import autotune


@triton.heuristics(
    {
        "HAS_BIAS": lambda args: args["B"] is not None,
    }
)
@triton.jit
def gated_rmsnorm_forward_kernel(
    X,  # pointer to the input
    Y,  # pointer to the output
    W,  # pointer to the weights
    B,  # pointer to the biases
    Z,  # pointer to the other branch (required, not optional)
    Rstd,  # pointer to the 1/std
    stride_x_row,  # how much to increase the pointer when moving by 1 row
    stride_y_row,
    stride_z_row,
    M,  # number of rows in X
    N,  # number of columns in X
    eps,  # epsilon to avoid division by zero
    BLOCK_N: tl.constexpr,
    HAS_BIAS: tl.constexpr,
    NORM_BEFORE_GATE: tl.constexpr,
):
    # Map the program id to the row of X and Y it should compute.
    row = tl.program_id(0)
    group = tl.program_id(1)
    X += row * stride_x_row + group * N
    Y += row * stride_y_row + group * N
    Z += row * stride_z_row + group * N
    Rstd += group * M
    W += group * N
    if HAS_BIAS:
        B += group * N
    # Compute variance (RMS norm doesn't use mean)
    cols = tl.arange(0, BLOCK_N)
    x = tl.load(X + cols, mask=cols < N, other=0.0).to(tl.float32)
    if not NORM_BEFORE_GATE:
        z = tl.load(Z + cols, mask=cols < N).to(tl.float32)
        x *= z * tl.sigmoid(z)
    # RMS norm: compute variance directly without mean subtraction
    xbar = tl.where(cols < N, x, 0.0)
    var = tl.sum(xbar * xbar, axis=0) / N
    rstd = 1 / tl.sqrt(var + eps)
    tl.store(Rstd + row, rstd)
    # Normalize and apply linear transformation
    mask = cols < N
    w = tl.load(W + cols, mask=mask).to(tl.float32)
    if HAS_BIAS:
        b = tl.load(B + cols, mask=mask).to(tl.float32)
    # RMS norm: normalize without mean subtraction
    x_hat = x * rstd
    y = x_hat * w + b if HAS_BIAS else x_hat * w
    if NORM_BEFORE_GATE:
        z = tl.load(Z + cols, mask=mask).to(tl.float32)
        y *= z * tl.sigmoid(z)
    # Write output
    tl.store(Y + cols, y, mask=mask)


def _get_gated_rmsnorm_configs():
    """Generate configurations for autotuning gated RMSNorm kernel."""
    configs = []
    # Different BLOCK_N sizes (powers of 2)
    for block_n in [64, 128, 256, 512, 1024, 2048, 4096]:
        # Different number of warps
        for num_warps in [1, 2, 4, 8]:
            # Skip configurations that are likely to be inefficient
            if block_n >= 2048 and num_warps > 4:
                continue
            if block_n <= 128 and num_warps > 2:
                continue
            configs.append({"BLOCK_N": block_n, "num_warps": num_warps})
    return configs


def _get_gated_rmsnorm_static_key(x: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor):
    """Generate static key for caching autotuned configurations."""
    M, N = x.shape
    return {
        "x_dtype": str(x.dtype),
        "weight_dtype": str(weight.dtype),
        "N": N,
        "has_bias": bias is not None,
    }


@autotune(
    kernel_name="gated_rmsnorm_forward:v1",
    configs_gen_func=_get_gated_rmsnorm_configs,
    static_key_func=_get_gated_rmsnorm_static_key,
    run_key_func=lambda x: x.shape[0],
)
def gated_rmsnorm_forward(
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor,
    eps: float,
    z: torch.Tensor,
    out: torch.Tensor = None,
    group_size: int = None,
    norm_before_gate: bool = True,
    run_config: dict = None,
):
    M, N = x.shape
    if group_size is None:
        group_size = N
    assert N % group_size == 0
    ngroups = N // group_size
    assert x.stride(-1) == 1
    # z is required for gated_rmsnorm
    assert z is not None, "z cannot be None for gated_rmsnorm_forward"
    assert z.stride(-1) == 1
    assert z.shape == (M, N)
    assert weight.shape == (N,)
    assert weight.stride(-1) == 1
    if bias is not None:
        assert bias.stride(-1) == 1
        assert bias.shape == (N,)
    # allocate output
    if out is not None:
        assert out.shape == x.shape
    else:
        out = torch.empty_like(x)
    assert out.stride(-1) == 1
    # For RMS norm, we still need rstd for the kernel
    rstd = torch.empty((ngroups * M,), dtype=torch.float32, device=x.device)

    # Default heuristic when autotune is disabled or no config provided
    if not run_config:
        # Less than 64KB per feature: enqueue fused kernel
        MAX_FUSED_SIZE = 65536 // x.element_size()
        BLOCK_N = min(MAX_FUSED_SIZE, triton.next_power_of_2(group_size))
        if group_size > BLOCK_N:
            raise RuntimeError("This layer norm doesn't support feature dim >= 64KB.")
        # heuristics for number of warps
        num_warps = min(max(BLOCK_N // 256, 1), 8)
        run_config = {"BLOCK_N": BLOCK_N, "num_warps": num_warps}

    BLOCK_N = run_config["BLOCK_N"]
    num_warps = run_config["num_warps"]

    # Validate BLOCK_N against group_size
    if group_size > BLOCK_N:
        # Fall back to largest valid BLOCK_N
        MAX_FUSED_SIZE = 65536 // x.element_size()
        BLOCK_N = min(MAX_FUSED_SIZE, triton.next_power_of_2(group_size))
        if group_size > BLOCK_N:
            raise RuntimeError("This layer norm doesn't support feature dim >= 64KB.")

    grid = (M, ngroups)
    gated_rmsnorm_forward_kernel[grid](
        x,
        out,
        weight,
        bias,
        z,
        rstd,
        x.stride(0),
        out.stride(0),
        z.stride(0),
        M,
        group_size,
        eps,
        BLOCK_N=BLOCK_N,
        NORM_BEFORE_GATE=norm_before_gate,
        num_warps=num_warps,
    )
    return out
