import torch

import triton
import triton.language as tl
import os


@triton.jit
def _rms_norm_fwd_fused(
    X,  # pointer to the input
    W,  # pointer to the weights
    x_stride0,  # how much to increase the pointer when moving by 1 row
    x_stride1,
    N,  # number of columns in X
    eps,  # epsilon to avoid division by zero
    head_dim: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    # Map the program id to the row of X and Y it should compute.
    row = tl.program_id(0)
    head_idx = tl.program_id(1)

    X += row * x_stride0
    # Compute variance
    cols = (head_idx * head_dim + tl.arange(0, BLOCK_SIZE)) * x_stride1
    x = tl.load(X + cols).to(tl.float32)
    var = tl.sum(x * x, axis=0) / head_dim
    rstd = 1 / tl.sqrt(var + eps)
    # Normalize and apply linear transformation
    w = tl.load(W + tl.arange(0, BLOCK_SIZE))
    x_hat = x * rstd
    y = x_hat.to(W.dtype.element_ty) * w
    # Write output
    tl.store(X + cols, y.to(X.dtype.element_ty))


def qk_rmsnorm_forward(x: torch.Tensor, weight: torch.Tensor, eps):
    """
    This function is used to perform in-place RMSNorm on the input tensor,
    and to adapt the head_dim norm for Qwen3 MoE and the splited qk tensor layout.
    x: (M, N)
    weight: (head_dim,)
    eps: float
    return: x
    """
    assert weight.is_contiguous()
    # reshape input data into 2D tensor
    x_arg = x.view(-1, x.shape[-1])
    M, N = x_arg.shape
    head_dim = weight.shape[0]
    assert x.shape[-1] % head_dim == 0
    BLOCK_SIZE = triton.next_power_of_2(head_dim)
    assert BLOCK_SIZE <= head_dim, "head_dim must be the power of 2"
    # enqueue kernel
    _rms_norm_fwd_fused[(M, N // head_dim)](
        x_arg,
        weight,
        x_arg.stride(0),
        x_arg.stride(1),
        N,
        eps,
        head_dim=head_dim,
        BLOCK_SIZE=BLOCK_SIZE,
        num_warps=1,
    )
    return x
