import torch

import triton
import triton.language as tl
import os

rmsnorm_num_warps = int(os.getenv("RMSNORM_WARPS", "4"))


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
        w = tl.load(W + cols, mask=mask)
        x = tl.load(X + cols, mask=mask, other=0.0).to(tl.float32)
        x_hat = (x * rstd).to(tl.bfloat16)
        y = x_hat * w.to(tl.bfloat16)
        # Write output
        tl.store(Y + cols * y_stride1, y, mask=mask)


def rmsnorm_forward1(x: torch.Tensor, weight, eps, out=None):
    # allocate output
    y = torch.empty_like(x) if out is None else out
    # reshape input data into 2D tensor
    x_arg = x.view(-1, x.shape[-1])
    y_arg = y.view(-1, x.shape[-1])
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


def rmsnorm_forward(hidden_states, weight, eps, out=None):
    input_dtype = hidden_states.dtype
    hidden_states = hidden_states.to(torch.float32)
    variance = hidden_states.pow(2).mean(-1, keepdim=True)
    hidden_states = hidden_states * torch.rsqrt(variance + eps)
    out = weight * hidden_states.to(input_dtype)
    return out


def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
    input_dtype = hidden_states.dtype
    hidden_states = hidden_states.to(torch.float32)
    variance = hidden_states.pow(2).mean(-1, keepdim=True)
    hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
    print(f"norm weight dtype:{self.weight.dtype}")
    return self.weight * hidden_states.to(input_dtype)
