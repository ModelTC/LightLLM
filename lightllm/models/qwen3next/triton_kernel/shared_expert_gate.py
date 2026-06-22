import torch
import triton
import triton.language as tl


@triton.jit
def _sigmoid_mul_kernel(
    x,
    gate,
    stride_x_m: tl.constexpr,
    stride_x_n: tl.constexpr,
    stride_g_m: tl.constexpr,
    stride_g_n: tl.constexpr,
    N: tl.constexpr,
    GATE_N: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    row = tl.program_id(0)
    offs = tl.arange(0, BLOCK_N)
    mask = offs < N
    x_ptrs = x + row * stride_x_m + offs * stride_x_n
    x_vals = tl.load(x_ptrs, mask=mask, other=0.0).to(tl.float32)
    if GATE_N == 1:
        gate_vals = tl.load(gate + row * stride_g_m).to(tl.float32)
    else:
        gate_vals = tl.load(gate + row * stride_g_m + offs * stride_g_n, mask=mask, other=0.0).to(tl.float32)
    gate_vals = tl.sigmoid(gate_vals)
    tl.store(x_ptrs, (x_vals * gate_vals).to(x.dtype.element_ty), mask=mask)


@torch.no_grad()
def sigmoid_mul_(x: torch.Tensor, gate: torch.Tensor) -> torch.Tensor:
    x_arg = x.view(-1, x.shape[-1])
    gate_arg = gate.view(-1, gate.shape[-1])
    assert gate_arg.shape[0] == x_arg.shape[0] and gate_arg.shape[1] in (1, x_arg.shape[1])
    _, n = x_arg.shape
    block_n = triton.next_power_of_2(n)
    _sigmoid_mul_kernel[(x_arg.shape[0],)](
        x=x_arg,
        gate=gate_arg,
        stride_x_m=x_arg.stride(0),
        stride_x_n=x_arg.stride(1),
        stride_g_m=gate_arg.stride(0),
        stride_g_n=gate_arg.stride(1),
        N=n,
        GATE_N=gate_arg.shape[1],
        BLOCK_N=block_n,
        num_warps=8,
    )
    return x
