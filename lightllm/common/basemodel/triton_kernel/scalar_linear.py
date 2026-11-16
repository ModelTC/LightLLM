import torch
import triton
import triton.language as tl


@triton.jit
def _scalar_linear(X, W, Y, K: tl.constexpr, XS0, XS1, WS, YS, BLOCK: tl.constexpr):
    row = tl.program_id(0)
    column = tl.arange(0, BLOCK)
    value = tl.load(X + row * XS0 + column * XS1, column < K, other=0).to(tl.float64)
    weight = tl.load(W + column * WS, column < K, other=0).to(tl.float64)
    tl.store(Y + row * YS, tl.sum(value * weight, 0))


def scalar_linear(input: torch.Tensor, weight: torch.Tensor, out=None, alloc_func=torch.empty) -> torch.Tensor:
    if input.ndim != 2 or weight.shape != (1, input.shape[1]):
        raise ValueError("Scalar projection requires (tokens, hidden) input and (1, hidden) weight")
    if not input.is_cuda or weight.device != input.device or weight.dtype != input.dtype:
        raise ValueError("Scalar projection requires CUDA input and matching weight dtype/device")
    if not input.is_floating_point():
        raise ValueError("Scalar projection requires floating-point activations")
    shape = (input.shape[0], 1)
    if out is None:
        out = alloc_func(shape, dtype=input.dtype, device=input.device)
    elif out.shape != shape or out.dtype != input.dtype or out.device != input.device:
        raise ValueError("Scalar projection output must match the unpadded shape, dtype and device")
    if input.shape[0]:
        block = triton.next_power_of_2(max(1, input.shape[1]))
        _scalar_linear[(input.shape[0],)](
            input,
            weight,
            out,
            input.shape[1],
            input.stride(0),
            input.stride(1),
            weight.stride(1),
            out.stride(0),
            block,
            num_warps=4 if block <= 8192 else 8,
            enable_fp_fusion=False,
        )
    return out
