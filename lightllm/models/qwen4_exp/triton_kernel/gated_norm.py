import torch
import triton
import triton.language as tl


@triton.jit
def _gated_norm(X, Gate, Weight, Out, D: tl.constexpr, EPS: tl.constexpr, SIGMOID: tl.constexpr, BLOCK: tl.constexpr):
    row = tl.program_id(0).to(tl.int64)
    col = tl.arange(0, BLOCK)
    x = tl.load(X + row * D + col, col < D, other=0).to(tl.float32)
    inv = tl.rsqrt(tl.sum(x * x, 0) / D + EPS)

    normalized = (x * inv).to(X.dtype.element_ty).to(tl.float32)
    weight = tl.load(Weight + col, col < D, other=0).to(tl.float32)
    weighted = (normalized * weight).to(X.dtype.element_ty).to(tl.float32)
    gate = tl.load(Gate + row * D + col, col < D, other=0).to(tl.float32)
    activation = tl.sigmoid(gate)
    if not SIGMOID:
        activation = gate * activation
    tl.store(Out + row * D + col, weighted * activation, col < D)


@torch.no_grad()
def gated_norm(input: torch.Tensor, gate: torch.Tensor, weight: torch.Tensor, eps: float, activation: str, out=None):
    if activation not in ("sigmoid", "silu"):
        raise ValueError("Qwen4Exp GDN output gate must be sigmoid or silu")
    if (
        input.shape != gate.shape
        or input.shape[-1] != weight.numel()
        or not input.is_contiguous()
        or not gate.is_contiguous()
    ):
        raise ValueError("GDN norm requires contiguous, matching input/gate tensors and a per-head norm weight")
    for tensor in (input, gate, weight):
        if not tensor.is_cuda or tensor.dtype != input.dtype or tensor.device != input.device:
            raise ValueError("GDN norm tensors must be CUDA tensors with matching device and dtype")
    if weight.ndim != 1 or not weight.is_contiguous() or input.ndim < 2 or eps <= 0:
        raise ValueError("GDN norm requires a weight vector, token/head rows and positive epsilon")
    result = torch.empty_like(input) if out is None else out
    if (
        result.shape != input.shape
        or result.dtype != input.dtype
        or result.device != input.device
        or not result.is_contiguous()
    ):
        raise ValueError("GDN norm output must be contiguous and match the input")
    rows = input.numel() // input.shape[-1]
    if rows:
        _gated_norm[(rows,)](
            input,
            gate,
            weight,
            result,
            input.shape[-1],
            eps,
            activation == "sigmoid",
            triton.next_power_of_2(input.shape[-1]),
            enable_fp_fusion=False,
        )
    return result
