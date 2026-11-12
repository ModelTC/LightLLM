import torch
import triton
import triton.language as tl


@triton.jit
def _gated_norm(
    X,
    Gate,
    Weight,
    Out,
    input_stride_row,
    gate_stride_token,
    gate_stride_head,
    out_stride_row,
    GATE_HEADS: tl.constexpr,
    D: tl.constexpr,
    EPS: tl.constexpr,
    SIGMOID: tl.constexpr,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0).to(tl.int64)
    col = tl.arange(0, BLOCK)
    x = tl.load(X + row * input_stride_row + col, col < D, other=0).to(tl.float32)
    inv = tl.rsqrt(tl.sum(x * x, 0) / D + EPS)

    normalized = (x * inv).to(X.dtype.element_ty).to(tl.float32)
    weight = tl.load(Weight + col, col < D, other=0).to(tl.float32)
    weighted = (normalized * weight).to(X.dtype.element_ty).to(tl.float32)
    gate_token = row // GATE_HEADS
    gate_head = row % GATE_HEADS
    gate = tl.load(
        Gate + gate_token * gate_stride_token + gate_head * gate_stride_head + col,
        col < D,
        other=0,
    ).to(tl.float32)
    activation = tl.sigmoid(gate)
    if not SIGMOID:
        activation = gate * activation
    tl.store(Out + row * out_stride_row + col, weighted * activation, col < D)


@torch.no_grad()
def gated_norm(input: torch.Tensor, gate: torch.Tensor, weight: torch.Tensor, eps: float, activation: str, out=None):
    if activation not in ("sigmoid", "silu"):
        raise ValueError("Qwen4Exp GDN output gate must be sigmoid or silu")
    if (
        input.ndim != 2
        or input.shape[-1] != weight.numel()
        or gate.ndim not in (2, 3)
        or gate.shape[-1] != input.shape[-1]
        or (gate.ndim == 2 and gate.shape != input.shape)
        or (gate.ndim == 3 and gate.shape[0] * gate.shape[1] != input.shape[0])
        or not input.is_contiguous()
        or gate.stride(-1) != 1
    ):
        raise ValueError("GDN norm requires a contiguous input, a packed gate, and a per-head norm weight")
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
    rows = input.shape[0]
    gate_heads = 1 if gate.ndim == 2 else gate.shape[1]
    gate_stride_token = gate.stride(0)
    gate_stride_head = 0 if gate.ndim == 2 else gate.stride(1)
    if rows:
        _gated_norm[(rows,)](
            input,
            gate,
            weight,
            result,
            input.stride(0),
            gate_stride_token,
            gate_stride_head,
            result.stride(0),
            GATE_HEADS=gate_heads,
            D=input.shape[-1],
            EPS=eps,
            SIGMOID=activation == "sigmoid",
            BLOCK=triton.next_power_of_2(input.shape[-1]),
            enable_fp_fusion=False,
        )
    return result
