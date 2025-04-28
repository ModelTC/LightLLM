import torch
import triton
import triton.language as tl
from torch import Tensor
from lightllm.common.basemodel.layer_infer.cache_tensor_manager import g_cache_manager


@triton.jit
def rms_norm_kernel(
    input,
    weight,
    output,
    in_row_stride: tl.constexpr,
    in_col_stride: tl.constexpr,
    out_row_stride: tl.constexpr,
    out_col_stride: tl.constexpr,
    eps: tl.constexpr,
    N_COLS: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    """Rms norm kernel."""
    prog_id = tl.program_id(0)
    offsets = tl.arange(0, BLOCK_N)

    w = tl.load(weight + offsets, mask=offsets < N_COLS, other=0.0)

    x_ptr = input + prog_id * in_row_stride
    x = tl.load(x_ptr + offsets * in_col_stride, mask=offsets < N_COLS, other=0.0)
    xf = x.to(tl.float32)

    var = tl.sum(xf * xf, 0) * float(1.0 / N_COLS)
    out = xf / tl.sqrt(var + eps)
    out = (w * out).to(x.dtype)

    out_ptr = output + prog_id * out_row_stride
    tl.store(out_ptr + offsets * out_col_stride, out, mask=offsets < N_COLS)


def rms_norm(hidden_states: Tensor, weight: Tensor, eps: float = 1e-5, use_custom_tensor_mananger: bool = False):
    """Rms norm."""

    assert hidden_states.is_contiguous(), "hidden_states must be contiguous"

    origin_shape = hidden_states.shape
    hidden_dim = weight.shape[0]
    assert hidden_dim == origin_shape[-1], f"hidden_dim {hidden_dim} != {origin_shape[-1]}"

    rows = hidden_states.numel() // hidden_dim
    if hidden_states.dim() == 3:  # (bs, seq_len, hidden_dim)
        hidden_states = hidden_states.view(rows, hidden_dim)

    in_row_stride, in_col_stride = hidden_states.stride(0), hidden_states.stride(1)

    BLOCK_N = triton.next_power_of_2(hidden_dim)
    if use_custom_tensor_mananger:
        shape = hidden_states.shape
        dtype = hidden_states.dtype
        device = hidden_states.device
        output = g_cache_manager.alloc_tensor(shape, dtype, device=device)
    else:
        output = torch.empty_like(hidden_states)

    out_row_stride, out_col_stride = output.stride(0), output.stride(1)
    grid = (rows,)
    rms_norm_kernel[grid](
        hidden_states,
        weight,
        output,
        in_row_stride,
        in_col_stride,
        out_row_stride,
        out_col_stride,
        eps=eps,
        N_COLS=hidden_dim,
        BLOCK_N=BLOCK_N,
        num_warps=4,
        num_stages=3,
    )
    return output.reshape(origin_shape)


def test():
    def _rms_norm_ref(x: torch.Tensor, weight: torch.Tensor, eps: float):
        var = (x.float() ** 2).mean(dim=-1, keepdim=True)
        y = x.float() / torch.sqrt(var + eps)
        return (y * weight).to(x.dtype)

    torch.manual_seed(0)
    device, dtype = "cuda", torch.float16
    bs, seq_len, hidden = 3, 1025, 3200
    eps = 1e-5
    weight = torch.randn(hidden, device=device, dtype=dtype)

    # 2-D contiguous
    x2 = torch.randn(seq_len, hidden, device=device, dtype=dtype).contiguous()
    assert torch.allclose(rms_norm(x2, weight, eps), _rms_norm_ref(x2, weight, eps), atol=1e-3, rtol=1e-3)

    # 3-D contiguous
    x3 = torch.randn(bs, seq_len, hidden, device=device, dtype=dtype).contiguous()
    assert torch.allclose(rms_norm(x3, weight, eps), _rms_norm_ref(x3, weight, eps), atol=1e-3, rtol=1e-3)

    print("all tests pass")
