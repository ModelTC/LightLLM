import torch
import triton
import triton.language as tl

# LayerNorm adapted from triton tutorial, used for Cohere q, k norm
# X [N, head_num, head_dim]
# W [head_num, head_dim]
@triton.jit
def _layer_norm_fwd_kernel(
    X,  # pointer to the input
    W,  # pointer to the weights
    Y,
    stride_x_N,
    stride_x_hn,
    stride_x_hd,
    stride_y_N,
    stride_y_hn,
    stride_y_hd,
    stride_w_hn,
    stride_w_hd,
    N,  # number of columns in X
    eps,  # epsilon to avoid division by zero
    BLOCK_SIZE: tl.constexpr,
):
    Seq = tl.program_id(0)
    H = tl.program_id(1)

    X += Seq * stride_x_N + H * stride_x_hn
    Y += Seq * stride_y_N + H * stride_y_hn
    W += H * stride_w_hn

    _mean = tl.zeros([BLOCK_SIZE], dtype=tl.float32)
    for off in range(0, N, BLOCK_SIZE):
        cols = off + tl.arange(0, BLOCK_SIZE)
        a = tl.load(X + cols, mask=cols < N, other=0.0).to(tl.float32)
        _mean += a
    mean = tl.sum(_mean, axis=0) / N

    _var = tl.zeros([BLOCK_SIZE], dtype=tl.float32)
    for off in range(0, N, BLOCK_SIZE):
        cols = off + tl.arange(0, BLOCK_SIZE)
        x = tl.load(X + cols, mask=cols < N, other=0.0).to(tl.float32)
        x = tl.where(cols < N, x - mean, 0.0)
        _var += x * x
    var = tl.sum(_var, axis=0) / N
    rstd = 1 / tl.sqrt(var + eps)

    for off in range(0, N, BLOCK_SIZE):
        cols = off + tl.arange(0, BLOCK_SIZE)
        mask = cols < N
        w = tl.load(W + cols, mask=mask).to(tl.float32)
        x = tl.load(X + cols, mask=mask, other=0.0).to(tl.float32)
        x_hat = (x - mean) * rstd
        y = x_hat * w

        tl.store(Y + cols, y.to(X.dtype.element_ty), mask=mask)


def layernorm_forward(
    X,  # pointer to the input
    W,  # pointer to the weights
    eps,  # epsilon to avoid division by zero
):
    assert len(X.shape) == 3
    assert len(W.shape) == 2
    assert X.shape[-1] == W.shape[-1]
    assert X.shape[-2] == W.shape[-2]

    y = torch.empty_like(X)

    stride_x_N = X.stride(0)
    stride_x_hn = X.stride(1)
    stride_x_hd = X.stride(2)

    stride_y_N = y.stride(0)
    stride_y_hn = y.stride(1)
    stride_y_hd = y.stride(2)

    stride_w_hn = W.stride(0)
    stride_w_hd = W.stride(1)

    N = X.shape[-1]
    BLOCK_SIZE = 128

    grid = (X.shape[0], X.shape[1])
    _layer_norm_fwd_kernel[grid](
        X,
        W,
        y,
        stride_x_N,
        stride_x_hn,
        stride_x_hd,
        stride_y_N,
        stride_y_hn,
        stride_y_hd,
        stride_w_hn,
        stride_w_hd,
        N,
        eps,
        BLOCK_SIZE,
    )

    return y


def torch_layernorm(x, weight, eps):
    inp_dtype = x.dtype
    x = x.to(torch.float32)
    mean = x.mean(-1, keepdim=True)
    variance = (x - mean).pow(2).mean(-1, keepdim=True)
    x = (x - mean) * torch.rsqrt(variance + eps)
    x = weight.to(torch.float32) * x
    return x.to(inp_dtype)


def test_layernorm(eps=1e-5):
    # create data
    dtype = torch.float16
    x_shape = (5, 1, 128)
    w_shape = (x_shape[-2], x_shape[-1])
    weight = torch.rand(w_shape, dtype=dtype, device="cuda")
    x = -2.3 + 0.5 * torch.randn(x_shape, dtype=dtype, device="cuda")
    # forward pass
    y_ref = torch_layernorm(x, weight, eps).to(dtype)
    y_out = layernorm_forward(x, weight, eps)

    # compare
    print("type:", y_out.dtype, y_ref.dtype)
    print("max delta:", torch.max(torch.abs(y_out - y_ref)))
    assert torch.allclose(y_out, y_ref, atol=1e-2, rtol=0)
    return
