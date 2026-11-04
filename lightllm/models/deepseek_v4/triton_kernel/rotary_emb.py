import torch
import triton
import triton.language as tl


@triton.jit
def _inverse_rotary_kernel(
    X,
    Cos,
    Sin,
    stride_xt,
    stride_xh,
    stride_xd,
    stride_ct,
    stride_cd,
    stride_st,
    stride_sd,
    num_rows,
    HEADS: tl.constexpr,
    PAIRS: tl.constexpr,
    BLOCK_ROWS: tl.constexpr,
    BLOCK_PAIRS: tl.constexpr,
):
    row = tl.program_id(0) * BLOCK_ROWS + tl.arange(0, BLOCK_ROWS)
    pair = tl.arange(0, BLOCK_PAIRS)
    token = row // HEADS
    head = row % HEADS
    mask = (row[:, None] < num_rows) & (pair[None, :] < PAIRS)
    offset = token[:, None] * stride_xt + head[:, None] * stride_xh + pair[None, :] * 2 * stride_xd
    real = tl.load(X + offset, mask=mask, other=0).to(tl.float32)
    imag = tl.load(X + offset + stride_xd, mask=mask, other=0).to(tl.float32)
    cos = tl.load(Cos + token[:, None] * stride_ct + pair[None, :] * stride_cd, mask=mask, other=1).to(tl.float32)
    sin = tl.load(Sin + token[:, None] * stride_st + pair[None, :] * stride_sd, mask=mask, other=0).to(tl.float32)
    tl.store(X + offset, real * cos + imag * sin, mask=mask)
    tl.store(X + offset + stride_xd, imag * cos - real * sin, mask=mask)


@torch.no_grad()
def inverse_rotary_emb(x, cos, sin):
    """Apply inverse interleaved RoPE in place to a strided [tokens, heads, rotary_dim] view.

    cos and sin are [tokens, rotary_dim // 2], including real/imag views of complex tables.
    """
    rows = x.shape[0] * x.shape[1]
    if rows:
        _inverse_rotary_kernel[(triton.cdiv(rows, 4),)](
            x,
            cos,
            sin,
            *x.stride(),
            *cos.stride(),
            *sin.stride(),
            rows,
            HEADS=x.shape[1],
            PAIRS=x.shape[2] // 2,
            BLOCK_ROWS=4,
            BLOCK_PAIRS=triton.next_power_of_2(x.shape[2] // 2),
            num_warps=4,
        )
