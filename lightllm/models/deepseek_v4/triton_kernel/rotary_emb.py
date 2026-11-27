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
    INTERLEAVED: tl.constexpr,
):
    row = tl.program_id(0) * BLOCK_ROWS + tl.arange(0, BLOCK_ROWS)
    pair = tl.arange(0, BLOCK_PAIRS)
    token = row // HEADS
    head = row % HEADS
    pair_mask = (row[:, None] < num_rows) & (pair[None, :] < PAIRS)
    base = token[:, None] * stride_xt + head[:, None] * stride_xh
    if INTERLEAVED:
        dim = tl.arange(0, 2 * BLOCK_PAIRS)
        dim_mask = row[:, None] < num_rows
        offset = base + dim[None, :]
        values = tl.load(X + offset, dim_mask, 0).to(tl.float32)
        real, imag = tl.split(tl.reshape(values, (BLOCK_ROWS, BLOCK_PAIRS, 2)))
    else:
        offset = base + pair[None, :] * 2 * stride_xd
        real = tl.load(X + offset, mask=pair_mask, other=0).to(tl.float32)
        imag = tl.load(X + offset + stride_xd, mask=pair_mask, other=0).to(tl.float32)
    cos = tl.load(Cos + token[:, None] * stride_ct + pair[None, :] * stride_cd, pair_mask, 1).to(tl.float32)
    sin = tl.load(Sin + token[:, None] * stride_st + pair[None, :] * stride_sd, pair_mask, 0).to(tl.float32)
    if INTERLEAVED:
        rotated = tl.join(real * cos + imag * sin, imag * cos - real * sin)
        tl.store(X + offset, tl.reshape(rotated, (BLOCK_ROWS, 2 * BLOCK_PAIRS)), dim_mask)
    else:
        tl.store(X + offset, real * cos + imag * sin, mask=pair_mask)
        tl.store(X + offset + stride_xd, imag * cos - real * sin, mask=pair_mask)


@torch.no_grad()
def inverse_rotary_emb(x, cos, sin):
    """Apply inverse interleaved RoPE in place to a strided [tokens, heads, rotary_dim] view.

    cos and sin are [tokens, rotary_dim // 2], including real/imag views of complex tables.
    """
    rows = x.shape[0] * x.shape[1]
    if rows:
        interleaved = x.dtype == torch.bfloat16 and x.shape[2] == 64 and x.stride(-1) == 1
        block_rows = 16 if interleaved else 4
        _inverse_rotary_kernel[(triton.cdiv(rows, block_rows),)](
            x,
            cos,
            sin,
            *x.stride(),
            *cos.stride(),
            *sin.stride(),
            rows,
            HEADS=x.shape[1],
            PAIRS=x.shape[2] // 2,
            BLOCK_ROWS=block_rows,
            BLOCK_PAIRS=triton.next_power_of_2(x.shape[2] // 2),
            INTERLEAVED=interleaved,
            num_warps=4,
        )
