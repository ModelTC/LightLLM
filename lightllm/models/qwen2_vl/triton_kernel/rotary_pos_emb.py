import math
import torch
import triton
import triton.language as tl


@triton.jit
def rotary_kernel(
    inp_ptr,
    cos_ptr,
    sin_ptr,
    out_ptr,
    stride_b,
    stride_l,
    stride_h,
    stride_cos_l,
    stride_sin_l,
    H: tl.constexpr,
    D: tl.constexpr,
    HALF_D: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    pid_bh = tl.program_id(0)
    pid_l = tl.program_id(1)
    pid_blk = tl.program_id(2)

    b = pid_bh // H
    h = pid_bh - b * H

    offs_d = tl.arange(0, BLOCK_D)
    d = pid_blk * BLOCK_D + offs_d
    mask = d < D

    # int64计算基址，防止b*stride_b溢出
    base = (
        tl.full([], b, tl.int64) * tl.full([], stride_b, tl.int64)
        + tl.full([], pid_l, tl.int64) * tl.full([], stride_l, tl.int64)
        + tl.full([], h, tl.int64) * tl.full([], stride_h, tl.int64)
    )

    in_ptr = inp_ptr + base + d
    cos_ptr_ = cos_ptr + tl.full([], pid_l, tl.int64) * tl.full([], stride_cos_l, tl.int64) + d
    sin_ptr_ = sin_ptr + tl.full([], pid_l, tl.int64) * tl.full([], stride_sin_l, tl.int64) + d

    x = tl.load(in_ptr, mask=mask)
    cos = tl.load(cos_ptr_, mask=mask)
    sin = tl.load(sin_ptr_, mask=mask)

    partner_d = tl.where(d < HALF_D, d + HALF_D, d - HALF_D)
    partner_ptr = inp_ptr + base + partner_d
    partner_val = tl.load(partner_ptr, mask=mask)
    rotated = tl.where(d < HALF_D, -partner_val, partner_val)

    y = x * cos + rotated * sin

    out_ptr_ = out_ptr + base + d
    tl.store(out_ptr_, y, mask=mask)


def apply_rotary_pos_emb_triton(
    tensor: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor, BLOCK_D: int = 128
) -> torch.Tensor:
    assert tensor.is_cuda and cos.is_cuda and sin.is_cuda
    if tensor.ndim != 4:
        raise RuntimeError("tensor shape should be [B, L, H, D]")
    orig_dtype = tensor.dtype
    x = tensor.float()

    cos = cos.unsqueeze(1).repeat(1, 1, 2).view(cos.size(0), -1).contiguous().float()
    sin = sin.unsqueeze(1).repeat(1, 1, 2).view(sin.size(0), -1).contiguous().float()

    B, L, H, D = x.shape
    HALF_D = D // 2
    y = torch.empty_like(x)

    stride_b, stride_l, stride_h, _ = x.stride()
    stride_cos_l, stride_sin_l = cos.stride(0), sin.stride(0)

    grid = (B * H, L, math.ceil(D / BLOCK_D))

    rotary_kernel[grid](
        x,
        cos,
        sin,
        y,
        stride_b,
        stride_l,
        stride_h,
        stride_cos_l,
        stride_sin_l,
        H,
        D,
        HALF_D,
        BLOCK_D=BLOCK_D,
    )

    return y.to(orig_dtype)
