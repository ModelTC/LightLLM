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
    stride_l,
    stride_h,
    stride_d,
    stride_cos_l,
    stride_cos_d,
    stride_sin_l,
    stride_sin_d,
    max_L,
    D: tl.constexpr,
    HALF_D: tl.constexpr,
    BLOCK_D: tl.constexpr,
    BLOCK_L: tl.constexpr,
):
    pid_h = tl.program_id(0).to(tl.int64)
    pid_l_blk = tl.program_id(1).to(tl.int64)
    pid_blk_d = tl.program_id(2).to(tl.int64)

    offs_d = tl.arange(0, BLOCK_D)
    d = pid_blk_d * BLOCK_D + offs_d
    mask = d < D

    seq_start = pid_l_blk * BLOCK_L
    seq_end = (pid_l_blk + 1) * BLOCK_L
    seq_end = tl.where(seq_end < max_L, seq_end, max_L)

    for seq in tl.range(seq_start, seq_end):
        base = seq * stride_l + pid_h * stride_h

        in_ptr = inp_ptr + base + d * stride_d
        cos_ptr_ = cos_ptr + seq * stride_cos_l + d * stride_cos_d
        sin_ptr_ = sin_ptr + seq * stride_sin_l + d * stride_sin_d

        x = tl.load(in_ptr, mask=mask, other=0.0)
        cos = tl.load(cos_ptr_, mask=mask, other=0.0)
        sin = tl.load(sin_ptr_, mask=mask, other=0.0)

        partner_d = tl.where(d < HALF_D, d + HALF_D, d - HALF_D)
        partner_ptr = inp_ptr + base + partner_d * stride_d
        partner_val = tl.load(partner_ptr, mask=mask, other=0.0)
        rotated = tl.where(d < HALF_D, -partner_val, partner_val)

        y = x * cos + rotated * sin
        out_ptr_ = out_ptr + base + d * stride_d
        tl.store(out_ptr_, y, mask=mask)


def apply_rotary_pos_emb_triton(
    tensor: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor, BLOCK_D: int = 128
) -> torch.Tensor:
    assert tensor.is_cuda and cos.is_cuda and sin.is_cuda
    assert cos.is_contiguous() and sin.is_contiguous()
    if tensor.ndim != 3:
        raise RuntimeError("tensor shape should be [L, H, D]")
    orig_dtype = tensor.dtype
    x = tensor.float()

    cos = cos.repeat(1, 2).view(cos.size(0), -1).contiguous().float()
    sin = sin.repeat(1, 2).view(sin.size(0), -1).contiguous().float()

    L, H, D = x.shape
    HALF_D = D // 2
    y = torch.empty_like(x)

    BLOCK_L = 16
    grid = (H, triton.cdiv(L, BLOCK_L), triton.cdiv(D, BLOCK_D))

    rotary_kernel[grid](
        inp_ptr=x,
        cos_ptr=cos,
        sin_ptr=sin,
        out_ptr=y,
        stride_l=x.stride(0),
        stride_h=x.stride(1),
        stride_d=x.stride(2),
        stride_cos_l=cos.stride(0),
        stride_cos_d=cos.stride(1),
        stride_sin_l=sin.stride(0),
        stride_sin_d=sin.stride(1),
        max_L=L,
        D=D,
        HALF_D=HALF_D,
        BLOCK_D=BLOCK_D,
        BLOCK_L=BLOCK_L,
    )

    return y.to(orig_dtype)
