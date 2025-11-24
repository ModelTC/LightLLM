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
    D: tl.constexpr,
    HALF_D: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    # program_id(0) : seq idx)
    # program_id(1) : head idx)
    # program_id(2) : 块化的 d 维
    pid_l = tl.program_id(0).to(tl.int64)
    pid_h = tl.program_id(1).to(tl.int64)
    pid_blk = tl.program_id(2).to(tl.int64)

    # 当前 block 处理的 d 下标
    offs_d = tl.arange(0, BLOCK_D)
    d = pid_blk * BLOCK_D + offs_d
    mask = d < D

    # base 是 (l, h, 0) 的起始地址
    base = pid_l * stride_l + pid_h * stride_h

    # ---- load x / cos / sin ----
    in_ptr = inp_ptr + base + d * stride_d

    cos_ptr_ = cos_ptr + pid_l * stride_cos_l + d * stride_cos_d
    sin_ptr_ = sin_ptr + pid_l * stride_sin_l + d * stride_sin_d

    x = tl.load(in_ptr, mask=mask, other=0.0)
    cos = tl.load(cos_ptr_, mask=mask, other=0.0)
    sin = tl.load(sin_ptr_, mask=mask, other=0.0)

    # ---- rotary partner (偶/奇半维互换) ----
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
    x = tensor.float()  # [L, H, D]

    # cos/sin: [L, D/2] -> [L, D]
    cos = cos.repeat(1, 2).view(cos.size(0), -1).contiguous().float()
    sin = sin.repeat(1, 2).view(sin.size(0), -1).contiguous().float()

    L, H, D = x.shape
    HALF_D = D // 2
    y = torch.empty_like(x)

    grid = (L, H, triton.cdiv(D, BLOCK_D))

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
        D=D,
        HALF_D=HALF_D,
        BLOCK_D=BLOCK_D,
    )

    return y.to(orig_dtype)
