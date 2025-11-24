import triton
import triton.language as tl
import torch


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
    L,
    H,
    D,
    BLOCK_SEQ: tl.constexpr,
    BLOCK_HEAD: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    pid_head_blk = tl.program_id(0)
    pid_seq_blk = tl.program_id(1)

    offs_h = pid_head_blk * BLOCK_HEAD + tl.arange(0, BLOCK_HEAD)
    offs_l = pid_seq_blk * BLOCK_SEQ + tl.arange(0, BLOCK_SEQ)
    offs_d = tl.arange(0, BLOCK_D)

    offs_h = offs_h.to(tl.int64)
    offs_l = offs_l.to(tl.int64)
    offs_d = offs_d.to(tl.int64)

    mask_h = offs_h < H
    mask_l = offs_l < L
    mask_d = offs_d < D

    HALF_D = D // 2

    l_b = offs_l[:, None, None]
    h_b = offs_h[None, :, None]
    d_b = offs_d[None, None, :]

    mask = mask_l[:, None, None] & mask_h[None, :, None] & mask_d[None, None, :]

    base = l_b * stride_l + h_b * stride_h + d_b * stride_d
    x = tl.load(inp_ptr + base, mask=mask, other=0.0)

    cos_base_2d = offs_l[:, None] * stride_cos_l + offs_d[None, :] * stride_cos_d
    sin_base_2d = offs_l[:, None] * stride_sin_l + offs_d[None, :] * stride_sin_d
    mask_ld = mask_l[:, None] & mask_d[None, :]

    cos_2d = tl.load(cos_ptr + cos_base_2d, mask=mask_ld, other=0.0)
    sin_2d = tl.load(sin_ptr + sin_base_2d, mask=mask_ld, other=0.0)

    cos = cos_2d[:, None, :]
    sin = sin_2d[:, None, :]

    partner_d = tl.where(offs_d < HALF_D, offs_d + HALF_D, offs_d - HALF_D)
    partner_d_b = partner_d[None, None, :]

    partner_base = l_b * stride_l + h_b * stride_h + partner_d_b * stride_d
    partner_val = tl.load(inp_ptr + partner_base, mask=mask, other=0.0)

    rotated = tl.where(d_b < HALF_D, -partner_val, partner_val)

    y = x * cos + rotated * sin

    tl.store(out_ptr + base, y, mask=mask)


def apply_rotary_pos_emb_triton(
    tensor: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
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
    y = torch.empty_like(x)

    BLOCK_SEQ = 16
    BLOCK_HEAD = 4
    BLOCK_D = triton.next_power_of_2(D)

    if D >= 128:
        num_warps = 8
    else:
        num_warps = 4

    grid = (
        triton.cdiv(H, BLOCK_HEAD),
        triton.cdiv(L, BLOCK_SEQ),
    )

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
        L=L,
        H=H,
        D=D,
        BLOCK_SEQ=BLOCK_SEQ,
        BLOCK_HEAD=BLOCK_HEAD,
        BLOCK_D=BLOCK_D,
        num_warps=num_warps,
    )

    return y.to(orig_dtype)
