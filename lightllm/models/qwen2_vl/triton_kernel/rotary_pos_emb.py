import math
import torch
import triton
import triton.language as tl


def rotate_half(x):
    """Rotates half the hidden dims of the input."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb_vision_ref(tensor: torch.Tensor, freqs: torch.Tensor) -> torch.Tensor:
    orig_dtype = tensor.dtype
    tensor = tensor.float()
    cos = freqs.cos()
    sin = freqs.sin()
    cos = cos.unsqueeze(1).repeat(1, 1, 2).unsqueeze(0).float()
    sin = sin.unsqueeze(1).repeat(1, 1, 2).unsqueeze(0).float()
    output = (tensor * cos) + (rotate_half(tensor) * sin)
    output = output.to(orig_dtype)
    return output


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

    # 64-bit 计算基址，防止 b*stride_b 溢出
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


def apply_rotary_pos_emb_triton(tensor: torch.Tensor, freqs: torch.Tensor, BLOCK_D: int = 128) -> torch.Tensor:
    assert tensor.is_cuda and freqs.is_cuda
    if tensor.ndim != 4:
        raise RuntimeError("tensor shape should be [B, L, H, D]")
    orig_dtype = tensor.dtype
    x = tensor.float()

    cos = freqs.cos().unsqueeze(1).repeat(1, 1, 2).view(freqs.size(0), -1).contiguous().float()
    sin = freqs.sin().unsqueeze(1).repeat(1, 1, 2).view(freqs.size(0), -1).contiguous().float()

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


def test_accuracy_and_speed(
    B: int = 16,
    L: int = 1296,
    H: int = 64,
    D: int = 80,
    warmup: int = 10,
    repeats: int = 50,
):
    torch.manual_seed(0)
    freqs = torch.randn(L, D // 2, device="cuda")
    x = torch.randn(B, L, H, D, device="cuda")

    # 误差
    y_ref = apply_rotary_pos_emb_vision_ref(x, freqs)
    y_tri = apply_rotary_pos_emb_triton(x, freqs)
    print("max abs error:", (y_ref - y_tri).abs().max().item())

    # 预热
    for _ in range(warmup):
        apply_rotary_pos_emb_vision_ref(x, freqs)
        apply_rotary_pos_emb_triton(x, freqs)
    torch.cuda.synchronize()

    # 计时
    def bench(fn):
        start = torch.cuda.Event(True)
        end = torch.cuda.Event(True)
        start.record()
        for _ in range(repeats):
            fn(x, freqs)
        end.record()
        torch.cuda.synchronize()
        return start.elapsed_time(end) / repeats  # ms

    print(f"PyTorch  : {bench(apply_rotary_pos_emb_vision_ref):.3f} ms")
    print(f"Triton   : {bench(apply_rotary_pos_emb_triton):.3f} ms")


# if __name__ == "__main__":
#     test_accuracy_and_speed()
