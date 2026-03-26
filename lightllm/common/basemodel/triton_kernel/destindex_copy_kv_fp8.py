import torch

import triton
import triton.language as tl


@triton.jit
def _fwd_kernel_destindex_copy_kv_per_head_fp8(
    KV,
    Dest_loc,
    Out,
    scale,
    stride_kv_bs,
    stride_kv_h,
    stride_kv_d,
    stride_o_bs,
    stride_o_h,
    stride_o_d,
    head_num,
    BLOCK_DMODEL: tl.constexpr,
    BLOCK_HEAD: tl.constexpr,
    FP8_MIN: tl.constexpr,
    FP8_MAX: tl.constexpr,
    IS_PER_TENSOR_QUANT: tl.constexpr,
):
    cur_index = tl.program_id(0)
    offs_h = tl.arange(0, BLOCK_HEAD)
    offs_d = tl.arange(0, BLOCK_DMODEL)

    dest_index = tl.load(Dest_loc + cur_index).to(tl.int64)

    kv_ptrs = KV + cur_index * stride_kv_bs + stride_kv_h * offs_h[:, None] + stride_kv_d * offs_d[None, :]
    o_ptrs = Out + dest_index * stride_o_bs + stride_o_h * offs_h[:, None] + stride_o_d * offs_d[None, :]

    # to fp8
    if not IS_PER_TENSOR_QUANT:
        scale_ptrs = scale + offs_h
        scales = tl.load(scale_ptrs, mask=offs_h < head_num, other=1.0)
    else:
        # k, v 各一个scale
        scale_ptrs = scale + tl.where(offs_h < head_num // 2, 0, 1)
        scales = tl.load(scale_ptrs)

    kv = tl.load(kv_ptrs, mask=offs_h[:, None] < head_num, other=0.0)
    kv_scale = kv / scales[:, None]
    kv_fp8 = tl.clamp(kv_scale, min=FP8_MIN, max=FP8_MAX).to(tl.float8e4nv)

    tl.store(o_ptrs, kv_fp8, mask=offs_h[:, None] < head_num)
    return


@torch.no_grad()
def destindex_copy_kv_fp8(KV, DestLoc, scales, Out, is_per_tensor_quant=False):
    """
    当 is_per_tensor_quant 为 False 时，为 为 per_head 量化。
    """
    seq_len = DestLoc.shape[0]
    head_num = KV.shape[1]
    assert head_num % 2 == 0
    head_dim = KV.shape[2]
    assert KV.shape[1] == Out.shape[1] and KV.shape[2] == Out.shape[2]
    BLOCK_HEAD = triton.next_power_of_2(head_num)
    grid = (seq_len,)
    num_warps = 1

    _fwd_kernel_destindex_copy_kv_per_head_fp8[grid](
        KV,
        DestLoc,
        Out,
        scales,
        KV.stride(0),
        KV.stride(1),
        KV.stride(2),
        Out.stride(0),
        Out.stride(1),
        Out.stride(2),
        head_num,
        BLOCK_DMODEL=head_dim,
        BLOCK_HEAD=BLOCK_HEAD,
        FP8_MIN=torch.finfo(torch.float8_e4m3fn).min,
        FP8_MAX=torch.finfo(torch.float8_e4m3fn).max,
        IS_PER_TENSOR_QUANT=is_per_tensor_quant,
        num_warps=num_warps,
        num_stages=1,
    )


if __name__ == "__main__":
    import torch.nn.functional as F
    from lightllm.utils.vllm_utils import vllm_ops

    B, N_CTX, H, HEAD_DIM = 32, 1024, 16, 128
    dtype = torch.bfloat16
    NUM = B
    dest_loc = torch.arange(NUM).cuda() * 2
    kv = torch.randn((len(dest_loc), H, HEAD_DIM), dtype=dtype).cuda()
    out = torch.zeros((B * N_CTX, H, HEAD_DIM), dtype=torch.uint8).cuda()
    scale = kv.abs().amax(dim=(0, 2)).to(torch.float32) / 448
    destindex_copy_kv_fp8(kv, dest_loc, scale, out.view(torch.float8_e4m3fn))

    assert torch.allclose(
        out[:, :, :HEAD_DIM][dest_loc].view(torch.float8_e4m3fn).float() * scale.view(H, 1).expand(NUM, H, 1),
        kv.float(),
        atol=1e-5,
        rtol=1e-1,
    )
