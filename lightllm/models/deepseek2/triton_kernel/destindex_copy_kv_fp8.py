import torch

import triton
import triton.language as tl


@triton.jit
def _fwd_kernel_destindex_copy_kv_fp8(
    KV_nope,
    KV_rope,
    Dest_loc,
    O_nope,
    O_rope,
    O_scale,
    stride_kv_nope_bs,
    stride_kv_nope_h,
    stride_kv_nope_d,
    stride_kv_rope_bs,
    stride_kv_rope_h,
    stride_kv_rope_d,
    stride_o_nope_bs,
    stride_o_nope_h,
    stride_o_nope_d,
    stride_o_rope_bs,
    stride_o_rope_h,
    stride_o_rope_d,
    stride_o_scale_bs,
    stride_o_scale_h,
    stride_o_scale_d,
    BLOCK_DMODEL_NOPE: tl.constexpr,
    BLOCK_DMODEL_ROPE: tl.constexpr,
    FP8_MIN: tl.constexpr,
    FP8_MAX: tl.constexpr,
):
    cur_index = tl.program_id(0)
    offs_d_nope = tl.arange(0, BLOCK_DMODEL_NOPE)
    offs_d_rope = tl.arange(0, BLOCK_DMODEL_ROPE)

    dest_index = tl.load(Dest_loc + cur_index)

    kv_nope_ptrs = KV_nope + cur_index * stride_kv_nope_bs + stride_kv_nope_d * offs_d_nope[None, :]
    kv_rope_ptrs = KV_rope + cur_index * stride_kv_rope_bs + stride_kv_rope_d * offs_d_rope[None, :]

    o_nope_ptrs = O_nope + dest_index * stride_o_nope_bs + stride_o_nope_d * offs_d_nope[None, :]
    o_rope_ptrs = O_rope + dest_index * stride_o_rope_bs + stride_o_rope_d * offs_d_rope[None, :]

    # to fp8
    kv_nope = tl.load(kv_nope_ptrs)
    kv_rope = tl.load(kv_rope_ptrs)
    max_nope = tl.max(tl.abs(kv_nope), axis=1)
    max_rope = tl.max(tl.abs(kv_rope), axis=1)
    max_kv = tl.maximum(tl.maximum(max_nope, max_rope), 1e-12)
    kv_scale = (max_kv / FP8_MAX).to(kv_nope.dtype)
    kv_nope_fp8 = tl.clamp(kv_nope / kv_scale, min=FP8_MIN, max=FP8_MAX).to(tl.float8e4nv)
    kv_rope_fp8 = tl.clamp(kv_rope / kv_scale, min=FP8_MIN, max=FP8_MAX).to(tl.float8e4nv)

    # save kv_scale
    offs_d_scale = tl.arange(0, 1)
    o_scale_ptrs = O_scale + dest_index * stride_o_scale_bs + stride_o_scale_d * offs_d_scale
    tl.store(o_scale_ptrs, kv_scale)

    # save fp8 kv
    tl.store(o_nope_ptrs, kv_nope_fp8)
    tl.store(o_rope_ptrs, kv_rope_fp8)
    return


@torch.no_grad()
def destindex_copy_kv_fp8(KV_nope, KV_rope, DestLoc, O_nope, O_rope, O_scale):
    seq_len = DestLoc.shape[0]
    kv_nope_head_dim = KV_nope.shape[2]
    kv_rope_head_dim = KV_rope.shape[2]

    assert KV_nope.shape[1] == O_nope.shape[1]
    assert KV_nope.shape[2] == O_nope.shape[2]
    assert KV_rope.shape[1] == O_rope.shape[1]
    assert KV_rope.shape[2] == O_rope.shape[2]
    grid = (seq_len,)
    num_warps = 1

    _fwd_kernel_destindex_copy_kv_fp8[grid](
        KV_nope,
        KV_rope,
        DestLoc,
        O_nope,
        O_rope,
        O_scale,
        KV_nope.stride(0),
        KV_nope.stride(1),
        KV_nope.stride(2),
        KV_rope.stride(0),
        KV_rope.stride(1),
        KV_rope.stride(2),
        O_nope.stride(0),
        O_nope.stride(1),
        O_nope.stride(2),
        O_rope.stride(0),
        O_rope.stride(1),
        O_rope.stride(2),
        O_scale.stride(0),
        O_scale.stride(1),
        O_scale.stride(2),
        BLOCK_DMODEL_NOPE=kv_nope_head_dim,
        BLOCK_DMODEL_ROPE=kv_rope_head_dim,
        FP8_MIN=torch.finfo(torch.float8_e4m3fn).min,
        FP8_MAX=torch.finfo(torch.float8_e4m3fn).max,
        num_warps=num_warps,
        num_stages=1,
    )
    return


def test():
    import torch.nn.functional as F
    from .destindex_copy_kv import destindex_copy_kv

    B, N_CTX, H, NOPE_HEAD, ROPE_HEAD = 32, 1024, 1, 512, 64
    dtype = torch.bfloat16
    dest_loc = torch.randint(0, 100, (50,), device="cuda").unique()
    kv = torch.randn((len(dest_loc), H, NOPE_HEAD + ROPE_HEAD), dtype=dtype).cuda()
    O_nope = torch.zeros((B * N_CTX, H, NOPE_HEAD), dtype=dtype).cuda()
    O_rope = torch.zeros((B * N_CTX, H, ROPE_HEAD), dtype=dtype).cuda()

    kv_nope = kv[:, :, :NOPE_HEAD]
    kv_rope = kv[:, :, NOPE_HEAD:]
    destindex_copy_kv(kv_nope, kv_rope, dest_loc, O_nope, O_rope)

    assert torch.allclose(O_nope[dest_loc], kv_nope, atol=1e-2, rtol=0)
    assert torch.allclose(O_rope[dest_loc], kv_rope, atol=1e-2, rtol=0)


if __name__ == "__main__":
    import torch.nn.functional as F

    B, N_CTX, H, NOPE_HEAD, ROPE_HEAD = 32, 1024, 1, 512, 64
    dtype = torch.bfloat16
    NUM = 20
    dest_loc = torch.arange(NUM).cuda()
    kv = torch.randn((len(dest_loc), H, NOPE_HEAD + ROPE_HEAD), dtype=dtype).cuda()
    out = torch.zeros((B * N_CTX, H, NOPE_HEAD + ROPE_HEAD + 2), dtype=torch.uint8).cuda()

    fp8_type = torch.float8_e4m3fn
    kv_nope = kv[:, :, :NOPE_HEAD]
    kv_rope = kv[:, :, NOPE_HEAD:]
    O_nope = out[:, :, :NOPE_HEAD].view(fp8_type)
    O_rope = out[:, :, NOPE_HEAD:-2].view(fp8_type)
    O_scale = out[:, :, -2:].view(dtype)
    destindex_copy_kv_fp8(kv_nope, kv_rope, dest_loc, O_nope, O_rope, O_scale)

    cos1 = F.cosine_similarity(O_nope[:NUM].to(dtype) * O_scale[:NUM], kv_nope).mean()
    cos2 = F.cosine_similarity(O_rope[:NUM].to(dtype) * O_scale[:NUM], kv_rope).mean()
    print(cos1, cos2)
