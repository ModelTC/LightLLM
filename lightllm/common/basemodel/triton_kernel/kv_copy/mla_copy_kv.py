import torch

import triton
import triton.language as tl


def _is_power_of_two(n):
    return n > 0 and (n & (n - 1)) == 0


@triton.jit
def _fwd_kernel_destindex_copy_kv(
    KV_nope,
    KV_rope,
    Dest_loc,
    O_nope,
    O_rope,
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
    BLOCK_DMODEL_NOPE: tl.constexpr,
    BLOCK_DMODEL_ROPE: tl.constexpr,
):
    cur_index = tl.program_id(0)
    offs_d_nope = tl.arange(0, BLOCK_DMODEL_NOPE)
    offs_d_rope = tl.arange(0, BLOCK_DMODEL_ROPE)

    dest_index = tl.load(Dest_loc + cur_index).to(tl.int64)

    kv_nope_ptrs = KV_nope + cur_index * stride_kv_nope_bs + stride_kv_nope_d * offs_d_nope
    kv_rope_ptrs = KV_rope + cur_index * stride_kv_rope_bs + stride_kv_rope_d * offs_d_rope

    o_nope_ptrs = O_nope + dest_index * stride_o_nope_bs + stride_o_nope_d * offs_d_nope
    o_rope_ptrs = O_rope + dest_index * stride_o_rope_bs + stride_o_rope_d * offs_d_rope

    kv_nope = tl.load(kv_nope_ptrs)
    kv_rope = tl.load(kv_rope_ptrs)

    tl.store(o_nope_ptrs, kv_nope)
    tl.store(o_rope_ptrs, kv_rope)
    return


@torch.no_grad()
def destindex_copy_kv(KV_nope, KV_rope, DestLoc, O_nope, O_rope):
    seq_len = DestLoc.shape[0]
    kv_nope_head_dim = KV_nope.shape[2]
    kv_rope_head_dim = KV_rope.shape[2]

    assert KV_nope.shape[1] == O_nope.shape[1]
    assert KV_nope.shape[2] == O_nope.shape[2]
    assert KV_rope.shape[1] == O_rope.shape[1]
    assert KV_rope.shape[2] == O_rope.shape[2]
    assert triton.next_power_of_2(kv_nope_head_dim) == kv_nope_head_dim
    assert triton.next_power_of_2(kv_rope_head_dim) == kv_rope_head_dim

    grid = (seq_len,)
    num_warps = 1

    _fwd_kernel_destindex_copy_kv[grid](
        KV_nope,
        KV_rope,
        DestLoc,
        O_nope,
        O_rope,
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
        BLOCK_DMODEL_NOPE=kv_nope_head_dim,
        BLOCK_DMODEL_ROPE=kv_rope_head_dim,
        num_warps=num_warps,
        num_stages=1,
    )
    return


if __name__ == "__main__":
    import torch.nn.functional as F

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
