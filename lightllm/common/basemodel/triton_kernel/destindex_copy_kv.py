import torch

import triton
import triton.language as tl


@triton.jit
def _fwd_kernel_destindex_copy_kv(
    K, Dest_loc,
    Out,
    stride_k_bs, stride_k_h, stride_k_d,
    stride_o_bs, stride_o_h, stride_o_d,
    head_num,
    BLOCK_DMODEL: tl.constexpr,
    BLOCK_HEAD: tl.constexpr
):
    cur_index = tl.program_id(0)
    offs_h = tl.arange(0, BLOCK_HEAD)
    offs_d = tl.arange(0, BLOCK_DMODEL)

    dest_index = tl.load(Dest_loc + cur_index)

    k_ptrs = K + cur_index * stride_k_bs + stride_k_h * offs_h[:, None] + stride_k_d * offs_d[None, :]
    o_ptrs = Out + dest_index * stride_o_bs + stride_o_h * offs_h[:, None] + stride_o_d * offs_d[None, :]

    k = tl.load(k_ptrs, mask=offs_h[:, None] < head_num, other=0.0)
    tl.store(o_ptrs, k, mask=offs_h[:, None] < head_num)
    return


@torch.no_grad()
def destindex_copy_kv(K, DestLoc, Out):
    seq_len = DestLoc.shape[0]
    head_num = K.shape[1]
    head_dim = K.shape[2]
    assert K.shape[1] == Out.shape[1] and K.shape[2] == Out.shape[2]
    BLOCK_HEAD = triton.next_power_of_2(head_num)
    grid = (seq_len,)
    num_warps = 1

    _fwd_kernel_destindex_copy_kv[grid](
        K, DestLoc, Out,
        K.stride(0), K.stride(1), K.stride(2),
        Out.stride(0), Out.stride(1), Out.stride(2),
        head_num,
        BLOCK_DMODEL=head_dim,
        BLOCK_HEAD=BLOCK_HEAD,
        num_warps=num_warps,
        num_stages=1,
    )
    return


@triton.jit
def _fwd_kernel_destindex_copy_quantize_kv(
    K, Dest_loc, Out, Out_scale,
    stride_k_bs, stride_k_h, stride_k_d,
    stride_o_bs, stride_o_h, stride_o_d,
    stride_os_bs, stride_os_h, stride_os_d,
    head_num,
    BLOCK_DMODEL: tl.constexpr,
    BLOCK_HEAD: tl.constexpr
):
    cur_index = tl.program_id(0)
    offs_h = tl.arange(0, BLOCK_HEAD)
    offs_d = tl.arange(0, BLOCK_DMODEL)

    dest_index = tl.load(Dest_loc + cur_index)
    src_data = tl.load(K + cur_index * stride_k_bs + offs_h[:, None] * stride_k_h + stride_k_d * offs_d[None, :], 
                       mask=offs_h[:, None] < head_num, other=0.0)
    abs_data = tl.abs(src_data)
    data_scale = (tl.max(abs_data, axis=1) / 127.).to(tl.float16)[:, None]
    q_src_data = (src_data / data_scale).to(tl.int8)
    o_ptrs = Out + dest_index * stride_o_bs + stride_o_h * offs_h[:, None] + stride_o_d * offs_d[None, :]
    os_ptrs = Out_scale + dest_index * stride_os_bs + stride_os_h * offs_h[:, None]
    tl.store(o_ptrs, q_src_data, mask=offs_h[:, None] < head_num)
    tl.store(os_ptrs, data_scale, mask=offs_h[:, None] < head_num)


@torch.no_grad()
def destindex_copy_quantize_kv(K, DestLoc, Out, Out_scale):
    seq_len = DestLoc.shape[0]
    head_num = K.shape[1]
    head_dim = K.shape[2]
    assert K.shape[1] == Out.shape[1] and K.shape[2] == Out.shape[2]
    BLOCK_HEAD = triton.next_power_of_2(head_num)
    grid = (seq_len,)
    num_warps = 1

    _fwd_kernel_destindex_copy_quantize_kv[grid](
        K, DestLoc, Out, Out_scale,
        K.stride(0), K.stride(1), K.stride(2),
        Out.stride(0), Out.stride(1), Out.stride(2),
        Out_scale.stride(0), Out_scale.stride(1), Out_scale.stride(2),
        head_num,
        BLOCK_DMODEL=head_dim,
        BLOCK_HEAD=BLOCK_HEAD,
        num_warps=num_warps,
        num_stages=1,
    )
    return


def test1():
    import time

    B, N_CTX, H, D = 32, 1024, 12, 128
    dest = torch.randn((B * N_CTX, H, D), dtype=torch.float16).cuda()
    src = torch.randn((B * N_CTX, H, D), dtype=torch.float16).cuda()
    dest_loc = torch.arange(0, B * N_CTX, dtype=torch.int32, device="cuda")

    for _ in range(10):
        destindex_copy_kv(src, dest_loc, dest)
    torch.cuda.synchronize()
    t1 = time.time()
    for _ in range(1000):
        destindex_copy_kv(src, dest_loc, dest)
    torch.cuda.synchronize()
    t2 = time.time()

    print("Time cost ", t2 - t1)
    print("max ", torch.max(torch.abs(dest - src)))
    print("mean ", torch.mean(torch.abs(dest - src)))
    assert torch.allclose(src, dest, atol=1e-2, rtol=0)


def test2():
    import time

    B, N_CTX, H, D = 32, 1024, 12, 128
    src = torch.randn((B * N_CTX, H, D), dtype=torch.float16).cuda()
    dest_loc = torch.arange(0, B * N_CTX, dtype=torch.int32).cuda()
    value_dest = torch.randn((B * N_CTX, H, D), dtype=torch.float16).cuda().to(torch.int8)
    scale_dest = torch.randn((B * N_CTX, H, 1), dtype=torch.float16).cuda()

    for _ in range(10):
        destindex_copy_quantize_kv(src, dest_loc, value_dest, scale_dest)
    torch.cuda.synchronize()
    t1 = time.time()
    for _ in range(1000):
        destindex_copy_quantize_kv(src, dest_loc, value_dest, scale_dest)
    torch.cuda.synchronize()
    t2 = time.time()

    print("Time cost ", t2 - t1)
    print("max ", torch.max(torch.abs(value_dest * scale_dest - src)))
    print("mean ", torch.mean(torch.abs(value_dest * scale_dest - src)))
    cos = torch.nn.CosineSimilarity(0)
    print("cos ", cos(src.flatten().to(torch.float32), (value_dest * scale_dest).flatten().to(torch.float32)))


if __name__ == '__main__':
    test1()
    test2()
