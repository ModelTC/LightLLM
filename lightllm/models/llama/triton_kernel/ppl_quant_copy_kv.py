import torch

import triton
import triton.language as tl


@triton.jit
def _fwd_kernel_destindex_copy_quantize_kv(
    K, V, Dest_loc, Out_k, Out_scale_k, Out_v, Out_scale_v,
    stride_k_bs, stride_k_h, stride_k_g, stride_k_d,
    stride_v_bs, stride_v_h, stride_v_g, stride_v_d,
    stride_o_k_bs, stride_o_k_h, stride_o_k_g, stride_o_k_d,
    stride_os_k_bs, stride_os_k_h, stride_os_k_g,
    stride_o_v_bs, stride_o_v_h, stride_o_v_g, stride_o_v_d,
    stride_os_v_bs, stride_os_v_h, stride_os_v_g,
    group_size,
    BLOCK_GROUP_NUM: tl.constexpr,
    BLOCK_GROUP_DIM: tl.constexpr 
):
    cur_index = tl.program_id(0)
    cur_head = tl.program_id(1)
     
    offs_g = tl.arange(0, BLOCK_GROUP_NUM)
    offs_d = tl.arange(0, BLOCK_GROUP_DIM)

    dest_index = tl.load(Dest_loc + cur_index)

    src_data = tl.load(K + cur_index * stride_k_bs + cur_head * stride_k_h + offs_g[:, None] * stride_k_g + offs_d[None, :], 
                       mask=offs_g[:, None] < group_size, other=0.0)
    abs_data = tl.abs(src_data)
    data_scale = (tl.max(abs_data, axis=1) / 127.).to(tl.float16)
    q_src_data = (src_data / data_scale[:, None]).to(tl.int8)

    o_k_ptrs = Out_k + dest_index * stride_o_k_bs + cur_head * stride_o_k_h + offs_g[:, None] * stride_o_k_g  +  offs_d[None, :]
    os_k_ptrs = Out_scale_k + dest_index * stride_os_k_bs + cur_head * stride_os_k_h + offs_g
    tl.store(o_k_ptrs, q_src_data, mask=offs_g[:, None]<group_size)
    tl.store(os_k_ptrs, data_scale)

    src_data = tl.load(V + cur_index * stride_v_bs + cur_head * stride_v_h + offs_g[:, None] * stride_v_g + offs_d[None, :],
                        mask=offs_g[:, None] < group_size, other=0.0)
    abs_data = tl.abs(src_data)
    data_scale = (tl.max(abs_data, axis=1) / 127.).to(tl.float16)
    q_src_data = (src_data / data_scale[:, None]).to(tl.int8)

    o_v_ptrs = Out_v + dest_index * stride_o_v_bs + cur_head * stride_o_v_h + offs_g[:, None] * stride_o_v_g + offs_d[None, :]
    os_v_ptrs = Out_scale_v + dest_index * stride_os_v_bs + cur_head * stride_os_v_h + offs_g
    tl.store(o_v_ptrs, q_src_data, mask=offs_g[:, None] < group_size)
    tl.store(os_v_ptrs, data_scale)
    return


@torch.no_grad()
def destindex_copy_quantize_kv(K, V, DestLoc, OutK, Out_scale_k, OutV, Out_scale_v):
    seq_len = DestLoc.shape[0]
    head_num = K.shape[1]
    head_dim = K.shape[2]
    quant_group_dim = 8

    assert head_dim % quant_group_dim == 0, "error head dim, can not been supported to copy quant kv"
    grid = (seq_len, head_num)
    num_warps = 1

    group_size = head_dim // quant_group_dim
    group_dim = quant_group_dim

    K = K.view((K.shape[0], K.shape[1], group_size, group_dim))
    V = V.view((V.shape[0], V.shape[1], group_size, group_dim))
    OutK = OutK.view(OutK.shape[0], OutK.shape[1], group_size, group_dim)
    OutV = OutV.view(OutV.shape[0], OutV.shape[1], group_size, group_dim)

    _fwd_kernel_destindex_copy_quantize_kv[grid](
        K, V, DestLoc, OutK, Out_scale_k, OutV, Out_scale_v,
        K.stride(0), K.stride(1), K.stride(2), K.stride(3),
        V.stride(0), V.stride(1), V.stride(2), V.stride(3),
        OutK.stride(0), OutK.stride(1), OutK.stride(2), OutK.stride(3),
        Out_scale_k.stride(0), Out_scale_k.stride(1), Out_scale_k.stride(2),
        OutV.stride(0), OutV.stride(1), OutV.stride(2), OutV.stride(3),
        Out_scale_v.stride(0), Out_scale_v.stride(1), Out_scale_v.stride(2),
        group_size,
        BLOCK_GROUP_NUM=triton.next_power_of_2(group_size),
        BLOCK_GROUP_DIM=group_dim, 
        num_warps=num_warps,
        num_stages=1,
    )
    return


def test2():
    import time

    B, N_CTX, H, D = 4,512, 12, 128
    src = torch.randn((B * N_CTX, H, D), dtype=torch.float16).cuda()
    src_v = torch.randn((B * N_CTX, H, D), dtype=torch.float16).cuda()
    dest_loc = torch.arange(0, B * N_CTX, dtype=torch.int32).cuda()
    value_dest = torch.randn((B * N_CTX, H, D), dtype=torch.float16).cuda().to(torch.int8)
    value_dest_v = torch.randn((B * N_CTX, H, D), dtype=torch.float16).cuda().to(torch.int8)
    scale_dest = torch.randn((B * N_CTX, H, D // 8), dtype=torch.float16).cuda()
    scale_dest_v = torch.randn((B * N_CTX, H, D // 8), dtype=torch.float16).cuda()

    for _ in range(10):
        destindex_copy_quantize_kv(src, src_v, dest_loc, value_dest, scale_dest, value_dest_v, scale_dest_v)
    torch.cuda.synchronize()
    t1 = time.time()
    for _ in range(1000):
        destindex_copy_quantize_kv(src, src_v, dest_loc, value_dest, scale_dest, value_dest_v, scale_dest_v)
    torch.cuda.synchronize()
    t2 = time.time()

    print("Time cost ", t2 - t1)
    value_dest = value_dest.view((B * N_CTX, H, D // 8, 8))
    scale_dest = scale_dest.view((B * N_CTX, H, D // 8, 1))
    print("max ", torch.max(torch.abs((value_dest * scale_dest).view(B * N_CTX, H, D) - src)))
    print("mean ", torch.mean(torch.abs((value_dest * scale_dest).view(B * N_CTX, H, D) - src)))
    cos = torch.nn.CosineSimilarity(0)
    print("cos ", cos(src.flatten().to(torch.float32), (value_dest * scale_dest).flatten().to(torch.float32)))


if __name__ == '__main__':
    test2()
