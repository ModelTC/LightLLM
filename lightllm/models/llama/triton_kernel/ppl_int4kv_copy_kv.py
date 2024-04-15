import torch

import triton
import triton.language as tl


@triton.jit
def _fwd_kernel_destindex_copy_quantize_int4_kv(
    K,
    Dest_loc,
    Out,
    Out_scale,
    stride_k_bs,
    stride_k_h,
    stride_k_g,
    stride_k_d,
    stride_o_bs,
    stride_o_h,
    stride_o_g,
    stride_o_d,
    stride_os_bs,
    stride_os_h,
    stride_os_g,
    group_size,
    BLOCK_GROUP_NUM: tl.constexpr,
    BLOCK_GROUP_DIM: tl.constexpr,
):
    cur_index = tl.program_id(0)
    cur_head = tl.program_id(1)

    offs_g = tl.arange(0, BLOCK_GROUP_NUM)
    offs_d = tl.arange(0, BLOCK_GROUP_DIM // 2)

    dest_index = tl.load(Dest_loc + cur_index)

    src_data_0 = tl.load(
        K + cur_index * stride_k_bs + cur_head * stride_k_h + offs_g[:, None] * stride_k_g + offs_d[None, :] * 2,
        mask=offs_g[:, None] < group_size,
        other=0.0,
    )
    src_data_1 = tl.load(
        K + cur_index * stride_k_bs + cur_head * stride_k_h + offs_g[:, None] * stride_k_g + offs_d[None, :] * 2 + 1,
        mask=offs_g[:, None] < group_size,
        other=0.0,
    )

    abs_data_0 = tl.abs(src_data_0)
    abs_data_1 = tl.abs(src_data_1)

    data_scale = (tl.maximum(tl.max(abs_data_0, axis=1), tl.max(abs_data_1, axis=1)) / 7.0).to(Out_scale.dtype.element_ty)
    q_src_data_0 = (src_data_0 / data_scale[:, None]).to(tl.int8)
    q_src_data_0 = tl.where(q_src_data_0 > 7, 7, q_src_data_0)
    q_src_data_0 = tl.where(q_src_data_0 < -7, -7, q_src_data_0)

    q_src_data_1 = (src_data_1 / data_scale[:, None]).to(tl.int8)
    q_src_data_1 = tl.where(q_src_data_1 > 7, 7, q_src_data_1)
    q_src_data_1 = tl.where(q_src_data_1 < -7, -7, q_src_data_1)

    low_4 = ((q_src_data_0 & 0x80) >> 4) | (q_src_data_0 & 0xF)
    high_4 = (((q_src_data_1 & 0x80) >> 4) | (q_src_data_1 & 0xF)) << 4

    # tl.device_print(low_4)
    # tl.device_print(high_4)

    out_data = low_4 | high_4

    o_ptrs = Out + dest_index * stride_o_bs + cur_head * stride_o_h + offs_g[:, None] * stride_o_g + offs_d[None, :]
    os_ptrs = Out_scale + dest_index * stride_os_bs + cur_head * stride_os_h + offs_g
    tl.store(o_ptrs, out_data, mask=offs_g[:, None] < group_size)
    tl.store(os_ptrs, data_scale, mask=offs_g < group_size)
    return


@torch.no_grad()
def destindex_copy_int4kv(K, DestLoc, Out, Out_scale):
    # seq_len = DestLoc.shape[0]
    # head_num = K.shape[1]
    head_dim = K.shape[2]
    quant_group_dim = 8

    assert head_dim % quant_group_dim == 0, "error head dim, can not been supported to copy quant kv"
    # grid = (seq_len, head_num)
    # num_warps = 1

    group_size = head_dim // quant_group_dim
    group_dim = quant_group_dim

    K = K.view((K.shape[0], K.shape[1], group_size, group_dim))
    Out = Out.view(
        Out.shape[0], Out.shape[1], group_size, group_dim // 2
    )  # OUt 是 int8 类型， 两个int4组一个int8，所以 group_dim // 2

    from lightllm_ppl_int4kv_flashdecoding_kernel import group8_copy_int4_kv

    group8_copy_int4_kv(Out, Out_scale, K, DestLoc, 4)

    # _fwd_kernel_destindex_copy_quantize_int4_kv[grid](
    #     K,
    #     DestLoc,
    #     Out,
    #     Out_scale,
    #     K.stride(0),
    #     K.stride(1),
    #     K.stride(2),
    #     K.stride(3),
    #     Out.stride(0),
    #     Out.stride(1),
    #     Out.stride(2),
    #     Out.stride(3),
    #     Out_scale.stride(0),
    #     Out_scale.stride(1),
    #     Out_scale.stride(2),
    #     group_size,
    #     BLOCK_GROUP_NUM=triton.next_power_of_2(group_size),
    #     BLOCK_GROUP_DIM=group_dim,
    #     num_warps=num_warps,
    #     num_stages=1,
    # )
    return


def test2():
    import time

    src = torch.randn((1, 1, 8), dtype=torch.float16).cuda()
    src[0, 0, :] = torch.tensor([1, -2, 2, 0, 4, 5, 6, 7]).cuda()
    dest_loc = torch.arange(0, 1, dtype=torch.int32).cuda()
    value_dest = torch.randn((1, 1, 4), dtype=torch.float16).cuda().to(torch.int8)
    scale_dest = torch.randn((1, 1, 1), dtype=torch.float16).cuda()

    destindex_copy_int4kv(src, dest_loc, value_dest, scale_dest)

    print(value_dest)
    print(scale_dest)


if __name__ == "__main__":
    test2()
