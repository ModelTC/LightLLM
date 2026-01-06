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
    group_count,
    token_num,
    HEAD_NUM: tl.constexpr,
    BLOCK_GROUP_COUNT: tl.constexpr,
    BLOCK_GROUP_DIM: tl.constexpr,
):
    start_index = tl.program_id(0)

    for cur_index in range(start_index, token_num, step=tl.num_programs(axis=0)):
        offs_g = tl.arange(0, BLOCK_GROUP_COUNT) % group_count
        offs_d = tl.arange(0, BLOCK_GROUP_DIM // 2)

        dest_index = tl.load(Dest_loc + cur_index).to(tl.int64)

        for cur_head in tl.static_range(HEAD_NUM, step=1):
            src_data_0 = tl.load(
                K
                + cur_index * stride_k_bs
                + cur_head * stride_k_h
                + offs_g[:, None] * stride_k_g
                + offs_d[None, :] * 2,
                other=0.0,
            )
            src_data_1 = tl.load(
                K
                + cur_index * stride_k_bs
                + cur_head * stride_k_h
                + offs_g[:, None] * stride_k_g
                + offs_d[None, :] * 2
                + 1,
                other=0.0,
            )

            abs_data_0 = tl.abs(src_data_0)
            abs_data_1 = tl.abs(src_data_1)

            data_scale = (tl.maximum(tl.max(abs_data_0, axis=1), tl.max(abs_data_1, axis=1)) / 7.0).to(
                Out_scale.dtype.element_ty
            )
            q_src_data_0 = (src_data_0 / data_scale[:, None]).to(tl.int8)
            q_src_data_0 = tl.where(q_src_data_0 > 7, 7, q_src_data_0)
            q_src_data_0 = tl.where(q_src_data_0 < -7, -7, q_src_data_0)

            q_src_data_1 = (src_data_1 / data_scale[:, None]).to(tl.int8)
            q_src_data_1 = tl.where(q_src_data_1 > 7, 7, q_src_data_1)
            q_src_data_1 = tl.where(q_src_data_1 < -7, -7, q_src_data_1)

            low_4 = ((q_src_data_0 & 0x80) >> 4) | (q_src_data_0 & 0xF)
            high_4 = (((q_src_data_1 & 0x80) >> 4) | (q_src_data_1 & 0xF)) << 4

            out_data = low_4 | high_4

            o_ptrs = (
                Out + dest_index * stride_o_bs + cur_head * stride_o_h + offs_g[:, None] * stride_o_g + offs_d[None, :]
            )
            os_ptrs = Out_scale + dest_index * stride_os_bs + cur_head * stride_os_h + offs_g
            tl.store(o_ptrs, out_data)
            tl.store(os_ptrs, data_scale)
    return


@torch.no_grad()
def destindex_copy_int4kv(
    KV: torch.Tensor,
    DestLoc: torch.Tensor,
    KV_buffer: torch.Tensor,
    KV_scale_buffer: torch.Tensor,
    quant_group_size: int,
):
    head_num = KV.shape[1]
    head_dim = KV.shape[2]

    assert head_dim % quant_group_size == 0, "error head dim, can not been supported to copy quant kv"

    group_count = head_dim // quant_group_size
    group_dim = quant_group_size

    assert triton.next_power_of_2(group_dim) == group_dim

    KV = KV.view((KV.shape[0], head_num, group_count, group_dim))
    KV_buffer = KV_buffer.view(
        KV_buffer.shape[0], KV_buffer.shape[1], group_count, group_dim // 2
    )  # OUt 是 int8 类型， 两个int4组一个int8，所以 group_dim // 2
    KV_scale_buffer = KV_scale_buffer.view(KV_scale_buffer.shape[0], KV_scale_buffer.shape[1], group_count)
    if len(DestLoc) < 1024:
        grid = (len(DestLoc),)
    else:
        grid = (1024,)

    _fwd_kernel_destindex_copy_quantize_int4_kv[grid](
        K=KV,
        Dest_loc=DestLoc,
        Out=KV_buffer,
        Out_scale=KV_scale_buffer,
        stride_k_bs=KV.stride(0),
        stride_k_h=KV.stride(1),
        stride_k_g=KV.stride(2),
        stride_k_d=KV.stride(3),
        stride_o_bs=KV_buffer.stride(0),
        stride_o_h=KV_buffer.stride(1),
        stride_o_g=KV_buffer.stride(2),
        stride_o_d=KV_buffer.stride(3),
        stride_os_bs=KV_scale_buffer.stride(0),
        stride_os_h=KV_scale_buffer.stride(1),
        stride_os_g=KV_scale_buffer.stride(2),
        group_count=group_count,
        token_num=len(DestLoc),
        HEAD_NUM=head_num,
        BLOCK_GROUP_COUNT=triton.next_power_of_2(group_count),
        BLOCK_GROUP_DIM=triton.next_power_of_2(group_dim),
        num_warps=4,
        num_stages=1,
    )
    return
