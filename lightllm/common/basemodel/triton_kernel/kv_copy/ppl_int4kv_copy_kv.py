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
            )
            src_data_1 = tl.load(
                K
                + cur_index * stride_k_bs
                + cur_head * stride_k_h
                + offs_g[:, None] * stride_k_g
                + offs_d[None, :] * 2
                + 1,
            )

            abs_data_0 = tl.abs(src_data_0)
            abs_data_1 = tl.abs(src_data_1)

            data_scale = (tl.maximum(tl.max(abs_data_0, axis=1), tl.max(abs_data_1, axis=1)) / 7.0).to(
                Out_scale.dtype.element_ty
            )
            q_src_data_0 = (src_data_0 / data_scale[:, None]).to(tl.int8)
            q_src_data_0 = tl.where(q_src_data_0 > 7, 7, q_src_data_0)
            q_src_data_0 = tl.where(q_src_data_0 < -7, -7, q_src_data_0)
            q_src_data_0 += 7
            q_src_data_0 = q_src_data_0.to(tl.uint8, bitcast=True)

            q_src_data_1 = (src_data_1 / data_scale[:, None]).to(tl.int8)
            q_src_data_1 = tl.where(q_src_data_1 > 7, 7, q_src_data_1)
            q_src_data_1 = tl.where(q_src_data_1 < -7, -7, q_src_data_1)
            q_src_data_1 += 7
            q_src_data_1 = q_src_data_1.to(tl.uint8, bitcast=True)

            low_4 = q_src_data_0 & 0xF
            high_4 = (q_src_data_1 & 0xF) << 4

            out_data = (low_4 | high_4).to(Out.dtype.element_ty, bitcast=True)

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
        BLOCK_GROUP_DIM=group_dim,
        num_warps=4,
        num_stages=1,
    )
    return


@triton.jit
def int4_to_float(k_int8, offs_d):
    k_int8 = k_int8.to(tl.uint8, bitcast=True)
    k_high = (k_int8 & 0xF0) >> 4
    k_low = k_int8 & 0x0F
    k_high = k_high.to(tl.int8, bitcast=True)
    k_low = k_low.to(tl.int8, bitcast=True)
    k_high -= 7
    k_low -= 7

    k_int4 = tl.where(
        offs_d[None, None, :] % 2 == 0,
        k_low,
        k_high,
    )
    return k_int4


@triton.jit
def _fwd_dequantize_int4kv(
    k,
    k_ss,
    k_sh,
    k_sg,
    k_sd,
    k_scale,
    k_scale_ss,
    k_scale_sh,
    k_scale_sg,
    k_scale_sd,
    v,
    v_ss,
    v_sh,
    v_sg,
    v_sd,
    v_scale,
    v_scale_ss,
    v_scale_sh,
    v_scale_sg,
    v_scale_sd,
    req_to_token_indexs,
    stride_req_to_tokens_b,
    stride_req_to_tokens_s,
    b_seq_len,
    b_req_idx,
    b_kv_start_loc,
    k_out,
    k_out_ss,
    k_out_sh,
    k_out_sg,
    k_out_sd,
    v_out,
    v_out_ss,
    v_out_sh,
    v_out_sg,
    v_out_sd,
    k_head_num,
    v_head_num,
    group_count,
    group_dim,
    SEQ_BLOCK_SIZE: tl.constexpr,
    GROUP_COUNT_BLOCK_SIZE: tl.constexpr,
    BLOCK_GROUP_DIM: tl.constexpr,
):
    start_block_index = tl.program_id(0)
    cur_batch = tl.program_id(1)
    cur_batch_req_idx = tl.load(b_req_idx + cur_batch)
    cur_seq_len = tl.load(b_seq_len + cur_batch)
    if start_block_index * SEQ_BLOCK_SIZE >= cur_seq_len:
        return

    out_start_loc = tl.load(b_kv_start_loc + cur_batch)

    offs_kv_loc = (start_block_index * SEQ_BLOCK_SIZE + tl.arange(0, SEQ_BLOCK_SIZE)) % cur_seq_len
    kv_loc = tl.load(req_to_token_indexs + cur_batch_req_idx * stride_req_to_tokens_b + offs_kv_loc).to(tl.int64)

    offs_d = tl.arange(0, BLOCK_GROUP_DIM)
    offs_scale_d = tl.arange(0, 1)
    group_offs = tl.arange(0, GROUP_COUNT_BLOCK_SIZE) % group_count

    for k_head_index in tl.range(0, k_head_num, step=1, num_stages=3):
        k_int8 = tl.load(
            k
            + kv_loc[:, None, None] * k_ss
            + k_head_index * k_sh
            + group_offs[None, :, None] * k_sg
            + offs_d[None, None, :] // 2
        )
        k_int4 = int4_to_float(k_int8, offs_d)

        k_scale_data = tl.load(
            k_scale
            + kv_loc[:, None, None] * k_scale_ss
            + k_head_index * k_scale_sh
            + group_offs[None, :, None] * k_scale_sg
            + offs_scale_d[None, None, :]
        )
        k_out_data = k_int4.to(k_out.dtype.element_ty) * k_scale_data
        tl.store(
            k_out
            + (out_start_loc + offs_kv_loc[:, None, None]) * k_out_ss
            + k_head_index * k_out_sh
            + group_offs[None, :, None] * k_out_sg
            + offs_d[None, None, :],
            k_out_data,
        )

    for v_head_index in tl.range(0, v_head_num, step=1, num_stages=3):
        v_int8 = tl.load(
            v
            + kv_loc[:, None, None] * v_ss
            + v_head_index * v_sh
            + group_offs[None, :, None] * v_sg
            + offs_d[None, None, :] // 2
        )
        v_int4 = int4_to_float(v_int8, offs_d)
        v_scale_data = tl.load(
            v_scale
            + kv_loc[:, None, None] * v_scale_ss
            + v_head_index * v_scale_sh
            + group_offs[None, :, None] * v_scale_sg
            + offs_scale_d[None, None, :]
        )
        v_out_data = v_int4.to(v_out.dtype.element_ty) * v_scale_data
        tl.store(
            v_out
            + (out_start_loc + offs_kv_loc[:, None, None]) * v_out_ss
            + v_head_index * v_out_sh
            + group_offs[None, :, None] * v_out_sg
            + offs_d[None, None, :],
            v_out_data,
        )
    return


@torch.no_grad()
def dequantize_int4kv(
    k: torch.Tensor,
    k_scale: torch.Tensor,
    v: torch.Tensor,
    v_scale: torch.Tensor,
    req_to_token_indexs: torch.Tensor,
    b_seq_len: torch.Tensor,
    b_req_idx: torch.Tensor,
    b_kv_start_loc: torch.Tensor,
    k_out: torch.Tensor,
    v_out: torch.Tensor,
    max_len_in_batch: int,
    quant_group_size: int,
):
    batch_size = b_seq_len.shape[0]
    k_head_num = k.shape[1]
    k_head_dim = k.shape[2] * 2
    v_head_num = v.shape[1]
    v_head_dim = v.shape[2] * 2
    assert k_head_dim % quant_group_size == 0, "error head dim, can not been supported to copy quant kv"
    assert v_head_dim % quant_group_size == 0, "error head dim, can not been supported to copy quant kv"
    assert k_head_dim == v_head_dim, "error head dim, can not been supported to copy quant kv"
    assert k_head_dim // v_scale.shape[2] == quant_group_size, "error head dim, can not been supported to copy quant kv"
    assert k_head_dim in [64, 128, 256]

    group_count = k_head_dim // quant_group_size
    group_dim = quant_group_size

    assert triton.next_power_of_2(group_dim) == group_dim

    k = k.view((k.shape[0], k.shape[1], group_count, group_dim // 2))  # int4kv 以 int8 存储的
    v = v.view((v.shape[0], v.shape[1], group_count, group_dim // 2))
    k_scale = k_scale.view((k_scale.shape[0], k_scale.shape[1], group_count, 1))
    v_scale = v_scale.view((v_scale.shape[0], v_scale.shape[1], group_count, 1))

    # 使拆分的grid 具有足够的并行度
    SEQ_BLOCK_SIZE = 128
    while triton.cdiv(max_len_in_batch, SEQ_BLOCK_SIZE) * batch_size < 512:
        SEQ_BLOCK_SIZE = SEQ_BLOCK_SIZE // 2
        if SEQ_BLOCK_SIZE <= 1:
            break

    if SEQ_BLOCK_SIZE <= 1:
        SEQ_BLOCK_SIZE = 8

    grid = (triton.cdiv(max_len_in_batch, SEQ_BLOCK_SIZE), batch_size)
    num_warps = 4
    k_out = k_out.view((k_out.shape[0], k_out.shape[1], group_count, group_dim))
    v_out = v_out.view((v_out.shape[0], v_out.shape[1], group_count, group_dim))

    _fwd_dequantize_int4kv[grid](
        k=k,
        k_ss=k.stride(0),
        k_sh=k.stride(1),
        k_sg=k.stride(2),
        k_sd=k.stride(3),
        k_scale=k_scale,
        k_scale_ss=k_scale.stride(0),
        k_scale_sh=k_scale.stride(1),
        k_scale_sg=k_scale.stride(2),
        k_scale_sd=k_scale.stride(2),
        v=v,
        v_ss=v.stride(0),
        v_sh=v.stride(1),
        v_sg=v.stride(2),
        v_sd=v.stride(3),
        v_scale=v_scale,
        v_scale_ss=v_scale.stride(0),
        v_scale_sh=v_scale.stride(1),
        v_scale_sg=v_scale.stride(2),
        v_scale_sd=v_scale.stride(3),
        req_to_token_indexs=req_to_token_indexs,
        stride_req_to_tokens_b=req_to_token_indexs.stride(0),
        stride_req_to_tokens_s=req_to_token_indexs.stride(1),
        b_seq_len=b_seq_len,
        b_req_idx=b_req_idx,
        b_kv_start_loc=b_kv_start_loc,
        k_out=k_out,
        k_out_ss=k_out.stride(0),
        k_out_sh=k_out.stride(1),
        k_out_sg=k_out.stride(2),
        k_out_sd=k_out.stride(3),
        v_out=v_out,
        v_out_ss=v_out.stride(0),
        v_out_sh=v_out.stride(1),
        v_out_sg=v_out.stride(2),
        v_out_sd=v_out.stride(3),
        k_head_num=k_head_num,
        v_head_num=v_head_num,
        group_count=group_count,
        group_dim=group_dim,
        SEQ_BLOCK_SIZE=SEQ_BLOCK_SIZE,
        GROUP_COUNT_BLOCK_SIZE=triton.next_power_of_2(group_count),
        BLOCK_GROUP_DIM=group_dim,
        num_warps=num_warps,
        num_stages=1,
    )
    return
