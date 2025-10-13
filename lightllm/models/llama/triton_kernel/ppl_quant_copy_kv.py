import torch

import triton
import triton.language as tl


@triton.jit
def _fwd_kernel_destindex_copy_quantize_kv(
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
    offs_d = tl.arange(0, BLOCK_GROUP_DIM)

    dest_index = tl.load(Dest_loc + cur_index).to(tl.int64)

    src_data = tl.load(
        K + cur_index * stride_k_bs + cur_head * stride_k_h + offs_g[:, None] * stride_k_g + offs_d[None, :],
        mask=offs_g[:, None] < group_size,
        other=0.0,
    )
    abs_data = tl.abs(src_data)
    data_scale = (tl.max(abs_data, axis=1) / 127.0).to(Out_scale.dtype.element_ty)
    q_src_data = (src_data / data_scale[:, None]).to(tl.int8)

    o_ptrs = Out + dest_index * stride_o_bs + cur_head * stride_o_h + offs_g[:, None] * stride_o_g + offs_d[None, :]
    os_ptrs = Out_scale + dest_index * stride_os_bs + cur_head * stride_os_h + offs_g
    tl.store(o_ptrs, q_src_data, mask=offs_g[:, None] < group_size)
    tl.store(os_ptrs, data_scale, mask=offs_g < group_size)
    return


@torch.no_grad()
def destindex_copy_quantize_kv(K, DestLoc, Out, Out_scale):
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
    Out = Out.view(Out.shape[0], Out.shape[1], group_size, group_dim)

    _fwd_kernel_destindex_copy_quantize_kv[grid](
        K,
        DestLoc,
        Out,
        Out_scale,
        K.stride(0),
        K.stride(1),
        K.stride(2),
        K.stride(3),
        Out.stride(0),
        Out.stride(1),
        Out.stride(2),
        Out.stride(3),
        Out_scale.stride(0),
        Out_scale.stride(1),
        Out_scale.stride(2),
        group_size,
        BLOCK_GROUP_NUM=triton.next_power_of_2(group_size),
        BLOCK_GROUP_DIM=group_dim,
        num_warps=num_warps,
        num_stages=1,
    )
    return


@triton.jit
def _fwd_kernel_destindex_copy_dequantize_kv(
    mem_kv_buffer,
    mem_kv_scale,
    req_to_token_indexs,
    b_seq_len,
    b_req_idx,
    Out,
    stride_kv_b,
    stride_kv_h,
    stride_kv_g,
    stride_kv_d,
    stride_o_bh,
    stride_o_l,
    stride_o_g,
    stride_o_d,
    stride_s_b,
    stride_s_h,
    stride_s_g,
    stride_req_to_tokens_b,
    stride_req_to_tokens_s,
    group_size,
    head_num: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
    BLOCK_GROUP_NUM: tl.constexpr,
    BLOCK_GROUP_DIM: tl.constexpr,
):
    cur_group = tl.program_id(0)
    start_m = tl.program_id(1)
    cur_bh = tl.program_id(2)
    cur_batch = cur_bh // head_num
    cur_head = cur_bh % head_num

    block_start_loc = BLOCK_SIZE * start_m
    cur_batch_req_idx = tl.load(b_req_idx + cur_batch)
    cur_seq_len = tl.load(b_seq_len + cur_batch)

    # initialize offsets
    offs_kv_loc = block_start_loc + tl.arange(0, BLOCK_SIZE)

    # offs_g = tl.arange(0, BLOCK_GROUP_NUM)
    offs_d = tl.arange(0, BLOCK_GROUP_DIM)

    kv_loc = tl.load(
        req_to_token_indexs + cur_batch_req_idx * stride_req_to_tokens_b + offs_kv_loc, mask=offs_kv_loc < cur_seq_len
    ).to(tl.int64)
    offs_kv = kv_loc[:, None] * stride_kv_b + cur_head * stride_kv_h + cur_group * stride_kv_g + offs_d[None, :]

    src_data = tl.load(
        mem_kv_buffer + offs_kv,
        mask=offs_kv_loc[:, None] < cur_seq_len,
        other=0.0,
    ).to(Out.dtype.element_ty)

    s_ptrs = mem_kv_scale + kv_loc * stride_s_b + cur_head * stride_s_h + cur_group * stride_s_g
    data_scale = tl.load(
        s_ptrs,
        mask=offs_kv_loc < cur_seq_len,
    )

    out_data = src_data * data_scale[:, None]
    o_ptrs = Out + cur_bh * stride_o_bh + offs_kv_loc[:, None] * stride_o_l + cur_group * stride_o_g + offs_d[None, :]
    tl.store(o_ptrs, out_data, mask=offs_kv_loc[:, None] < cur_seq_len)
    return


@torch.no_grad()
def destindex_copy_dequantize_kv(
    mem_kv_buffer, mem_kv_scale, req_to_token_indexs, b_seq_len, b_req_idx, max_len_in_batch, Out
):
    batch_size = b_seq_len.shape[0]
    head_num = mem_kv_buffer.shape[1]
    head_dim = mem_kv_buffer.shape[2]
    quant_group_dim = 8
    BLOCK_SIZE = 128
    group_size = head_dim // quant_group_dim
    group_dim = quant_group_dim
    assert head_dim % quant_group_dim == 0, "error head dim, can not been supported to copy quant kv"
    grid = (group_size, triton.cdiv(max_len_in_batch, BLOCK_SIZE), batch_size * head_num)
    num_warps = 1
    mem_kv_buffer = mem_kv_buffer.view((mem_kv_buffer.shape[0], mem_kv_buffer.shape[1], group_size, group_dim))
    mem_kv_scale = mem_kv_scale.view((mem_kv_buffer.shape[0], mem_kv_buffer.shape[1], -1))
    Out = Out.view(Out.shape[0] * Out.shape[1], -1, group_size, group_dim)

    _fwd_kernel_destindex_copy_dequantize_kv[grid](
        mem_kv_buffer,
        mem_kv_scale,
        req_to_token_indexs,
        b_seq_len,
        b_req_idx,
        Out,
        mem_kv_buffer.stride(0),
        mem_kv_buffer.stride(1),
        mem_kv_buffer.stride(2),
        mem_kv_buffer.stride(3),
        Out.stride(0),
        Out.stride(1),
        Out.stride(2),
        Out.stride(3),
        mem_kv_scale.stride(0),
        mem_kv_scale.stride(1),
        mem_kv_scale.stride(2),
        req_to_token_indexs.stride(0),
        req_to_token_indexs.stride(1),
        group_size,
        head_num=head_num,
        BLOCK_SIZE=BLOCK_SIZE,
        BLOCK_GROUP_NUM=triton.next_power_of_2(group_size),
        BLOCK_GROUP_DIM=group_dim,
        num_warps=num_warps,
        num_stages=1,
    )
    return


def test2():
    import time

    B, N_CTX, H, D = 1, 3, 12, 128
    src = torch.randn((B * N_CTX, H, D), dtype=torch.float16).cuda()
    dest_loc = torch.arange(0, B * N_CTX, dtype=torch.int32).cuda()
    value_dest = torch.randn((B * N_CTX, H, D), dtype=torch.float16).cuda().to(torch.int8)
    scale_dest = torch.randn((B * N_CTX, H, D // 8), dtype=torch.float16).cuda()

    for _ in range(10):
        destindex_copy_quantize_kv(src, dest_loc, value_dest, scale_dest)
    torch.cuda.synchronize()
    t1 = time.time()
    for _ in range(1000):
        destindex_copy_quantize_kv(src, dest_loc, value_dest, scale_dest)
    torch.cuda.synchronize()
    t2 = time.time()

    print("Time cost ", t2 - t1)
    value_dest = value_dest.view((B * N_CTX, H, D // 8, 8))
    scale_dest = scale_dest.view((B * N_CTX, H, D // 8, 1))
    print("max ", torch.max(torch.abs((value_dest * scale_dest).view(B * N_CTX, H, D) - src)))
    print("mean ", torch.mean(torch.abs((value_dest * scale_dest).view(B * N_CTX, H, D) - src)))
    cos = torch.nn.CosineSimilarity(0)
    print("cos ", cos(src.flatten().to(torch.float32), (value_dest * scale_dest).flatten().to(torch.float32)))


def torch_dequant(kv, kv_scale, o, b_req_idx, b_seq_len, req_to_token_indexs):

    batch = b_req_idx.shape[0]
    for i in range(batch):
        req_idx = b_req_idx[i]
        seq_len = b_seq_len[i]
        print(seq_len, b_seq_len)
        kv_loc = req_to_token_indexs[req_idx, :seq_len]
        head_num = kv.shape[1]
        cur_kv = kv[kv_loc, :, :].reshape(seq_len, head_num, -1, 8).to(o.dtype)
        cur_scale = kv_scale[kv_loc, :, :].reshape(seq_len, head_num, -1, 1)
        out = cur_kv * cur_scale
        o[i, :seq_len, :, :] = out.reshape(out.shape[0], out.shape[1], -1)


def test3():
    import time
    import numpy as np

    Z, H, N_CTX, D_HEAD = 1, 16, 3, 128
    dtype = torch.bfloat16
    kv = torch.empty((Z * N_CTX + 100, 2 * H, D_HEAD), dtype=torch.int8, device="cuda")
    kv_scale = torch.randn((Z * N_CTX + 100, 2 * H, D_HEAD // 8), dtype=dtype, device="cuda")
    out = torch.empty((Z, 2 * H, N_CTX, D_HEAD), dtype=dtype, device="cuda").normal_(mean=0.4, std=0.2)
    torch_out = torch.empty((Z, N_CTX, 2 * H, D_HEAD), dtype=dtype, device="cuda").normal_(mean=0.3, std=0.2)
    req_to_token_indexs = torch.empty((1000, N_CTX + 7000), dtype=torch.int32, device="cuda")
    max_input_len = N_CTX
    b_seq_len = torch.ones((Z,), dtype=torch.int32, device="cuda")
    b_req_idx = torch.ones((Z,), dtype=torch.int32, device="cuda")
    for i in range(Z):
        seq_len = N_CTX - i * 100
        b_seq_len[i] = seq_len
        b_req_idx[i] = i
        req_to_token_indexs[i][:seq_len] = (
            torch.tensor(np.arange(seq_len), dtype=torch.int32).cuda() + b_seq_len[0:i].sum()
        )
    print(b_seq_len)
    destindex_copy_dequantize_kv(kv, kv_scale, req_to_token_indexs, b_seq_len, b_req_idx, max_input_len, out)
    torch_dequant(kv, kv_scale, torch_out, b_req_idx, b_seq_len, req_to_token_indexs)
    torch.cuda.synchronize()
    t1 = time.time()
    for _ in range(1000):
        destindex_copy_dequantize_kv(kv, kv_scale, req_to_token_indexs, b_seq_len, b_req_idx, max_input_len, out)
    torch.cuda.synchronize()
    t2 = time.time()
    print((t2 - t1))
    torch_out = torch_out.transpose(1, 2)
    for i in range(Z):
        print("max ", torch.max(torch.abs(torch_out - out)[i][:, : b_seq_len[i]]))
        print("mean ", torch.mean(torch.abs(torch_out - out)[i][:, : b_seq_len[i]]))
        assert torch.allclose(torch_out[i][:, : b_seq_len[i]], out[i][:, : b_seq_len[i]], atol=1e-2, rtol=0)
    # print("max ", torch.max(torch.abs((value_dest * scale_dest).view(B * N_CTX, H, D) - src)))
    # print("mean ", torch.mean(torch.abs((value_dest * scale_dest).view(B * N_CTX, H, D) - src)))
    # cos = torch.nn.CosineSimilarity(0)
    # print("cos ", cos(src.flatten().to(torch.float32), (value_dest * scale_dest).flatten().to(torch.float32)))


if __name__ == "__main__":
    test3()
