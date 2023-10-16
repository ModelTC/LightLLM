import torch

import triton
import triton.language as tl
import math


@triton.jit
def _fwd_kernel_token_att1(
    Q, K, sm_scale, B_Loc, B_Start_Loc, B_Seqlen, max_input_len,
    Att_Out,
    stride_b_loc_b, stride_b_loc_s,
    stride_qbs, stride_qh, stride_qd,
    stride_kbs, stride_kh, stride_kd,
    att_stride_h, att_stride_bs,

    BLOCK_DMODEL: tl.constexpr,
    BLOCK_N: tl.constexpr
):
    cur_batch = tl.program_id(0)
    cur_head = tl.program_id(1)
    start_n = tl.program_id(2)

    offs_d = tl.arange(0, BLOCK_DMODEL)
    cur_batch_seq_len = tl.load(B_Seqlen + cur_batch)
    cur_batch_in_all_start_index = tl.load(B_Start_Loc + cur_batch)

    cur_batch_start_index = max_input_len - cur_batch_seq_len
    cur_batch_end_index = max_input_len

    off_q = cur_batch * stride_qbs + cur_head * stride_qh + offs_d * stride_qd

    offs_n = start_n * BLOCK_N + tl.arange(0, BLOCK_N)

    block_stard_index = start_n * BLOCK_N
    block_mask = tl.where(block_stard_index < cur_batch_seq_len, 1, 0)

    for start_mark in range(0, block_mask, 1):
        q = tl.load(Q + off_q + start_mark)
        offs_n_new = cur_batch_start_index + offs_n
        k_loc = tl.load(B_Loc + stride_b_loc_b * cur_batch + stride_b_loc_s * offs_n_new, mask=offs_n_new < cur_batch_end_index, other=0)
        off_k = k_loc[:, None] * stride_kbs + cur_head * stride_kh + offs_d[None, :] * stride_kd
        k = tl.load(K + off_k, mask=offs_n_new[:, None] < cur_batch_end_index, other=0.0)
        att_value = tl.sum(q[None, :] * k, 1)
        att_value *= sm_scale
        off_o = cur_head * att_stride_h + (cur_batch_in_all_start_index + offs_n) * att_stride_bs
        tl.store(Att_Out + off_o, att_value, mask=offs_n_new < cur_batch_end_index)
    return


@torch.no_grad()
def token_att_fwd(q, k, att_out, B_Loc, B_Start_Loc, B_Seqlen, max_input_len):
    BLOCK = 32
    # shape constraints
    Lq, Lk = q.shape[-1], k.shape[-1]
    assert Lq == Lk
    assert Lk in {16, 32, 64, 128}
    sm_scale = 1.0 / (Lk ** 0.5)

    batch, head_num = B_Loc.shape[0], q.shape[1]

    grid = (batch, head_num, triton.cdiv(max_input_len, BLOCK))

    num_warps = 4
    
    _fwd_kernel_token_att1[grid](
        q, k, sm_scale, B_Loc, B_Start_Loc, B_Seqlen, max_input_len,
        att_out,
        B_Loc.stride(0), B_Loc.stride(1),
        q.stride(0), q.stride(1), q.stride(2),
        k.stride(0), k.stride(1), k.stride(2),
        att_out.stride(0), att_out.stride(1),
        BLOCK_DMODEL=Lk,
        BLOCK_N=BLOCK,
        num_warps=num_warps,
        num_stages=1,
    )
    return


@triton.jit
def _fwd_kernel_token_att1_int8(
    Q, K, K_scale, sm_scale, B_Loc, B_Start_Loc, B_Seqlen, max_input_len,
    Att_Out,
    stride_b_loc_b, stride_b_loc_s,
    stride_qbs, stride_qh, stride_qd,
    stride_kbs, stride_kh, stride_kd,
    stride_ksbs, stride_ksh, stride_ksd,
    att_stride_h, att_stride_bs,
    BLOCK_DMODEL: tl.constexpr,
    BLOCK_N: tl.constexpr
):
    cur_batch = tl.program_id(0)
    cur_head = tl.program_id(1)
    start_n = tl.program_id(2)

    offs_d = tl.arange(0, BLOCK_DMODEL)
    cur_batch_seq_len = tl.load(B_Seqlen + cur_batch)
    cur_batch_in_all_start_index = tl.load(B_Start_Loc + cur_batch)

    cur_batch_start_index = max_input_len - cur_batch_seq_len
    cur_batch_end_index = max_input_len

    off_q = cur_batch * stride_qbs + cur_head * stride_qh + offs_d * stride_qd

    offs_n = start_n * BLOCK_N + tl.arange(0, BLOCK_N)

    block_stard_index = start_n * BLOCK_N
    block_mask = tl.where(block_stard_index < cur_batch_seq_len, 1, 0)

    for start_mark in range(0, block_mask, 1):
        q = tl.load(Q + off_q + start_mark)
        offs_n_new = cur_batch_start_index + offs_n
        k_loc = tl.load(B_Loc + stride_b_loc_b * cur_batch + stride_b_loc_s * offs_n_new, mask=offs_n_new < cur_batch_end_index, other=0)
        off_k = k_loc[:, None] * stride_kbs + cur_head * stride_kh + offs_d[None, :] * stride_kd
        k = tl.load(K + off_k, mask=offs_n_new[:, None] < cur_batch_end_index, other=0.0)
        off_ks = k_loc[:, None] * stride_ksbs + cur_head * stride_ksh
        k_scale = tl.load(K_scale + off_ks, mask=offs_n_new[:, None] < cur_batch_end_index, other=0.0)
        att_value = tl.sum(q[None, :] * k * k_scale, 1)
        att_value *= sm_scale
        off_o = cur_head * att_stride_h + (cur_batch_in_all_start_index + offs_n) * att_stride_bs
        tl.store(Att_Out + off_o, att_value, mask=offs_n_new < cur_batch_end_index)
    return


@torch.no_grad()
def token_att_fwd_int8k(q, k, k_scale, att_out, B_Loc, B_Start_Loc, B_Seqlen, max_input_len):
    BLOCK = 32
    # shape constraints
    Lq, Lk = q.shape[-1], k.shape[-1]
    assert Lq == Lk
    assert Lk in {16, 32, 64, 128}
    sm_scale = 1.0 / (Lk ** 0.5)

    batch, head_num = B_Loc.shape[0], q.shape[1]

    grid = (batch, head_num, triton.cdiv(max_input_len, BLOCK))

    num_warps = 4 if Lk <= 64 else 8
    num_warps = 2

    _fwd_kernel_token_att1_int8[grid](
        q, k, k_scale, sm_scale, B_Loc, B_Start_Loc, B_Seqlen, max_input_len,
        att_out,
        B_Loc.stride(0), B_Loc.stride(1),
        q.stride(0), q.stride(1), q.stride(2),
        k.stride(0), k.stride(1), k.stride(2),
        k_scale.stride(0), k_scale.stride(1), k_scale.stride(2),
        att_out.stride(0), att_out.stride(1),
        BLOCK_DMODEL=Lk,
        BLOCK_N=BLOCK,
        num_warps=num_warps,
        num_stages=1,
    )
    return


def torch_att(xq, xk, bs, seqlen, num_head, head_dim):
    xq = xq.view(bs, 1, num_head, head_dim)
    xk = xk.view(bs, seqlen, num_head, head_dim)
    keys = xk
    xq = xq.transpose(1, 2)
    keys = keys.transpose(1, 2)
    scores = (torch.matmul(xq, keys.transpose(2, 3)) / math.sqrt(head_dim)).squeeze().transpose(0, 1).reshape(num_head, -1)
    print("s  ", scores.shape)
    return scores


def torch_att1(xq, xk, seqlen, num_head, head_dim):
    xq = xq.view(1, num_head, head_dim)
    xk = xk.view(seqlen, num_head, head_dim)
    logics = torch.sum(xq * xk, dim=-1, keepdim=False)

    logics = logics.transpose(0, 1) / math.sqrt(head_dim)
    return logics


def test1():
    import time

    B, N_CTX, H, D = 17, 1025, 12, 128

    dtype = torch.float16

    q = torch.empty((B, H, D), dtype=dtype, device="cuda").normal_(mean=0.1, std=0.2)
    k = torch.empty((B * N_CTX, H, D), dtype=dtype, device="cuda").normal_(mean=0.1, std=0.2)
    att_out = torch.empty((H, B * N_CTX), dtype=dtype, device="cuda")

    # print(att_out)

    b_loc = torch.zeros((B, N_CTX), dtype=torch.int32, device="cuda")
    b_start_loc = torch.zeros((B,), dtype=torch.int32, device="cuda")
    b_seq_len = torch.zeros((B,), dtype=torch.int32, device="cuda")

    for i in range(B):
        b_start_loc[i] = i * N_CTX
        b_seq_len[i] = N_CTX
        b_loc[i] = i * N_CTX + torch.arange(0, N_CTX, dtype=torch.int32, device="cuda")
        print(b_loc[i])

    # Warm up
    for _ in range(10):
        token_att_fwd(q, k, att_out, b_loc, b_start_loc, b_seq_len, N_CTX)
    run_iter = 1000
    torch.cuda.synchronize()
    t1 = time.time()
    for _ in range(run_iter):
         token_att_fwd(q, k, att_out, b_loc, b_start_loc, b_seq_len, N_CTX)
    torch.cuda.synchronize()
    t2 = time.time()
    print("Time cost {}".format((t2 - t1) / run_iter))

    torch_out = torch_att(q, k, B, N_CTX, H, D).squeeze()
    o = att_out.squeeze()
    print("max ", torch.max(torch.abs(torch_out - o)))
    print("mean ", torch.mean(torch.abs(torch_out - o)))
    assert torch.allclose(torch_out, o, atol=1e-2, rtol=0)


def test2():
    import time

    B, N_CTX, H, D = 17, 1025, 12, 128
    dtype = torch.float16

    q = torch.empty((B, H, D), dtype=dtype, device="cuda").normal_(mean=0.0, std=1)
    k = torch.empty((B * N_CTX, H, D), dtype=dtype, device="cuda").normal_(mean=0.0, std=1)
    k_scale = (k.abs().max(-1, keepdim=True)[0] / 127.)
    int_k = (k / k_scale).to(torch.int8).half()

    att_out = torch.empty((H, B * N_CTX), dtype=dtype, device="cuda")

    b_loc = torch.zeros((B, N_CTX), dtype=torch.int32, device="cuda")
    b_start_loc = torch.zeros((B,), dtype=torch.int32, device="cuda")
    b_seq_len = torch.zeros((B,), dtype=torch.int32, device="cuda")

    for i in range(B):
        b_start_loc[i] = i * N_CTX
        b_seq_len[i] = N_CTX
        b_loc[i] = i * N_CTX + torch.arange(0, N_CTX, dtype=torch.int32, device="cuda")

    # Warm up
    for _ in range(10):
        token_att_fwd_int8k(q, int_k, k_scale, att_out, b_loc, b_start_loc, b_seq_len, N_CTX)
    run_iter = 1000
    torch.cuda.synchronize()
    t1 = time.time()
    for _ in range(run_iter):
        token_att_fwd_int8k(q, int_k, k_scale, att_out, b_loc, b_start_loc, b_seq_len, N_CTX)
    torch.cuda.synchronize()
    t2 = time.time()
    print("Time cost {}".format((t2 - t1) / run_iter))
    print(att_out.min(), att_out.max())
    torch_out = torch_att(q, k, B, N_CTX, H, D).squeeze()
    o = att_out.squeeze()
    print("max ", torch.max(torch.abs(torch_out - o)))
    print("mean ", torch.mean(torch.abs(torch_out - o)))
    cos = torch.nn.CosineSimilarity(0)
    print("cos ", cos(torch_out.flatten().to(torch.float32), o.flatten().to(torch.float32)))


if __name__ == '__main__':
    test1()
    test2()
