import torch

import triton
import triton.language as tl


@triton.jit
def _fwd_kernel_token_att2(
    Prob, V, Out, B_Loc, B_Start_Loc, B_Seqlen, max_input_len,  # B_Start_Loc 保存的是如果连续存储时候的累加输入和
    stride_b_loc_b, stride_b_loc_s,
    stride_ph, stride_pbs,
    stride_vbs, stride_vh, stride_vd,
    stride_obs, stride_oh, stride_od,
    BLOCK_DMODEL: tl.constexpr,
    BLOCK_N: tl.constexpr
):
    cur_batch = tl.program_id(0)
    cur_head = tl.program_id(1)

    offs_n = tl.arange(0, BLOCK_N)
    offs_d = tl.arange(0, BLOCK_DMODEL)
    cur_batch_seq_len = tl.load(B_Seqlen + cur_batch)
    cur_batch_start_index = max_input_len - cur_batch_seq_len
    cur_batch_end_index = cur_batch_seq_len
    cur_batch_in_all_start_index = tl.load(B_Start_Loc + cur_batch)

    v_loc_off = cur_batch * stride_b_loc_b + (cur_batch_start_index + offs_n) * stride_b_loc_s
    p_offs = cur_head * stride_ph + (cur_batch_in_all_start_index + offs_n) * stride_pbs
    v_offs = cur_head * stride_vh + offs_d[None, :] * stride_vd

    acc = tl.zeros([BLOCK_DMODEL], dtype=tl.float32)
    for start_n in range(0, cur_batch_seq_len, BLOCK_N):
        start_n = tl.multiple_of(start_n, BLOCK_N)
        p_value = tl.load(Prob + p_offs + start_n * stride_b_loc_s, mask=(start_n + offs_n) < cur_batch_seq_len, other=0.0)
        v_loc = tl.load(B_Loc + v_loc_off + start_n * stride_b_loc_s, mask=(start_n + offs_n) < cur_batch_seq_len, other=0.0)
        v_value = tl.load(V + v_offs + v_loc[:, None] * stride_vbs, mask=(start_n + offs_n[:, None]) < cur_batch_seq_len, other=0.0)
        acc += tl.sum(p_value[:, None] * v_value, 0)

    acc = acc.to(tl.float16)
    off_o = cur_batch * stride_obs + cur_head * stride_oh + offs_d * stride_od
    out_ptrs = Out + off_o
    tl.store(out_ptrs, acc)
    return


@torch.no_grad()
def token_att_fwd2(prob, v, out, B_Loc, B_Start_Loc, B_Seqlen, max_input_len):
    if triton.__version__ >= "2.1.0":
        BLOCK = 128
    else:
        BLOCK = 64
    batch, head = B_Loc.shape[0], v.shape[1]
    grid = (batch, head)
    num_warps = 4
    dim = v.shape[-1]

    _fwd_kernel_token_att2[grid](
        prob, v, out, B_Loc, B_Start_Loc, B_Seqlen, max_input_len,
        B_Loc.stride(0), B_Loc.stride(1),
        prob.stride(0), prob.stride(1),
        v.stride(0), v.stride(1), v.stride(2),
        out.stride(0), out.stride(1), out.stride(2),
        BLOCK_DMODEL=dim,
        BLOCK_N=BLOCK,
        num_warps=num_warps,
        num_stages=1,
    )
    return


@triton.jit
def _fwd_kernel_token_att2_int8v(
    Prob, V, V_scale, Out, B_Loc, B_Start_Loc, B_Seqlen, max_input_len,  # B_Start_Loc 保存的是如果连续存储时候的累加输入和
    stride_b_loc_b, stride_b_loc_s,
    stride_ph, stride_pbs,
    stride_vbs, stride_vh, stride_vd,
    stride_vsbs, stride_vsh, stride_vsd,
    stride_obs, stride_oh, stride_od,
    BLOCK_DMODEL: tl.constexpr,
    BLOCK_N: tl.constexpr
):
    cur_batch = tl.program_id(0)
    cur_head = tl.program_id(1)

    offs_n = tl.arange(0, BLOCK_N)
    offs_d = tl.arange(0, BLOCK_DMODEL)
    cur_batch_seq_len = tl.load(B_Seqlen + cur_batch)
    cur_batch_start_index = max_input_len - cur_batch_seq_len
    cur_batch_end_index = cur_batch_seq_len
    cur_batch_in_all_start_index = tl.load(B_Start_Loc + cur_batch)

    v_loc_off = cur_batch * stride_b_loc_b + (cur_batch_start_index + offs_n) * stride_b_loc_s
    p_offs = cur_head * stride_ph + (cur_batch_in_all_start_index + offs_n) * stride_pbs
    v_offs = cur_head * stride_vh + offs_d[None, :] * stride_vd
    vs_offs = cur_head * stride_vsh

    acc = tl.zeros([BLOCK_DMODEL], dtype=tl.float32)
    for start_n in range(0, cur_batch_seq_len, BLOCK_N):
        start_n = tl.multiple_of(start_n, BLOCK_N)
        p_value = tl.load(Prob + p_offs + start_n * stride_b_loc_s, mask=(start_n + offs_n) < cur_batch_seq_len, other=0.0)
        v_loc = tl.load(B_Loc + v_loc_off + start_n * stride_b_loc_s, mask=(start_n + offs_n) < cur_batch_seq_len, other=0.0)
        v_value = tl.load(V + v_offs + v_loc[:, None] * stride_vbs, mask=(start_n + offs_n[:, None]) < cur_batch_seq_len, other=0.0)
        vs_value = tl.load(V_scale + vs_offs + v_loc[:, None] * stride_vsbs, mask=(start_n + offs_n[:, None]) < cur_batch_seq_len, other=0.0)
        acc += tl.sum(p_value[:, None] * v_value * vs_value, 0)

    acc = acc.to(tl.float16)
    off_o = cur_batch * stride_obs + cur_head * stride_oh + offs_d * stride_od
    out_ptrs = Out + off_o
    tl.store(out_ptrs, acc)
    return


@torch.no_grad()
def token_att_fwd2_int8v(prob, v, v_scale, out, B_Loc, B_Start_Loc, B_Seqlen, max_input_len):
    if max_input_len < 512:
        BLOCK = triton.next_power_of_2(max_input_len)
    else:
        BLOCK = 512
    batch, head = B_Loc.shape[0], v.shape[1]
    grid = (batch, head)
    num_warps = 4
    dim = v.shape[-1]

    _fwd_kernel_token_att2_int8v[grid](
        prob, v, v_scale, out, B_Loc, B_Start_Loc, B_Seqlen, max_input_len,
        B_Loc.stride(0), B_Loc.stride(1),
        prob.stride(0), prob.stride(1),
        v.stride(0), v.stride(1), v.stride(2),
        v_scale.stride(0), v_scale.stride(1), v_scale.stride(2),
        out.stride(0), out.stride(1), out.stride(2),
        BLOCK_DMODEL=dim,
        BLOCK_N=BLOCK,
        num_warps=num_warps,
        num_stages=1,
    )
    return


def torch_att(V, P, bs, seqlen, num_head, head_dim):
    V = V.view(bs, seqlen, num_head, head_dim).transpose(1, 2)
    P = P.reshape(num_head, bs, 1, seqlen).transpose(0, 1)
    out = torch.matmul(P, V)

    return out


def test1():
    import time

    B, N_CTX, H, D = 17, 1025, 12, 128
    dtype = torch.float16

    V = torch.empty((B * N_CTX, H, D), dtype=dtype, device="cuda").normal_(mean=0.1, std=10)
    Prob = torch.empty((H, B * N_CTX), dtype=dtype, device="cuda").normal_(mean=0.4,
                                                                           std=0.2).reshape(H, B, N_CTX).softmax(-1).reshape(H, B * N_CTX)
    Out = torch.empty((B, H, D), dtype=dtype, device="cuda")

    b_start_loc = torch.zeros((B,), dtype=torch.int32, device="cuda")
    b_seq_len = torch.zeros((B,), dtype=torch.int32, device="cuda")
    b_loc = torch.zeros((B, N_CTX), dtype=torch.int32, device="cuda")
    for i in range(B):
        b_start_loc[i] = i * N_CTX
        b_seq_len[i] = N_CTX
        b_loc[i] = i * N_CTX + torch.arange(0, N_CTX, dtype=torch.int32, device="cuda")

    # Warm up
    for _ in range(10):
        token_att_fwd2(Prob, V, Out, b_loc, b_start_loc, b_seq_len, N_CTX)
    run_iter = 1000
    torch.cuda.synchronize()
    t1 = time.time()
    for _ in range(run_iter):
        token_att_fwd2(Prob, V, Out, b_loc, b_start_loc, b_seq_len, N_CTX)
    torch.cuda.synchronize()
    t2 = time.time()
    print("Time cost {}".format((t2 - t1) / run_iter))
    torch_out = torch_att(V, Prob, B, N_CTX, H, D).squeeze()
    o = Out
    print("max ", torch.max(torch.abs(torch_out - o)))
    print("mean ", torch.mean(torch.abs(torch_out - o)))
    assert torch.allclose(torch_out, o, atol=1e-2, rtol=0)


def test2():
    import time

    B, N_CTX, H, D = 17, 1025, 12, 128
    dtype = torch.float16

    V = torch.empty((B * N_CTX, H, D), dtype=dtype, device="cuda").normal_(mean=0.1, std=10)
    V_scale = (V.abs().max(-1, keepdim=True)[0] / 127.)
    int_V = (V / V_scale).to(torch.int8)

    Prob = torch.empty((H, B * N_CTX), dtype=dtype, device="cuda").normal_(mean=0.4,
                                                                           std=0.2).reshape(H, B, N_CTX).softmax(-1).reshape(H, B * N_CTX)
    Out = torch.empty((B, H, D), dtype=dtype, device="cuda")

    b_start_loc = torch.zeros((B,), dtype=torch.int32, device="cuda")
    b_seq_len = torch.zeros((B,), dtype=torch.int32, device="cuda")
    b_loc = torch.zeros((B, N_CTX), dtype=torch.int32, device="cuda")
    for i in range(B):
        b_start_loc[i] = i * N_CTX
        b_seq_len[i] = N_CTX
        b_loc[i] = i * N_CTX + torch.arange(0, N_CTX, dtype=torch.int32, device="cuda")

    for _ in range(10):
        token_att_fwd2_int8v(Prob, int_V, V_scale, Out, b_loc, b_start_loc, b_seq_len, N_CTX)

    run_iter = 1000
    torch.cuda.synchronize()
    t1 = time.time()
    for _ in range(run_iter):
        token_att_fwd2_int8v(Prob, int_V, V_scale, Out, b_loc, b_start_loc, b_seq_len, N_CTX)
    torch.cuda.synchronize()
    t2 = time.time()
    print("Time cost {}".format((t2 - t1) / run_iter))
    torch_out = torch_att(V, Prob, B, N_CTX, H, D).squeeze()
    o = Out
    print("max ", torch.max(torch.abs(torch_out - o)))
    print("mean ", torch.mean(torch.abs(torch_out - o)))
    cos = torch.nn.CosineSimilarity(0)
    print("cos ", cos(torch_out.flatten().to(torch.float32), o.flatten().to(torch.float32)))


if __name__ == '__main__':
    test1()
    test2()
