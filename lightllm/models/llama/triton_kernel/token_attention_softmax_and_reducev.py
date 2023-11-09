import torch
import triton
import triton.language as tl
import torch.nn.functional as F


@triton.jit
def _fwd_kernel(
    Logics, V, Out,
    Req_to_tokens, B_req_idx, B_Start_Loc, B_Seqlen,
    stride_logic_h, stride_logic_bs,
    stride_vbs, stride_vh, stride_vd,
    stride_obs, stride_oh, stride_od,
    stride_req_to_token_b, stride_req_to_token_s,
    other_kv_index, # 避免读取到nan的数据
    kv_group_num,
    BLOCK_DMODEL: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    cur_batch = tl.program_id(0)
    cur_head = tl.program_id(1)

    cur_kv_head = cur_head // kv_group_num

    cur_batch_seq_len = tl.load(B_Seqlen + cur_batch)
    cur_batch_start_loc = tl.load(B_Start_Loc + cur_batch)
    cur_batch_req_idx = tl.load(B_req_idx + cur_batch)

    offs_n = tl.arange(0, BLOCK_N)
    offs_d = tl.arange(0, BLOCK_DMODEL)

    off_v = cur_kv_head * stride_vh + offs_d[None, :] * stride_vd
    v_ptrs = V + off_v

    e_max = float("-inf")
    e_sum = 0.0
    acc = tl.zeros([BLOCK_DMODEL], dtype=tl.float32)

    for start_n in range(0, cur_batch_seq_len, BLOCK_N):
        start_n = tl.multiple_of(start_n, BLOCK_N)
        v_index = tl.load(Req_to_tokens + cur_batch_req_idx * stride_req_to_token_b + 
                          (start_n + offs_n) * stride_req_to_token_s, 
                          mask=(start_n + offs_n) < cur_batch_seq_len, other=other_kv_index)

        qk = tl.load(Logics + cur_head * stride_logic_h + (cur_batch_start_loc + start_n + offs_n) * stride_logic_bs, 
                     mask=start_n + offs_n < cur_batch_seq_len, other=float("-inf"))
    
        n_e_max = tl.maximum(tl.max(qk, 0), e_max)
        old_scale = tl.exp(e_max - n_e_max)
        p = tl.exp(qk - n_e_max)
        e_sum = e_sum * old_scale + tl.sum(p, 0)
        v = tl.load(v_ptrs + v_index[:, None] * stride_vbs)
        acc = acc * old_scale + tl.sum(p[:, None] * v, 0)
        e_max = n_e_max

    acc = acc / e_sum
    off_o = cur_batch * stride_obs + cur_head * stride_oh + offs_d * stride_od
    out_ptrs = Out + off_o
    tl.store(out_ptrs, acc)
    return


@torch.no_grad()
def token_softmax_reducev_fwd(logics, v, o, req_to_tokens, b_req_idx, b_start_loc, b_seq_len, other_kv_index):
    BLOCK = 64
    batch, head = b_seq_len.shape[0], logics.shape[0]
    grid = (batch, head)
    kv_group_num = logics.shape[0] // v.shape[1]

    num_warps = 1
    _fwd_kernel[grid](
        logics, v, o, req_to_tokens, b_req_idx, b_start_loc, b_seq_len,
        logics.stride(0), logics.stride(1),
        v.stride(0), v.stride(1), v.stride(2),
        o.stride(0), o.stride(1), o.stride(2),
        req_to_tokens.stride(0), req_to_tokens.stride(1),
        other_kv_index,
        kv_group_num,
        BLOCK_DMODEL=v.shape[-1],
        BLOCK_N=BLOCK,
        num_warps=num_warps,
        num_stages=3
    )
    return