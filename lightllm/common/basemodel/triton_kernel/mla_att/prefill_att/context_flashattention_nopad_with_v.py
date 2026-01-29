import torch

import triton
import triton.language as tl
import math
import torch.nn.functional as F
from lightllm.utils.device_utils import is_tesla


@triton.jit
def _fwd_kernel_with_v(
    Q_nope,
    Q_rope,
    K_nope,
    K_rope,
    V,
    sm_scale,
    B_Start_Loc,
    B_Kv_Start_Loc,
    B_Seqlen,  # B_LOC 内部记录每个batch 输入的真实位置， B_SEQ_len 记录当前输入的真实长度
    Out,
    stride_q_bs,
    stride_q_h,
    stride_q_rope_bs,
    stride_q_rope_h,
    stride_k_bs,
    stride_k_h,
    stride_k_rope_bs,
    stride_k_rope_h,
    stride_vbs,
    stride_vh,
    stride_obs,
    stride_oh,
    b_prompt_cache_len,
    BLOCK_M: tl.constexpr,
    BLOCK_DMODEL: tl.constexpr,
    BLOCK_ROPE_DMODEL: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    cur_batch = tl.program_id(0)
    cur_head = tl.program_id(1)
    start_m = tl.program_id(2)

    cur_k_head = cur_head

    cur_batch_in_q_start_index = tl.load(B_Start_Loc + cur_batch)
    cur_batch_in_kv_start_index = tl.load(B_Kv_Start_Loc + cur_batch)
    prompt_cache_len = tl.load(b_prompt_cache_len + cur_batch)
    cur_batch_seq_len = tl.load(B_Seqlen + cur_batch) - prompt_cache_len

    block_start_loc = BLOCK_M * start_m

    # initialize offsets
    offs_n = tl.arange(0, BLOCK_N)
    offs_d = tl.arange(0, BLOCK_DMODEL)
    offs_rope_d = tl.arange(0, BLOCK_ROPE_DMODEL)
    offs_m = start_m * BLOCK_M + tl.arange(0, BLOCK_M)
    off_q = (cur_batch_in_q_start_index + offs_m[:, None]) * stride_q_bs + cur_head * stride_q_h + offs_d[None, :]
    off_q_rope = (
        (cur_batch_in_q_start_index + offs_m[:, None]) * stride_q_rope_bs
        + cur_head * stride_q_rope_h
        + offs_rope_d[None, :]
    )
    off_k = offs_n[None, :] * stride_k_bs + cur_k_head * stride_k_h + offs_d[:, None]
    off_k_rope = offs_n[None, :] * stride_k_rope_bs + offs_rope_d[:, None]
    off_v = offs_n[:, None] * stride_vbs + cur_k_head * stride_vh + offs_d[None, :]

    q = tl.load(Q_nope + off_q, mask=offs_m[:, None] < cur_batch_seq_len, other=0.0)
    q_rope = tl.load(Q_rope + off_q_rope, mask=offs_m[:, None] < cur_batch_seq_len, other=0.0)

    k_ptrs = K_nope + off_k
    k_rope_ptrs = K_rope + off_k_rope
    v_ptrs = V + off_v

    # initialize pointer to m and l
    m_i = tl.zeros([BLOCK_M], dtype=tl.float32) - float("inf")
    l_i = tl.zeros([BLOCK_M], dtype=tl.float32)
    acc = tl.zeros([BLOCK_M, BLOCK_DMODEL], dtype=tl.float32)

    block_mask = tl.where(block_start_loc < cur_batch_seq_len, 1, 0)
    block_end_loc = tl.minimum((start_m + 1) * BLOCK_M + prompt_cache_len, cur_batch_seq_len + prompt_cache_len)

    for start_n in range(0, block_mask * block_end_loc, BLOCK_N):
        start_n = tl.multiple_of(start_n, BLOCK_N)
        # -- compute qk ----
        k = tl.load(
            k_ptrs + (cur_batch_in_kv_start_index + start_n) * stride_k_bs,
            mask=(start_n + offs_n[None, :]) < block_end_loc,
            other=0.0,
        )
        k_rope = tl.load(
            k_rope_ptrs + (cur_batch_in_kv_start_index + start_n) * stride_k_rope_bs,
            mask=(start_n + offs_n[None, :]) < block_end_loc,
            other=0.0,
        )

        qk = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32)
        qk += tl.dot(q, k)
        qk += tl.dot(q_rope, k_rope)
        qk *= sm_scale
        qk = tl.where(offs_m[:, None] + prompt_cache_len >= start_n + offs_n[None, :], qk, float("-100000000.0"))

        # -- compute m_ij, p, l_ij
        m_ij = tl.maximum(m_i, tl.max(qk, 1))
        qk -= m_ij[:, None]
        p = tl.math.exp2(qk)
        l_ij = tl.sum(p, 1)

        # -- update m_i and l_i
        alpha = tl.math.exp2(m_i - m_ij)
        l_i = l_i * alpha + l_ij
        # -- update output accumulator --
        acc = acc * alpha[:, None]
        # update acc
        v = tl.load(
            v_ptrs + (cur_batch_in_kv_start_index + start_n) * stride_vbs,
            mask=(start_n + offs_n[:, None]) < block_end_loc,
            other=0.0,
        )
        p = p.to(v.dtype)
        acc = tl.dot(p, v, acc)
        # update m_i and l_i
        m_i = m_ij

    acc = acc / l_i[:, None]
    # initialize pointers to output
    off_o = (cur_batch_in_q_start_index + offs_m[:, None]) * stride_obs + cur_head * stride_oh + offs_d[None, :]
    out_ptrs = Out + off_o
    tl.store(out_ptrs, acc, mask=offs_m[:, None] < cur_batch_seq_len)
    return


@torch.no_grad()
def context_attention_fwd_with_v(
    q_nope,
    q_rope,
    k_nope,
    k_rope,
    v,
    o,
    b_start_loc,
    b_kv_start_loc,
    b_seq_len,
    b_prompt_cache_len,
    max_input_len,
    softmax_scale,
):

    BLOCK = 128 if not is_tesla() else 64
    q_nope_dim = q_nope.shape[-1]
    q_rope_dim = q_rope.shape[-1]
    assert q_nope_dim == k_nope.shape[-1]
    assert q_rope_dim == k_rope.shape[-1]
    assert q_nope_dim in {16, 32, 64, 128, 256, 512}
    assert q_rope_dim in {16, 32, 64, 128, 256}
    assert q_nope_dim == v.shape[-1]

    if q_nope_dim >= 512:
        BLOCK = 64 if not is_tesla() else 32
    else:
        BLOCK = 128 if not is_tesla() else 64

    if q_nope.dtype == torch.float32:
        BLOCK = BLOCK // 4

    sm_scale = softmax_scale * 1.4426950408889634
    batch, head = b_seq_len.shape[0], q_nope.shape[1]

    grid = (batch, head, triton.cdiv(max_input_len, BLOCK))  # batch, head,
    num_warps = 4 if q_nope_dim <= 64 else 8

    _fwd_kernel_with_v[grid](
        q_nope,
        q_rope,
        k_nope,
        k_rope,
        v,
        sm_scale,
        b_start_loc,
        b_kv_start_loc,
        b_seq_len,
        o,
        q_nope.stride(0),
        q_nope.stride(1),
        q_rope.stride(0),
        q_rope.stride(1),
        k_nope.stride(0),
        k_nope.stride(1),
        k_rope.stride(0),
        k_rope.stride(1),
        v.stride(0),
        v.stride(1),
        o.stride(0),
        o.stride(1),
        b_prompt_cache_len=b_prompt_cache_len,
        BLOCK_M=BLOCK,
        BLOCK_DMODEL=q_nope_dim,
        BLOCK_ROPE_DMODEL=q_rope_dim,
        BLOCK_N=BLOCK,
        num_warps=num_warps,
        num_stages=1,
    )
    return


if __name__ == "__main__":
    import torch
    import flashinfer

    Z, N_CTX, H, D_HEAD, ROPE_HEAD = 32, 1024, 16, 128, 64
    dtype = torch.bfloat16

    k_nope = torch.randn((Z * N_CTX, H, D_HEAD), dtype=dtype, device="cuda")
    k_rope = torch.randn((Z * N_CTX, 1, ROPE_HEAD), dtype=dtype, device="cuda")
    k = torch.cat([k_nope, torch.repeat_interleave(k_rope, H, dim=-2)], dim=-1)
    v = torch.randn((Z * N_CTX, H, D_HEAD), dtype=dtype, device="cuda")

    max_input_len = Z * N_CTX
    softmax_scale = 0.117
    b_seq_len = torch.ones((Z,), dtype=torch.int32, device="cuda") * N_CTX
    b_prompt_cache_len = torch.zeros_like(b_seq_len, dtype=torch.int32, device="cuda")
    b_prompt_cache_len = torch.randint_like(b_seq_len, high=N_CTX - 1, dtype=torch.int32, device="cuda")
    q_lens = b_seq_len - b_prompt_cache_len
    q_start_loc = q_lens.cumsum(0) - q_lens
    kv_start_loc = b_seq_len.cumsum(0) - b_seq_len

    q_nope = torch.randn((q_lens.sum(), H, D_HEAD), dtype=dtype, device="cuda")
    q_rope = torch.randn((q_lens.sum(), H, ROPE_HEAD), dtype=dtype, device="cuda")
    q = torch.cat([q_nope, q_rope], dim=-1)

    o = torch.empty((q_lens.sum(), H, D_HEAD), dtype=dtype, device="cuda")
    o1 = torch.empty((q_lens.sum(), H, D_HEAD), dtype=dtype, device="cuda")
    o2 = torch.empty((q_lens.sum(), H, D_HEAD), dtype=dtype, device="cuda")

    fn1 = lambda: context_attention_fwd_with_v(
        q_nope,
        q_rope,
        k_nope,
        k_rope,
        v,
        o,
        q_start_loc,
        kv_start_loc,
        b_seq_len,
        b_prompt_cache_len,
        max_input_len,
        softmax_scale,
    )

    q_starts = torch.zeros((Z + 1,)).int().cuda()
    q_starts[1:] = torch.cumsum(b_seq_len - b_prompt_cache_len, dim=0)
    kv_starts = torch.zeros_like(q_starts)
    kv_starts[1:] = torch.cumsum(b_seq_len, dim=0)
    kv_layout = "NHD"
    batch_size = Z
    q_indptr = q_starts
    kv_indptr = kv_starts
    workspace_buffer = torch.empty(128 * 1024 * 1024, dtype=torch.int8).to(0)
    wrapper = flashinfer.prefill.BatchPrefillWithRaggedKVCacheWrapper(workspace_buffer, kv_layout)
    wrapper.plan(
        qo_indptr=q_indptr,
        kv_indptr=kv_indptr,
        num_qo_heads=H,
        num_kv_heads=H,
        head_dim_qk=D_HEAD + ROPE_HEAD,
        head_dim_vo=D_HEAD,
        q_data_type=dtype,
        causal=True,
        sm_scale=softmax_scale,
    )
    fn2 = lambda: wrapper.run(q, k, v, out=o1)

    ms1 = triton.testing.do_bench(fn1)
    ms2 = triton.testing.do_bench(fn2)
    cos_sim1 = F.cosine_similarity(o, o1).mean()
    print(cos_sim1)
    print(ms1)
    print(ms2)
