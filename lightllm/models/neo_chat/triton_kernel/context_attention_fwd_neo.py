import math
import torch
import triton
import triton.language as tl

from lightllm.utils.device_utils import is_tesla


@triton.jit
def _fwd_kernel(
    Q,
    K,
    V,
    sm_scale,
    Out,
    position_ids,  # 1D: packed like Q (only NEW tokens), length == Q.shape[0]
    B_Start_Loc,
    B_Seqlen,
    Req_to_tokens,
    B_req_idx,
    stride_qbs,
    stride_qh,
    stride_qd,
    stride_kbs,
    stride_kh,
    stride_kd,
    stride_vbs,
    stride_vh,
    stride_vd,
    stride_obs,
    stride_oh,
    stride_od,
    stride_req_to_tokens_b,
    stride_req_to_tokens_s,
    kv_group_num,
    b_prompt_cache_len,
    H: tl.constexpr,
    BLOCK_DMODEL: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    start_m = tl.program_id(0)
    cur_bh = tl.program_id(1)
    cur_batch = cur_bh // H
    cur_head = cur_bh % H

    cur_kv_head = cur_head // kv_group_num

    cur_batch_in_all_start_index = tl.load(B_Start_Loc + cur_batch)
    prompt_cache_len = tl.load(b_prompt_cache_len + cur_batch)
    total_len = tl.load(B_Seqlen + cur_batch)
    cur_batch_seq_len = total_len - prompt_cache_len  # NEW len
    cur_batch_req_idx = tl.load(B_req_idx + cur_batch)

    block_start_loc = BLOCK_M * start_m

    offs_n = tl.arange(0, BLOCK_N)
    offs_d = tl.arange(0, BLOCK_DMODEL)
    offs_m = block_start_loc + tl.arange(0, BLOCK_M)

    # Q pointers
    off_q = (
        (cur_batch_in_all_start_index + offs_m[:, None]) * stride_qbs
        + cur_head * stride_qh
        + offs_d[None, :] * stride_qd
    )

    q_valid = offs_m < cur_batch_seq_len
    q = tl.load(Q + off_q, mask=q_valid[:, None], other=0.0)

    # online softmax state
    m_i = tl.zeros([BLOCK_M], dtype=tl.float32) - float("inf")
    l_i = tl.zeros([BLOCK_M], dtype=tl.float32)
    acc = tl.zeros([BLOCK_M, BLOCK_DMODEL], dtype=tl.float32)

    block_mask = tl.where(block_start_loc < cur_batch_seq_len, 1, 0)
    block_end_loc = total_len

    # absolute q positions in the request
    q_pos = prompt_cache_len + offs_m  # [M]

    # q_gid from packed position_ids (aligned with Q rows)
    q_gid = tl.load(
        position_ids + cur_batch_in_all_start_index + offs_m,
        mask=q_valid,
        other=-2147483648,
    ).to(tl.int32)

    BIG = tl.full([BLOCK_N], 1000000000, tl.int32)  # ensure != any normal gid

    for start_n in range(0, block_mask * block_end_loc, BLOCK_N):
        start_n = tl.multiple_of(start_n, BLOCK_N)

        k_pos = start_n + offs_n  # [N]
        k_valid = k_pos < block_end_loc

        # map logical pos -> mem_index (for K/V)
        kv_loc = tl.load(
            Req_to_tokens + stride_req_to_tokens_b * cur_batch_req_idx + stride_req_to_tokens_s * k_pos,
            mask=k_valid,
            other=0,
        ).to(tl.int64)

        # load K
        off_k = kv_loc[None, :] * stride_kbs + cur_kv_head * stride_kh + offs_d[:, None] * stride_kd
        k = tl.load(K + off_k, mask=k_valid[None, :], other=0.0)

        qk = tl.dot(q, k)

        # k_gid:
        # - for cached keys (k_pos < prompt_cache_len): set to BIG + k_pos so equality is always false
        # - for new keys (k_pos >= prompt_cache_len): read from packed position_ids by (k_pos - prompt_cache_len)
        k_in_new = k_pos >= prompt_cache_len
        k_new_idx = (k_pos - prompt_cache_len).to(tl.int32)  # [N] valid only when k_in_new
        k_gid_new = tl.load(
            position_ids + cur_batch_in_all_start_index + k_new_idx,
            mask=k_valid & k_in_new,
            other=-2147483647,
        ).to(tl.int32)

        k_gid = tl.where(
            k_in_new,
            k_gid_new,
            (k_pos.to(tl.int32) + BIG),
        )

        # mask: causal OR same gid (only possible inside NEW part)
        mask = (q_pos[:, None] >= k_pos[None, :]) | (q_gid[:, None] == k_gid[None, :])
        mask = mask & q_valid[:, None] & k_valid[None, :]

        qk = tl.where(mask, qk * sm_scale, -1.0e8)

        # online softmax
        m_ij = tl.maximum(m_i, tl.max(qk, 1))
        qk -= m_ij[:, None]
        p = tl.math.exp2(qk)
        l_ij = tl.sum(p, 1)

        alpha = tl.math.exp2(m_i - m_ij)
        l_i = l_i * alpha + l_ij
        acc = acc * alpha[:, None]

        # load V
        off_v = kv_loc[:, None] * stride_vbs + cur_kv_head * stride_vh + offs_d[None, :] * stride_vd
        v = tl.load(V + off_v, mask=k_valid[:, None], other=0.0)

        p = p.to(v.dtype)
        acc = tl.dot(p, v, acc)

        m_i = m_ij

    acc = acc / l_i[:, None]

    off_o = (
        (cur_batch_in_all_start_index + offs_m[:, None]) * stride_obs
        + cur_head * stride_oh
        + offs_d[None, :] * stride_od
    )
    tl.store(Out + off_o, acc, mask=q_valid[:, None])


@torch.no_grad()
def context_attention_fwd_neo(
    q,
    k,
    v,
    o,
    position_ids,  # 1D packed like q (only NEW tokens)
    b_req_idx,
    b_start_loc,
    b_seq_len,
    b_prompt_cache_len,
    max_input_len,
    req_to_token_indexs,
):
    # minimal safety: position_ids must cover packed q rows
    assert position_ids.numel() >= q.shape[0], (position_ids.numel(), q.shape[0])

    BLOCK_M = 128 if not is_tesla() else 64

    Lq, Lk, Lv = q.shape[-1], k.shape[-1], v.shape[-1]
    assert Lq == Lk and Lk == Lv
    assert Lk in {16, 32, 64, 128, 256}
    base_head_dim = Lq // 2
    sm_scale = 1.0 / (base_head_dim ** 0.5) * 1.4426950408889634

    batch, head = b_seq_len.shape[0], q.shape[1]
    kv_group_num = q.shape[1] // k.shape[1]

    grid = lambda meta: (triton.cdiv(max_input_len, meta["BLOCK_M"]), batch * head, 1)

    BLOCK_N = BLOCK_M
    num_warps = 4 if Lk <= 64 else 8
    num_stages = 1

    _fwd_kernel[grid](
        q,
        k,
        v,
        sm_scale,
        o,
        position_ids,
        b_start_loc,
        b_seq_len,
        req_to_token_indexs,
        b_req_idx,
        q.stride(0),
        q.stride(1),
        q.stride(2),
        k.stride(0),
        k.stride(1),
        k.stride(2),
        v.stride(0),
        v.stride(1),
        v.stride(2),
        o.stride(0),
        o.stride(1),
        o.stride(2),
        req_to_token_indexs.stride(0),
        req_to_token_indexs.stride(1),
        kv_group_num=kv_group_num,
        b_prompt_cache_len=b_prompt_cache_len,
        H=head,
        BLOCK_DMODEL=Lk,
        BLOCK_M=BLOCK_M,
        BLOCK_N=BLOCK_N,
        num_warps=num_warps,
        num_stages=num_stages,
    )


def reference_attention(
    q,
    k,
    v,
    position_ids_q,  # 1D packed like q (only NEW tokens)
    b_req_idx,
    b_start_loc,
    b_seq_len,
    b_prompt_cache_len,
    req_to_token_indexs,
):
    device = q.device
    dtype = q.dtype
    sum_q, Hq, D = q.shape
    Hk = k.shape[1]
    kv_group_num = Hq // Hk

    batch = b_seq_len.shape[0]
    out = torch.empty_like(q)
    scale = 1.0 / math.sqrt(D)

    for b in range(batch):
        req = int(b_req_idx[b].item())
        total_len = int(b_seq_len[b].item())
        prompt_len = int(b_prompt_cache_len[b].item())
        new_len = total_len - prompt_len

        q_start = int(b_start_loc[b].item())
        q_blk = q[q_start : q_start + new_len]  # [M, Hq, D]
        gid_new = position_ids_q[q_start : q_start + new_len].to(torch.int64)  # [M]

        # gather K/V for full request by logical pos -> mem_index
        token_locs = req_to_token_indexs[req, :total_len].to(torch.int64)  # [L]
        k_blk = k[token_locs]  # [L, Hk, D]
        v_blk = v[token_locs]  # [L, Hk, D]

        # expand kv heads to q heads (GQA)
        k_hq = k_blk.repeat_interleave(kv_group_num, dim=1)  # [L, Hq, D]
        v_hq = v_blk.repeat_interleave(kv_group_num, dim=1)  # [L, Hq, D]

        # positions
        q_pos = torch.arange(prompt_len, total_len, device=device, dtype=torch.int64)  # [M]
        k_pos = torch.arange(0, total_len, device=device, dtype=torch.int64)  # [L]

        # build allow mask:
        # causal always
        allow = k_pos[None, :] <= q_pos[:, None]

        # full-attn only inside NEW part by gid
        # compare only when k_pos in NEW
        k_in_new = k_pos >= prompt_len
        k_rel = (k_pos - prompt_len).clamp_min(0)  # [L]
        # map k_rel to gid_new, but only valid where k_in_new
        k_gid = torch.empty((total_len,), device=device, dtype=torch.int64)
        k_gid[:] = 10 ** 12 + k_pos  # never equal to gid_new
        k_gid[k_in_new] = gid_new[k_rel[k_in_new]]

        allow = allow | (gid_new[q_pos - prompt_len][:, None] == k_gid[None, :])

        # scores: [Hq, M, L]
        q_t = q_blk.permute(1, 0, 2).to(torch.float32)  # [Hq, M, D]
        k_t = k_hq.permute(1, 2, 0).to(torch.float32)  # [Hq, D, L]
        scores = torch.matmul(q_t, k_t) * scale  # [Hq, M, L]

        neg = torch.tensor(-1.0e9, device=device, dtype=torch.float32)
        scores = torch.where(allow[None, :, :], scores, neg)

        p = torch.softmax(scores, dim=-1).to(torch.float32)  # [Hq, M, L]
        v_t = v_hq.permute(1, 0, 2).to(torch.float32)  # [Hq, L, D]
        out_hq = torch.matmul(p, v_t)  # [Hq, M, D]
        out_blk = out_hq.permute(1, 0, 2).to(dtype)  # [M, Hq, D]

        out[q_start : q_start + new_len] = out_blk

    return out


def make_test_case(
    device="cuda",
    dtype=torch.float16,
    batch=3,
    Hq=8,
    Hk=4,
    D=64,
    seed=0,
    base_index=50000,
):
    torch.manual_seed(seed)

    # prompt (cached) len and new len
    prompt_lens = torch.randint(low=2, high=8, size=(batch,), device=device)
    new_lens = torch.randint(low=1, high=8, size=(batch,), device=device)
    total_lens = (prompt_lens + new_lens).to(torch.int32)

    max_total_len = int(total_lens.max().item())
    max_new_len = int(new_lens.max().item())

    # packed q start
    b_start_loc = torch.zeros((batch,), device=device, dtype=torch.int32)
    cur = 0
    for b in range(batch):
        b_start_loc[b] = cur
        cur += int(new_lens[b].item())
    sum_q = cur

    b_seq_len = total_lens
    b_prompt_cache_len = prompt_lens.to(torch.int32)

    # one req per batch
    num_req = batch
    b_req_idx = torch.arange(batch, device=device, dtype=torch.int32)

    # global KV space large, indices not small
    sum_kv = int(total_lens.sum().item())
    kv_size = base_index + sum_kv + 1024
    pool = torch.randperm(kv_size - base_index, device=device, dtype=torch.int64)[:sum_kv] + base_index

    # Req_to_tokens [num_req, max_total_len]
    req_to_token_indexs = torch.zeros((num_req, max_total_len), device=device, dtype=torch.int32)
    p = 0
    for r in range(num_req):
        L = int(total_lens[r].item())
        req_to_token_indexs[r, :L] = pool[p : p + L].to(torch.int32)
        p += L

    # position_ids_q: only NEW tokens, packed like q
    position_ids_q = torch.empty((sum_q,), device=device, dtype=torch.int32)
    for b in range(batch):
        M = int(new_lens[b].item())
        start = int(b_start_loc[b].item())

        gid = torch.arange(M, device=device, dtype=torch.int32)

        # make one repeated block inside NEW part to simulate image tokens
        if M >= 4 and torch.rand((), device=device).item() > 0.3:
            s = int(torch.randint(0, M - 2, (1,), device=device).item())
            e = min(M, s + 3)
            gid[s:e] = gid[s]

        position_ids_q[start : start + M] = gid

    q = torch.randn((sum_q, Hq, D), device=device, dtype=dtype)
    k = torch.randn((kv_size, Hk, D), device=device, dtype=dtype)
    v = torch.randn((kv_size, Hk, D), device=device, dtype=dtype)
    o = torch.empty((sum_q, Hq, D), device=device, dtype=dtype)

    return (
        q,
        k,
        v,
        o,
        position_ids_q,
        b_req_idx,
        b_start_loc,
        b_seq_len,
        b_prompt_cache_len,
        max_new_len,
        req_to_token_indexs,
    )


def check_once(device="cuda", dtype=torch.float16, seed=0):
    (
        q,
        k,
        v,
        o,
        position_ids_q,
        b_req_idx,
        b_start_loc,
        b_seq_len,
        b_prompt_cache_len,
        max_new_len,
        req_to_token_indexs,
    ) = make_test_case(device=device, dtype=dtype, seed=seed)

    context_attention_fwd_neo(
        q,
        k,
        v,
        o,
        position_ids_q,
        b_req_idx,
        b_start_loc,
        b_seq_len,
        b_prompt_cache_len,
        max_new_len,
        req_to_token_indexs,
    )

    ref = reference_attention(
        q,
        k,
        v,
        position_ids_q,
        b_req_idx,
        b_start_loc,
        b_seq_len,
        b_prompt_cache_len,
        req_to_token_indexs,
    )

    diff = (o - ref).abs()
    max_abs = diff.max().item()
    denom = ref.abs().max().item() + 1e-6
    max_rel = max_abs / denom

    print(f"seed={seed}, dtype={dtype}")
    print(f"max_abs_error = {max_abs:.6e}")
    print(f"max_rel_error = {max_rel:.6e}")
    print("allclose(fp16 tol)?", torch.allclose(o, ref, atol=5e-2, rtol=5e-2))


if __name__ == "__main__":
    if not torch.cuda.is_available():
        print("No CUDA, skip.")
    else:
        torch.cuda.synchronize()
        check_once(dtype=torch.bfloat16, seed=0)
        check_once(dtype=torch.bfloat16, seed=1)
        check_once(dtype=torch.bfloat16, seed=2)
