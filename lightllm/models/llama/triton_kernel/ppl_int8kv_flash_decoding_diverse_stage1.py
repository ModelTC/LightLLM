import torch
import triton
import triton.language as tl


@triton.jit
def _fwd_kernel_flash_decode_stage1(
    Q,
    stride_qbs,
    stride_qh,
    stride_qd,
    K,
    K_scale,
    stride_kbs,
    stride_kh,
    stride_kd,
    V,
    V_scale,
    stride_vbs,
    stride_vh,
    stride_vd,
    sm_scale,
    Req_to_tokens,
    stride_req_to_tokens_b,
    stride_req_to_tokens_s,
    B_req_idx,
    b_shared_seq_len,
    b_mark_shared_group,
    Mid_O,  # [batch, head, seq_block_num, head_dim]
    stride_mid_ob,
    stride_mid_oh,
    stride_mid_os,
    stride_mid_od,
    Mid_O_LogExpSum,  # [batch, head, seq_block_num]
    stride_mid_o_eb,
    stride_mid_o_eh,
    stride_mid_o_es,
    gqa_group_size,
    BLOCK_HEAD: tl.constexpr,
    BLOCK_SEQ: tl.constexpr,
    BLOCK_HEADDIM: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_BATCH: tl.constexpr,
    KV_QUANT_GROUP_SIZE: tl.constexpr,
):
    cur_batch = tl.program_id(0)
    shared_batch_group_size = tl.load(b_mark_shared_group + cur_batch)
    if shared_batch_group_size == 0:
        return
    cur_batch = cur_batch - shared_batch_group_size
    cur_kv_head = tl.program_id(1)
    seq_start_block = tl.program_id(2)

    cur_q_head_range = cur_kv_head * gqa_group_size + tl.arange(0, BLOCK_HEAD)
    q_head_end_index = (cur_kv_head + 1) * gqa_group_size
    cur_q_head_range = tl.where(cur_q_head_range < q_head_end_index, cur_q_head_range, cur_kv_head * gqa_group_size)

    offs_d = tl.arange(0, BLOCK_HEADDIM)
    cur_batch_seq_len = tl.load(b_shared_seq_len + cur_batch)
    cur_batch_req_idx = tl.load(B_req_idx + cur_batch)
    cur_batch_start_index = seq_start_block * BLOCK_SEQ
    cur_batch_end_index = tl.minimum(cur_batch_seq_len, cur_batch_start_index + BLOCK_SEQ)

    offs_batch = cur_batch + tl.arange(0, BLOCK_BATCH)
    offs_batch = tl.where(offs_batch < cur_batch + shared_batch_group_size, offs_batch, cur_batch)

    off_q = offs_batch[:, None, None] * stride_qbs + cur_q_head_range[None, :, None] * stride_qh + offs_d[None, None, :]

    block_n_size = tl.cdiv(
        tl.where(cur_batch_end_index - cur_batch_start_index <= 0, 0, cur_batch_end_index - cur_batch_start_index),
        BLOCK_N,
    )

    if block_n_size == 0:
        return

    offs_n = cur_batch_start_index + tl.arange(0, BLOCK_N)
    Q_BATCH_HEAD_NUM: tl.constexpr = BLOCK_BATCH * BLOCK_HEAD
    q = tl.load(Q + off_q, other=0.0).view(Q_BATCH_HEAD_NUM, BLOCK_HEADDIM)

    sum_exp = tl.zeros([Q_BATCH_HEAD_NUM], dtype=tl.float32)
    max_logic = tl.zeros([Q_BATCH_HEAD_NUM], dtype=tl.float32) - float("inf")
    acc = tl.zeros([Q_BATCH_HEAD_NUM, BLOCK_HEADDIM], dtype=tl.float32)

    for start_n in range(0, block_n_size, 1):
        offs_n_new = start_n * BLOCK_N + offs_n
        n_mask = offs_n_new < cur_batch_end_index
        k_loc = tl.load(
            Req_to_tokens + stride_req_to_tokens_b * cur_batch_req_idx + offs_n_new,
            mask=offs_n_new < cur_batch_end_index,
            other=0,
        ).to(tl.int64)
        off_k = k_loc[None, :] * stride_kbs + cur_kv_head * stride_kh + offs_d[:, None]
        off_k_scale = off_k // KV_QUANT_GROUP_SIZE
        k = tl.load(K + off_k, mask=n_mask[None, :], other=0)
        k_scale = tl.load(K_scale + off_k_scale, mask=n_mask[None, :], other=0.0)
        k = k * k_scale
        att_value = tl.dot(q, k)
        att_value *= sm_scale
        att_value = tl.where(offs_n_new[None, :] < cur_batch_end_index, att_value, -1000000000.0)
        v = tl.load(
            V + off_k.T,
            mask=n_mask[:, None],
            other=0,
        )
        v_scale = tl.load(
            V + off_k_scale.T,
            mask=n_mask[:, None],
            other=0.0,
        )
        v = v * v_scale

        cur_max_logic = tl.max(att_value, axis=1)
        new_max_logic = tl.maximum(cur_max_logic, max_logic)

        exp_logic = tl.exp(att_value - new_max_logic[:, None])
        logic_scale = tl.exp(max_logic - new_max_logic)
        acc *= logic_scale[:, None]
        acc += tl.dot(exp_logic.to(v.dtype), v)

        sum_exp = sum_exp * logic_scale + tl.sum(exp_logic, axis=1)
        max_logic = new_max_logic

    need_store = tl.where(block_n_size == 0, 0, 1)
    for _ in range(0, need_store, 1):
        off_mid_o = (
            offs_batch[:, None, None] * stride_mid_ob
            + cur_q_head_range[None, :, None] * stride_mid_oh
            + seq_start_block * stride_mid_os
            + offs_d[None, None, :]
        )
        off_mid_o_logexpsum = (
            offs_batch[:, None] * stride_mid_o_eb + cur_q_head_range[None, :] * stride_mid_o_eh + seq_start_block
        )
        tl.store(
            Mid_O + off_mid_o,
            (acc / sum_exp[:, None]).view(BLOCK_BATCH, BLOCK_HEAD, BLOCK_HEADDIM),
        )
        tl.store(
            Mid_O_LogExpSum + off_mid_o_logexpsum,
            (max_logic + tl.log(sum_exp)).view(BLOCK_BATCH, BLOCK_HEAD),
        )
    return


@torch.no_grad()
def flash_decode_stage1(
    q, k, v, Req_to_tokens, B_req_idx, B_Seqlen, max_len_in_batch, mid_out, mid_out_logsumexp, block_seq
):
    BLOCK_SEQ = block_seq
    BLOCK_N = 16
    assert BLOCK_SEQ % BLOCK_N == 0
    # shape constraints
    Lq, Lk = q.shape[-1], k.shape[-1]
    assert Lq == Lk
    assert Lk in {16, 32, 64, 128}
    sm_scale = 1.0 / (Lk ** 0.5)
    batch, kv_head_num = B_req_idx.shape[0], k.shape[1]
    grid = (batch, kv_head_num, triton.cdiv(max_len_in_batch, BLOCK_SEQ))
    gqa_group_size = q.shape[1] // k.shape[1]
    assert triton.next_power_of_2(k.shape[-1]) == k.shape[-1]

    _fwd_kernel_flash_decode_stage1[grid](
        q,
        k,
        v,
        sm_scale,
        Req_to_tokens,
        B_req_idx,
        B_Seqlen,
        mid_out,
        mid_out_logsumexp,
        Req_to_tokens.stride(0),
        Req_to_tokens.stride(1),
        q.stride(0),
        q.stride(1),
        q.stride(2),
        k.stride(0),
        k.stride(1),
        k.stride(2),
        v.stride(0),
        v.stride(1),
        v.stride(2),
        mid_out.stride(0),
        mid_out.stride(1),
        mid_out.stride(2),
        mid_out.stride(3),
        mid_out_logsumexp.stride(0),
        mid_out_logsumexp.stride(1),
        mid_out_logsumexp.stride(2),
        gqa_group_size,
        BLOCK_HEAD=max(16, triton.next_power_of_2(gqa_group_size)),
        BLOCK_SEQ=BLOCK_SEQ,
        BLOCK_HEADDIM=Lk,
        BLOCK_N=BLOCK_N,
        num_warps=2,
        num_stages=2,
    )
    return
