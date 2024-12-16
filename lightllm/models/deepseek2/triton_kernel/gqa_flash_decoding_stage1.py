import os
import torch
import triton
import triton.language as tl


@triton.jit
def _fwd_kernel_flash_decode_stage1(
    Q_nope,
    Q_rope,
    KV_nope,
    KV_rope,
    sm_scale,
    Req_to_tokens,
    B_req_idx,
    B_Seqlen,
    Mid_O,  # [batch, head, seq_block_num, head_dim]
    Mid_O_LogExpSum,  # [batch, head, seq_block_num]
    stride_req_to_tokens_b,
    stride_req_to_tokens_s,
    stride_q_bs,
    stride_q_h,
    stride_q_d,
    stride_q_rope_bs,
    stride_q_rope_h,
    stride_q_rope_d,
    stride_kv_bs,
    stride_kv_h,
    stride_kv_d,
    stride_kv_rope_bs,
    stride_kv_rope_h,
    stride_kv_rope_d,
    stride_mid_ob,
    stride_mid_oh,
    stride_mid_os,
    stride_mid_od,
    stride_mid_o_eb,
    stride_mid_o_eh,
    stride_mid_o_es,
    Q_HEAD_NUM: tl.constexpr,
    BLOCK_SEQ: tl.constexpr,
    BLOCK_DMODEL: tl.constexpr,
    BLOCK_ROPE_DMODEL: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    seq_start_block = tl.program_id(0)
    cur_q_head = tl.program_id(1)
    cur_batch = tl.program_id(2)

    cur_q_head_offs = tl.arange(0, Q_HEAD_NUM)
    cur_q_head_range = cur_q_head * Q_HEAD_NUM + cur_q_head_offs

    offs_d = tl.arange(0, BLOCK_DMODEL)
    offs_rope_d = tl.arange(0, BLOCK_ROPE_DMODEL)
    cur_batch_seq_len = tl.load(B_Seqlen + cur_batch)
    cur_batch_req_idx = tl.load(B_req_idx + cur_batch)
    cur_batch_start_index = seq_start_block * BLOCK_SEQ
    cur_batch_end_index = tl.minimum(cur_batch_seq_len, cur_batch_start_index + BLOCK_SEQ)

    off_q = cur_batch * stride_q_bs + cur_q_head_range[:, None] * stride_q_h + offs_d[None, :]
    off_rope_q = cur_batch * stride_q_rope_bs + cur_q_head_range[:, None] * stride_q_rope_h + offs_rope_d[None, :]
    q = tl.load(Q_nope + off_q)
    q_rope = tl.load(Q_rope + off_rope_q)
    block_n_size = (
        tl.where(
            cur_batch_end_index - cur_batch_start_index <= 0,
            0,
            cur_batch_end_index - cur_batch_start_index + BLOCK_N - 1,
        )
        // BLOCK_N
    )

    offs_n = cur_batch_start_index + tl.arange(0, BLOCK_N)
    sum_exp = tl.zeros([Q_HEAD_NUM], dtype=tl.float32)
    max_logic = tl.zeros([Q_HEAD_NUM], dtype=tl.float32) - float("inf")
    acc = tl.zeros([Q_HEAD_NUM, BLOCK_DMODEL], dtype=tl.float32)
    for start_n in range(0, block_n_size, 1):
        offs_n_new = start_n * BLOCK_N + offs_n
        kv_loc = tl.load(
            Req_to_tokens + stride_req_to_tokens_b * cur_batch_req_idx + offs_n_new,
            mask=offs_n_new < cur_batch_end_index,
            other=0,
        )
        off_kv = kv_loc[None, :] * stride_kv_bs + offs_d[:, None]
        kv = tl.load(KV_nope + off_kv, mask=offs_n_new[None, :] < cur_batch_end_index, other=0.0)
        att_value = tl.dot(q, kv)
        off_rope_kv = kv_loc[None, :] * stride_kv_rope_bs + offs_rope_d[:, None]
        rope_kv = tl.load(KV_rope + off_rope_kv, mask=offs_n_new[None, :] < cur_batch_end_index, other=0.0)
        att_value += tl.dot(q_rope, rope_kv)

        att_value *= sm_scale
        att_value = tl.where(offs_n_new[None, :] < cur_batch_end_index, att_value, float("-inf"))

        cur_max_logic = tl.max(att_value, axis=1)
        new_max_logic = tl.maximum(cur_max_logic, max_logic)

        exp_logic = tl.exp(att_value - new_max_logic[:, None])
        logic_scale = tl.exp(max_logic - new_max_logic)
        acc *= logic_scale[:, None]
        acc += tl.dot(exp_logic.to(kv.dtype), tl.trans(kv))

        sum_exp = sum_exp * logic_scale + tl.sum(exp_logic, axis=1)
        max_logic = new_max_logic

    need_store = tl.where(block_n_size == 0, 0, 1)
    for _ in range(0, need_store, 1):
        off_mid_o = (
            cur_batch * stride_mid_ob
            + cur_q_head_range[:, None] * stride_mid_oh
            + seq_start_block * stride_mid_os
            + offs_d[None, :]
        )
        off_mid_o_logexpsum = cur_batch * stride_mid_o_eb + cur_q_head_range * stride_mid_o_eh + seq_start_block
        tl.store(
            Mid_O + off_mid_o,
            acc / sum_exp[:, None],
        )
        tl.store(
            Mid_O_LogExpSum + off_mid_o_logexpsum,
            max_logic + tl.log(sum_exp),
        )
    return


@triton.jit
def _fwd_kernel_flash_decode_stage1_padding(
    Q_nope,
    Q_rope,
    KV_nope,
    KV_rope,
    sm_scale,
    Req_to_tokens,
    B_req_idx,
    B_Seqlen,
    Mid_O,  # [batch, head, seq_block_num, head_dim]
    Mid_O_LogExpSum,  # [batch, head, seq_block_num]
    stride_req_to_tokens_b,
    stride_req_to_tokens_s,
    stride_q_bs,
    stride_q_h,
    stride_q_d,
    stride_q_rope_bs,
    stride_q_rope_h,
    stride_q_rope_d,
    stride_kv_bs,
    stride_kv_h,
    stride_kv_d,
    stride_kv_rope_bs,
    stride_kv_rope_h,
    stride_kv_rope_d,
    stride_mid_ob,
    stride_mid_oh,
    stride_mid_os,
    stride_mid_od,
    stride_mid_o_eb,
    stride_mid_o_eh,
    stride_mid_o_es,
    gqa_group_size,
    Q_HEAD_NUM: tl.constexpr,
    BLOCK_SEQ: tl.constexpr,
    BLOCK_DMODEL: tl.constexpr,
    BLOCK_ROPE_DMODEL: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    seq_start_block = tl.program_id(0)
    cur_kv_head = tl.program_id(1)
    cur_batch = tl.program_id(2)

    cur_q_head_offs = tl.arange(0, Q_HEAD_NUM)
    cur_q_head_range = cur_kv_head * gqa_group_size + cur_q_head_offs

    offs_d = tl.arange(0, BLOCK_DMODEL)
    offs_rope_d = tl.arange(0, BLOCK_ROPE_DMODEL)
    cur_batch_seq_len = tl.load(B_Seqlen + cur_batch)
    cur_batch_req_idx = tl.load(B_req_idx + cur_batch)
    cur_batch_start_index = seq_start_block * BLOCK_SEQ
    cur_batch_end_index = tl.minimum(cur_batch_seq_len, cur_batch_start_index + BLOCK_SEQ)

    off_q = cur_batch * stride_q_bs + cur_q_head_range[:, None] * stride_q_h + offs_d[None, :]
    off_rope_q = cur_batch * stride_q_rope_bs + cur_q_head_range[:, None] * stride_q_rope_h + offs_rope_d[None, :]
    q = tl.load(Q_nope + off_q, mask=cur_q_head_range[:, None] < (cur_kv_head + 1) * gqa_group_size, other=0.0)
    q_rope = tl.load(
        Q_rope + off_rope_q, mask=cur_q_head_range[:, None] < (cur_kv_head + 1) * gqa_group_size, other=0.0
    )
    block_n_size = (
        tl.where(
            cur_batch_end_index - cur_batch_start_index <= 0,
            0,
            cur_batch_end_index - cur_batch_start_index + BLOCK_N - 1,
        )
        // BLOCK_N
    )

    offs_n = cur_batch_start_index + tl.arange(0, BLOCK_N)
    sum_exp = tl.zeros([Q_HEAD_NUM], dtype=tl.float32)
    max_logic = tl.zeros([Q_HEAD_NUM], dtype=tl.float32) - float("inf")
    acc = tl.zeros([Q_HEAD_NUM, BLOCK_DMODEL], dtype=tl.float32)
    for start_n in range(0, block_n_size, 1):
        offs_n_new = start_n * BLOCK_N + offs_n
        kv_loc = tl.load(
            Req_to_tokens + stride_req_to_tokens_b * cur_batch_req_idx + offs_n_new,
            mask=offs_n_new < cur_batch_end_index,
            other=0,
        )
        off_kv = kv_loc[None, :] * stride_kv_bs + offs_d[:, None]
        kv = tl.load(KV_nope + off_kv, mask=offs_n_new[None, :] < cur_batch_end_index, other=0.0)
        att_value = tl.dot(q, kv)
        off_rope_kv = kv_loc[None, :] * stride_kv_rope_bs + offs_rope_d[:, None]
        rope_kv = tl.load(KV_rope + off_rope_kv, mask=offs_n_new[None, :] < cur_batch_end_index, other=0.0)
        att_value += tl.dot(q_rope, rope_kv)

        att_value *= sm_scale
        att_value = tl.where(offs_n_new[None, :] < cur_batch_end_index, att_value, float("-inf"))

        cur_max_logic = tl.max(att_value, axis=1)
        new_max_logic = tl.maximum(cur_max_logic, max_logic)

        exp_logic = tl.exp(att_value - new_max_logic[:, None])
        logic_scale = tl.exp(max_logic - new_max_logic)
        acc *= logic_scale[:, None]
        acc += tl.dot(exp_logic.to(kv.dtype), tl.trans(kv))

        sum_exp = sum_exp * logic_scale + tl.sum(exp_logic, axis=1)
        max_logic = new_max_logic

    need_store = tl.where(block_n_size == 0, 0, 1)
    for _ in range(0, need_store, 1):
        off_mid_o = (
            cur_batch * stride_mid_ob
            + cur_q_head_range[:, None] * stride_mid_oh
            + seq_start_block * stride_mid_os
            + offs_d[None, :]
        )
        off_mid_o_logexpsum = cur_batch * stride_mid_o_eb + cur_q_head_range * stride_mid_o_eh + seq_start_block
        tl.store(
            Mid_O + off_mid_o,
            acc / sum_exp[:, None],
            mask=cur_q_head_range[:, None] < (cur_kv_head + 1) * gqa_group_size,
        )
        tl.store(
            Mid_O_LogExpSum + off_mid_o_logexpsum,
            max_logic + tl.log(sum_exp),
            mask=cur_q_head_range < (cur_kv_head + 1) * gqa_group_size,
        )
    return


@torch.no_grad()
def flash_decode_stage1(
    q_nope,
    q_rope,
    kv_nope,
    kv_rope,
    Req_to_tokens,
    B_req_idx,
    B_Seqlen,
    max_len_in_batch,
    mid_out,
    mid_out_logsumexp,
    block_seq,
    softmax_scale,
):
    if hasattr(os, "tuning_config"):
        BLOCK_SEQ = os.tuning_config["BLOCK_SEQ"]
        BLOCK_N = os.tuning_config["BLOCK_N"]
        BLOCK_Q_HEAD = os.tuning_config["BLOCK_Q_HEAD"]
        num_warps = os.tuning_config["stage1_num_warps"]
        num_stages = os.tuning_config["stage1_num_stages"]
    else:
        BLOCK_SEQ = block_seq
        BLOCK_N = 16
        BLOCK_Q_HEAD = 16
        num_warps = 4
        num_stages = 2

    assert BLOCK_SEQ % BLOCK_N == 0
    # shape constraints
    q_nope_dim = q_nope.shape[-1]
    q_rope_dim = q_rope.shape[-1]

    assert q_nope_dim == kv_nope.shape[-1]
    assert q_rope_dim == kv_rope.shape[-1]
    assert q_nope_dim in {16, 32, 64, 128, 256, 512}
    assert q_rope_dim in {16, 32, 64, 128, 256}

    sm_scale = softmax_scale  # 计算scale系数
    batch, q_head_num = B_req_idx.shape[0], q_nope.shape[1]
    if q_head_num % BLOCK_Q_HEAD == 0:
        grid = (triton.cdiv(max_len_in_batch, BLOCK_SEQ), q_head_num // BLOCK_Q_HEAD, batch)
        _fwd_kernel_flash_decode_stage1[grid](
            q_nope,
            q_rope,
            kv_nope,
            kv_rope,
            sm_scale,
            Req_to_tokens,
            B_req_idx,
            B_Seqlen,
            mid_out,
            mid_out_logsumexp,
            Req_to_tokens.stride(0),
            Req_to_tokens.stride(1),
            q_nope.stride(0),
            q_nope.stride(1),
            q_nope.stride(2),
            q_rope.stride(0),
            q_rope.stride(1),
            q_rope.stride(2),
            kv_nope.stride(0),
            kv_nope.stride(1),
            kv_nope.stride(2),
            kv_rope.stride(0),
            kv_rope.stride(1),
            kv_rope.stride(2),
            mid_out.stride(0),
            mid_out.stride(1),
            mid_out.stride(2),
            mid_out.stride(3),
            mid_out_logsumexp.stride(0),
            mid_out_logsumexp.stride(1),
            mid_out_logsumexp.stride(2),
            Q_HEAD_NUM=BLOCK_Q_HEAD,
            BLOCK_SEQ=BLOCK_SEQ,
            BLOCK_DMODEL=q_nope_dim,
            BLOCK_ROPE_DMODEL=q_rope_dim,
            BLOCK_N=BLOCK_N,
            num_warps=num_warps,
            num_stages=num_stages,
        )
    else:
        assert q_head_num < BLOCK_Q_HEAD
        kv_head_num = kv_nope.shape[1]
        grid = (triton.cdiv(max_len_in_batch, BLOCK_SEQ), kv_head_num, batch)
        gqa_group_size = q_head_num // kv_head_num
        _fwd_kernel_flash_decode_stage1_padding[grid](
            q_nope,
            q_rope,
            kv_nope,
            kv_rope,
            sm_scale,
            Req_to_tokens,
            B_req_idx,
            B_Seqlen,
            mid_out,
            mid_out_logsumexp,
            Req_to_tokens.stride(0),
            Req_to_tokens.stride(1),
            q_nope.stride(0),
            q_nope.stride(1),
            q_nope.stride(2),
            q_rope.stride(0),
            q_rope.stride(1),
            q_rope.stride(2),
            kv_nope.stride(0),
            kv_nope.stride(1),
            kv_nope.stride(2),
            kv_rope.stride(0),
            kv_rope.stride(1),
            kv_rope.stride(2),
            mid_out.stride(0),
            mid_out.stride(1),
            mid_out.stride(2),
            mid_out.stride(3),
            mid_out_logsumexp.stride(0),
            mid_out_logsumexp.stride(1),
            mid_out_logsumexp.stride(2),
            gqa_group_size,
            Q_HEAD_NUM=max(16, triton.next_power_of_2(gqa_group_size)),
            BLOCK_SEQ=BLOCK_SEQ,
            BLOCK_DMODEL=q_nope_dim,
            BLOCK_ROPE_DMODEL=q_rope_dim,
            BLOCK_N=BLOCK_N,
            num_warps=num_warps,
            num_stages=num_stages,
        )
    return
