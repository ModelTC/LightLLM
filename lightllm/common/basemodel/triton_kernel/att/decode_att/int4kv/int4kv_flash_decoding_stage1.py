import torch
import triton
import triton.language as tl


@triton.jit
def int4_to_float(k_int8, k_scale, offs_d):
    k_int8 = k_int8.to(tl.uint8, bitcast=True)
    k_high = (k_int8 & 0xF0) >> 4
    k_low = k_int8 & 0x0F
    k_high = k_high.to(tl.int8, bitcast=True)
    k_low = k_low.to(tl.int8, bitcast=True)
    k_high -= 7
    k_low -= 7
    k_int4 = tl.where(
        offs_d[None, :] % 2 == 0,
        k_low,
        k_high,
    )
    k = k_int4.to(k_scale.dtype) * k_scale
    return k


@triton.jit
def _fwd_kernel_flash_decode_stage1(
    Q,
    K,
    K_scale,
    V,
    V_scale,
    sm_scale,
    Req_to_tokens,
    B_req_idx,
    B_Seqlen,
    Mid_O,  # [batch, head, seq_block_num, head_dim]
    Mid_O_LogExpSum,  # [batch, head, seq_block_num]
    stride_req_to_tokens_b,
    stride_req_to_tokens_s,
    stride_qbs,
    stride_qh,
    stride_qd,
    stride_kbs,
    stride_kh,
    stride_kd,
    stride_vbs,
    stride_vh,
    stride_vd,
    stride_mid_ob,
    stride_mid_oh,
    stride_mid_os,
    stride_mid_od,
    stride_mid_o_eb,
    stride_mid_o_eh,
    stride_mid_o_es,
    gqa_group_size,
    quant_group_size,
    BLOCK_HEAD: tl.constexpr,
    BLOCK_SEQ: tl.constexpr,
    BLOCK_DMODEL: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    cur_batch = tl.program_id(0)
    cur_kv_head = tl.program_id(1)
    block_index = tl.program_id(2)
    grid_block_num = tl.num_programs(2)

    cur_batch_seq_len = tl.load(B_Seqlen + cur_batch)
    req_total_block_num = tl.cdiv(cur_batch_seq_len, BLOCK_SEQ)
    if block_index >= req_total_block_num:
        return

    cur_batch_req_idx = tl.load(B_req_idx + cur_batch)
    off_head = cur_kv_head * gqa_group_size + tl.arange(0, BLOCK_HEAD)
    off_head = tl.where(tl.arange(0, BLOCK_HEAD) < gqa_group_size, off_head, cur_kv_head * gqa_group_size)
    offs_d = tl.arange(0, BLOCK_DMODEL)

    tl.device_assert(stride_qd == 1)
    off_q = cur_batch * stride_qbs + off_head[:, None] * stride_qh + offs_d[None, :]
    q = tl.load(Q + off_q)

    sum_exp = tl.zeros([BLOCK_HEAD], dtype=tl.float32)
    max_logic = tl.zeros([BLOCK_HEAD], dtype=tl.float32) - float("inf")
    acc = tl.zeros([BLOCK_HEAD, BLOCK_DMODEL], dtype=tl.float32)

    for iter_block_index in range(block_index, req_total_block_num, grid_block_num):
        cur_batch_start_index = iter_block_index * BLOCK_SEQ
        cur_batch_end_index = tl.minimum(cur_batch_seq_len, cur_batch_start_index + BLOCK_SEQ)
        block_n_size = tl.cdiv(cur_batch_end_index - cur_batch_start_index, BLOCK_N)

        offs_n = cur_batch_start_index + tl.arange(0, BLOCK_N)

        for start_n in range(0, block_n_size, 1):
            offs_n_new = start_n * BLOCK_N + offs_n
            k_loc = tl.load(
                Req_to_tokens + stride_req_to_tokens_b * cur_batch_req_idx + offs_n_new,
                mask=offs_n_new < cur_batch_end_index,
                other=0,
            )
            k_loc = k_loc.to(tl.int64)
            off_k = k_loc[:, None] * stride_kbs + cur_kv_head * stride_kh + offs_d[None, :] // 2
            off_k_scale = off_k // (quant_group_size // 2)
            k_int8 = tl.load(K + off_k, mask=offs_n_new[:, None] < cur_batch_end_index, other=0)
            k_scale = tl.load(K_scale + off_k_scale, mask=offs_n_new[:, None] < cur_batch_end_index, other=0.0)
            k = int4_to_float(k_int8, k_scale, offs_d)

            att_value = tl.dot(q, k.T)
            att_value *= sm_scale
            att_value = tl.where((offs_n_new[None, :] < cur_batch_end_index), att_value, float("-inf"))
            v_int8 = tl.load(V + off_k, mask=offs_n_new[:, None] < cur_batch_end_index, other=0)
            v_scale = tl.load(V_scale + off_k_scale, mask=offs_n_new[:, None] < cur_batch_end_index, other=0.0)
            v = int4_to_float(v_int8, v_scale, offs_d)

            cur_max_logic = tl.max(att_value, axis=1)
            new_max_logic = tl.maximum(cur_max_logic, max_logic)

            exp_logic = tl.exp(att_value - new_max_logic[:, None])
            logic_scale = tl.exp(max_logic - new_max_logic)
            acc *= logic_scale[:, None]
            acc += tl.dot(exp_logic.to(v.dtype), v)

            sum_exp = sum_exp * logic_scale + tl.sum(exp_logic, axis=1)
            max_logic = new_max_logic

        off_mid_o = (
            cur_batch * stride_mid_ob
            + off_head[:, None] * stride_mid_oh
            + block_index * stride_mid_os
            + offs_d[None, :]
        )
        off_mid_o_logexpsum = cur_batch * stride_mid_o_eb + off_head * stride_mid_o_eh + block_index
        tl.store(Mid_O + off_mid_o, acc / sum_exp[:, None])
        tl.store(Mid_O_LogExpSum + off_mid_o_logexpsum, max_logic + tl.log(sum_exp))
    return


@torch.no_grad()
def int4kv_flash_decode_stage1(
    q,
    k,
    k_scale,
    v,
    v_scale,
    Req_to_tokens,
    B_req_idx,
    B_Seqlen,
    max_kv_seq_len,
    mid_out,
    mid_out_logsumexp,
    block_seq,
):
    BLOCK_SEQ = block_seq
    BLOCK_N = 16
    assert BLOCK_SEQ % BLOCK_N == 0
    # shape constraints
    Lq, Lk = q.shape[-1], k.shape[-1] * 2
    assert Lq == Lk
    assert Lk in {16, 32, 64, 128}
    sm_scale = 1.0 / (Lk ** 0.5)
    batch, kv_head_num = B_req_idx.shape[0], k.shape[1]
    grid_block_num = mid_out.shape[2]
    grid = (batch, kv_head_num, grid_block_num)
    gqa_group_size = q.shape[1] // k.shape[1]
    quant_group_size = Lk // k_scale.shape[-1]
    assert triton.next_power_of_2(quant_group_size) == quant_group_size
    assert k.stride() == v.stride()
    # TODO 优化为gqa使用tensor core的实现，速度更快。
    _fwd_kernel_flash_decode_stage1[grid](
        q,
        k,
        k_scale,
        v,
        v_scale,
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
        gqa_group_size=gqa_group_size,
        quant_group_size=quant_group_size,
        BLOCK_HEAD=triton.next_power_of_2(gqa_group_size),
        BLOCK_SEQ=BLOCK_SEQ,
        BLOCK_DMODEL=Lk,
        BLOCK_N=BLOCK_N,
        num_warps=4,
        num_stages=2,
    )
    return
