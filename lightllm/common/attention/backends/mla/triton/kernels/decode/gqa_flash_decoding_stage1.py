import torch
import triton
import triton.language as tl
from lightllm.utils.device_utils import calcu_kernel_best_vsm_count


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
    Mid_O,  # [head, seq_block_num, head_dim]
    Mid_O_LogExpSum,  # [head, seq_block_num]
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
    stride_mid_oh,
    stride_mid_os,
    stride_mid_od,
    stride_mid_o_eh,
    stride_mid_o_es,
    block_size_ptr,
    num_sm,
    head_group_num,
    head_num,
    batch_size,
    Q_HEAD_NUM: tl.constexpr,
    BLOCK_DMODEL: tl.constexpr,
    BLOCK_ROPE_DMODEL: tl.constexpr,
    BLOCK_N: tl.constexpr,
    NUM_STAGES: tl.constexpr,
    NEED_HEAD_MASK: tl.constexpr,
):
    # cur_kv_head = 0
    sm_id = tl.program_id(0).to(tl.int64)
    out_batch_start_index = tl.cast(0, tl.int64)
    block_seq = tl.load(block_size_ptr, eviction_policy="evict_last")

    cur_q_head_offs = tl.arange(0, Q_HEAD_NUM)
    offs_d = tl.arange(0, BLOCK_DMODEL)
    offs_rope_d = tl.arange(0, BLOCK_ROPE_DMODEL)

    for cur_batch in range(batch_size):
        cur_batch_seq_len = tl.load(B_Seqlen + cur_batch, eviction_policy="evict_last")
        cur_block_num = tl.cdiv(cur_batch_seq_len, block_seq) * head_group_num
        cur_batch_req_idx = tl.load(B_req_idx + cur_batch, eviction_policy="evict_last")
        req_to_tokens_ptr = Req_to_tokens + stride_req_to_tokens_b * cur_batch_req_idx

        while sm_id < cur_block_num:
            loop_head_group_index = sm_id % head_group_num
            loop_seq_block_index = sm_id // head_group_num

            cur_q_head_range = loop_head_group_index * Q_HEAD_NUM + cur_q_head_offs
            if NEED_HEAD_MASK:
                head_mask = cur_q_head_range < head_num

            cur_batch_start_index = block_seq * loop_seq_block_index
            cur_batch_end_index = tl.minimum(cur_batch_seq_len, cur_batch_start_index + block_seq)

            off_q = cur_batch * stride_q_bs + cur_q_head_range[:, None] * stride_q_h + offs_d[None, :]
            off_rope_q = (
                cur_batch * stride_q_rope_bs + cur_q_head_range[:, None] * stride_q_rope_h + offs_rope_d[None, :]
            )
            if NEED_HEAD_MASK:
                q = tl.load(
                    Q_nope + off_q,
                    mask=head_mask[:, None],
                    other=0.0,
                )
                q_rope = tl.load(
                    Q_rope + off_rope_q,
                    mask=head_mask[:, None],
                    other=0.0,
                )
            else:
                q = tl.load(Q_nope + off_q)
                q_rope = tl.load(Q_rope + off_rope_q)

            block_n_size = tl.cdiv(cur_batch_end_index - cur_batch_start_index, BLOCK_N)

            offs_n = cur_batch_start_index + tl.arange(0, BLOCK_N)
            sum_exp = tl.zeros([Q_HEAD_NUM], dtype=tl.float32)
            max_logic = tl.zeros([Q_HEAD_NUM], dtype=tl.float32) - float("inf")
            acc = tl.zeros([Q_HEAD_NUM, BLOCK_DMODEL], dtype=tl.float32)
            for start_n in tl.range(0, block_n_size, 1, num_stages=NUM_STAGES):
                offs_n_new = start_n * BLOCK_N + offs_n
                seq_n_mask = offs_n_new < cur_batch_end_index
                kv_loc = tl.load(
                    req_to_tokens_ptr + offs_n_new,
                    mask=seq_n_mask,
                    other=0,
                ).to(tl.int64)
                off_kv = kv_loc[None, :] * stride_kv_bs + offs_d[:, None]
                kv = tl.load(KV_nope + off_kv, mask=seq_n_mask[None, :], other=0.0)
                att_value = tl.dot(q, kv)
                off_rope_kv = kv_loc[None, :] * stride_kv_rope_bs + offs_rope_d[:, None]
                rope_kv = tl.load(KV_rope + off_rope_kv, mask=seq_n_mask[None, :], other=0.0)
                att_value += tl.dot(q_rope, rope_kv)

                att_value *= sm_scale
                att_value = tl.where(seq_n_mask[None, :], att_value, float("-inf"))

                cur_max_logic = tl.max(att_value, axis=1)
                new_max_logic = tl.maximum(cur_max_logic, max_logic)

                exp_logic = tl.exp(att_value - new_max_logic[:, None])
                logic_scale = tl.exp(max_logic - new_max_logic)
                acc *= logic_scale[:, None]
                acc += tl.dot(exp_logic.to(kv.dtype), tl.trans(kv))

                sum_exp = sum_exp * logic_scale + tl.sum(exp_logic, axis=1)
                max_logic = new_max_logic

            off_mid_o = (
                cur_q_head_range[:, None] * stride_mid_oh
                + (out_batch_start_index + loop_seq_block_index) * stride_mid_os
                + offs_d[None, :]
            )
            off_mid_o_logexpsum = cur_q_head_range * stride_mid_o_eh + out_batch_start_index + loop_seq_block_index
            if NEED_HEAD_MASK:
                tl.store(
                    Mid_O + off_mid_o,
                    acc / sum_exp[:, None],
                    mask=head_mask[:, None],
                )
                tl.store(
                    Mid_O_LogExpSum + off_mid_o_logexpsum,
                    max_logic + tl.log(sum_exp),
                    mask=head_mask,
                )
            else:
                tl.store(
                    Mid_O + off_mid_o,
                    acc / sum_exp[:, None],
                )
                tl.store(
                    Mid_O_LogExpSum + off_mid_o_logexpsum,
                    max_logic + tl.log(sum_exp),
                )
            sm_id += num_sm

        out_batch_start_index += cur_block_num // head_group_num
        sm_id -= cur_block_num
    return


@torch.no_grad()
def flash_decode_stage1(
    in_block_seq: torch.Tensor,
    q_nope,
    q_rope,
    kv_nope,
    kv_rope,
    Req_to_tokens,
    B_req_idx,
    B_Seqlen,
    mid_out,
    mid_out_logsumexp,
    softmax_scale,
    get_sm_count: bool = False,
    **run_config,
):
    if run_config:
        Q_HEAD_NUM = run_config["BLOCK_Q_HEAD"]
        BLOCK_N = run_config["BLOCK_N"]
        num_warps = run_config["stage1_num_warps"]
        num_stages = run_config["stage1_num_stages"]

    # shape constraints
    q_nope_dim = q_nope.shape[-1]
    q_rope_dim = q_rope.shape[-1]

    assert q_nope_dim == kv_nope.shape[-1]
    assert q_rope_dim == kv_rope.shape[-1]
    assert q_nope_dim in {16, 32, 64, 128, 256, 512}
    assert q_rope_dim in {16, 32, 64, 128, 256}
    assert kv_nope.shape[1] == 1

    batch_size, q_head_num = B_req_idx.shape[0], q_nope.shape[1]
    head_group_num = triton.cdiv(q_head_num, Q_HEAD_NUM)
    NEED_HEAD_MASK = (q_head_num % Q_HEAD_NUM) != 0

    kernel = _fwd_kernel_flash_decode_stage1_padding.warmup(
        q_nope,
        q_rope,
        kv_nope,
        kv_rope,
        softmax_scale,
        Req_to_tokens,
        B_req_idx,
        B_Seqlen,
        mid_out,
        mid_out_logsumexp,
        *Req_to_tokens.stride(),
        *q_nope.stride(),
        *q_rope.stride(),
        *kv_nope.stride(),
        *kv_rope.stride(),
        *mid_out.stride(),
        *mid_out_logsumexp.stride(),
        in_block_seq,
        num_sm=1,
        head_group_num=head_group_num,
        head_num=q_head_num,
        batch_size=batch_size,
        Q_HEAD_NUM=Q_HEAD_NUM,
        BLOCK_DMODEL=q_nope_dim,
        BLOCK_ROPE_DMODEL=q_rope_dim,
        BLOCK_N=BLOCK_N,
        NEED_HEAD_MASK=NEED_HEAD_MASK,
        NUM_STAGES=num_stages,
        num_warps=num_warps,
        num_stages=1,
        grid=(1,),
    )

    kernel._init_handles()
    num_sm = calcu_kernel_best_vsm_count(kernel, num_warps=num_warps)
    grid = (num_sm,)
    if get_sm_count:
        return num_sm

    assert num_sm * 4 + batch_size <= mid_out.shape[1]

    _fwd_kernel_flash_decode_stage1_padding[grid](
        q_nope,
        q_rope,
        kv_nope,
        kv_rope,
        softmax_scale,
        Req_to_tokens,
        B_req_idx,
        B_Seqlen,
        mid_out,
        mid_out_logsumexp,
        *Req_to_tokens.stride(),
        *q_nope.stride(),
        *q_rope.stride(),
        *kv_nope.stride(),
        *kv_rope.stride(),
        *mid_out.stride(),
        *mid_out_logsumexp.stride(),
        in_block_seq,
        num_sm=num_sm,
        head_group_num=head_group_num,
        head_num=q_head_num,
        batch_size=batch_size,
        Q_HEAD_NUM=Q_HEAD_NUM,
        BLOCK_DMODEL=q_nope_dim,
        BLOCK_ROPE_DMODEL=q_rope_dim,
        BLOCK_N=BLOCK_N,
        NEED_HEAD_MASK=NEED_HEAD_MASK,
        NUM_STAGES=num_stages,
        num_warps=num_warps,
        num_stages=1,
    )

    return
