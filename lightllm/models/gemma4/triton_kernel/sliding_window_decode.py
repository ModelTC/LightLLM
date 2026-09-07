"""Gemma sliding decode over a request ring and the current token's scratch KV."""

import torch
import triton
import triton.language as tl

from lightllm.common.basemodel.triton_kernel.att.decode_att.gqa.flash_decoding.gqa_flash_decoding_stage2 import (
    flash_decode_stage2,
)


@triton.jit
def _sliding_window_decode_stage1(
    Q,
    K,
    V,
    BReqIdx,
    BSeqLen,
    BQStartLoc,
    MidO,
    MidLogSumExp,
    sm_scale,
    stride_qb,
    stride_qh,
    stride_qd,
    stride_kt,
    stride_kh,
    stride_kd,
    stride_vt,
    stride_vh,
    stride_vd,
    stride_ob,
    stride_oh,
    stride_os,
    stride_od,
    stride_lb,
    stride_lh,
    stride_ls,
    gqa_group_size,
    WINDOW: tl.constexpr,
    SCRATCH_START: tl.constexpr,
    Q_HEAD_NUM: tl.constexpr,
    BLOCK_SEQ: tl.constexpr,
    BLOCK_DMODEL: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    batch_idx = tl.program_id(0)
    kv_head = tl.program_id(1)
    block_idx = tl.program_id(2)
    grid_block_num = tl.num_programs(2)

    seq_len = tl.load(BSeqLen + batch_idx).to(tl.int64)
    kv_start = tl.maximum(seq_len - WINDOW, 0)
    window_len = seq_len - kv_start
    total_blocks = tl.cdiv(window_len, BLOCK_SEQ)
    if block_idx >= total_blocks:
        return

    req_idx = tl.load(BReqIdx + batch_idx).to(tl.int64)
    q_start = tl.load(BQStartLoc + batch_idx).to(tl.int64)
    scratch_token = tl.full((), SCRATCH_START, tl.int64) + q_start
    head_offsets = tl.arange(0, Q_HEAD_NUM)
    q_heads = kv_head * gqa_group_size + head_offsets
    q_heads = tl.where(head_offsets < gqa_group_size, q_heads, kv_head * gqa_group_size)
    offs_d = tl.arange(0, BLOCK_DMODEL)
    q = tl.load(Q + batch_idx * stride_qb + q_heads[:, None] * stride_qh + offs_d[None, :] * stride_qd)

    # Match the common GQA stage1's tiling and online-softmax arithmetic.
    sum_exp = tl.zeros([Q_HEAD_NUM], dtype=tl.float32)
    max_logic = tl.zeros([Q_HEAD_NUM], dtype=tl.float32) - float("inf")
    acc = tl.zeros([Q_HEAD_NUM, BLOCK_DMODEL], dtype=tl.float32)
    for block in range(block_idx, total_blocks, grid_block_num):
        block_start = block * BLOCK_SEQ
        block_end = tl.minimum(window_len, block_start + BLOCK_SEQ)
        offs_n = block_start + tl.arange(0, BLOCK_N)
        for tile in range(0, tl.cdiv(block_end - block_start, BLOCK_N)):
            positions = tile * BLOCK_N + offs_n
            mask = positions < block_end
            token_pos = kv_start + positions
            k_loc = tl.where(token_pos < seq_len - 1, req_idx * WINDOW + token_pos % WINDOW, scratch_token)
            k = tl.load(
                K + k_loc[None, :] * stride_kt + kv_head * stride_kh + offs_d[:, None] * stride_kd,
                mask=mask[None, :],
                other=0.0,
            )
            att_value = tl.dot(q, k.to(q.dtype))
            att_value *= sm_scale
            att_value = tl.where(mask[None, :], att_value, float("-inf"))
            v = tl.load(
                V + k_loc[:, None] * stride_vt + kv_head * stride_vh + offs_d[None, :] * stride_vd,
                mask=mask[:, None],
                other=0.0,
            )
            cur_max_logic = tl.max(att_value, axis=1)
            new_max_logic = tl.maximum(cur_max_logic, max_logic)
            exp_logic = tl.exp(att_value - new_max_logic[:, None])
            logic_scale = tl.exp(max_logic - new_max_logic)
            acc *= logic_scale[:, None]
            acc += tl.dot(exp_logic.to(v.dtype), v)
            sum_exp = sum_exp * logic_scale + tl.sum(exp_logic, axis=1)
            max_logic = new_max_logic

    out_offsets = (
        batch_idx * stride_ob + q_heads[:, None] * stride_oh + block_idx * stride_os + offs_d[None, :] * stride_od
    )
    log_offsets = batch_idx * stride_lb + q_heads * stride_lh + block_idx * stride_ls
    tl.store(MidO + out_offsets, acc / sum_exp[:, None], mask=(head_offsets < gqa_group_size)[:, None])
    tl.store(MidLogSumExp + log_offsets, max_logic + tl.log(sum_exp), mask=head_offsets < gqa_group_size)


@torch.no_grad()
def sliding_window_decode_attention(
    q,
    k,
    v,
    b_req_idx,
    b_seq_len,
    b_q_start_loc,
    sliding_window: int,
    scratch_start: int,
    out=None,
    alloc_tensor_func=torch.empty,
):
    """Decode one token per request without a token-to-sliding-KV index table."""
    batch_size, q_head_num, head_dim = q.shape
    assert k.shape == v.shape and k.shape[-1] == head_dim
    assert head_dim in {16, 32, 64, 128, 256, 512}
    assert q_head_num % k.shape[1] == 0
    assert b_req_idx.shape == b_seq_len.shape == b_q_start_loc.shape == (batch_size,)
    assert sliding_window > 0 and scratch_start >= sliding_window
    assert q.dtype == k.dtype == v.dtype

    # Keep the common GQA wrapper's launch and reduction schedule unchanged.
    block_seq = 256
    block_num = 128 if batch_size <= 16 else (64 if batch_size <= 64 else 32)
    mid_o = alloc_tensor_func([batch_size, q_head_num, block_num, head_dim], dtype=q.dtype, device=q.device)
    mid_logsumexp = alloc_tensor_func([batch_size, q_head_num, block_num], dtype=torch.float32, device=q.device)
    out = alloc_tensor_func(q.shape, dtype=q.dtype, device=q.device) if out is None else out
    group_size = q_head_num // k.shape[1]
    _sliding_window_decode_stage1[(batch_size, k.shape[1], block_num)](
        q,
        k,
        v,
        b_req_idx,
        b_seq_len,
        b_q_start_loc,
        mid_o,
        mid_logsumexp,
        1.0 / (head_dim ** 0.5),
        *q.stride(),
        *k.stride(),
        *v.stride(),
        *mid_o.stride(),
        *mid_logsumexp.stride(),
        group_size,
        WINDOW=sliding_window,
        SCRATCH_START=scratch_start,
        Q_HEAD_NUM=max(16, triton.next_power_of_2(group_size)),
        BLOCK_SEQ=block_seq,
        BLOCK_DMODEL=head_dim,
        BLOCK_N=16,
        num_warps=4,
        num_stages=2,
    )
    flash_decode_stage2(
        mid_out=mid_o,
        mid_out_logexpsum=mid_logsumexp,
        B_Seqlen=b_seq_len,
        out=out,
        block_seq=block_seq,
        sliding_window=(sliding_window - 1, 0),
    )
    return out
