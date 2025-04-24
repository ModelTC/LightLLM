import torch

import triton
import triton.language as tl
import numpy as np


@triton.jit
def _fwd_kernel_apply_penalty(
    Logits,
    presence_penalty,
    freqency_penalty,
    repetition_penalty,
    p_token_ids,
    p_token_counts,
    p_cumsum_seq_len,
    exponential_decay_length_penalties,
    length_penalty_idx,
    eos_ids,
    mask_eos_reqs,
    stride_logit_b,
    stride_logit_s,
    BLOCK_P: tl.constexpr,
    EOS_ID_NUM: tl.constexpr,
):
    cur_batch = tl.program_id(0)
    cur_freqency = tl.load(freqency_penalty + cur_batch)
    cur_presence = tl.load(presence_penalty + cur_batch)
    cur_repetition = tl.load(repetition_penalty + cur_batch)

    cur_batch_start_index = tl.load(p_cumsum_seq_len + cur_batch)
    cur_batch_end_index = tl.load(p_cumsum_seq_len + cur_batch + 1)
    for block_start_index in range(cur_batch_start_index, cur_batch_end_index, BLOCK_P):
        cur_batch_id_offset = block_start_index + tl.arange(0, BLOCK_P)
        batch_ids = tl.load(p_token_ids + cur_batch_id_offset, mask=cur_batch_id_offset < cur_batch_end_index, other=0)
        batch_ids_count = tl.load(
            p_token_counts + cur_batch_id_offset, mask=cur_batch_id_offset < cur_batch_end_index, other=0
        )

        row_start_ptr = Logits + cur_batch * stride_logit_b
        cur_offset = row_start_ptr + batch_ids
        cur_logits = tl.load(cur_offset, mask=cur_batch_id_offset < cur_batch_end_index, other=0.0)
        rep_logits = tl.where(cur_logits > 0, cur_logits / cur_repetition, cur_logits * cur_repetition)
        freq_logits = rep_logits - batch_ids_count * cur_freqency
        pre_logits = freq_logits - cur_presence
        output_ptr = Logits + cur_batch * stride_logit_b + batch_ids
        tl.store(output_ptr, pre_logits, mask=cur_batch_id_offset < cur_batch_end_index)

    mask_eos = tl.load(mask_eos_reqs + cur_batch)
    exponential_decay_length_penalty = tl.load(exponential_decay_length_penalties + cur_batch)
    length_penalty = tl.load(length_penalty_idx + cur_batch)
    penalty_scale = tl.exp2(tl.log2(exponential_decay_length_penalty) * length_penalty) - 1

    for eos_index in range(EOS_ID_NUM):
        eos_id = tl.load(eos_ids + eos_index)
        cur_eos_logit_ptr = Logits + cur_batch * stride_logit_b + eos_id
        cur_eos_logit = tl.load(cur_eos_logit_ptr)
        cur_eos_logit = cur_eos_logit + tl.abs(cur_eos_logit) * penalty_scale
        cur_eos_logit = tl.where(mask_eos, -10000000.0, cur_eos_logit)
        tl.store(cur_eos_logit_ptr, cur_eos_logit)
    return


@torch.no_grad()
def apply_penalty(
    Logits,
    presence_penalty,
    freqency_penalty,
    repetition_penalty,
    p_token_ids,
    p_token_counts,
    p_cumsum_seq_len,
    exponential_decay_length_penalties,
    length_penalty_idx,
    eos_ids,
    mask_eos_reqs,
):
    assert Logits.is_contiguous()
    BLOCK_P = 1024
    num_warps = 8
    _fwd_kernel_apply_penalty[(Logits.shape[0],)](
        Logits,
        presence_penalty,
        freqency_penalty,
        repetition_penalty,
        p_token_ids,
        p_token_counts,
        p_cumsum_seq_len,
        exponential_decay_length_penalties,
        length_penalty_idx,
        eos_ids,
        mask_eos_reqs,
        Logits.stride(0),
        Logits.stride(1),
        num_warps=num_warps,
        BLOCK_P=BLOCK_P,
        EOS_ID_NUM=eos_ids.shape[0],
    )
    return
