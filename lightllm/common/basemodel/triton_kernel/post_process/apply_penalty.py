import torch
import triton
import triton.language as tl
import numpy as np
from lightllm.common.req_manager import ReqSamplingParamsManager


@triton.jit
def _fwd_kernel_apply_penalty(
    Logits,
    stride_logit_b,
    stride_logit_s,
    b_req_idx,
    req_to_presence_penalty,
    req_to_frequency_penalty,
    req_to_repetition_penalty,
    req_to_exponential_decay_length_penalty,
    b_length_penalty_param,
    p_token_ids,
    p_token_counts,
    p_cumsum_seq_len,
    eos_ids,
    b_mask_eos_reqs,
    vocab_size,
    BLOCK_P: tl.constexpr,
    EOS_ID_NUM: tl.constexpr,
):
    cur_batch = tl.program_id(0)
    cur_req_idx = tl.load(b_req_idx + cur_batch)
    cur_freqency = tl.load(req_to_frequency_penalty + cur_req_idx)
    cur_presence = tl.load(req_to_presence_penalty + cur_req_idx)
    cur_repetition = tl.load(req_to_repetition_penalty + cur_req_idx)

    cur_batch_start_index = tl.load(p_cumsum_seq_len + cur_batch)
    cur_batch_end_index = tl.load(p_cumsum_seq_len + cur_batch + 1)
    for block_start_index in range(cur_batch_start_index, cur_batch_end_index, BLOCK_P):
        cur_batch_id_offset = block_start_index + tl.arange(0, BLOCK_P)
        token_ids = tl.load(p_token_ids + cur_batch_id_offset, mask=cur_batch_id_offset < cur_batch_end_index, other=0)
        token_ids_count = tl.load(
            p_token_counts + cur_batch_id_offset, mask=cur_batch_id_offset < cur_batch_end_index, other=0
        )

        row_start_ptr = Logits + cur_batch * stride_logit_b
        cur_offset = row_start_ptr + token_ids
        cur_logits = tl.load(
            cur_offset, mask=(cur_batch_id_offset < cur_batch_end_index) & (token_ids < vocab_size), other=0.0
        )
        rep_logits = tl.where(cur_logits > 0, cur_logits / cur_repetition, cur_logits * cur_repetition)
        freq_logits = rep_logits - token_ids_count * cur_freqency
        pre_logits = freq_logits - cur_presence
        output_ptr = Logits + cur_batch * stride_logit_b + token_ids
        tl.store(output_ptr, pre_logits, mask=(cur_batch_id_offset < cur_batch_end_index) & (token_ids < vocab_size))

    mask_eos = tl.load(b_mask_eos_reqs + cur_batch)
    exponential_decay_length_penalty = tl.load(req_to_exponential_decay_length_penalty + cur_req_idx)
    length_penalty = tl.load(b_length_penalty_param + cur_batch)
    penalty_scale = tl.exp2(tl.log2(exponential_decay_length_penalty) * length_penalty) - 1

    for eos_index in range(EOS_ID_NUM):
        eos_id = tl.load(eos_ids + eos_index)
        cur_eos_logit_ptr = Logits + cur_batch * stride_logit_b + eos_id
        cur_eos_logit = tl.load(cur_eos_logit_ptr)
        cur_eos_logit = cur_eos_logit + tl.abs(cur_eos_logit) * penalty_scale
        cur_eos_logit = tl.where(mask_eos != 0, -10000000.0, cur_eos_logit)
        tl.store(cur_eos_logit_ptr, cur_eos_logit)
    return


@torch.no_grad()
def apply_penalty(
    Logits: torch.Tensor,
    b_req_idx: torch.Tensor,
    b_length_penalty_param: torch.Tensor,
    b_mask_eos_reqs: torch.Tensor,
    p_token_ids: torch.Tensor,
    p_token_counts: torch.Tensor,
    p_cumsum_seq_len: torch.Tensor,
    eos_ids: torch.Tensor,
    sampling_params_manager: ReqSamplingParamsManager,
):
    assert Logits.is_contiguous()
    BLOCK_P = 1024
    num_warps = 8
    _fwd_kernel_apply_penalty[(Logits.shape[0],)](
        Logits=Logits,
        stride_logit_b=Logits.stride(0),
        stride_logit_s=Logits.stride(1),
        b_req_idx=b_req_idx,
        req_to_presence_penalty=sampling_params_manager.req_to_presence_penalty,
        req_to_frequency_penalty=sampling_params_manager.req_to_frequency_penalty,
        req_to_repetition_penalty=sampling_params_manager.req_to_repetition_penalty,
        req_to_exponential_decay_length_penalty=sampling_params_manager.req_to_exponential_decay_length_penalty,
        b_length_penalty_param=b_length_penalty_param,
        p_token_ids=p_token_ids,
        p_token_counts=p_token_counts,
        p_cumsum_seq_len=p_cumsum_seq_len,
        eos_ids=eos_ids,
        b_mask_eos_reqs=b_mask_eos_reqs,
        vocab_size=sampling_params_manager.vocab_size,
        num_warps=num_warps,
        BLOCK_P=BLOCK_P,
        EOS_ID_NUM=eos_ids.shape[0],
    )
    return
