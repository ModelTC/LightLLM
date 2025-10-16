import torch
import triton
import triton.language as tl
import torch.nn.functional as F
import numpy as np
from lightllm.common.req_manager import ReqSamplingParamsManager


@triton.jit
def _fwd_kernel_apply_penalty_cache(
    Logits,
    stride_logit_b,
    stride_logit_s,
    b_req_idx,
    req_to_presence_penalty,
    req_to_frequency_penalty,
    req_to_repetition_penalty,
    req_to_out_token_id_counter,
    stride_counter_r,
    stride_counter_s,
    vocab_size,
    BLOCK_P: tl.constexpr,
):
    cur_batch = tl.program_id(0)
    cur_req_idx = tl.load(b_req_idx + cur_batch)
    block_idx = tl.program_id(1)
    cur_freqency = tl.load(req_to_frequency_penalty + cur_req_idx)
    cur_presence = tl.load(req_to_presence_penalty + cur_req_idx)
    cur_repetition = tl.load(req_to_repetition_penalty + cur_req_idx)

    token_ids = BLOCK_P * block_idx + tl.arange(0, BLOCK_P)
    mask = token_ids < vocab_size
    token_ids_count = tl.load(
        req_to_out_token_id_counter + cur_req_idx * stride_counter_r + token_ids,
        mask=mask,
        other=0,
    )
    row_start_ptr = Logits + cur_batch * stride_logit_b
    cur_offset = row_start_ptr + token_ids
    origin_logits = tl.load(cur_offset, mask=mask, other=0.0)
    p_logits = tl.where(origin_logits > 0, origin_logits / cur_repetition, origin_logits * cur_repetition)
    p_logits = tl.where(token_ids_count > 0, p_logits, origin_logits)
    p_logits = p_logits - token_ids_count * cur_freqency
    p_logits = p_logits - tl.where(token_ids_count > 0, cur_presence, 0.0)
    output_ptr = Logits + cur_batch * stride_logit_b + token_ids
    tl.store(output_ptr, p_logits, mask=mask)
    return


@triton.jit
def _eos_penalty(
    Logits,
    stride_logit_b,
    stride_logit_s,
    b_req_idx,
    req_to_exponential_decay_length_penalty,
    b_length_penalty_param,
    eos_ids,
    b_mask_eos_reqs,
    batch_size,
    BLOCK: tl.constexpr,
    EOS_ID_NUM: tl.constexpr,
):
    block_index = tl.program_id(0)
    offs = block_index * BLOCK + tl.arange(0, BLOCK)
    mask = offs < batch_size
    req_idxes = tl.load(b_req_idx + offs, mask=mask, other=0)
    exponential_decay_length_penalty = tl.load(
        req_to_exponential_decay_length_penalty + req_idxes, mask=mask, other=1.0
    )
    length_penalty = tl.load(b_length_penalty_param + offs, mask=mask, other=0)
    penalty_scale = tl.exp2(tl.log2(exponential_decay_length_penalty) * length_penalty) - 1
    mask_eos = tl.load(b_mask_eos_reqs + offs, mask=mask, other=True)
    for eos_index in range(EOS_ID_NUM):
        eos_id = tl.load(eos_ids + eos_index)
        cur_eos_logit_ptr = Logits + offs * stride_logit_b + eos_id
        cur_eos_logit = tl.load(cur_eos_logit_ptr, mask=mask, other=0.0)
        cur_eos_logit = cur_eos_logit + tl.abs(cur_eos_logit) * penalty_scale
        cur_eos_logit = tl.where(mask_eos != 0, -10000000.0, cur_eos_logit)
        tl.store(cur_eos_logit_ptr, cur_eos_logit, mask=mask)
    return


@torch.no_grad()
def apply_penalty_gpu_cache(
    Logits: torch.Tensor,
    b_req_idx: torch.Tensor,
    b_length_penalty_param: torch.Tensor,
    b_mask_eos_reqs: torch.Tensor,
    eos_ids: torch.Tensor,
    sampling_params_manager: ReqSamplingParamsManager,
):
    assert Logits.is_contiguous()
    BLOCK_P = 2048
    num_warps = 8
    vocab_size = sampling_params_manager.vocab_size
    req_to_out_token_id_counter = sampling_params_manager.req_to_out_token_id_counter
    _fwd_kernel_apply_penalty_cache[(Logits.shape[0], triton.cdiv(vocab_size, BLOCK_P))](
        Logits=Logits,
        stride_logit_b=Logits.stride(0),
        stride_logit_s=Logits.stride(1),
        b_req_idx=b_req_idx,
        req_to_presence_penalty=sampling_params_manager.req_to_presence_penalty,
        req_to_frequency_penalty=sampling_params_manager.req_to_frequency_penalty,
        req_to_repetition_penalty=sampling_params_manager.req_to_repetition_penalty,
        req_to_out_token_id_counter=req_to_out_token_id_counter,
        stride_counter_r=req_to_out_token_id_counter.stride(0),
        stride_counter_s=req_to_out_token_id_counter.stride(1),
        vocab_size=vocab_size,
        BLOCK_P=BLOCK_P,
        num_warps=num_warps,
    )

    BLOCK = 128
    grid = (triton.cdiv(Logits.shape[0], BLOCK),)
    _eos_penalty[grid](
        Logits=Logits,
        stride_logit_b=Logits.stride(0),
        stride_logit_s=Logits.stride(1),
        b_req_idx=b_req_idx,
        req_to_exponential_decay_length_penalty=sampling_params_manager.req_to_exponential_decay_length_penalty,
        b_length_penalty_param=b_length_penalty_param,
        eos_ids=eos_ids,
        b_mask_eos_reqs=b_mask_eos_reqs,
        batch_size=Logits.shape[0],
        BLOCK=BLOCK,
        EOS_ID_NUM=eos_ids.shape[0],
        num_warps=1,
    )
    return
