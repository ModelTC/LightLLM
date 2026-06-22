from typing import List, Optional, Tuple
import torch

from lightllm.common.basemodel.triton_kernel.post_process.apply_invalid_token import (
    apply_invalid_token_ids as _triton_apply_invalid_token_ids,
)
from lightllm.common.basemodel.triton_kernel.post_process.apply_penalty import (
    apply_penalty as _triton_apply_penalty,
)
from lightllm.common.basemodel.triton_kernel.post_process.apply_penalty_gpu_cache import (
    apply_penalty_gpu_cache as _triton_apply_penalty_gpu_cache,
)
from lightllm.server.router.model_infer.mode_backend.generic_post_process import (
    top_p_top_k_sample_triton,
    top_p_top_k_sample_sglang_kernel,
)
from lightllm.platform.base.sampling import register_sampling_op


@register_sampling_op("cuda_like")
def apply_invalid_token_ids(
    *,
    logits: torch.Tensor,
    invalid_token_ids: torch.Tensor,
    cu_invalid_token_num: torch.Tensor,
) -> None:
    _triton_apply_invalid_token_ids(
        Logits=logits,
        invalid_token_ids=invalid_token_ids,
        cu_invalid_token_num=cu_invalid_token_num,
    )


@register_sampling_op("cuda_like")
def apply_penalty(
    *,
    logits: torch.Tensor,
    b_req_idx: torch.Tensor,
    b_length_penalty_param: torch.Tensor,
    b_mask_eos_reqs: torch.Tensor,
    p_token_ids: torch.Tensor,
    p_token_counts: torch.Tensor,
    p_cumsum_seq_len: torch.Tensor,
    eos_ids: torch.Tensor,
    req_to_presence_penalty: torch.Tensor,
    req_to_frequency_penalty: torch.Tensor,
    req_to_repetition_penalty: torch.Tensor,
    req_to_exponential_decay_length_penalty: torch.Tensor,
    vocab_size: int,
) -> None:
    _triton_apply_penalty(
        Logits=logits,
        b_req_idx=b_req_idx,
        b_length_penalty_param=b_length_penalty_param,
        b_mask_eos_reqs=b_mask_eos_reqs,
        p_token_ids=p_token_ids,
        p_token_counts=p_token_counts,
        p_cumsum_seq_len=p_cumsum_seq_len,
        eos_ids=eos_ids,
        req_to_presence_penalty=req_to_presence_penalty,
        req_to_frequency_penalty=req_to_frequency_penalty,
        req_to_repetition_penalty=req_to_repetition_penalty,
        req_to_exponential_decay_length_penalty=req_to_exponential_decay_length_penalty,
        vocab_size=vocab_size,
    )

@register_sampling_op("cuda_like")
def apply_penalty_gpu_cache(
    *,
    logits: torch.Tensor,
    b_req_idx: torch.Tensor,
    b_length_penalty_param: torch.Tensor,
    b_mask_eos_reqs: torch.Tensor,
    eos_ids: torch.Tensor,
    req_to_presence_penalty: torch.Tensor,
    req_to_frequency_penalty: torch.Tensor,
    req_to_repetition_penalty: torch.Tensor,
    req_to_out_token_id_counter: torch.Tensor,
    req_to_exponential_decay_length_penalty: torch.Tensor,
    vocab_size: int,
) -> None:
    _triton_apply_penalty_gpu_cache(
        Logits=logits,
        b_req_idx=b_req_idx,
        b_length_penalty_param=b_length_penalty_param,
        b_mask_eos_reqs=b_mask_eos_reqs,
        eos_ids=eos_ids,
        req_to_presence_penalty=req_to_presence_penalty,
        req_to_frequency_penalty=req_to_frequency_penalty,
        req_to_repetition_penalty=req_to_repetition_penalty,
        req_to_out_token_id_counter=req_to_out_token_id_counter,
        req_to_exponential_decay_length_penalty=req_to_exponential_decay_length_penalty,
        vocab_size=vocab_size,
    )


@register_sampling_op("cuda_like")
def top_p_top_k_sample(
    *,
    probs: torch.Tensor,
    top_ps: torch.Tensor,
    top_ks: torch.Tensor,
    generators: List[Optional[torch.Generator]] | None = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    return top_p_top_k_sample_triton(probs=probs, top_ps=top_ps, top_ks=top_ks, generators=generators)


@register_sampling_op("cuda_like", sampling_backend="sglang_kernel")
def top_p_top_k_sample(
    *,
    probs: torch.Tensor,
    top_ps: torch.Tensor,
    top_ks: torch.Tensor,
    generators: List[Optional[torch.Generator]] | None = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    return top_p_top_k_sample_sglang_kernel(probs=probs, top_ps=top_ps, top_ks=top_ks, generators=generators)
