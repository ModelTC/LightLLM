import re
import torch
from typing import List, Tuple
from lightllm.server.router.model_infer.infer_batch import InferBatch
from lightllm.common.basemodel.triton_kernel.apply_penalty import apply_penalty


def sample(logits, reqs, eos_id: List[int] = [2]):
    logits = logits.contiguous()
    (
        presence_penalties,
        frequency_penalties,
        repetition_penalties,
        exponential_decay_length_penalties,
        temperatures,
        top_ps,
        top_ks,
        p_token_ids,
        p_token_counts,
        p_cumsum_seq_len,
        p_max_len_in_batch,
        length_penalty_idx,
        mask_eos_reqs,
    ) = _get_post_sample_tensors(reqs)

    apply_penalty(
        logits,
        presence_penalties,
        frequency_penalties,
        repetition_penalties,
        p_token_ids,
        p_token_counts,
        p_cumsum_seq_len,
        p_max_len_in_batch,
    )
    logits[:, eos_id] = logits[:, eos_id] + torch.abs(logits[:, eos_id]) * (
        torch.pow(exponential_decay_length_penalties, length_penalty_idx).view((-1, 1)) - 1
    )
    logits[mask_eos_reqs, eos_id] = -1000000.0
    logits.div_(temperatures.view((-1, 1)))
    probs = torch.softmax(logits, dim=-1)
    probs_sort, probs_idx = _top_p_top_k(probs, top_ps, top_ks)
    sampled_index = torch.multinomial(probs_sort, num_samples=1, replacement=True)

    batch_next_token_ids = torch.gather(probs_idx, dim=1, index=sampled_index)
    batch_next_token_probs = torch.gather(probs_sort, dim=1, index=sampled_index)

    return batch_next_token_ids.view(-1), batch_next_token_probs.view(-1)


def _top_p_top_k(probs: torch.Tensor, top_ps: torch.Tensor, top_ks: torch.Tensor):
    probs_sort, probs_idx = probs.sort(dim=-1, descending=True)

    probs_sum = torch.cumsum(probs_sort, dim=-1)
    probs_sort[(probs_sum - probs_sort) > top_ps.view(-1, 1)] = 0.0

    probs_sort[torch.arange(0, probs.shape[-1], device="cuda").view(1, -1) >= top_ks.view(-1, 1)] = 0.0

    return probs_sort, probs_idx


def _get_post_sample_tensors(reqs):
    presence_penalties: List[float] = []
    frequency_penalties: List[float] = []
    repetition_penalties: List[float] = []
    exponential_decay_length_penalties: List[float] = []
    temperatures: List[float] = []
    top_ps: List[float] = []
    top_ks: List[int] = []
    p_token_ids: List[int] = []
    p_token_counts: List[int] = []
    p_seq_len: List[int] = [
        0,
    ]
    p_max_len_in_batch: int = 0
    length_penalty_idx: List[int] = []
    mask_eos_reqs: List[bool] = []
    for i, req_obj in enumerate(reqs):
        id_to_count = req_obj.out_token_id_count
        sample_param = req_obj.sampling_param
        presence_penalties.append(sample_param.presence_penalty)
        frequency_penalties.append(sample_param.frequency_penalty)
        repetition_penalties.append(sample_param.repetition_penalty)
        exponential_decay_length_penalties.append(sample_param.exponential_decay_length_penalty[1])
        out_token_len = len(req_obj.input_token_ids) - req_obj.prompt_len
        length_penalty_idx.append(max(out_token_len - sample_param.exponential_decay_length_penalty[0], 0))
        mask_eos_reqs.append(out_token_len < sample_param.min_new_tokens - 1)

        temperatures.append(sample_param.temperature)
        top_ps.append(sample_param.top_p)
        top_ks.append(sample_param.top_k)

        for token_id, count in id_to_count.items():
            p_token_ids.append(token_id)
            p_token_counts.append(count)
        p_seq_len.append(len(id_to_count))
        p_max_len_in_batch = max(p_max_len_in_batch, len(id_to_count))

    presence_penalties = torch.tensor(presence_penalties, dtype=torch.float, device="cuda")
    frequency_penalties = torch.tensor(frequency_penalties, dtype=torch.float, device="cuda")
    repetition_penalties = torch.tensor(repetition_penalties, dtype=torch.float, device="cuda")
    exponential_decay_length_penalties = torch.tensor(
        exponential_decay_length_penalties, dtype=torch.float, device="cuda"
    )
    temperatures = torch.tensor(temperatures, dtype=torch.float, device="cuda")
    top_ps = torch.tensor(top_ps, dtype=torch.float, device="cuda")
    top_ks = torch.tensor(top_ks, dtype=torch.int32, device="cuda")
    p_token_ids = torch.tensor(p_token_ids, dtype=torch.int32, device="cuda")
    p_token_counts = torch.tensor(p_token_counts, dtype=torch.int32, device="cuda")
    p_seq_len = torch.tensor(p_seq_len, dtype=torch.int32, device="cuda")
    p_cumsum_seq_len = torch.cumsum(p_seq_len, dim=0, dtype=torch.int32)
    length_penalty_idx = torch.tensor(length_penalty_idx, dtype=torch.int32, device="cuda")
    mask_eos_reqs = torch.tensor(mask_eos_reqs, dtype=torch.bool, device="cuda")
    return (
        presence_penalties,
        frequency_penalties,
        repetition_penalties,
        exponential_decay_length_penalties,
        temperatures,
        top_ps,
        top_ks,
        p_token_ids,
        p_token_counts,
        p_cumsum_seq_len,
        p_max_len_in_batch,
        length_penalty_idx,
        mask_eos_reqs,
    )
