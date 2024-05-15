import os
import re
import torch
from typing import List, Tuple
from lightllm.server.router.model_infer.infer_batch import InferBatch, group_mapping, requests_mapping
from lightllm.common.basemodel.triton_kernel.apply_penalty import apply_penalty
from lightllm.server.io_struct import FinishStatus
from lightllm.server.router.model_infer.mode_backend.continues_batch.post_process import _top_p_top_k
SPLIT_TOKEN = int(os.getenv("SPLIT_TOKEN", -1))
global_topp = [0.7, 0.9, 0.9]
global_repetition = [1.12, 1.15, 1.14]


def sample(logits, req_groups, is_prefill, vocab_size, req_manager, eos_id: List[int] = [2]):
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
    ) = _get_post_sample_tensors(req_groups, is_prefill)
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

    batch_next_token_ids = []
    batch_next_token_logprobs = []
    start = 0
    for i in range(len(req_groups)):
        req_group = req_groups[i]
        best_of = req_group.best_of
        end = start + 1 if is_prefill else start + best_of
        if best_of > 1 and not req_group.has_beam:            
            next_token_id, next_token_logprob = diverse_sample(probs[start:end], req_group, is_prefill, req_manager)
        else:
            probs_sort, probs_idx = _top_p_top_k(probs[start:end], top_ps[start:end], top_ks[start:end])
            next_token_id, next_token_prob = random_sample(probs_sort, probs_idx)
            next_token_id = next_token_id.view(-1).detach().cpu().numpy()
            next_token_logprob = torch.log(next_token_prob).view(-1).detach().cpu().numpy()


        batch_next_token_ids.append(next_token_id)
        batch_next_token_logprobs.append(next_token_logprob)
        start = end
    return batch_next_token_ids, batch_next_token_logprobs

def random_sample(probs_sort, probs_idx):
    sampled_index = torch.multinomial(probs_sort, num_samples=1, replacement=True)
    batch_next_token_ids = torch.gather(probs_idx, dim=1, index=sampled_index)
    batch_next_token_probs = torch.gather(probs_sort, dim=1, index=sampled_index)
    return batch_next_token_ids, batch_next_token_probs

def diverse_sample(probs, req_group, is_prefill, req_manager):
    best_of = req_group.best_of
    valid_beams = 0
    logprobs = torch.log(probs)
    best_of = req_group.best_of
    next_token_logprob, next_token_id = torch.topk(logprobs[0].view(-1), best_of, dim=0, largest=True, sorted=True)
    next_token_logprob = next_token_logprob.detach().cpu().numpy()
    next_token_id = next_token_id.detach().cpu().numpy()
    if is_prefill and next_token_id[0] == SPLIT_TOKEN:
        next_token_id = [next_token_id[0]] * best_of
        next_token_logprob = [next_token_logprob[0]] * best_of
    else:
        req_group.has_beam = True
    if is_prefill:
        req_group.beam_copy(req_manager, is_prefill)
    return next_token_id, next_token_logprob

def _get_post_sample_tensors(req_groups, is_prefill):
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
    for i, req_group_obj in enumerate(req_groups):
        for j in range(req_group_obj.best_of):
            req_obj = req_group_obj.get_req(j)
            relative_idx = req_group_obj.get_relative_index(j)
            id_to_count = req_obj.out_token_id_count
            sample_param = req_obj.sampling_param
            presence_penalties.append(sample_param.presence_penalty)
            frequency_penalties.append(sample_param.frequency_penalty)
            repetition_penalties.append(global_repetition[relative_idx % len(global_repetition)])
            exponential_decay_length_penalties.append(sample_param.exponential_decay_length_penalty[1])
            out_token_len = len(req_obj.input_token_ids) - req_obj.prompt_len
            length_penalty_idx.append(max(out_token_len - sample_param.exponential_decay_length_penalty[0], 0))
            mask_eos_reqs.append(out_token_len < sample_param.min_new_tokens - 1)

            temperatures.append(sample_param.temperature)
            top_ps.append(global_topp[relative_idx % len(global_topp)])
            top_ks.append(sample_param.top_k)

            for token_id, count in id_to_count.items():
                p_token_ids.append(token_id)
                p_token_counts.append(count)
            p_seq_len.append(len(id_to_count))
            p_max_len_in_batch = max(p_max_len_in_batch, len(id_to_count))
            if is_prefill:
                break

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
