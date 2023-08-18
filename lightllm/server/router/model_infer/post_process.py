import re
import torch
from typing import List
from lightllm.server.router.model_infer.infer_batch import InferBatch
from lightllm.common.basemodel.triton_kernel.apply_penalty import apply_penalty

def sample(logits, batch:InferBatch):
    logits = logits.contiguous()
    presence_penalties, frequency_penalties, temperatures, top_ps, top_ks, p_token_ids, p_token_counts, p_cumsum_seq_len, p_max_len_in_batch = batch.get_post_sample_tensors()
    
    apply_penalty(logits, presence_penalties, frequency_penalties, p_token_ids, p_token_counts, p_cumsum_seq_len, p_max_len_in_batch) 
    logits.div_(temperatures.view((-1, 1)))
    probs = torch.softmax(logits, dim=-1)
    probs_sort, probs_idx = _top_p_top_k(probs, top_ps, top_ks)
    sampled_index = torch.multinomial(probs_sort, num_samples=1, replacement=True)
    
    batch_next_token_ids = torch.gather(probs_idx, dim=1, index=sampled_index)
    
    return batch_next_token_ids.view(-1)

def _top_p_top_k(probs: torch.Tensor, top_ps: torch.Tensor, top_ks: torch.Tensor):    
    probs_sort, probs_idx = probs.sort(dim=-1, descending=True)
    
    probs_sum = torch.cumsum(probs_sort, dim=-1)
    probs_sort[(probs_sum - probs_sort) > top_ps.view(-1, 1)] = 0.0

    probs_sort[torch.arange(0, probs.shape[-1], device="cuda").view(1, -1) >= top_ks.view(-1, 1)] = 0.0

    return probs_sort, probs_idx
