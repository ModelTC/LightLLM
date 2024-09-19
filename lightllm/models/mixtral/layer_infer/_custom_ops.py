import functools
import json
import os
from typing import Any, Dict, Optional, Tuple

import torch
import triton
import triton.language as tl
from lightllm.utils.log_utils import init_logger

logger = init_logger(__name__)

# Pytorch version
# Triton version in progress
def topk_softmax(
    topk_weights,
    topk_ids,
    token_expert_indicies,
    gating_output,
    topk=2,
):
    scores = torch.softmax(gating_output, dim=-1)
    topk_weights, topk_ids = torch.topk(scores, k=topk, dim=-1, sorted=False)
    return topk_weights, topk_ids


def fused_topk(
    hidden_states: torch.Tensor,
    gating_output: torch.Tensor,
    topk: int,
    renormalize: bool,
):
    assert hidden_states.shape[0] == gating_output.shape[0], "Number of tokens mismatch"

    M, _ = hidden_states.shape

    topk_weights = torch.empty(M, topk, dtype=torch.float32, device=hidden_states.device)
    topk_ids = torch.empty(M, topk, dtype=torch.int32, device=hidden_states.device)
    token_expert_indicies = torch.empty(M, topk, dtype=torch.int32, device=hidden_states.device)
    topk_weights, topk_ids = topk_softmax(topk_weights, topk_ids, token_expert_indicies, gating_output.float(), topk)
    del token_expert_indicies  # Not used. Will be used in the future.

    if renormalize:
        topk_weights = topk_weights / topk_weights.sum(dim=-1, keepdim=True)
    return topk_weights, topk_ids
