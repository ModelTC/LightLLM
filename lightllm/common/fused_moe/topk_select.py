# Adapted from
# https://github.com/vllm-project/vllm/blob/v0.6.4.post1/vllm/model_executor/layers/fused_moe/fused_moe.py
# of the vllm-project/vllm GitHub repository.
#
# Copyright 2023 ModelTC Team
# Copyright 2023 vLLM Team
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import torch
from lightllm.common.vllm_kernel import _custom_ops as ops


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

    ops.topk_softmax(
        topk_weights,
        topk_ids,
        token_expert_indicies,
        gating_output.float(),  # TODO(woosuk): Optimize this.
    )
    del token_expert_indicies  # Not used. Will be used in the future.

    if renormalize:
        topk_weights = topk_weights / topk_weights.sum(dim=-1, keepdim=True)

    return topk_weights, topk_ids


# This is used by the Deepseek-V2 model
def grouped_topk(
    hidden_states: torch.Tensor,
    gating_output: torch.Tensor,
    topk: int,
    renormalize: bool,
    num_expert_group: int = 0,
    topk_group: int = 0,
):

    assert hidden_states.shape[0] == gating_output.shape[0], "Number of tokens mismatch"

    scores = torch.softmax(gating_output, dim=-1)
    num_token = scores.shape[0]
    group_scores = scores.view(num_token, num_expert_group, -1).max(dim=-1).values  # [n, n_group]
    group_idx = torch.topk(group_scores, k=topk_group, dim=-1, sorted=False)[1]  # [n, top_k_group]
    group_mask = torch.zeros_like(group_scores)  # [n, n_group]
    group_mask.scatter_(1, group_idx, 1)  # [n, n_group]
    score_mask = (
        group_mask.unsqueeze(-1)
        .expand(num_token, num_expert_group, scores.shape[-1] // num_expert_group)
        .reshape(num_token, -1)
    )  # [n, e]
    tmp_scores = scores.masked_fill(~score_mask.bool(), 0.0)  # [n, e]
    topk_weights, topk_ids = torch.topk(tmp_scores, k=topk, dim=-1, sorted=False)

    if renormalize:
        topk_weights = topk_weights / topk_weights.sum(dim=-1, keepdim=True)

    return topk_weights.to(torch.float32), topk_ids.to(torch.int32)
