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

import os
import torch
from lightllm.utils.sgl_utils import sgl_ops
from lightllm.utils.light_utils import light_ops
from typing import Callable, List, Optional, Tuple
from lightllm.common.fused_moe.softmax_topk import softmax_topk
from lightllm.common.triton_utils.autotuner import Autotuner

use_cuda_grouped_topk = os.getenv("LIGHTLLM_CUDA_GROUPED_TOPK", "False").upper() in ["ON", "TRUE", "1"]


def fused_topk(
    hidden_states: torch.Tensor,
    gating_output: torch.Tensor,
    topk: int,
    renormalize: bool,
):
    assert hidden_states.shape[0] == gating_output.shape[0], "Number of tokens mismatch"

    if sgl_ops is None:
        return softmax_topk(gating_output, topk, renorm=renormalize)
    M, _ = hidden_states.shape

    topk_weights = torch.empty(M, topk, dtype=torch.float32, device=hidden_states.device)
    topk_ids = torch.empty(M, topk, dtype=torch.int32, device=hidden_states.device)

    sgl_ops.topk_softmax(
        topk_weights,
        topk_ids,
        gating_output.float(),  # TODO(woosuk): Optimize this.
        renormalize=renormalize,
    )
    return topk_weights, topk_ids


# This is used by the Deepseek-V2 model
def grouped_topk(
    hidden_states: torch.Tensor,
    gating_output: torch.Tensor,
    correction_bias: torch.Tensor,
    topk: int,
    renormalize: bool,
    num_expert_group: int = 0,
    topk_group: int = 0,
    scoring_func: str = "softmax",
):
    assert hidden_states.shape[0] == gating_output.shape[0], "Number of tokens mismatch"
    if scoring_func == "sigmoid":
        scores = torch.sigmoid(gating_output)
    else:
        scores = torch.softmax(gating_output, dim=-1)
    old_scores = scores
    if correction_bias is not None:
        scores = scores + correction_bias

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
    _, topk_ids = torch.topk(tmp_scores, k=topk, dim=-1, sorted=False)
    topk_weights = old_scores.gather(1, topk_ids)
    if renormalize:
        topk_weights = topk_weights / topk_weights.sum(dim=-1, keepdim=True)

    return topk_weights.to(torch.float32), topk_ids.to(torch.int32)


# biased_grouped_topk adapt from sgl-project/sglang/python/sglang/srt/layers/moe/topk.py
def biased_grouped_topk(
    hidden_states: torch.Tensor,
    gating_output: torch.Tensor,
    correction_bias: torch.Tensor,
    topk: int,
    renormalize: bool,
    num_expert_group: int = 0,
    topk_group: int = 0,
    scoring_func: str = "sigmoid",
):
    assert hidden_states.shape[0] == gating_output.shape[0], "Number of tokens mismatch"
    scores = gating_output.sigmoid()
    num_token = scores.shape[0]
    scores_for_choice = scores.view(num_token, -1) + correction_bias.unsqueeze(0)
    group_scores = (
        scores_for_choice.view(num_token, num_expert_group, -1).topk(2, dim=-1)[0].sum(dim=-1)
    )  # [n, n_group]
    group_idx = torch.topk(group_scores, k=topk_group, dim=-1, sorted=False)[1]  # [n, top_k_group]
    group_mask = torch.zeros_like(group_scores)  # [n, n_group]
    group_mask.scatter_(1, group_idx, 1)  # [n, n_group]
    score_mask = (
        group_mask.unsqueeze(-1)
        .expand(num_token, num_expert_group, scores.shape[-1] // num_expert_group)
        .reshape(num_token, -1)
    )  # [n, e]
    tmp_scores = scores_for_choice.masked_fill(~score_mask.bool(), 0.0)  # [n, e]
    _, topk_ids = torch.topk(tmp_scores, k=topk, dim=-1, sorted=False)
    topk_weights = scores.gather(1, topk_ids)

    if renormalize:
        topk_weights = topk_weights / topk_weights.sum(dim=-1, keepdim=True)

    return topk_weights.to(torch.float32), topk_ids.to(torch.int32)


# This is used by the Deepseek-V2 model
def cuda_grouped_topk(
    hidden_states: torch.Tensor,
    gating_output: torch.Tensor,
    correction_bias: torch.Tensor,
    topk: int,
    renormalize: bool,
    num_expert_group: int = 0,
    topk_group: int = 0,
    scoring_func: str = "softmax",
):
    assert hidden_states.shape[0] == gating_output.shape[0], "Number of tokens mismatch"
    assert light_ops is not None, "lightllm_kernel is not installed."

    num_tokens = gating_output.shape[0]
    topk_weights = torch.empty(num_tokens, topk, device=hidden_states.device, dtype=torch.float32)
    topk_indices = torch.empty(num_tokens, topk, device=hidden_states.device, dtype=torch.int32)
    token_expert_indices = torch.empty(num_tokens, topk_group, device=hidden_states.device, dtype=torch.int32)
    group_scores = torch.empty(num_tokens, num_expert_group, device=hidden_states.device, dtype=torch.float32)
    if correction_bias is None:
        correction_bias = torch.zeros_like(gating_output, dtype=torch.float32)
    light_ops.grouped_topk(
        topk_weights,
        correction_bias,
        topk_indices,
        token_expert_indices,
        gating_output.float(),
        num_expert_group,
        topk_group,
        topk,
        renormalize,
        scoring_func,
        group_scores,
    )

    return topk_weights, topk_indices


def select_experts(
    hidden_states: torch.Tensor,
    router_logits: torch.Tensor,
    correction_bias: torch.Tensor,
    top_k: int,
    use_grouped_topk: bool,
    renormalize: bool,
    topk_group: Optional[int] = None,
    num_expert_group: Optional[int] = None,
    scoring_func: str = "softmax",
    custom_routing_function: Optional[Callable] = None,
):
    from lightllm.common.fused_moe.topk_select import fused_topk
    from lightllm.common.fused_moe.grouped_topk import triton_grouped_topk

    # DeekSeekv2 uses grouped_top_k
    if use_grouped_topk:
        assert topk_group is not None
        assert num_expert_group is not None
        if use_cuda_grouped_topk:
            topk_weights, topk_ids = cuda_grouped_topk(
                hidden_states=hidden_states,
                gating_output=router_logits,
                correction_bias=correction_bias,
                topk=top_k,
                renormalize=renormalize,
                num_expert_group=num_expert_group,
                topk_group=topk_group,
                scoring_func=scoring_func,
            )
        else:
            group_score_topk_num = 1
            # for deepseek v3
            if topk_group == 4 and num_expert_group == 8 and top_k == 8:
                group_score_topk_num = 2

            topk_weights, topk_ids = triton_grouped_topk(
                hidden_states=hidden_states,
                gating_output=router_logits,
                correction_bias=correction_bias,
                topk=top_k,
                renormalize=renormalize,
                num_expert_group=num_expert_group,
                topk_group=topk_group,
                scoring_func=scoring_func,
                group_score_used_topk_num=group_score_topk_num,
            )

    elif custom_routing_function is None:
        topk_weights, topk_ids = fused_topk(
            hidden_states=hidden_states, gating_output=router_logits, topk=top_k, renormalize=renormalize
        )
    else:
        topk_weights, topk_ids = custom_routing_function(
            hidden_states=hidden_states, gating_output=router_logits, topk=top_k, renormalize=renormalize
        )

    ######################################## warning ##################################################
    # here is used to match autotune feature, make topk_ids more random
    if Autotuner.is_autotune_warmup():
        rand_gen = torch.Generator(device="cuda")
        rand_gen.manual_seed(router_logits.shape[0])
        router_logits = torch.randn(size=router_logits.shape, generator=rand_gen, dtype=torch.float32, device="cuda")
        _, topk_ids = torch.topk(router_logits, k=top_k, dim=1)

    return topk_weights, topk_ids
