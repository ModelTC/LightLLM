# SPDX-License-Identifier: Apache-2.0

from typing import Callable

import torch

from lightllm.distributed.communication_op import all_gather_into_tensor
from lightllm.models.llama.layer_infer.post_layer_infer import LlamaPostLayerInfer


def vocab_parallel_top1(
    local_logits: torch.Tensor,
    local_vocab_start_id: int,
    tp_world_size: int,
    dist_group,
    alloc_func: Callable,
) -> torch.Tensor:
    """Return global greedy token ids without materializing global logits."""

    assert local_logits.ndim == 2
    local_max_values, local_max_indexes = torch.max(local_logits, dim=0)
    local_token_ids = local_max_indexes + local_vocab_start_id
    if tp_world_size == 1:
        return local_token_ids

    local_winners = torch.stack(
        [local_max_values.float(), local_token_ids.float()],
        dim=-1,
    ).contiguous()
    token_num = local_winners.shape[0]
    gathered_winners = alloc_func(
        (tp_world_size * token_num, 2),
        dtype=torch.float32,
        device=local_logits.device,
    )
    all_gather_into_tensor(
        gathered_winners,
        local_winners,
        group=dist_group,
        async_op=False,
    )
    gathered_winners = gathered_winners.view(tp_world_size, token_num, 2)
    winning_ranks = torch.argmax(gathered_winners[:, :, 0], dim=0)
    token_rows = torch.arange(token_num, dtype=torch.long, device=local_logits.device)
    return gathered_winners[winning_ranks, token_rows, 1].long()


def vocab_parallel_top1_and_prob(
    local_logits: torch.Tensor,
    local_vocab_start_id: int,
    tp_world_size: int,
    dist_group,
    alloc_func: Callable,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return global Top-1 ids and exact softmax probabilities.

    Only three FP32 values per token and rank are gathered: the local maximum,
    its global token id, and the local log-sum-exp.  This replaces the normal
    all-gather of every vocabulary logit while preserving the result of a
    global FP32 argmax/softmax reduction.
    """

    assert local_logits.ndim == 2
    local_logits_fp32 = local_logits.float()
    local_max_values, local_max_indexes = torch.max(local_logits_fp32, dim=0)
    local_logsumexp = torch.logsumexp(local_logits_fp32, dim=0)
    local_token_ids = local_max_indexes + local_vocab_start_id

    if tp_world_size == 1:
        return local_token_ids, torch.exp(local_max_values - local_logsumexp)

    local_stats = torch.stack(
        [local_max_values, local_token_ids.float(), local_logsumexp],
        dim=-1,
    ).contiguous()
    token_num = local_stats.shape[0]
    gathered_stats = alloc_func(
        (tp_world_size * token_num, 3),
        dtype=torch.float32,
        device=local_logits.device,
    )
    all_gather_into_tensor(
        gathered_stats,
        local_stats,
        group=dist_group,
        async_op=False,
    )
    gathered_stats = gathered_stats.view(tp_world_size, token_num, 3)
    winning_ranks = torch.argmax(gathered_stats[:, :, 0], dim=0)
    token_rows = torch.arange(token_num, dtype=torch.long, device=local_logits.device)
    winning_stats = gathered_stats[winning_ranks, token_rows]
    global_logsumexp = torch.logsumexp(gathered_stats[:, :, 2], dim=0)
    return winning_stats[:, 1].long(), torch.exp(winning_stats[:, 0] - global_logsumexp)


class Glm5NextMTPPostLayerInfer(LlamaPostLayerInfer):
    """GLM NextN uses the common exact vocab-parallel greedy output path."""
