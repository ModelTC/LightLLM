"""Collect sparse candidates directly from tensor-parallel vocabulary shards."""

import os

import torch

from lightllm.distributed.communication_op import all_gather_into_tensor
from lightllm.utils.envs_utils import enable_env_vars


VOCAB_PARALLEL_TOPK_ENV = "LIGHTLLM_VOCAB_PARALLEL_TOPK"
VOCAB_PARALLEL_TOPK_SIZE_ENV = "LIGHTLLM_VOCAB_PARALLEL_TOPK_SIZE"
DEFAULT_VOCAB_PARALLEL_TOPK = 128


def is_vocab_parallel_topk_enabled() -> bool:
    """Whether target-model greedy batches may use sparse vocabulary output."""

    return enable_env_vars(VOCAB_PARALLEL_TOPK_ENV)


def get_vocab_parallel_topk_size() -> int:
    topk = int(os.getenv(VOCAB_PARALLEL_TOPK_SIZE_ENV, str(DEFAULT_VOCAB_PARALLEL_TOPK)))
    assert topk > 0, f"{VOCAB_PARALLEL_TOPK_SIZE_ENV} must be positive, got {topk}"
    return topk


@torch.no_grad()
def vocab_parallel_topk(
    local_logits: torch.Tensor,
    *,
    vocab_size: int,
    vocab_start_id: int,
    topk: int,
    tp_world_size: int,
    group,
    alloc_func,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Gather each TP rank's local top-k logits and their global token ids.

    The returned width is ``tp_world_size * topk``. It intentionally keeps the
    union of local candidates: greedy selection remains exact, while probability
    calculations over the sparse result are an inexpensive approximation.
    """

    assert local_logits.ndim == 2 and local_logits.is_cuda and local_logits.is_contiguous()
    local_vocab_size, token_num = local_logits.shape
    # Collectives require every rank to contribute the same shape. Vocabulary
    # shards can differ by one row, so cap against the smallest possible shard.
    local_topk = min(topk, vocab_size // tp_world_size)
    assert local_topk > 0
    assert local_vocab_size >= local_topk

    local_values, local_indexes = torch.topk(local_logits, k=local_topk, dim=0, sorted=False)
    local_values = local_values.float()
    local_token_ids = local_indexes.to(torch.int32).add_(int(vocab_start_id))

    if tp_world_size == 1:
        candidate_values = local_values.permute(1, 0).contiguous()
        candidate_token_ids = local_token_ids.permute(1, 0).contiguous()
    else:
        # Values and ids are both four bytes. Bit-packing the ids into the FP32
        # payload keeps the operation to one fixed-shape collective.
        local_payload = alloc_func(
            (local_topk * 2, token_num),
            dtype=torch.float32,
            device=local_logits.device,
        )
        local_payload[:local_topk].copy_(local_values)
        local_payload[local_topk:].view(torch.int32).copy_(local_token_ids)

        gathered_payload = alloc_func(
            (tp_world_size, local_topk * 2, token_num),
            dtype=torch.float32,
            device=local_logits.device,
        )
        all_gather_into_tensor(
            output_=gathered_payload,
            input_=local_payload,
            group=group,
            async_op=False,
        )
        candidate_values = gathered_payload[:, :local_topk, :].permute(2, 0, 1).reshape(token_num, -1)
        candidate_token_ids = (
            gathered_payload[:, local_topk:, :].view(torch.int32).permute(2, 0, 1).reshape(token_num, -1)
        )

    output_logits = alloc_func(
        candidate_values.shape,
        dtype=torch.float32,
        device=local_logits.device,
    )
    output_logits.copy_(candidate_values)
    return output_logits, candidate_token_ids.to(torch.int64).contiguous()
