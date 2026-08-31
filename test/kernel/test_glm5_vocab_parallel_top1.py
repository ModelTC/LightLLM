# SPDX-License-Identifier: Apache-2.0

import os

import pytest
import torch
import torch.distributed as dist

from lightllm.models.glm5_next_mtp.layer_infer.post_layer_infer import (
    vocab_parallel_top1,
    vocab_parallel_top1_and_prob,
)


pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")


def test_vocab_parallel_top1_matches_full_vocab_softmax():
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    torch.cuda.set_device(local_rank)
    initialized_here = not dist.is_initialized()
    if initialized_here:
        dist.init_process_group(backend="nccl", device_id=torch.device("cuda", local_rank))

    try:
        rank = dist.get_rank()
        world_size = dist.get_world_size()
        assert world_size >= 2

        torch.manual_seed(20260828)
        token_num = 8
        vocab_size = 154880
        assert vocab_size % world_size == 0
        full_logits = torch.randn(
            (vocab_size, token_num),
            dtype=torch.bfloat16,
            device="cuda",
        )
        # Exercise rank-boundary winners and deterministic ties.  Global
        # argmax must choose the lower vocabulary id just like torch.argmax.
        shard_size = vocab_size // world_size
        full_logits[shard_size - 1, 0] = 20
        full_logits[shard_size, 0] = 20
        full_logits[-1, 1] = 21

        local_start = rank * shard_size
        local_logits = full_logits[local_start : local_start + shard_size].contiguous()

        def alloc_func(shape, dtype, device):
            return torch.empty(shape, dtype=dtype, device=device)

        token_ids, token_probs = vocab_parallel_top1_and_prob(
            local_logits=local_logits,
            local_vocab_start_id=local_start,
            tp_world_size=world_size,
            dist_group=dist.group.WORLD,
            alloc_func=alloc_func,
        )

        expected_probs, expected_ids = torch.softmax(full_logits.float().t(), dim=-1).max(dim=-1)
        torch.testing.assert_close(token_ids, expected_ids, rtol=0, atol=0)
        torch.testing.assert_close(token_probs, expected_probs, rtol=2e-5, atol=2e-7)
        torch.testing.assert_close(
            vocab_parallel_top1(
                local_logits,
                local_start,
                world_size,
                dist.group.WORLD,
                alloc_func,
            ),
            expected_ids,
            rtol=0,
            atol=0,
        )

        if os.environ.get("LIGHTLLM_TEST_SKIP_CUDA_GRAPH") == "1":
            return

        # The draft head normally runs inside a decode CUDA Graph.  Replay
        # with changed logits to ensure the tiny collective and returned
        # tensors are captured as live outputs rather than stale warmup data.
        graph_logits = local_logits.clone()
        for _ in range(2):
            vocab_parallel_top1_and_prob(
                graph_logits,
                local_start,
                world_size,
                dist.group.WORLD,
                alloc_func,
            )
        dist.barrier()
        torch.cuda.synchronize()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            graph_ids, graph_probs = vocab_parallel_top1_and_prob(
                graph_logits,
                local_start,
                world_size,
                dist.group.WORLD,
                alloc_func,
            )

        torch.manual_seed(20260829)
        replay_full_logits = torch.randn_like(full_logits)
        replay_full_logits[shard_size * 2 - 1, 2] = 22
        graph_logits.copy_(replay_full_logits[local_start : local_start + shard_size])
        graph.replay()
        torch.cuda.synchronize()
        expected_probs, expected_ids = torch.softmax(replay_full_logits.float().t(), dim=-1).max(dim=-1)
        torch.testing.assert_close(graph_ids, expected_ids, rtol=0, atol=0)
        torch.testing.assert_close(graph_probs, expected_probs, rtol=2e-5, atol=2e-7)
    finally:
        if initialized_here:
            dist.destroy_process_group()
