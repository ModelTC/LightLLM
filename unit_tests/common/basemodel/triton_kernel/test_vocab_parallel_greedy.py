import importlib

import pytest
import torch

from lightllm.common.basemodel.triton_kernel.post_process.greedy_sample import (
    greedy_sample_local_stats,
)


pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required for Triton kernels")


@pytest.mark.parametrize("token_num", [1, 7, 64])
def test_vocab_parallel_greedy_matches_full_logits(monkeypatch, token_num):
    module = importlib.import_module("lightllm.common.basemodel.triton_kernel.post_process.vocab_parallel_greedy")
    tp_world_size = 4
    local_vocab_size = 8192
    vocab_size = tp_world_size * local_vocab_size
    generator = torch.Generator(device="cuda").manual_seed(20260826 + token_num)
    local_logits_by_rank = [
        torch.randn(
            (local_vocab_size, token_num),
            dtype=torch.bfloat16,
            device="cuda",
            generator=generator,
        )
        for _ in range(tp_world_size)
    ]

    # Exercise deterministic tie-breaking both between local reduction blocks
    # and across tensor-parallel ranks. The smallest global token id must win.
    local_logits_by_rank[0][4097, 0] = 20.0
    local_logits_by_rank[0][3, 0] = 20.0
    local_logits_by_rank[3][2, 0] = 20.0

    local_stats_by_rank = [
        greedy_sample_local_stats(local_logits.transpose(0, 1).contiguous()) for local_logits in local_logits_by_rank
    ]

    def fake_all_gather_into_tensor(output_, input_, **_kwargs):
        for output, local_stats in zip(output_, local_stats_by_rank):
            output.copy_(local_stats)

    monkeypatch.setattr(module, "all_gather_into_tensor", fake_all_gather_into_tensor)
    actual_logits, actual_ids, actual_logsumexp = module.vocab_parallel_greedy(
        local_logits_by_rank[0],
        vocab_size=vocab_size,
        tp_world_size=tp_world_size,
        group=None,
        alloc_func=torch.empty,
    )
    actual_ids = actual_ids.view(-1)
    actual_logprobs = actual_logits.view(-1) - actual_logsumexp

    full_logits = torch.cat(local_logits_by_rank, dim=0).transpose(0, 1).float()
    expected_ids = full_logits.argmax(dim=1)
    expected_logits = full_logits.gather(1, expected_ids[:, None]).view(-1)
    expected_logsumexp = torch.logsumexp(full_logits, dim=1)
    expected_logprobs = torch.log_softmax(full_logits, dim=1).gather(1, expected_ids[:, None]).squeeze(1)

    torch.testing.assert_close(actual_ids, expected_ids, rtol=0, atol=0)
    torch.testing.assert_close(actual_logits.view(-1), expected_logits, rtol=0, atol=0)
    torch.testing.assert_close(actual_logsumexp, expected_logsumexp, rtol=2e-4, atol=2e-4)
    torch.testing.assert_close(actual_logprobs, expected_logprobs, rtol=2e-4, atol=2e-4)
