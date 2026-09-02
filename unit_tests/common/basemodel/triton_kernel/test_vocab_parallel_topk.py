import importlib

import pytest
import torch


pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")


@pytest.mark.parametrize("token_num", [1, 7, 64])
def test_vocab_parallel_topk_collects_candidates_and_preserves_global_argmax(monkeypatch, token_num):
    module = importlib.import_module("lightllm.common.basemodel.triton_kernel.post_process.vocab_parallel_topk")
    tp_world_size = 4
    local_vocab_size = 1024
    vocab_size = tp_world_size * local_vocab_size
    local_topk = 16
    generator = torch.Generator(device="cuda").manual_seed(20260902 + token_num)
    local_logits_by_rank = [
        torch.randn(
            (local_vocab_size, token_num),
            dtype=torch.bfloat16,
            device="cuda",
            generator=generator,
        )
        for _ in range(tp_world_size)
    ]
    local_logits_by_rank[3][2, 0] = 20.0

    def pack_rank(rank):
        values, indexes = torch.topk(local_logits_by_rank[rank], k=local_topk, dim=0, sorted=False)
        payload = torch.empty((local_topk * 2, token_num), dtype=torch.float32, device="cuda")
        payload[:local_topk].copy_(values.float())
        payload[local_topk:].view(torch.int32).copy_(indexes.to(torch.int32) + rank * local_vocab_size)
        return payload

    payloads = [pack_rank(rank) for rank in range(tp_world_size)]

    def fake_all_gather_into_tensor(output_, input_, **_kwargs):
        assert output_.shape == (tp_world_size, local_topk * 2, token_num)
        assert input_.shape == (local_topk * 2, token_num)
        for output, payload in zip(output_, payloads):
            output.copy_(payload)

    monkeypatch.setattr(module, "all_gather_into_tensor", fake_all_gather_into_tensor)
    actual_logits, actual_ids = module.vocab_parallel_topk(
        local_logits_by_rank[0],
        vocab_size=vocab_size,
        vocab_start_id=0,
        topk=local_topk,
        tp_world_size=tp_world_size,
        group=None,
        alloc_func=torch.empty,
    )

    assert actual_logits.shape == (token_num, tp_world_size * local_topk)
    assert actual_ids.shape == actual_logits.shape
    assert actual_logits.dtype == torch.float32
    assert actual_ids.dtype == torch.int64

    full_logits = torch.cat(local_logits_by_rank, dim=0).transpose(0, 1).float()
    candidate_indexes = actual_logits.argmax(dim=1, keepdim=True)
    actual_argmax_ids = actual_ids.gather(1, candidate_indexes).view(-1)
    torch.testing.assert_close(actual_argmax_ids, full_logits.argmax(dim=1), rtol=0, atol=0)


def test_vocab_parallel_topk_single_rank_caps_width_to_vocabulary():
    module = importlib.import_module("lightllm.common.basemodel.triton_kernel.post_process.vocab_parallel_topk")
    local_logits = torch.tensor([[1.0], [3.0], [2.0]], device="cuda")

    logits, token_ids = module.vocab_parallel_topk(
        local_logits,
        vocab_size=3,
        vocab_start_id=0,
        topk=128,
        tp_world_size=1,
        group=None,
        alloc_func=torch.empty,
    )

    assert logits.shape == (1, 3)
    torch.testing.assert_close(token_ids.sort(dim=1).values, torch.tensor([[0, 1, 2]], device="cuda"))
