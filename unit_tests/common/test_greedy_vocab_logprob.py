import pytest
import torch
from types import SimpleNamespace

from lightllm.common.basemodel.triton_kernel.local_vocab_topk import local_vocab_greedy_logprob


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.parametrize("batch,vocab", [(1, 127), (7, 2051), (64, 129280), (256, 129280)])
@pytest.mark.parametrize("scale", [1.0, 30.0])
def test_greedy_ids_and_logprobs(batch, vocab, scale):
    torch.manual_seed(1013)
    logits = torch.randn(batch, vocab, dtype=torch.float32, device="cuda") * scale
    expected_ids = logits.argmax(dim=-1)
    expected_logprobs = torch.log(torch.softmax(logits, dim=-1).gather(1, expected_ids[:, None])).flatten()
    ids, logprobs = local_vocab_greedy_logprob(logits)
    torch.testing.assert_close(ids, expected_ids, rtol=0, atol=0)
    torch.testing.assert_close(logprobs, expected_logprobs, rtol=1e-5, atol=2e-6)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_ties_and_large_common_offset():
    logits = torch.full((3, 1031), 1e30, dtype=torch.float32, device="cuda")
    ids, logprobs = local_vocab_greedy_logprob(logits)
    assert torch.all(ids == 0)
    torch.testing.assert_close(logprobs, torch.full_like(logprobs, -torch.log(torch.tensor(1031.0)).item()))
    logits.zero_()
    logits[:, 3] = logits[:, -1] = 10
    ids, logprobs = local_vocab_greedy_logprob(logits)
    assert torch.all(ids == 3)
    expected = torch.log(torch.softmax(logits, dim=-1)[:, 3])
    torch.testing.assert_close(logprobs, expected, rtol=1e-5, atol=2e-6)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_graph_replay():
    logits = torch.zeros(16, 2051, dtype=torch.float32, device="cuda")
    local_vocab_greedy_logprob(logits)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        ids, logprobs = local_vocab_greedy_logprob(logits)
    for winner in [2050, 1]:
        logits.zero_()
        logits[:, winner] = 20
        graph.replay()
        expected = torch.log(torch.softmax(logits, dim=-1)[:, winner])
        assert torch.all(ids == winner)
        torch.testing.assert_close(logprobs, expected, rtol=1e-5, atol=2e-6)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_sampler_keeps_penalty_then_temperature_order(monkeypatch):
    from lightllm.server.router.model_infer.mode_backend import generic_post_process as sampler

    logits = torch.randn(4, 2051, device="cuda", dtype=torch.float32)
    temperatures = torch.tensor([0.25, 0.5, 1.0, 2.0], device="cuda")
    req_ids = torch.arange(4, device="cuda", dtype=torch.int32)
    zeros = torch.zeros_like(req_ids)
    params = SimpleNamespace(penalty_counter_mode="gpu_counter")
    monkeypatch.setattr(
        sampler, "g_infer_context", SimpleNamespace(req_manager=SimpleNamespace(req_sampling_params_manager=params))
    )
    monkeypatch.setattr(
        sampler,
        "_get_post_sample_tensors",
        lambda reqs: (
            req_ids,
            temperatures,
            temperatures,
            req_ids,
            zeros,
            zeros,
            None,
            None,
            True,
            False,
            False,
            True,
            False,
        ),
    )

    def penalty(**kwargs):
        kwargs["Logits"][:, 0].sub_(2.5)

    monkeypatch.setattr(sampler, "apply_penalty_gpu_cache", penalty)
    expected = logits.clone()
    expected[:, 0].sub_(2.5)
    expected.div_(temperatures[:, None])
    expected_ids = expected.argmax(dim=-1)
    expected_logprobs = torch.log(torch.softmax(expected, dim=-1).gather(1, expected_ids[:, None])).flatten()
    ids, logprobs = sampler.sample(logits, [None] * 4, [0])
    torch.testing.assert_close(ids, expected_ids, rtol=0, atol=0)
    torch.testing.assert_close(logprobs, expected_logprobs, rtol=1e-5, atol=2e-6)
