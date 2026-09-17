from types import SimpleNamespace

import pytest
import torch

from lightllm.server.router.model_infer.mode_backend import generic_post_process as sampling
from unit_tests.server.router.model_infer.test_generic_greedy import make_context


pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="需要 CUDA")


def reference_filter(
    probs: torch.Tensor, top_ps: torch.Tensor, top_ks: torch.Tensor, **kwargs
) -> tuple[torch.Tensor, torch.Tensor]:
    values, ids = probs.sort(dim=-1, descending=True)
    cumulative = values.cumsum(dim=-1)
    values[(cumulative - values) > top_ps[:, None]] = 0.0
    values[torch.arange(probs.shape[1], device=probs.device)[None, :] >= top_ks[:, None]] = 0.0
    return values, ids


def requests(top_ks: list[int]) -> list[SimpleNamespace]:
    return [
        SimpleNamespace(generator=None, sampling_param=SimpleNamespace(shm_param=SimpleNamespace(top_k=k)))
        for k in top_ks
    ]


@pytest.mark.parametrize("vocab", [31, 1024, 4097, 151936])
@pytest.mark.parametrize("limit", [2, 8, 128])
def test_filter_matches_full_sort(vocab: int, limit: int) -> None:
    torch.manual_seed(29)
    # 使用互异分数比较全局 ID，避免把 torch.topk 未承诺的同分次序当成契约。
    row = torch.linspace(-8.0, 8.0, vocab, device="cuda")[torch.randperm(vocab, device="cuda")]
    probs = row.expand(4, -1).softmax(-1)
    limit = min(vocab, limit)
    top_ks = torch.tensor([1, min(2, limit), min(7, limit), limit], device="cuda")
    top_ps = torch.tensor([0.1, 0.5, 0.9, 1.0], device="cuda")
    actual, ids = sampling._top_p_top_k(probs, top_ps, top_ks, max_top_k=limit)
    expected, expected_ids = reference_filter(probs, top_ps, top_ks)
    restored = torch.zeros_like(probs).scatter_(1, ids, actual)
    reference = torch.zeros_like(probs).scatter_(1, expected_ids, expected)
    torch.testing.assert_close(restored, reference, atol=0, rtol=0)
    assert actual.shape[1] == (limit if limit * 4 <= vocab else vocab)


@pytest.mark.parametrize("top_p", [0.25, 0.5, 0.75, 1.0])
def test_top_p_uses_original_probability_mass(top_p: float) -> None:
    probs = torch.zeros((1, 64), device="cuda")
    probs[0, :4] = torch.tensor([0.5, 0.25, 0.125, 0.125], device="cuda")
    top_ps = torch.tensor([top_p], device="cuda")
    top_ks = torch.tensor([2], device="cuda")
    actual, ids = sampling._top_p_top_k(probs, top_ps, top_ks, max_top_k=2)
    expected, _ = reference_filter(probs, top_ps, top_ks)
    torch.testing.assert_close(actual, expected[:, :2], atol=0, rtol=0)
    torch.testing.assert_close(ids, torch.tensor([[0, 1]], device="cuda"), atol=0, rtol=0)


def test_ties_preserve_selected_scores() -> None:
    probs = torch.full((4, 1024), 1.0 / 1024, device="cuda")
    top_ps = torch.ones(4, device="cuda")
    top_ks = torch.tensor([1, 2, 4, 8], device="cuda")
    actual, ids = sampling._top_p_top_k(probs, top_ps, top_ks, max_top_k=8)
    expected, _ = reference_filter(probs, top_ps, top_ks)
    torch.testing.assert_close(actual, expected[:, :8], atol=0, rtol=0)
    assert all(torch.unique(row).numel() == 8 for row in ids)


@pytest.mark.parametrize("case", ["seeded", "pure_top_p", "large_k", "fp16", "mixed_greedy", "single"])
def test_fallback_preserves_samples_and_rng(monkeypatch: pytest.MonkeyPatch, case: str) -> None:
    monkeypatch.setattr(sampling, "get_env_start_args", lambda: SimpleNamespace(sampling_backend="triton"))
    vocab = 1024
    top_ks = [8, 16, 32] if case == "seeded" else [vocab if case == "pure_top_p" else 512] * 3
    if case == "mixed_greedy":
        top_ks = [1, 8, 32]
    elif case == "single":
        top_ks = [8]
    reqs = requests(top_ks)
    if case == "seeded":
        reqs[1].generator = torch.Generator(device="cuda").manual_seed(47)
    probs = torch.randn((len(reqs), vocab), device="cuda").softmax(-1)
    if case == "fp16":
        probs = probs.half()
        top_ks = [8, 16, 32]
        reqs = requests(top_ks)
    ps = torch.full((len(reqs),), 0.9, device="cuda")
    ks = torch.tensor(top_ks, device="cuda")
    global_state = torch.cuda.get_rng_state()
    req_state = reqs[1].generator.get_state() if case == "seeded" else None
    actual = sampling._top_p_top_k_sample(reqs, probs, ps, ks, case == "seeded")
    actual_global = torch.cuda.get_rng_state()
    actual_req = reqs[1].generator.get_state() if case == "seeded" else None
    torch.cuda.set_rng_state(global_state)
    if case == "seeded":
        reqs[1].generator.set_state(req_state)
    monkeypatch.setattr(sampling, "_top_p_top_k", reference_filter)
    expected = sampling._top_p_top_k_sample(reqs, probs, ps, ks, case == "seeded")
    torch.testing.assert_close(actual[0], expected[0], atol=0, rtol=0)
    torch.testing.assert_close(actual[1], expected[1], atol=0, rtol=0)
    torch.testing.assert_close(actual_global, torch.cuda.get_rng_state(), atol=0, rtol=0)
    if case == "seeded":
        torch.testing.assert_close(actual_req, reqs[1].generator.get_state(), atol=0, rtol=0)


def test_compact_sampling_distribution_and_logprob(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(sampling, "get_env_start_args", lambda: SimpleNamespace(sampling_backend="triton"))
    torch.manual_seed(47)
    batch = 20000
    probs = torch.zeros((batch, 64), device="cuda")
    probs[:, 0] = 0.5
    probs[:, 1] = 0.25
    probs[:, 2] = 0.25
    reqs = requests([2] * batch)
    ids, logprobs = sampling._top_p_top_k_sample(
        reqs, probs, torch.full((batch,), 0.5, device="cuda"), torch.full((batch,), 2, device="cuda"), False
    )
    # 第二项存在同分，top-k 可以选取 ID1 或 ID2；最高项仍应以 2/3 的条件概率出现。
    assert abs((ids == 0).float().mean().item() - 2.0 / 3.0) < 0.015
    assert (ids <= 2).all()
    expected = probs.gather(1, ids[:, None]).flatten().log()
    torch.testing.assert_close(logprobs, expected, atol=0, rtol=0)


@pytest.mark.parametrize("counter_mode", ["cpu_counter", "gpu_counter"])
@pytest.mark.parametrize("case", ["normal", "penalty", "eos", "invalid", "constraint", "length", "candidates"])
def test_full_sampler_preserves_preprocessing(monkeypatch: pytest.MonkeyPatch, counter_mode: str, case: str) -> None:
    torch.manual_seed(29)
    manager, reqs = make_context(3, 4097, counter_mode)
    monkeypatch.setattr(
        sampling, "g_infer_context", SimpleNamespace(req_manager=SimpleNamespace(req_sampling_params_manager=manager))
    )
    monkeypatch.setattr(sampling, "get_env_start_args", lambda: SimpleNamespace(sampling_backend="triton"))
    for req, k in zip(reqs, [2, 8, 16]):
        req.sampling_param.shm_param.top_k = k
        req.sampling_param.shm_param.top_p = 0.9
    logits = torch.linspace(-5.0, 5.0, 4097, device="cuda")[torch.randperm(4097, device="cuda")].expand(3, -1).clone()
    logits[:, 2], logits[:, 5] = 10.0, 9.0
    if case == "penalty":
        manager.req_to_repetition_penalty.fill_(2.0)
        manager.req_to_frequency_penalty.fill_(0.4)
        manager.req_to_presence_penalty.fill_(0.6)
    elif case == "eos":
        for req in reqs:
            req.sampling_param.shm_param.min_new_tokens = 10
    elif case == "invalid":
        for req in reqs:
            req.sampling_param.invalid_token_ids = [2, 5]
    elif case == "constraint":
        logits[:, :1024] = -1e6
    elif case == "length":
        manager.req_to_exponential_decay_length_penalty.fill_(1.5)
    elif case == "candidates":
        values, ids = logits.topk(128, dim=-1)
        logits.fill_(-1e7).scatter_(1, ids, values)
    processed = logits.clone()
    actual_ids, actual_logprobs = sampling.sample(processed, reqs, eos_id=[2])
    monkeypatch.setattr(sampling, "_top_p_top_k", reference_filter)
    expected_processed = logits.clone()
    sampling.sample(expected_processed, reqs, eos_id=[2])
    torch.testing.assert_close(processed, expected_processed, atol=0, rtol=0)
    probs = processed.softmax(-1)
    values, ids = reference_filter(probs, torch.full((3,), 0.9, device="cuda"), torch.tensor([2, 8, 16], device="cuda"))
    support = torch.zeros_like(probs).scatter_(1, ids, values)
    assert (support.gather(1, actual_ids[:, None]) > 0).all()
    expected_logprobs = probs.gather(1, actual_ids[:, None]).flatten().log()
    torch.testing.assert_close(actual_logprobs, expected_logprobs, atol=1e-6, rtol=1e-6)
