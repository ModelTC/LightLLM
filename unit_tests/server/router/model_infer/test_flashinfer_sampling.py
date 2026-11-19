import sys
from types import ModuleType, SimpleNamespace
from unittest.mock import Mock

import pytest
import torch

from lightllm.server.router.model_infer.mode_backend import generic_post_process as sampling


@pytest.fixture
def flashinfer_mock(monkeypatch):
    module = ModuleType("flashinfer")
    module.sampling = ModuleType("flashinfer.sampling")
    module.sampling.sampling_from_probs = Mock()
    module.sampling.top_k_top_p_sampling_from_probs = Mock()
    monkeypatch.setitem(sys.modules, "flashinfer", module)
    monkeypatch.setitem(sys.modules, "flashinfer.sampling", module.sampling)
    monkeypatch.setattr(sampling, "get_env_start_args", lambda: SimpleNamespace(sampling_backend="flashinfer"))
    return module.sampling


@pytest.fixture
def cpu_sampling(monkeypatch):
    manager = SimpleNamespace(penalty_counter_mode="gpu_counter", req_to_bitmask_enabled=None)
    monkeypatch.setattr(sampling.g_infer_context, "req_manager", SimpleNamespace(req_sampling_params_manager=manager))
    monkeypatch.setattr(sampling, "apply_penalty_gpu_cache", lambda **kwargs: None)
    monkeypatch.setattr(
        sampling.g_pin_mem_manager,
        "gen_from_list",
        lambda key, data, dtype: SimpleNamespace(cuda=lambda **kwargs: torch.tensor(data, dtype=dtype)),
    )


def make_request(req_idx, top_k, top_p, generator=None):
    params = SimpleNamespace(
        temperature=1.0,
        top_k=top_k,
        top_p=top_p,
        min_new_tokens=0,
        output_top_logprobs=-1,
        exponential_decay_length_penalty=SimpleNamespace(to_tuple=lambda: (1, 1.0)),
    )
    return SimpleNamespace(
        req_idx=req_idx,
        vocab_size=3,
        generator=generator,
        shm_req=SimpleNamespace(input_len=10),
        get_cur_total_len=lambda: 10,
        sampling_param=SimpleNamespace(shm_param=params, invalid_token_ids=[]),
    )


@pytest.mark.parametrize("filtered", [False, True])
def test_flashinfer_sample_preserves_token_dtype_and_invalid_rows(cpu_sampling, flashinfer_mock, filtered):
    top_k, top_p = (2, 0.9) if filtered else (3, 1.0)
    reqs = [make_request(i, top_k, top_p) for i in range(4)]
    probs = torch.tensor([[0.1, 0.8, 0.1], [0.0, 0.0, 0.0], [0.2, 0.3, 0.5], [1.0, 0.0, 0.0]])
    logits = probs.log()
    samples = torch.tensor([1, 0, 999, 0], dtype=torch.int32)
    valid = torch.tensor([True, False, False, True])
    filtered_kernel = flashinfer_mock.top_k_top_p_sampling_from_probs
    unfiltered_kernel = flashinfer_mock.sampling_from_probs
    selected, unused = (filtered_kernel, unfiltered_kernel) if filtered else (unfiltered_kernel, filtered_kernel)
    selected.return_value = (samples, valid)

    token_ids, logprobs, _ = sampling.sample(logits, reqs, recover_bad_samples=True)

    selected.assert_called_once()
    unused.assert_not_called()
    assert selected.call_args.kwargs["return_valid"] is True
    if filtered:
        assert selected.call_args.kwargs["filter_apply_order"] == "joint"
    assert token_ids.dtype == torch.int64
    assert token_ids.tolist() == [1, 0, 0, 0]
    torch.testing.assert_close(logprobs, torch.tensor([0.8, 0.0, 0.0, 1.0]).log())


@pytest.mark.parametrize("filtered", [False, True])
def test_flashinfer_keeps_seeded_batch_on_torch(cpu_sampling, flashinfer_mock, monkeypatch, filtered):
    top_k, top_p = (2, 0.9) if filtered else (3, 1.0)
    generator = torch.Generator().manual_seed(21)
    reqs = [make_request(0, top_k, top_p, generator), make_request(1, top_k, top_p)]
    logits = torch.tensor([[0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]).log()
    original_sampler = Mock(wraps=sampling._random_sample)
    monkeypatch.setattr(sampling, "_random_sample", original_sampler)

    token_ids, logprobs, _ = sampling.sample(logits, reqs, recover_bad_samples=True)

    flashinfer_mock.sampling_from_probs.assert_not_called()
    flashinfer_mock.top_k_top_p_sampling_from_probs.assert_not_called()
    original_sampler.assert_called_once()
    assert token_ids.tolist() == [1, 2]
    torch.testing.assert_close(logprobs, torch.zeros(2))


@pytest.mark.skipif(not torch.cuda.is_available(), reason="FlashInfer sampling requires CUDA")
@pytest.mark.parametrize("filtered", [False, True])
def test_flashinfer_cuda_handles_degenerate_probabilities(filtered):
    probs = torch.tensor(
        [[0.0, 1.0, 0.0], [0.0, 0.0, 0.0], [torch.nan, torch.nan, torch.nan], [0.0, 0.0, 1.0]],
        device="cuda",
    )
    top_ps = torch.full((4,), 0.9, device=probs.device) if filtered else None
    top_ks = torch.full((4,), 2, dtype=torch.int32, device=probs.device) if filtered else None

    token_ids, logprobs = sampling._flashinfer_sample(probs, top_ps, top_ks)

    assert token_ids.dtype == torch.int64
    assert token_ids.tolist() == [1, 0, 0, 2]
    torch.testing.assert_close(logprobs.cpu(), torch.tensor([0.0, -torch.inf, -torch.inf, 0.0]))
