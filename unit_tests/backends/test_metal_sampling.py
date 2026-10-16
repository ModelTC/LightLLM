"""Fused greedy sampling must retain penalties, mutations, ties and logprobs."""

from types import SimpleNamespace

import pytest
import torch

from lightllm.backends.metal.kernels import apply_penalty_gpu_cache
from lightllm.backends.metal.sampling import greedy


pytestmark = pytest.mark.skipif(
    not torch.backends.mps.is_available() or not hasattr(torch.mps, "compile_shader"),
    reason="Requires MPS and torch.mps.compile_shader",
)


def manager(states, vocab, generator, defaults=False):
    return SimpleNamespace(
        penalty_counter_mode="gpu_counter",
        req_to_out_token_id_counter=torch.zeros(states, vocab, dtype=torch.int32)
        if defaults
        else torch.randint(0, 5, (states, vocab), dtype=torch.int32, generator=generator),
        req_to_repetition_penalty=torch.ones(states) if defaults else torch.rand(states, generator=generator) + 0.7,
        req_to_frequency_penalty=torch.zeros(states)
        if defaults
        else torch.rand(states, generator=generator) * 0.5 - 0.2,
        req_to_presence_penalty=torch.zeros(states)
        if defaults
        else torch.rand(states, generator=generator) * 0.5 - 0.2,
        req_to_exponential_decay_length_penalty=torch.ones(states)
        if defaults
        else torch.rand(states, generator=generator) * 0.1 + 1,
    )


def reference(logits, requests, temperatures, length_penalty, mask, eos, params):
    """Keep FP32 penalty mutations; use independent FP64 for probability accuracy."""
    apply_penalty_gpu_cache(logits, requests, length_penalty, mask, eos, params)
    logits.div_(temperatures[:, None])
    tokens = logits.argmax(-1)
    return tokens, logits.double().log_softmax(-1).gather(1, tokens[:, None]).view(-1)


def on_device(params, strided=False):
    result = SimpleNamespace(penalty_counter_mode=params.penalty_counter_mode)
    for name, value in vars(params).items():
        if not isinstance(value, torch.Tensor):
            continue
        if strided:
            if value.ndim == 1:
                storage = torch.empty(value.numel() * 2 + 1, dtype=value.dtype, device="mps")
                gpu = storage[1::2]
            else:
                storage = torch.empty(value.shape[0] + 1, value.shape[1] * 2 + 1, dtype=value.dtype, device="mps")
                gpu = storage[1:, 1::2]
            gpu.copy_(value)
        else:
            gpu = value.to("mps")
        setattr(result, name, gpu)
    return result


@pytest.mark.parametrize(
    "batch,vocab", [(1, 1), (4, 31), (4, 2047), (4, 2048), (4, 2049), (16, 4097), (1, 248320), (16, 248320)]
)
@pytest.mark.parametrize("strided", [False, True])
def test_greedy_matches_cpu_with_penalties_and_strides(batch, vocab, strided):
    generator = torch.Generator().manual_seed(37181)
    states = batch + 3
    params = manager(states, vocab, generator)
    requests = torch.arange(batch, dtype=torch.int64 if strided else torch.int32) + 1
    requests[-1] = -1
    temperatures = torch.rand(batch, generator=generator) + 0.5
    length_penalty = torch.arange(batch, dtype=torch.int64 if strided else torch.int32) % 9
    mask = (torch.arange(batch) % 2 == 0).to(torch.int32 if strided else torch.bool)
    eos = torch.tensor([-1, 0, -1], dtype=requests.dtype)
    host = torch.randn(batch + 1, vocab * 2 + 3, generator=generator) * 2
    expected_storage = host.clone()
    expected_logits = expected_storage[1:, 1 : 2 * vocab + 1 : 2]
    expected_tokens, expected_scores = reference(
        expected_logits, requests, temperatures, length_penalty, mask, eos, params
    )
    gpu_storage = host.to("mps")
    logits = gpu_storage[1:, 1 : 2 * vocab + 1 : 2]

    def device(value):
        if not strided:
            return value.to("mps")
        storage = torch.empty(value.numel() * 2 + 1, dtype=value.dtype, device="mps")
        view = storage[1::2]
        view.copy_(value)
        return view

    gpu_params = on_device(params, strided)
    result = greedy(
        logits, device(requests), device(temperatures), device(length_penalty), device(mask), device(eos), gpu_params
    )
    assert result is not None
    tokens, scores = result
    torch.testing.assert_close(tokens.cpu(), expected_tokens, atol=0, rtol=0)
    torch.testing.assert_close(scores.cpu().double(), expected_scores, atol=3e-6, rtol=1e-6)
    torch.testing.assert_close(gpu_storage.cpu(), expected_storage, atol=2e-5, rtol=3e-6)
    torch.testing.assert_close(
        gpu_params.req_to_out_token_id_counter.cpu(), params.req_to_out_token_id_counter, atol=0, rtol=0
    )


def test_greedy_ties_and_nonfinite_rows():
    generator = torch.Generator().manual_seed(37182)
    vocab = 4100
    logits = torch.zeros(4, vocab)
    logits[0, 1] = logits[0, 2050] = 5
    logits[1].fill_(-float("inf"))
    logits[2, 2052] = logits[2, 3000] = float("inf")
    logits[3, 4088] = logits[3, 9] = float("nan")
    params = manager(4, vocab, generator, defaults=True)
    args = [
        torch.arange(4, dtype=torch.int32),
        torch.ones(4),
        torch.zeros(4, dtype=torch.int32),
        torch.zeros(4, dtype=torch.bool),
        torch.empty(0, dtype=torch.int32),
    ]
    expected_logits = logits.clone()
    tokens, scores = reference(expected_logits, *args, params)
    gpu_logits = logits.to("mps")
    actual = greedy(gpu_logits, *(value.to("mps") for value in args), on_device(params))
    assert actual is not None
    torch.testing.assert_close(actual[0].cpu(), tokens, atol=0, rtol=0)
    torch.testing.assert_close(actual[1].cpu().double(), scores, atol=3e-6, rtol=1e-6, equal_nan=True)
    torch.testing.assert_close(gpu_logits.cpu(), expected_logits, atol=0, rtol=0, equal_nan=True)


def test_unsupported_greedy_is_read_only():
    generator = torch.Generator().manual_seed(37183)
    params = on_device(manager(4, 32, generator))
    logits = torch.randn(4, 32, device="mps")
    args = [
        torch.arange(4, dtype=torch.int32, device="mps"),
        torch.ones(4, device="mps"),
        torch.zeros(4, dtype=torch.int32, device="mps"),
        torch.zeros(4, dtype=torch.bool, device="mps"),
        torch.tensor([0, 31], dtype=torch.int32, device="mps"),
    ]
    before = logits.cpu().clone()
    assert greedy(logits.half(), *args, params) is None
    params.penalty_counter_mode = "cpu_counter"
    assert greedy(logits, *args, params) is None
    params.penalty_counter_mode = "gpu_counter"
    aliased_args = [logits.view(torch.int32)[0, :4], *args[1:]]
    assert greedy(logits, *aliased_args, params) is None
    torch.testing.assert_close(logits.cpu(), before, atol=0, rtol=0)


def test_eos_mask_overrides_nonfinite_values_before_reduction():
    generator = torch.Generator().manual_seed(37184)
    logits = torch.randn(3, 4100, generator=generator)
    logits[:, 0] = float("nan")
    logits[:, -1] = float("inf")
    params = manager(3, 4100, generator, defaults=True)
    args = [
        torch.arange(3, dtype=torch.int32),
        torch.ones(3),
        torch.zeros(3, dtype=torch.int32),
        torch.tensor([True, False, True]),
        torch.tensor([0, -1, 0], dtype=torch.int64),
    ]
    expected = logits.clone()
    tokens, scores = reference(expected, *args, params)
    actual_logits = logits.to("mps")
    actual = greedy(actual_logits, *(t.to("mps") for t in args), on_device(params))
    assert actual is not None
    torch.testing.assert_close(actual[0].cpu(), tokens, atol=0, rtol=0)
    torch.testing.assert_close(actual[1].cpu().double(), scores, atol=3e-6, rtol=1e-6, equal_nan=True)
    torch.testing.assert_close(actual_logits.cpu(), expected, atol=0, rtol=0, equal_nan=True)
