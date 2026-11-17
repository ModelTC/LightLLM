from collections import Counter
from types import SimpleNamespace

import pytest
import torch
from triton.runtime.jit import JITFunction

from lightllm.common.basemodel.triton_kernel import gather_token_id as token_ops
from lightllm.common.basemodel.triton_kernel import gen_sampling_params as sampling_ops
from lightllm.common.basemodel.triton_kernel.post_process.apply_penalty_gpu_cache import (
    apply_penalty_gpu_cache,
)


pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")


def _observe(monkeypatch, kernels):
    misses = Counter()
    hashes = {kernel: set() for kernel in kernels}
    original_compile = JITFunction._do_compile

    def compile_kernel(self, *args, **kwargs):
        if self in hashes:
            misses[self] += 1
        return original_compile(self, *args, **kwargs)

    monkeypatch.setattr(JITFunction, "_do_compile", compile_kernel)
    for kernel in kernels:
        original_run = kernel.run

        def run(*args, _kernel=kernel, _original=original_run, **kwargs):
            compiled = _original(*args, **kwargs)
            hashes[_kernel].add(compiled.hash)
            return compiled

        monkeypatch.setattr(kernel, "run", run)
    return misses, hashes


def _assert_reused(misses, hashes):
    assert all(len(values) == 1 for values in hashes.values())
    assert all(misses[kernel] <= 1 for kernel in hashes)


@pytest.mark.parametrize("has_mask", [False, True])
@pytest.mark.parametrize("width", [1, 3, 16, 17])
def test_token_helpers_reuse_runtime_counts_with_fixed_strides(monkeypatch, has_mask, width):
    # MTP width and vocabulary are fixed for each service. Preserve their stride
    # specializations while varying the request count across integer classes.
    kernels = [
        token_ops._fwd_kernel_scatter,
        token_ops._fwd_kernel_gather,
        token_ops._fwd_kernel_gather_prefill_decode_mixed,
        sampling_ops._token_id_counter_update_kernel,
    ]
    misses, hashes = _observe(monkeypatch, kernels)
    table = torch.zeros((128, width), dtype=torch.int64, device="cuda")
    counters = torch.zeros((128, width), dtype=torch.int32, device="cuda")
    for batch in [1, 3, 16, 17, 33, 65]:
        requests = torch.arange(batch, dtype=torch.int32, device="cuda")
        columns = requests % width
        values = requests.long() + 100
        enabled = requests % 3 != 1
        mask = enabled if has_mask else None
        table.zero_()
        token_ops.scatter_token(values, table, requests, columns, mask)
        actual = token_ops.gather_token(table, requests, columns)
        expected = torch.where(enabled, values, 0) if has_mask else values
        assert torch.equal(actual, expected)

        input_ids = torch.full((batch * 2,), -1, dtype=torch.int64, device="cuda")
        is_decode = requests % 2 == 0
        starts = requests * 2
        expected_input = input_ids.clone()
        expected_input[starts[is_decode]] = expected[is_decode]
        token_ops.gather_token_prefill_decode_mixed(input_ids, table, requests, columns, is_decode, starts)
        assert torch.equal(input_ids, expected_input)

        counters.zero_()
        owners = requests % 7
        sampling_ops.update_req_to_token_id_counter(owners, columns, counters, mask)
        included = enabled if has_mask else torch.ones_like(enabled)
        flat = owners[included].long() * width + columns[included]
        expected_counts = torch.bincount(flat, minlength=counters.numel()).reshape_as(counters)
        assert torch.equal(counters, expected_counts)
    _assert_reused(misses, hashes)


@pytest.mark.parametrize("vocab", [1, 3, 16, 17])
def test_sampling_parameter_and_prompt_counts_reuse_compilation(monkeypatch, vocab):
    kernels = [sampling_ops._gen_sampling_params_kernel, sampling_ops._token_id_counter_kernel]
    misses, hashes = _observe(monkeypatch, kernels)
    fields = [
        "presence_penalty",
        "frequency_penalty",
        "repetition_penalty",
        "temperature",
        "exponential_decay_length_penalty",
    ]
    manager = SimpleNamespace(
        **{
            f"req_to_{field}": torch.arange(128, device="cuda", dtype=torch.float32) + index
            for index, field in enumerate(fields)
        }
    )
    for count in [1, 3, 16, 17, 65]:
        requests = torch.arange(count, device="cuda", dtype=torch.int32).flip(0)
        actual = sampling_ops.gen_sampling_params(requests, manager)
        for value, field in zip(actual, fields):
            assert torch.equal(value, getattr(manager, f"req_to_{field}")[requests])
        tokens = torch.arange(count, device="cuda") % vocab
        counters = torch.zeros(vocab, dtype=torch.int32, device="cuda")
        sampling_ops.token_id_counter(tokens, counters)
        assert torch.equal(counters, torch.bincount(tokens, minlength=vocab))
    _assert_reused(misses, hashes)


def test_penalties_match_reference_across_shapes():
    # Preserve the specialized stride/EOS variants selected by performance
    # measurements; their outputs must still agree across shapes.
    for batch, vocab, eos_count in [(1, 1, 0), (3, 3, 1), (16, 16, 3), (17, 17, 17), (33, 3, 3)]:
        requests = torch.arange(batch, dtype=torch.int32, device="cuda")
        counters = (torch.arange(128 * vocab, device="cuda") % 3).int().reshape(128, vocab)
        manager = SimpleNamespace(
            vocab_size=vocab,
            req_to_out_token_id_counter=counters,
            req_to_presence_penalty=torch.full((128,), 0.5, device="cuda"),
            req_to_frequency_penalty=torch.full((128,), 0.25, device="cuda"),
            req_to_repetition_penalty=torch.full((128,), 2.0, device="cuda"),
            req_to_exponential_decay_length_penalty=torch.full((128,), 2.0, device="cuda"),
        )
        logits = ((torch.arange(batch * vocab, device="cuda") % 9 - 4).float() / 4).reshape(batch, vocab)
        counts = counters[requests]
        repeated = torch.where(logits > 0, logits / 2, logits * 2)
        expected = torch.where(counts > 0, repeated, logits) - counts * 0.25
        expected -= torch.where(counts > 0, 0.5, 0.0)
        length = requests % 3
        mask_eos = requests % 4 == 2
        eos_ids = torch.arange(eos_count, device="cuda") % vocab
        scales = torch.exp2(length.float()) - 1
        for eos in eos_ids.tolist():
            value = expected[:, eos]
            expected[:, eos] = torch.where(mask_eos, -10000000.0, value + value.abs() * scales)
        apply_penalty_gpu_cache(logits, requests, length, mask_eos, eos_ids, manager)
        torch.testing.assert_close(logits, expected, atol=0, rtol=0)
