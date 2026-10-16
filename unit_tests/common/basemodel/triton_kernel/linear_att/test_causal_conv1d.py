import builtins
from collections import defaultdict

import pytest
import torch
import torch.nn.functional as F

from lightllm.common.basemodel.triton_kernel.linear_att.causal_conv1d import (
    causal_conv1d_fn as dispatch_prefill,
    causal_conv1d_update as dispatch_update,
)
from lightllm.common.basemodel.triton_kernel.linear_att.causal_conv1d_triton import (
    _prefill_configs,
    _update_configs,
    causal_conv1d_fn,
    causal_conv1d_update,
)
from lightllm.common.triton_utils import autotuner


pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")


def _prefill_reference(x, weight, bias, states, lengths, indices, initial, activation):
    """Independent grouped-convolution reference, including cached prefixes."""
    width = weight.shape[-1]
    output = x.clone()
    start = 0
    for length, slot, use_initial in zip(lengths, indices, initial):
        if length and slot != -1:
            prefix = states[slot] if use_initial else torch.zeros_like(states[slot])
            extended = torch.cat((prefix, x[:, start : start + length]), dim=-1)
            with torch.backends.cudnn.flags(allow_tf32=False):
                convolved = F.conv1d(
                    extended.float().unsqueeze(0),
                    weight.float().unsqueeze(1),
                    bias.float() if bias is not None else None,
                    groups=x.shape[0],
                ).squeeze(0)
            output[:, start : start + length] = F.silu(convolved) if activation else convolved
            states[slot] = extended[:, -(width - 1) :]
        start += length
    return output


def _assert_output(actual, expected):
    if actual.dtype == torch.float32:
        torch.testing.assert_close(actual, expected, rtol=2e-5, atol=2e-6)
    else:
        torch.testing.assert_close(actual, expected, rtol=1e-2, atol=2e-3)


def _check_prefill_ragged_prefix(dtype, width, layout, run_config=None):
    torch.manual_seed(42)
    dim = 131  # Exercise masked channels and unaligned strides.
    lengths = [1, 0, 3, 19, 67]
    tokens = sum(lengths)
    if layout == "projection":
        x = torch.randn(tokens, dim + 32, device="cuda", dtype=dtype)[:, :dim].T
    else:
        x = torch.randn(dim, tokens * 2, device="cuda", dtype=dtype)[:, ::2]
    weight = torch.randn(dim, width * 2, device="cuda", dtype=dtype)[:, ::2] * 0.2
    bias = torch.randn(dim * 2, device="cuda", dtype=dtype)[::2] if layout == "projection" else None
    activation = "silu" if bias is not None else None
    state_storage = torch.randn(6, dim, width + 2, device="cuda", dtype=dtype)
    original_storage = state_storage.clone()
    states = state_storage[..., : width - 1]
    ref_states = states.clone()
    indices = [3, 1, -1, 4, 0]
    initial = [True, False, False, True, False]
    starts = torch.tensor([0, 1, 1, 4, 23, 90], device="cuda", dtype=torch.int32)
    expected = _prefill_reference(x, weight, bias, ref_states, lengths, indices, initial, activation)
    actual = causal_conv1d_fn(
        x,
        weight,
        bias,
        query_start_loc=starts,
        cache_indices=torch.tensor(indices, device="cuda"),
        has_initial_state=torch.tensor(initial, device="cuda"),
        conv_states=states,
        activation=activation,
        max_seqlen=max(lengths),
        run_config=run_config,
    )
    _assert_output(actual, expected)
    torch.testing.assert_close(states, ref_states, rtol=0, atol=0)
    torch.testing.assert_close(state_storage[..., width - 1 :], original_storage[..., width - 1 :], rtol=0, atol=0)


@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
@pytest.mark.parametrize("width", [2, 3, 4])
@pytest.mark.parametrize("layout", ["projection", "sequence"])
def test_prefill_ragged_prefix_and_strided_state(dtype, width, layout):
    _check_prefill_ragged_prefix(dtype, width, layout)


@pytest.mark.parametrize("run_config", _prefill_configs())
def test_prefill_autotune_configurations(run_config):
    _check_prefill_ragged_prefix(torch.bfloat16, 4, "projection", run_config)


def test_prefill_new_requests_ignore_stale_state():
    x = torch.randn(2, 65, 33, device="cuda")
    weight = torch.randn(65, 4, device="cuda")
    states = torch.randn(2, 65, 3, device="cuda")
    with torch.backends.cudnn.flags(allow_tf32=False):
        expected = F.silu(F.conv1d(F.pad(x, (3, 0)), weight[:, None], groups=65))
    actual = causal_conv1d_fn(
        x.transpose(0, 1).reshape(65, -1),
        weight,
        query_start_loc=torch.tensor([0, 33, 66], device="cuda", dtype=torch.int32),
        cache_indices=torch.arange(2, device="cuda", dtype=torch.int32),
        has_initial_state=torch.zeros(2, device="cuda", dtype=torch.bool),
        conv_states=states,
        activation="swish",
    )
    _assert_output(actual, expected.transpose(0, 1).reshape(65, -1))
    torch.testing.assert_close(states, x[..., -3:], rtol=0, atol=0)


def _update_reference(x, states, weight, bias, indices, cache_lengths, activation):
    output = x.clone()
    input_3d = x.unsqueeze(-1) if x.ndim == 2 else x
    output_3d = output.unsqueeze(-1) if output.ndim == 2 else output
    width, state_len = weight.shape[-1], states.shape[-1]
    for batch, slot in enumerate(indices):
        if slot == -1:
            continue
        for token in range(input_3d.shape[-1]):
            if cache_lengths is None:
                history = states[slot, :, -(width - 1) :].clone()
                states[slot] = torch.cat((states[slot, :, 1:], input_3d[batch, :, token : token + 1]), dim=-1)
            else:
                cursor = cache_lengths[batch] + token
                history = torch.stack(
                    [states[slot, :, (cursor - width + 1 + j) % state_len] for j in range(width - 1)], dim=-1
                )
                states[slot, :, cursor % state_len] = input_3d[batch, :, token]
            values = torch.cat((history, input_3d[batch, :, token : token + 1]), dim=-1).double()
            # Round after each FP32 FMA, independently of Triton's implementation.
            accumulator = bias.float().clone() if bias is not None else torch.zeros(input_3d.shape[1], device=x.device)
            for tap in range(width):
                accumulator = (accumulator.double() + values[:, tap] * weight[:, tap].double()).float()
            output_3d[batch, :, token] = F.silu(accumulator) if activation else accumulator
    return output


def _check_update_buffers(dtype, width, circular, tokens, run_config=None):
    torch.manual_seed(17)
    dim = 131
    x = torch.randn(3, dim + 16, tokens * 2, device="cuda", dtype=dtype)[:, :dim, ::2]
    if tokens == 1:
        x = x.squeeze(-1)
    weight = torch.randn(dim, width, device="cuda", dtype=dtype) * 0.2
    bias = torch.randn(dim * 2, device="cuda", dtype=dtype)[::2]
    state_len = width + 1
    storage = torch.randn(5, dim, state_len + 3, device="cuda", dtype=dtype)
    original = storage.clone()
    states = storage[..., :state_len]
    ref_states = states.clone()
    indices = [2, -1, 0]
    cache_lengths = [0, 3, 13] if circular else None
    activation = "silu" if tokens == 1 else None
    expected = _update_reference(x, ref_states, weight, bias, indices, cache_lengths, activation)
    actual = causal_conv1d_update(
        x,
        states,
        weight,
        bias,
        activation,
        torch.tensor(cache_lengths, device="cuda", dtype=torch.int32) if circular else None,
        torch.tensor(indices, device="cuda", dtype=torch.int32),
        run_config=run_config,
    )
    _assert_output(actual, expected)
    torch.testing.assert_close(states, ref_states, rtol=0, atol=0)
    torch.testing.assert_close(storage[..., state_len:], original[..., state_len:], rtol=0, atol=0)


@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
@pytest.mark.parametrize("width", [2, 3, 4])
@pytest.mark.parametrize("circular,tokens", [(False, 1), (False, 7), (True, 1), (True, 7)])
def test_update_linear_and_ring_buffers(dtype, width, circular, tokens):
    _check_update_buffers(dtype, width, circular, tokens)


@pytest.mark.parametrize("run_config", _update_configs())
@pytest.mark.parametrize("circular", [False, True])
def test_update_autotune_configurations(run_config, circular):
    _check_update_buffers(torch.bfloat16, 4, circular, 7, run_config)


@pytest.mark.parametrize("mode", ["prefill", "linear", "circular"])
def test_autotune_and_cached_config_update_state_once(monkeypatch, tmp_path, mode):
    torch.manual_seed(13)
    tuner = causal_conv1d_fn if mode == "prefill" else causal_conv1d_update
    monkeypatch.setattr(autotuner, "get_triton_autotune_level", lambda: autotuner.AutotuneLevel.ADAPTIVE_AUTOTUNE)
    monkeypatch.setattr(tuner, "_cache_dir", str(tmp_path), raising=False)
    monkeypatch.setattr(tuner, "cached_configs", {})
    monkeypatch.setattr(tuner, "fast_match_configs", defaultdict(dict))
    monkeypatch.setattr(tuner, "warmuped_configs_set", set())
    configs = tuner.configs_gen_func()
    monkeypatch.setattr(tuner, "configs_gen_func", lambda: [configs[0], configs[-1]])

    dim, state_len = 65, 3 if mode == "prefill" else 5
    storage = torch.randn(4, dim, state_len + 2, device="cuda", dtype=torch.bfloat16)
    original = storage.clone()
    states = storage[..., :state_len]
    reference_states = states.clone()
    weight = torch.randn(dim, 4, device="cuda", dtype=states.dtype) * 0.2
    indices = torch.tensor([3, 1], device="cuda", dtype=torch.int32)
    if mode == "prefill":
        x = torch.randn(3, dim + 16, device="cuda", dtype=states.dtype)[:, :dim].T
        starts = torch.tensor([0, 1, 3], device="cuda", dtype=torch.int32)
        initial = torch.tensor([True, True], device="cuda")

        def run():
            return tuner(
                x,
                weight,
                query_start_loc=starts,
                cache_indices=indices,
                has_initial_state=initial,
                conv_states=states,
                max_seqlen=2,
            )

        def reference():
            return _prefill_reference(x, weight, None, reference_states, [1, 2], [3, 1], [True, True], "silu")

    else:
        x = torch.randn(2, dim, device="cuda", dtype=states.dtype)
        lengths = [0, 13] if mode == "circular" else None
        cache_lengths = torch.tensor(lengths, device="cuda", dtype=torch.int32) if lengths is not None else None

        def run():
            return tuner(x, states, weight, activation="silu", cache_seqlens=cache_lengths, conv_state_indices=indices)

        def reference():
            return _update_reference(x, reference_states, weight, None, [3, 1], lengths, "silu")

    clone_mutated_args = tuner._mutate_args_clone
    scratch_states = []

    def check_clones(args, kwargs):
        new_args, new_kwargs, original_states, cloned_states = clone_mutated_args(args, kwargs)
        assert len(cloned_states) == 1 and original_states[0] is states
        assert cloned_states[0].data_ptr() != states.data_ptr()
        scratch_states.append(cloned_states[0])
        return new_args, new_kwargs, original_states, cloned_states

    monkeypatch.setattr(tuner, "_mutate_args_clone", check_clones)
    expected = reference()
    with autotuner.Autotuner.autotune_warmup(tuner.kernel_type):
        actual = run()
    assert len(scratch_states) == 2  # One protected copy per candidate.
    assert len(list(tmp_path.glob("*.json"))) == 1
    assert all(config is not None for cached in tuner.cached_configs.values() for config in cached.values())
    _assert_output(actual, expected)
    torch.testing.assert_close(states, reference_states, rtol=0, atol=0)

    # Historical-config warmup must also use protected state, including when
    # the first cache load is captured in a CUDA graph.
    monkeypatch.setattr(autotuner, "get_triton_autotune_level", lambda: autotuner.AutotuneLevel.USE_AUTOTUNE_HIS_CONFIG)
    monkeypatch.setattr(tuner, "cached_configs", {})
    monkeypatch.setattr(tuner, "fast_match_configs", defaultdict(dict))
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        output = run()
    expected = reference()
    graph.replay()
    _assert_output(output, expected)
    torch.testing.assert_close(states, reference_states, rtol=0, atol=0)
    torch.testing.assert_close(storage[..., state_len:], original[..., state_len:], rtol=0, atol=0)
    assert len(scratch_states) > 2


def test_prefill_graph_replays_with_changed_lengths_and_prefixes():
    torch.manual_seed(1)
    x = torch.randn(25, 96, device="cuda", dtype=torch.bfloat16).T
    weight = torch.randn(96, 4, device="cuda", dtype=x.dtype) * 0.2
    original = torch.randn(2, 96, 3, device="cuda", dtype=x.dtype)
    states = original.clone()
    starts = torch.tensor([0, 1, 25], dtype=torch.int32, device="cuda")
    initial = torch.tensor([True, False], device="cuda")
    indices = torch.arange(2, device="cuda", dtype=torch.int32)

    def run():
        return causal_conv1d_fn(
            x,
            weight,
            query_start_loc=starts,
            cache_indices=indices,
            has_initial_state=initial,
            conv_states=states,
            max_seqlen=25,
        )

    run()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        output = run()
    for lengths, flags in [([1, 24], [True, False]), ([25, 0], [False, True]), ([2, 23], [True, True])]:
        starts.copy_(torch.tensor([0, lengths[0], 25], dtype=starts.dtype, device="cuda"))
        initial.copy_(torch.tensor(flags, device="cuda"))
        states.copy_(original)
        reference_states = original.clone()
        expected = _prefill_reference(x, weight, None, reference_states, lengths, [0, 1], flags, "silu")
        graph.replay()
        _assert_output(output, expected)
        torch.testing.assert_close(states, reference_states, rtol=0, atol=0)


def test_dispatch_without_sgl_kernel_or_vllm(monkeypatch):
    import_fn = builtins.__import__

    def without_optional_kernels(name, *args, **kwargs):
        if name.split(".")[0] in ("sgl_kernel", "vllm"):
            raise ModuleNotFoundError(name)
        return import_fn(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", without_optional_kernels)
    x = torch.randn(13, 65, dtype=torch.bfloat16, device="cuda").T
    weight = torch.randn(65, 4, dtype=x.dtype, device="cuda") * 0.2
    states = torch.zeros(1, 65, 3, dtype=x.dtype, device="cuda")
    ref_states = states.clone()
    starts = torch.tensor([0, 13], dtype=torch.int32, device="cuda")
    expected = _prefill_reference(x, weight, None, ref_states, [13], [0], [False], "silu")
    _assert_output(
        dispatch_prefill(
            x,
            weight,
            query_start_loc=starts,
            conv_states=states,
            cache_indices=torch.zeros(1, device="cuda", dtype=torch.int32),
            has_initial_state=torch.zeros(1, device="cuda", dtype=torch.bool),
        ),
        expected,
    )
    token = torch.randn(1, 65, dtype=x.dtype, device="cuda")
    expected = _update_reference(token, ref_states, weight, None, [0], None, "silu")
    _assert_output(dispatch_update(token, states, weight, activation="silu"), expected)
    torch.testing.assert_close(states, ref_states, rtol=0, atol=0)


def test_bfloat16_products_accumulate_in_float32():
    # BF16 operands whose product cannot be represented in BF16. With a unit
    # residual the premature product rounding also changes the BF16 output.
    x = torch.tensor([[1.0078125, 1.0]], device="cuda", dtype=torch.bfloat16)
    weight = torch.tensor([[1.0078125, -1.0]], device="cuda", dtype=torch.bfloat16)
    expected = torch.tensor([[-1.0078125, 0.01568603515625]], device="cuda", dtype=x.dtype)
    actual = causal_conv1d_fn(
        x,
        weight,
        activation=None,
        query_start_loc=torch.tensor([0, 2], device="cuda", dtype=torch.int32),
        cache_indices=torch.zeros(1, device="cuda", dtype=torch.int32),
        has_initial_state=torch.zeros(1, device="cuda", dtype=torch.bool),
        conv_states=torch.ones(1, 1, 1, device="cuda", dtype=x.dtype),
    )
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)


def test_empty_inputs_do_not_touch_state():
    state = torch.randn(2, 17, 3, device="cuda")
    original = state.clone()
    weight = torch.randn(17, 4, device="cuda")
    output = causal_conv1d_fn(
        torch.empty(17, 0, device="cuda"),
        weight,
        conv_states=state,
        query_start_loc=torch.zeros(3, device="cuda", dtype=torch.int32),
        cache_indices=torch.arange(2, device="cuda", dtype=torch.int32),
        has_initial_state=torch.ones(2, device="cuda", dtype=torch.bool),
    )
    assert output.numel() == 0
    assert causal_conv1d_update(torch.empty(2, 17, 0, device="cuda"), state, weight).numel() == 0
    torch.testing.assert_close(state, original, rtol=0, atol=0)
