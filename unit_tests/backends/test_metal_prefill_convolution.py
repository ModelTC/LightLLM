"""Packed prefill convolution must preserve raw request cache history."""

import pytest
import torch

from lightllm.backends.metal.convolution import prefill
from lightllm.backends.metal.kernels import _causal_conv1d_mps


pytestmark = pytest.mark.skipif(
    not torch.backends.mps.is_available() or not hasattr(torch.mps, "compile_shader"),
    reason="Requires MPS and torch.mps.compile_shader",
)


@pytest.mark.parametrize("state_dtype", [torch.float16, torch.float32])
@pytest.mark.parametrize(
    "channels,width,lengths,slots,flags,activation,bias_dtype",
    [
        (6144, 4, [128], [2], [False], "silu", None),
        (6144, 4, [256], [2], [True], "silu", torch.float16),
        (17, 4, [5, 0, 2, 9], [2, 2, 1, 2], [False, True, True, True], "swish", torch.float32),
        (17, 2, [1, 3, 1, 5], [2, 2, -1, 2], [True, False, False, True], None, torch.float16),
        (17, 8, [0, 0, 0], [0, 1, 0], [True, False, False], "silu", None),
    ],
)
def test_prefill_matches_cpu_and_preserves_storage(
    state_dtype, channels, width, lengths, slots, flags, activation, bias_dtype
):
    generator = torch.Generator().manual_seed(79134)
    tokens = sum(lengths)
    # QKV is a strided view of the projection; other fields must stay intact.
    host_input = torch.randn(tokens + 1, 2 * channels + 3, dtype=torch.float16, generator=generator)
    host_state = torch.randn(6, 2 * channels + 1, 2 * (width - 1) + 1, dtype=state_dtype, generator=generator)
    host_weight = torch.randn(channels, 2 * width + 1, dtype=torch.float32, generator=generator) * 0.2
    host_bias = torch.randn(2 * channels + 1, dtype=bias_dtype, generator=generator) * 0.1 if bias_dtype else None
    expected_input, expected_state = host_input.clone(), host_state.clone()
    gpu_input, gpu_state, gpu_weight = host_input.to("mps"), host_state.to("mps"), host_weight.to("mps")
    weight = host_weight[:, 1::2]
    bias = host_bias[1::2] if host_bias is not None else None
    boundaries = torch.tensor([0, *torch.tensor(lengths).cumsum(0).tolist()], dtype=torch.int64)
    indexes = torch.tensor(slots, dtype=torch.int32)
    has_initial = torch.tensor(flags, dtype=torch.bool)
    input_view = expected_input[1:, 1 : 2 * channels + 1 : 2].t()
    state_view = expected_state[1:, 1::2, 1::2]
    with torch.inference_mode():
        _causal_conv1d_mps(input_view, weight, bias, boundaries, indexes, has_initial, state_view, activation, -1)
        # Use strides and storage offsets for metadata too.
        index_storage = torch.empty(2 * len(slots) + 1, dtype=indexes.dtype, device="mps")
        index_storage[1::2].copy_(indexes)
        boundary_storage = torch.empty(2 * len(boundaries) + 1, dtype=boundaries.dtype, device="mps")
        boundary_storage[1::2].copy_(boundaries)
        flag_storage = torch.empty(2 * len(flags) + 1, dtype=torch.bool, device="mps")
        flag_storage[1::2].copy_(has_initial)
        actual = gpu_input[1:, 1 : 2 * channels + 1 : 2].t()
        assert prefill(
            actual,
            gpu_weight[:, 1::2],
            host_bias.to("mps")[1::2] if host_bias is not None else None,
            boundary_storage[1::2],
            index_storage[1::2],
            flag_storage[1::2],
            gpu_state[1:, 1::2, 1::2],
            activation,
            -1,
        )
    torch.testing.assert_close(gpu_input.cpu(), expected_input, atol=0.002, rtol=0.003)
    torch.testing.assert_close(gpu_state.cpu(), expected_state, atol=0, rtol=0)


def test_unsupported_prefill_selects_existing_path():
    x = torch.randn(17, 8, device="mps", dtype=torch.float16)
    state = torch.zeros(3, 17, 3, device="mps", dtype=x.dtype)
    weight = torch.randn(17, 4, device="mps", dtype=x.dtype)
    boundaries = torch.tensor([0, 8], device="mps", dtype=torch.int32)
    indexes = torch.tensor([1], device="mps", dtype=torch.int32)
    flags = torch.tensor([True], device="mps")
    assert not prefill(x.float(), weight, None, boundaries, indexes, flags, state, "silu", -1)
    assert not prefill(x, weight, None, boundaries, None, flags, state, "silu", -1)
    assert not prefill(x, weight, None, boundaries, indexes, flags, state, "relu", -1)
