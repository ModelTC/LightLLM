# SPDX-License-Identifier: Apache-2.0

"""Check GLM-5 KDA's copy-free strided causal convolution path."""

import argparse

import torch

from lightllm.common.basemodel.triton_kernel.linear_att.causal_conv1d import (
    causal_conv1d_fn,
)


def _inputs(seq_lens: list[int], dim: int, cache_lines: int):
    total_tokens = sum(seq_lens)
    token_major = torch.randn((total_tokens, dim), device="cuda", dtype=torch.bfloat16)
    weight = torch.randn((dim, 4), device="cuda", dtype=torch.bfloat16)
    conv_states = torch.randn((cache_lines, dim, 3), device="cuda", dtype=torch.bfloat16)
    cache_indices = torch.arange(len(seq_lens), device="cuda", dtype=torch.int32)
    has_initial_state = torch.tensor(
        [(index % 2) == 1 for index in range(len(seq_lens))],
        device="cuda",
        dtype=torch.bool,
    )
    query_start_loc = torch.tensor(
        [0, *torch.tensor(seq_lens).cumsum(0).tolist()],
        device="cuda",
        dtype=torch.int32,
    )
    return (
        token_major,
        weight,
        conv_states,
        cache_indices,
        has_initial_state,
        query_start_loc,
    )


def _run(arguments, seq_lens: list[int], *, copy_free: bool):
    (
        token_major,
        weight,
        conv_states,
        cache_indices,
        has_initial_state,
        query_start_loc,
    ) = arguments
    return causal_conv1d_fn(
        token_major.transpose(0, 1),
        weight,
        query_start_loc=query_start_loc,
        cache_indices=cache_indices,
        has_initial_state=has_initial_state,
        conv_states=conv_states,
        activation="silu",
        seq_lens_cpu=seq_lens if copy_free else None,
    )


def _time_ms(function, iterations: int) -> float:
    for _ in range(3):
        function()
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(iterations):
        function()
    end.record()
    end.synchronize()
    return start.elapsed_time(end) / iterations


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--benchmark", action="store_true")
    parser.add_argument("--iterations", type=int, default=20)
    args = parser.parse_args()

    torch.manual_seed(1234)
    seq_lens = [7, 1, 19, 5]
    inputs = _inputs(seq_lens, dim=64, cache_lines=8)
    reference_inputs = tuple(value.clone() if isinstance(value, torch.Tensor) else value for value in inputs)
    actual = _run(inputs, seq_lens, copy_free=True)
    expected = _run(reference_inputs, seq_lens, copy_free=False)
    torch.cuda.synchronize()
    output_error = (actual.float() - expected.float()).abs().max().item()
    state_error = (inputs[2].float() - reference_inputs[2].float()).abs().max().item()
    # The two kernels accumulate the four taps in a different order; one BF16
    # ULP at this random input scale is expected.
    assert output_error <= 0.0625, output_error
    assert state_error == 0.0, state_error
    assert actual.transpose(0, 1).is_contiguous()
    print(f"PASS correctness output_error={output_error:.8f} " f"state_error={state_error:.8f}")

    if args.benchmark:
        seq_lens = [268] * 64
        strided_inputs = _inputs(seq_lens, dim=3072, cache_lines=64)
        copied_inputs = tuple(value.clone() if isinstance(value, torch.Tensor) else value for value in strided_inputs)
        strided_ms = _time_ms(
            lambda: _run(strided_inputs, seq_lens, copy_free=True),
            args.iterations,
        )
        copied_ms = _time_ms(
            lambda: _run(copied_inputs, seq_lens, copy_free=False),
            args.iterations,
        )
        print(
            f"PASS benchmark tokens={sum(seq_lens)} dim=3072 "
            f"strided_ms={strided_ms:.4f} copied_ms={copied_ms:.4f} "
            f"speedup={copied_ms / strided_ms:.2f}x"
        )


if __name__ == "__main__":
    main()
