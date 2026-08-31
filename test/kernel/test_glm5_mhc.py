# SPDX-License-Identifier: Apache-2.0

"""Numerical and launch-overhead check for the fused GLM-5 mHC kernels."""

import argparse

import torch

from lightllm.common.basemodel.triton_kernel.norm.rmsnorm import rmsnorm_forward
from lightllm.models.glm5_next.triton_kernel.mhc import (
    hc_post,
    hc_post_reference,
    hc_pre_norm,
    hc_pre_reference,
)


def _max_error(actual: torch.Tensor, expected: torch.Tensor) -> float:
    return (actual.float() - expected.float()).abs().max().item()


def _time_ms(function, iterations: int = 100) -> float:
    for _ in range(5):
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
    parser.add_argument("--large-post", action="store_true")
    parser.add_argument("--iterations", type=int, default=100)
    args = parser.parse_args()

    torch.manual_seed(1234)
    device = torch.device("cuda")
    streams = 4
    hidden = 4096
    for tokens in (1, 8, 48):
        x = torch.randn((tokens, streams * hidden), device=device, dtype=torch.bfloat16)
        fn = 0.005 * torch.randn(
            ((2 + streams) * streams, streams * hidden),
            device=device,
            dtype=torch.float32,
        )
        scale = torch.randn((3,), device=device, dtype=torch.float32)
        base = torch.randn(((2 + streams) * streams,), device=device, dtype=torch.float32)
        layer_output = torch.randn((tokens, hidden), device=device, dtype=torch.bfloat16)
        norm_weight = torch.randn((hidden,), device=device, dtype=torch.bfloat16)
        arguments = (x, fn, scale, base, streams, 1e-6, 1e-6, 20)

        expected_pre = hc_pre_reference(*arguments)
        expected_pre = (
            rmsnorm_forward(expected_pre[0], weight=norm_weight, eps=1e-6),
            expected_pre[1],
            expected_pre[2],
        )
        actual_pre = hc_pre_norm(
            x,
            fn,
            scale,
            base,
            norm_weight,
            streams,
            1e-6,
            1e-6,
            1e-6,
            20,
        )
        torch.cuda.synchronize()
        pre_errors = tuple(_max_error(actual, expected) for actual, expected in zip(actual_pre, expected_pre))
        assert pre_errors[0] <= 0.03125, pre_errors
        # DeepGEMM intentionally uses TF32 and a split-K reduction, matching
        # the optimized serving path rather than torch.mm's accumulation order.
        assert pre_errors[1] <= 5e-4, pre_errors
        assert pre_errors[2] <= 5e-4, pre_errors

        expected_post = hc_post_reference(layer_output, x, expected_pre[1], expected_pre[2], streams)
        actual_post = hc_post(layer_output, x, actual_pre[1], actual_pre[2], streams)
        torch.cuda.synchronize()
        post_error = _max_error(actual_post, expected_post)
        assert post_error <= 0.03125, post_error

        def reference_path():
            layer_input, residual_mix, post_mix = hc_pre_reference(*arguments)
            rmsnorm_forward(layer_input, weight=norm_weight, eps=1e-6)
            return hc_post_reference(layer_output, x, residual_mix, post_mix, streams)

        def fused_path():
            _, residual_mix, post_mix = hc_pre_norm(
                x,
                fn,
                scale,
                base,
                norm_weight,
                streams,
                1e-6,
                1e-6,
                1e-6,
                20,
            )
            return hc_post(layer_output, x, residual_mix, post_mix, streams)

        def fused_pre_path():
            return hc_pre_norm(
                x,
                fn,
                scale,
                base,
                norm_weight,
                streams,
                1e-6,
                1e-6,
                1e-6,
                20,
            )

        def fused_post_path():
            return hc_post(layer_output, x, actual_pre[1], actual_pre[2], streams)

        reference_ms = _time_ms(reference_path)
        fused_ms = _time_ms(fused_path)
        fused_pre_ms = _time_ms(fused_pre_path)
        fused_post_ms = _time_ms(fused_post_path)
        print(
            f"PASS tokens={tokens} pre_errors={pre_errors} "
            f"post_error={post_error:.8f} reference_ms={reference_ms:.4f} "
            f"fused_ms={fused_ms:.4f} pre_ms={fused_pre_ms:.4f} "
            f"post_ms={fused_post_ms:.4f} "
            f"speedup={reference_ms / fused_ms:.2f}x"
        )

    if args.large_post:
        from sglang.kernels.ops.layernorm.mhc import mhc_post_tilelang

        tokens = 17152
        residual = torch.randn((tokens, streams * hidden), device=device, dtype=torch.bfloat16)
        layer_output = torch.randn((tokens, hidden), device=device, dtype=torch.bfloat16)
        residual_mix = torch.randn((tokens, streams, streams), device=device, dtype=torch.float32)
        post_mix = torch.randn((tokens, streams), device=device, dtype=torch.float32)
        actual = hc_post(layer_output, residual, residual_mix, post_mix, streams)

        def sgl_mhc_post():
            output = torch.empty_like(residual.view(tokens, streams, hidden))
            mhc_post_tilelang(
                residual_mix,
                residual.view(tokens, streams, hidden),
                post_mix,
                layer_output,
                output,
                streams,
                hidden,
            )
            return output

        sglang_output = sgl_mhc_post().view(tokens, -1)
        torch.cuda.synchronize()
        cross_error = _max_error(actual, sglang_output)
        assert cross_error <= 0.0625, cross_error
        triton_ms = _time_ms(
            lambda: hc_post(layer_output, residual, residual_mix, post_mix, streams),
            args.iterations,
        )
        tilelang_ms = _time_ms(
            sgl_mhc_post,
            args.iterations,
        )
        print(
            f"PASS large_post tokens={tokens} cross_error={cross_error:.8f} "
            f"triton_ms={triton_ms:.4f} tilelang_ms={tilelang_ms:.4f} "
            f"triton_over_tilelang={tilelang_ms / triton_ms:.2f}x"
        )

        # Compare the full DeepGEMM prenorm + mHC-pre fusion used by both
        # runtimes.  The standalone kernel test has no SGLang process group,
        # so disable only its symmetric-allocation context.
        import contextlib
        import sglang.kernels.ops.layernorm.mhc as sgl_mhc

        sgl_mhc.get_tp_group = lambda: None
        sgl_mhc.is_allocation_symmetric = lambda: False
        sgl_mhc.use_symmetric_memory = lambda *_args, **_kwargs: contextlib.nullcontext()
        fn = 0.005 * torch.randn(
            ((2 + streams) * streams, streams * hidden),
            device=device,
            dtype=torch.float32,
        )
        scale = torch.randn((3,), device=device, dtype=torch.float32)
        base = torch.randn(((2 + streams) * streams,), device=device, dtype=torch.float32)
        norm_weight = torch.randn((hidden,), device=device, dtype=torch.bfloat16)

        def lightllm_pre():
            return hc_pre_norm(
                residual,
                fn,
                scale,
                base,
                norm_weight,
                streams,
                1e-6,
                1e-6,
                1e-6,
                20,
            )

        def sglang_pre():
            return sgl_mhc.mhc_pre(
                residual.view(tokens, streams, hidden),
                fn,
                scale,
                base,
                1e-6,
                1e-6,
                1e-6,
                2.0,
                20,
                norm_weight=norm_weight,
                norm_eps=1e-6,
            )

        lightllm_result = lightllm_pre()
        sglang_result = sglang_pre()
        torch.cuda.synchronize()
        pre_cross_errors = (
            _max_error(lightllm_result[0], sglang_result[2]),
            _max_error(lightllm_result[1], sglang_result[1]),
            _max_error(lightllm_result[2], sglang_result[0].squeeze(-1)),
        )
        assert pre_cross_errors[0] <= 0.0625, pre_cross_errors
        assert pre_cross_errors[1] <= 5e-4, pre_cross_errors
        assert pre_cross_errors[2] <= 5e-4, pre_cross_errors
        lightllm_pre_ms = _time_ms(lightllm_pre, args.iterations)
        sglang_pre_ms = _time_ms(sglang_pre, args.iterations)
        print(
            f"PASS large_pre tokens={tokens} errors={pre_cross_errors} "
            f"lightllm_ms={lightllm_pre_ms:.4f} "
            f"sglang_ms={sglang_pre_ms:.4f} "
            f"sglang_speedup={lightllm_pre_ms / sglang_pre_ms:.2f}x"
        )


if __name__ == "__main__":
    main()
