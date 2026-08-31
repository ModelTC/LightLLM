"""Compare GLM-5.3 sparse prefill kernels at serving-sized shapes.

This benchmark deliberately includes FlashMLA's TP head-padding allocation and
copy, because that work is part of every LightLLM prefill attention layer.
TileLang receives the native eight TP-local query heads.
"""

import argparse
import gc
import json
import statistics
from collections.abc import Callable

import torch


def make_causal_indices(tokens: int, sequence_length: int, topk: int) -> torch.Tensor:
    """Build the packed-request causal index layout used by the serving test."""

    token_ids = torch.arange(tokens, dtype=torch.int32, device="cuda")
    positions = token_ids.remainder(sequence_length)
    sequence_starts = token_ids - positions
    columns = torch.arange(topk, dtype=torch.int32, device="cuda").view(1, -1)
    indices = sequence_starts.view(-1, 1) + columns
    indices.masked_fill_(columns > positions.view(-1, 1), -1)
    return indices.unsqueeze(1)


def measure_ms(
    name: str,
    operation: Callable[[], torch.Tensor],
    warmup: int,
    iterations: int,
) -> dict[str, float | str]:
    result = None
    for _ in range(warmup):
        result = operation()
        torch.cuda.synchronize()
        del result

    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    baseline_bytes = torch.cuda.memory_allocated()
    samples = []
    for _ in range(iterations):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        result = operation()
        end.record()
        end.synchronize()
        samples.append(start.elapsed_time(end))
        del result

    peak_delta_bytes = torch.cuda.max_memory_allocated() - baseline_bytes
    return {
        "name": name,
        "median_ms": statistics.median(samples),
        "min_ms": min(samples),
        "max_ms": max(samples),
        "peak_delta_gib": peak_delta_bytes / 2 ** 30,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tokens", type=int, required=True)
    parser.add_argument("--sequence-length", type=int, default=1024)
    parser.add_argument("--topk", type=int, default=2048)
    parser.add_argument("--local-heads", type=int, default=8)
    parser.add_argument("--required-heads", type=int, default=64)
    parser.add_argument("--head-dim", type=int, default=512)
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--iterations", type=int, default=5)
    args = parser.parse_args()

    if args.topk % 64:
        raise ValueError("TileLang requires --topk to be divisible by 64")
    if args.required_heads % args.local_heads:
        raise ValueError("--required-heads must be divisible by --local-heads")

    from sgl_kernel.flash_mla import flash_mla_sparse_fwd
    from sglang.kernels.ops.attention.dsa.tilelang_kernel import tilelang_sparse_fwd

    torch.manual_seed(0)
    q = torch.randn(
        (args.tokens, args.local_heads, args.head_dim),
        dtype=torch.bfloat16,
        device="cuda",
    )
    kv = torch.randn(
        (args.tokens, 1, args.head_dim),
        dtype=torch.bfloat16,
        device="cuda",
    )
    indices = make_causal_indices(args.tokens, args.sequence_length, args.topk)
    scale = args.head_dim ** -0.5

    padded_q = q.new_zeros((args.tokens, args.required_heads, args.head_dim))
    padded_q[:, : args.local_heads].copy_(q)

    def flashmla_kernel() -> torch.Tensor:
        return flash_mla_sparse_fwd(
            padded_q,
            kv,
            indices,
            scale,
            d_v=args.head_dim,
        )[0]

    def flashmla_lightllm_path() -> torch.Tensor:
        q_input = q.new_zeros((args.tokens, args.required_heads, args.head_dim))
        q_input[:, : args.local_heads].copy_(q)
        return flash_mla_sparse_fwd(q_input, kv, indices, scale, d_v=args.head_dim,)[
            0
        ][:, : args.local_heads]

    def tilelang_lightllm_path() -> torch.Tensor:
        output = tilelang_sparse_fwd(
            q,
            kv,
            indices,
            scale,
            d_v=args.head_dim,
        )
        return output.squeeze(0) if output.ndim == 4 else output

    results = []
    for name, operation in (
        ("flashmla_padded_kernel", flashmla_kernel),
        ("flashmla_lightllm_path", flashmla_lightllm_path),
        ("tilelang_lightllm_path", tilelang_lightllm_path),
    ):
        results.append(measure_ms(name, operation, args.warmup, args.iterations))
        gc.collect()
        torch.cuda.empty_cache()

    flash_ms = next(r["median_ms"] for r in results if r["name"] == "flashmla_lightllm_path")
    tile_ms = next(r["median_ms"] for r in results if r["name"] == "tilelang_lightllm_path")
    print(
        json.dumps(
            {
                "shape": {
                    "tokens": args.tokens,
                    "sequence_length": args.sequence_length,
                    "topk": args.topk,
                    "local_heads": args.local_heads,
                    "required_heads": args.required_heads,
                    "head_dim": args.head_dim,
                },
                "results": results,
                "tilelang_speedup_over_lightllm_flashmla": flash_ms / tile_ms,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
