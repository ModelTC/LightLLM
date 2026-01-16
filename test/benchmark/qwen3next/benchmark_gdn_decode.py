"""Benchmark script for Qwen3Next GDN decode performance."""

import torch
import time
from typing import Callable


def benchmark_kernel(
    fn: Callable,
    warmup: int = 10,
    iterations: int = 100,
    sync: bool = True,
) -> float:
    """Benchmark a kernel function."""
    # Warmup
    for _ in range(warmup):
        fn()
        if sync:
            torch.cuda.synchronize()

    # Benchmark
    if sync:
        torch.cuda.synchronize()
    start = time.perf_counter()

    for _ in range(iterations):
        fn()
        if sync:
            torch.cuda.synchronize()

    elapsed = time.perf_counter() - start
    return elapsed / iterations * 1000  # ms


def main():
    """Run benchmarks."""
    if not torch.cuda.is_available():
        print("CUDA not available. This benchmark requires a GPU.")
        return

    print("Qwen3Next GDN Decode Benchmarks")
    print("=" * 50)

    # Test parameters matching real model
    batch_size = 32
    mtp_size = 2
    total_tokens = batch_size * mtp_size
    dim = 384  # typical qkv dim
    num_heads = 8

    device = "cuda"
    dtype = torch.bfloat16

    # Test data
    mixed_qkv = torch.randn(total_tokens, dim, device=device, dtype=dtype)

    # Benchmark: strided slice + contiguous
    def bench_contiguous():
        for step in range(mtp_size):
            _ = mixed_qkv[step::mtp_size].contiguous()

    # Benchmark: copy to pre-allocated buffer
    work_buffer = torch.empty(batch_size, dim, device=device, dtype=dtype)

    def bench_copy_to_buffer():
        for step in range(mtp_size):
            work_buffer.copy_(mixed_qkv[step::mtp_size])

    time_contiguous = benchmark_kernel(bench_contiguous)
    time_copy = benchmark_kernel(bench_copy_to_buffer)

    print(f"\n1. MTP Decode Buffer Strategy:")
    print(f"   Strided .contiguous(): {time_contiguous:.3f} ms")
    print(f"   Copy to buffer:        {time_copy:.3f} ms")
    print(f"   Speedup:               {time_contiguous / time_copy:.2f}x")

    # Benchmark: torch.cat elimination
    print(f"\n2. QKV Concatenation:")
    q = torch.randn(batch_size, dim // 3, device=device, dtype=dtype)
    k = torch.randn(batch_size, dim // 3, device=device, dtype=dtype)
    v = torch.randn(batch_size, dim // 3, device=device, dtype=dtype)

    def bench_torch_cat():
        return torch.cat([q, k, v], dim=-1)

    # Pre-concatenated (simulating the optimization)
    qkv_pre = torch.empty(batch_size, dim, device=device, dtype=dtype)

    def bench_pre_concat():
        qkv_pre[:, : dim // 3] = q
        qkv_pre[:, dim // 3 : 2 * dim // 3] = k
        qkv_pre[:, 2 * dim // 3 :] = v
        return qkv_pre

    time_cat = benchmark_kernel(bench_torch_cat)
    time_pre = benchmark_kernel(bench_pre_concat)

    print(f"   torch.cat():           {time_cat:.3f} ms")
    print(f"   Pre-allocated:         {time_pre:.3f} ms")
    print(f"   Speedup:               {time_cat / time_pre:.2f}x")

    # Benchmark: Fused gating kernel
    print(f"\n3. Fused Gating Kernel:")
    try:
        from lightllm.models.qwen3next.triton_kernel.fused_gdn_gating import (
            fused_gdn_gating,
        )
        from lightllm.models.qwen3next.triton_kernel.fused_qkv_gating import (
            fused_gdn_gating_v2,
        )

        a = torch.randn(batch_size, num_heads, device=device, dtype=dtype)
        b = torch.randn(batch_size, num_heads, device=device, dtype=dtype)
        A_log = torch.randn(num_heads, device=device, dtype=torch.float32)
        dt_bias = torch.randn(num_heads, device=device, dtype=torch.float32)

        def bench_original_gating():
            return fused_gdn_gating(A_log, a, b, dt_bias)

        g_out = torch.empty(1, batch_size, num_heads, device=device, dtype=torch.float32)
        beta_out = torch.empty(1, batch_size, num_heads, device=device, dtype=torch.float32)

        def bench_v2_gating():
            return fused_gdn_gating_v2(a, b, A_log, dt_bias, g_out, beta_out)

        time_orig = benchmark_kernel(bench_original_gating)
        time_v2 = benchmark_kernel(bench_v2_gating)

        print(f"   Original (allocates): {time_orig:.3f} ms")
        print(f"   V2 (pre-alloc):       {time_v2:.3f} ms")
        print(f"   Speedup:              {time_orig / time_v2:.2f}x")
    except ImportError as e:
        print(f"   Skipped (import error): {e}")

    print(f"\n" + "=" * 50)
    print("Benchmark complete.")


if __name__ == "__main__":
    main()
