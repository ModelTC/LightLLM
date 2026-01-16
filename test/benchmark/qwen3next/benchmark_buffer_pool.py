# test/benchmark/qwen3next/benchmark_buffer_pool.py
import torch
import time
from lightllm.models.qwen3next.buffer_pool import Qwen3NextBufferPool


def benchmark_allocations():
    """Compare raw allocation vs buffer pool."""
    shape = (1024, 2048)
    dtype = torch.bfloat16
    device = torch.device("cuda")
    iterations = 1000

    # Without pool
    torch.cuda.synchronize()
    start = time.perf_counter()
    for _ in range(iterations):
        _ = torch.empty(shape, dtype=dtype, device=device)  # noqa: F841
    torch.cuda.synchronize()
    no_pool_time = time.perf_counter() - start

    # With pool
    pool = Qwen3NextBufferPool()
    torch.cuda.synchronize()
    start = time.perf_counter()
    for _ in range(iterations):
        _ = pool.get_buffer(shape, dtype, device)  # noqa: F841
    pool.release_all()  # Release after timing loop
    torch.cuda.synchronize()
    pool_time = time.perf_counter() - start

    print(f"Without pool: {no_pool_time * 1000:.2f}ms")
    print(f"With pool: {pool_time * 1000:.2f}ms")
    print(f"Speedup: {no_pool_time / pool_time:.2f}x")


if __name__ == "__main__":
    benchmark_allocations()
