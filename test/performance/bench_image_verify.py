"""Benchmark: find the right LIGHTLLM_IMAGE_VERIFY_WORKERS value.

Methodology:
  - Generate N independent JPEGs once (random pixels so libjpeg can't cheat).
  - For each candidate pool size, create a FRESH ThreadPoolExecutor of that size,
    submit all N decodes concurrently (no semaphore), measure wall time.
  - This faithfully simulates production: at peak, many requests pile into
    run_in_executor at once and the pool size is the real bottleneck.

This lets us compare different LIGHTLLM_IMAGE_VERIFY_WORKERS settings in one run.

Usage:
    python test/performance/bench_image_verify.py
    python test/performance/bench_image_verify.py --size 4096 --num 128 --pool_sizes 1,2,4,8,16,32,64
"""
import argparse
import asyncio
import os
import time
from concurrent.futures import ThreadPoolExecutor
from io import BytesIO
from typing import List

import numpy as np
from PIL import Image

from lightllm.server.multimodal_params import _verify_image_bytes


def make_big_jpeg(size: int, seed: int) -> bytes:
    """Random-noise JPEG so decode time is real (flat images decode too fast)."""
    rng = np.random.default_rng(seed)
    arr = rng.integers(0, 256, (size, size, 3), dtype=np.uint8)
    buf = BytesIO()
    Image.fromarray(arr).save(buf, format="JPEG", quality=85)
    return buf.getvalue()


def bench_serial(images: List[bytes]) -> float:
    t0 = time.perf_counter()
    for img in images:
        _verify_image_bytes(img)
    return time.perf_counter() - t0


def bench_pool(images: List[bytes], pool_size: int) -> float:
    """Fresh pool of `pool_size`, submit all images concurrently, wait, time it."""
    pool = ThreadPoolExecutor(max_workers=pool_size, thread_name_prefix=f"bench-{pool_size}")
    try:
        # Pre-warm threads so we don't time thread spawn-up
        list(pool.map(lambda _: None, range(pool_size)))

        async def run():
            loop = asyncio.get_running_loop()
            futs = [loop.run_in_executor(pool, _verify_image_bytes, img) for img in images]
            await asyncio.gather(*futs)

        t0 = time.perf_counter()
        asyncio.run(run())
        return time.perf_counter() - t0
    finally:
        pool.shutdown(wait=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--size", type=int, default=2048, help="image edge length, e.g. 2048/4096")
    parser.add_argument("--num", type=int, default=64, help="total images to decode per run")
    parser.add_argument(
        "--pool_sizes",
        default="1,2,4,8,16,32,64",
        help="comma-separated pool sizes (LIGHTLLM_IMAGE_VERIFY_WORKERS candidates)",
    )
    parser.add_argument("--warmup", type=int, default=4)
    parser.add_argument("--repeat", type=int, default=2, help="repeats per pool size, takes the best")
    args = parser.parse_args()

    print(f"CPU count           : {os.cpu_count()}")
    print(f"Image size          : {args.size}x{args.size}")
    print(f"Images per run      : {args.num}")
    print(f"Pool sizes to test  : {args.pool_sizes}")
    print(f"Repeats per pool    : {args.repeat} (best time wins)\n")

    print("Generating distinct test images ...")
    images = [make_big_jpeg(args.size, seed=i) for i in range(args.num)]
    avg_kb = sum(len(b) for b in images) / len(images) / 1024
    print(f"  per-image encoded size ~ {avg_kb:.1f} KB\n")

    # Warmup libjpeg / page faults
    for _ in range(args.warmup):
        _verify_image_bytes(images[0])

    # Baseline
    serial_times = [bench_serial(images) for _ in range(args.repeat)]
    serial_t = min(serial_times)
    print(
        f"[serial]   {args.num} images in {serial_t * 1000:.1f} ms  "
        f"=> {args.num / serial_t:.1f} img/s, {serial_t / args.num * 1000:.2f} ms/img\n"
    )

    # Sweep pool size
    print("[threaded] — vary LIGHTLLM_IMAGE_VERIFY_WORKERS")
    print(f"  {'pool':>6} | {'time(ms)':>10} | {'img/s':>8} | {'speedup':>8} | {'efficiency':>10}")
    print(f"  {'-' * 6}-+-{'-' * 10}-+-{'-' * 8}-+-{'-' * 8}-+-{'-' * 10}")
    rows = []
    for p in [int(x) for x in args.pool_sizes.split(",")]:
        times = [bench_pool(images, p) for _ in range(args.repeat)]
        t = min(times)
        ips = args.num / t
        speedup = serial_t / t
        eff = speedup / p
        rows.append((p, t, ips, speedup, eff))
        print(f"  {p:>6} | {t * 1000:>10.1f} | {ips:>8.1f} | {speedup:>7.2f}x | {eff * 100:>9.1f}%")

    # Pick the sweet spot: largest speedup before efficiency drops below 50%
    best = max(rows, key=lambda r: r[3])
    knee = next((r for r in rows if r[4] < 0.5), rows[-1])
    print(f"\nBest absolute throughput : pool={best[0]}  ({best[2]:.1f} img/s, {best[3]:.2f}x)")
    print(f"Diminishing-returns knee : pool={knee[0]}  (efficiency drops <50% beyond here)")
    print("\nHints:")
    print("  - efficiency = speedup / pool_size. ~100% means perfect linear scaling.")
    print("  - You usually want the smallest pool size that still gets >80% of peak throughput,")
    print("    since extra threads only add scheduling + memory pressure.")
    print(f"  - Recommended:  export LIGHTLLM_IMAGE_VERIFY_WORKERS={knee[0]}")


if __name__ == "__main__":
    main()
