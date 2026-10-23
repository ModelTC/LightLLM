"""Measure page-cache token index pinned allocation, excluding data writes and H2D.

Example:
    CUDA_VISIBLE_DEVICES=4 PYTHONPATH=. python test/benchmark/benchmark_req_token_index_alloc.py \
        --cpu 72 --output-dir /tmp/req_token_index_alloc
"""

import argparse
import csv
import gc
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import statistics
import time

import torch

from lightllm.server.router.model_infer.infer_req_ext1 import PinMemPageCacheReqTokenIndex


SLAB_TOKENS = 1024 * 1024
CLOCK = time.perf_counter_ns


def allocate_direct(token_num):
    return torch.empty(token_num, dtype=torch.int32, device="cpu", pin_memory=True, requires_grad=False)


def summarize(samples):
    ordered = sorted(samples)
    return {
        "count": len(samples),
        "min_us": min(samples),
        "max_us": max(samples),
        "mean_us": statistics.mean(samples),
        "median_us": statistics.median(samples),
        "p95_us": ordered[int(0.95 * (len(ordered) - 1))],
    }


def measure_cached(mode, token_num, repeats, backing):
    allocate = allocate_direct if mode == "direct_cached" else PinMemPageCacheReqTokenIndex._alloc_pinned_tensor
    samples = []
    for _ in range(repeats):
        if mode == "slab_slice" and 0 < token_num <= SLAB_TOKENS:
            # 仅测切片路径：当前没有存活的返回值或异步传输，允许在计时外重置测试 backing。
            # 生产接口始终向后切分；slab_cached 模式保留其真实换块行为。
            if (
                PinMemPageCacheReqTokenIndex._pinned_buffer is None
                or PinMemPageCacheReqTokenIndex._pinned_offset + token_num > SLAB_TOKENS
            ):
                PinMemPageCacheReqTokenIndex._pinned_buffer = backing
                PinMemPageCacheReqTokenIndex._pinned_offset = 0
        start = CLOCK()
        tensor = allocate(token_num)
        samples.append((CLOCK() - start) / 1000)
        del tensor
    return samples


def measure_growth(mode, token_num, live_count):
    # 仅清理 HOST 空闲缓存；用于构造缓存未命中，不用于生产服务。
    PinMemPageCacheReqTokenIndex._pinned_buffer = None
    PinMemPageCacheReqTokenIndex._pinned_offset = 0
    gc.collect()
    torch.cuda.synchronize()
    torch._C._host_emptyCache()
    before = torch.cuda.memory.host_memory_stats()
    allocate = allocate_direct if mode == "direct" else PinMemPageCacheReqTokenIndex._alloc_pinned_tensor
    samples, held = [], []
    for _ in range(live_count):
        start = CLOCK()
        tensor = allocate(token_num)
        samples.append((CLOCK() - start) / 1000)
        held.append(tensor)
    after = torch.cuda.memory.host_memory_stats()
    host_calls = after["num_host_alloc"] - before["num_host_alloc"]
    expected_calls = live_count
    if mode == "slab" and token_num <= SLAB_TOKENS:
        expected_calls = math.ceil(live_count / (SLAB_TOKENS // token_num))
    assert host_calls == expected_calls, (mode, token_num, host_calls, expected_calls)
    assert all(tensor.is_pinned() and tensor.numel() == token_num for tensor in held)
    return {
        "samples_us": samples,
        "summary": summarize(samples),
        "host_allocations": host_calls,
        "allocated_bytes": after["allocated_bytes.current"] - before["allocated_bytes.current"],
        "host_alloc_time_us": after["host_alloc_time.total"] - before["host_alloc_time.total"],
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--cpu", type=int)
    parser.add_argument("--rounds", type=int, default=10)
    parser.add_argument("--repeats", type=int, default=2000)
    args = parser.parse_args()
    if args.cpu is not None:
        os.sched_setaffinity(0, {args.cpu})
    args.output_dir.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(1)
    torch.cuda.init()
    seed = allocate_direct(1)
    del seed
    torch._C._host_emptyCache()
    start = CLOCK()
    seed = PinMemPageCacheReqTokenIndex._alloc_pinned_tensor(1)
    initial_slab_us = (CLOCK() - start) / 1000
    backing = seed._base
    assert backing.numel() == SLAB_TOKENS and backing.is_pinned()
    del seed
    gc.collect()
    gc.disable()

    sizes = (0, 128, 1024, 2048, 8192, 32768, 262144, SLAB_TOKENS // 2, SLAB_TOKENS, 2 * SLAB_TOKENS)
    modes = ("direct_cached", "slab_slice", "slab_cached")
    result = {
        "metadata": {
            "torch": torch.__version__,
            "torch_git": torch.version.git_version,
            "python": platform.python_version(),
            "gpu": torch.cuda.get_device_name(0),
            "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
            "cpu_affinity": sorted(os.sched_getaffinity(0)),
            "torch_threads": torch.get_num_threads(),
            "allocator_env": {name: os.environ.get(name) for name in ("PYTORCH_ALLOC_CONF", "PYTORCH_CUDA_ALLOC_CONF")},
            "source_sha256": hashlib.sha256(
                Path(PinMemPageCacheReqTokenIndex._alloc_pinned_tensor.__code__.co_filename).read_bytes()
            ).hexdigest(),
            "slab_tokens": SLAB_TOKENS,
            "slab_bytes": SLAB_TOKENS * 4,
            "scope": "CPU allocation wall time; excludes release, fill, H2D, imports, CUDA init and GC.",
            "cached_modes": {
                "direct_cached": "torch.empty with warm HOST size buckets",
                "slab_slice": "slab slicing; backing reset outside timer when exhausted, no live or in-flight data",
                "slab_cached": "actual shared allocator, including warm HOST-cache slab replacements",
            },
            "growth": "Retain every returned tensor; clear unused HOST cache before each cohort; include fresh slabs.",
        },
        "initial_slab_allocation_us": initial_slab_us,
        "cached": {str(size): {mode: {"rounds_us": []} for mode in modes} for size in sizes},
        "growth": {},
    }

    # 预热所有直接申请桶，再准备两个可交替复用的 slab；验证计时区间没有调用 host 分配 API。
    for token_num in sizes:
        seed = allocate_direct(token_num)
        del seed
    warm = [allocate_direct(SLAB_TOKENS) for _ in range(2)]
    del warm
    before = torch.cuda.memory.host_memory_stats()["num_host_alloc"]
    for repeat in range(args.rounds):
        for token_num in sizes if repeat % 2 == 0 else reversed(sizes):
            for mode in modes if repeat % 2 == 0 else reversed(modes):
                samples = measure_cached(mode, token_num, args.repeats, backing)
                result["cached"][str(token_num)][mode]["rounds_us"].append(samples)
        print(f"Cached round {repeat + 1}/{args.rounds} complete", flush=True)
    result["cached_host_allocations"] = torch.cuda.memory.host_memory_stats()["num_host_alloc"] - before
    assert result["cached_host_allocations"] == 0
    for data in result["cached"].values():
        for measured in data.values():
            rounds = measured["rounds_us"]
            measured["summary"] = summarize([sample for samples in rounds for sample in samples])
            measured["round_medians_us"] = [statistics.median(samples) for samples in rounds]

    for token_num in (1024, 8192, 32768, 262144, 1048576, 2 * SLAB_TOKENS):
        live_count = min(256, max(16, 16 * 1024 * 1024 // token_num))
        data = {mode: {"rounds": []} for mode in ("direct", "slab")}
        data["live_count"] = live_count
        for repeat in range(args.rounds):
            for mode in ("direct", "slab") if repeat % 2 == 0 else ("slab", "direct"):
                data[mode]["rounds"].append(measure_growth(mode, token_num, live_count))
        for mode in ("direct", "slab"):
            rounds = data[mode]["rounds"]
            data[mode]["summary"] = summarize([sample for batch in rounds for sample in batch["samples_us"]])
            data[mode]["round_mean_us"] = [batch["summary"]["mean_us"] for batch in rounds]
        result["growth"][str(token_num)] = data
        print(f"Uncached growth for {token_num} tokens x {live_count} complete", flush=True)

    PinMemPageCacheReqTokenIndex._pinned_buffer = None
    PinMemPageCacheReqTokenIndex._pinned_offset = 0
    gc.enable()
    (args.output_dir / "result.json").write_text(json.dumps(result, indent=2) + "\n")
    with (args.output_dir / "timings.csv").open("w", newline="") as output:
        writer = csv.writer(output)
        fields = ("count", "min_us", "max_us", "mean_us", "median_us", "p95_us")
        writer.writerow(("scenario", "tokens", "bytes", "mode", *fields))
        for scenario in ("cached", "growth"):
            for tokens, data in result[scenario].items():
                for mode, measured in data.items():
                    if mode != "live_count":
                        writer.writerow(
                            (scenario, tokens, int(tokens) * 4, mode, *[measured["summary"][f] for f in fields])
                        )
    print(f"Results: {args.output_dir}", flush=True)


if __name__ == "__main__":
    main()
