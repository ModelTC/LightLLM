"""Compare torch.empty allocation latency while retaining every tensor.

Example (select an idle GPU and a CPU on the same NUMA node):
    CUDA_VISIBLE_DEVICES=4 python test/benchmark/benchmark_cpu_pinned_alloc_keepalive.py \
        --cpu 48 --malloc-usable-size --output-dir /tmp/cpu_pinned_alloc

Each mode/repeat runs in a fresh process. Warmup and measured tensors, across
all sizes, remain alive until process exit. No fill, H2D, or release is timed.
Ordinary CPU empty allocations need not fault in all physical pages. This is
an allocation API benchmark, not a comparison of ready-to-use memory bandwidth.
"""

import argparse
import csv
import ctypes
import gc
import json
import os
from pathlib import Path
import platform
import random
import statistics
import subprocess
import sys
import time


MIB = 1024 * 1024
DEFAULT_SIZES = [
    64,
    256,
    1024,
    4096,
    4097,
    16384,
    65536,
    262144,
    MIB,
    MIB + 1,
    4 * MIB,
    16 * MIB,
    64 * MIB,
    64 * MIB + 1,
]


def percentile(values, fraction):
    ordered = sorted(values)
    position = (len(ordered) - 1) * fraction
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    return ordered[lower] + (ordered[upper] - ordered[lower]) * (position - lower)


def rss_bytes():
    return int(Path("/proc/self/statm").read_text().split()[1]) * os.sysconf("SC_PAGE_SIZE")


def sample_count(size, args):
    # Both modes use the same count. Budget using pinned allocator rounding.
    rounded = 1 << (size - 1).bit_length()
    return max(args.min_count, min(args.max_count, args.case_mib * MIB // rounded))


def run_worker(args):
    if args.cpu is not None:
        os.sched_setaffinity(0, {args.cpu})
    import torch

    torch.set_num_threads(1)
    torch.cuda.init()
    pinned = args.mode == "pinned"
    # Retain warmup too: initialization is excluded without returning any blocks.
    warmup = [torch.empty(4096, dtype=torch.uint8, device="cpu", pin_memory=pinned) for _ in range(32)]
    usable_size = None
    if args.malloc_usable_size and not pinned:
        if platform.libc_ver()[0] != "glibc" or os.environ.get("LD_PRELOAD"):
            raise RuntimeError("--malloc-usable-size requires the default glibc CPU allocator without LD_PRELOAD")
        # Diagnostic only; never call this on CUDA host allocations. The caller
        # must use a PyTorch build with the default system CPU allocator.
        usable_size = ctypes.CDLL(None).malloc_usable_size
        usable_size.argtypes = [ctypes.c_void_p]
        usable_size.restype = ctypes.c_size_t

    sizes = list(args.sizes)
    random.Random(args.repeat).shuffle(sizes)
    kept = []
    rows = []
    gc.disable()
    for size in sizes:
        count = sample_count(size, args)
        tensors = [None] * count
        durations_ns = [0] * count
        before_rss = rss_bytes()
        before_stats = torch.cuda.memory.host_memory_stats() if pinned else None
        for index in range(count):
            start = time.perf_counter_ns()
            tensor = torch.empty(size, dtype=torch.uint8, device="cpu", pin_memory=pinned, requires_grad=False)
            end = time.perf_counter_ns()
            tensors[index] = tensor
            durations_ns[index] = end - start
        kept.append(tensors)
        after_stats = torch.cuda.memory.host_memory_stats() if pinned else None
        after_rss = rss_bytes()
        # Validation and diagnostics are outside the timed region.
        assert all(t.nbytes == size and t.untyped_storage().nbytes() == size for t in tensors)
        assert len({t.data_ptr() for t in tensors}) == count
        if pinned:
            actual_allocs = after_stats["num_host_alloc"] - before_stats["num_host_alloc"]
            assert actual_allocs == count, (size, count, actual_allocs)
            block_bytes = (after_stats["allocated_bytes.current"] - before_stats["allocated_bytes.current"]) // count
            assert block_bytes == 1 << (size - 1).bit_length(), (size, block_bytes)
        else:
            actual_allocs = block_bytes = None
        usable = [usable_size(t.data_ptr()) for t in tensors] if usable_size else []
        assert not usable or min(usable) >= size
        rows.append(
            {
                "mode": args.mode,
                "repeat": args.repeat,
                "requested_bytes": size,
                "count": count,
                "latency_ns": durations_ns,
                "median_us": statistics.median(durations_ns) / 1000,
                "mean_us": statistics.mean(durations_ns) / 1000,
                "p95_us": percentile(durations_ns, 0.95) / 1000,
                "rss_delta_bytes": after_rss - before_rss,
                "pinned_block_bytes": block_bytes,
                "new_host_allocations": actual_allocs,
                "cpu_malloc_usable_min": min(usable) if usable else None,
                "cpu_malloc_usable_median": statistics.median(usable) if usable else None,
                "cpu_malloc_usable_max": max(usable) if usable else None,
            }
        )

    metadata = {
        "python": platform.python_version(),
        "torch": torch.__version__,
        "torch_commit": torch.version.git_version,
        "cuda": torch.version.cuda,
        "gpu": torch.cuda.get_device_name(0),
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "cpu_affinity": sorted(os.sched_getaffinity(0)),
        "libc": platform.libc_ver(),
        "cpu_model": next(
            line.split(":", 1)[1].strip()
            for line in Path("/proc/cpuinfo").read_text().splitlines()
            if line.startswith("model name")
        ),
        "final_rss_bytes": rss_bytes(),
        "warmup_count": len(warmup),
        "retained_measured_tensors": sum(len(group) for group in kept),
        "size_order": sizes,
    }
    Path(args.worker_output).write_text(json.dumps({"metadata": metadata, "rows": rows}, indent=2) + "\n")
    # Exit while warmup/kept still own their storages; never release measured
    # tensors inside this process or let local-variable teardown recycle them.
    os._exit(0)


def summarize(trials, sizes):
    summary = []
    for size in sorted(sizes):
        row = {"requested_bytes": size}
        for mode in ("cpu", "pinned"):
            selected = [
                r for trial in trials for r in trial["rows"] if r["mode"] == mode and r["requested_bytes"] == size
            ]
            times = [value for item in selected for value in item["latency_ns"]]
            row[f"{mode}_count_per_repeat"] = selected[0]["count"]
            row[f"{mode}_median_us"] = statistics.median(times) / 1000
            row[f"{mode}_mean_us"] = statistics.mean(times) / 1000
            row[f"{mode}_p95_us"] = percentile(times, 0.95) / 1000
            row[f"{mode}_repeat_median_min_us"] = min(r["median_us"] for r in selected)
            row[f"{mode}_repeat_median_max_us"] = max(r["median_us"] for r in selected)
            if mode == "cpu":
                usable = [r for r in selected if r["cpu_malloc_usable_min"] is not None]
                row["cpu_malloc_usable_min"] = min(r["cpu_malloc_usable_min"] for r in usable) if usable else None
                row["cpu_malloc_usable_max"] = max(r["cpu_malloc_usable_max"] for r in usable) if usable else None
            else:
                row["pinned_block_bytes"] = selected[0]["pinned_block_bytes"]
        row["pinned_over_cpu_median"] = row["pinned_median_us"] / row["cpu_median_us"]
        summary.append(row)
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=Path("/tmp/cpu_pinned_alloc_keepalive"))
    parser.add_argument("--sizes", nargs="+", type=int, default=DEFAULT_SIZES, help="Allocation sizes in bytes")
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument(
        "--case-mib", type=int, default=128, help="Target retained pinned MiB per size; min-count may exceed it"
    )
    parser.add_argument("--min-count", type=int, default=8)
    parser.add_argument("--max-count", type=int, default=2048)
    parser.add_argument("--cpu", type=int, help="Bind workers to this logical CPU")
    parser.add_argument(
        "--malloc-usable-size", action="store_true", help="Probe CPU malloc sizes; only with default glibc allocator"
    )
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--mode", choices=["cpu", "pinned"], help=argparse.SUPPRESS)
    parser.add_argument("--repeat", type=int, default=0, help=argparse.SUPPRESS)
    parser.add_argument("--worker-output", help=argparse.SUPPRESS)
    args = parser.parse_args()
    if min(args.sizes + [args.repeats, args.case_mib, args.min_count]) < 1 or args.max_count < args.min_count:
        parser.error("Sizes/counts/repeats must be positive and max-count >= min-count")
    if len(set(args.sizes)) != len(args.sizes):
        parser.error("Sizes must be unique")
    for name in ("PYTORCH_ALLOC_CONF", "PYTORCH_CUDA_ALLOC_CONF"):
        if os.environ.get(name):
            parser.error(f"Unset {name} to measure default allocator behavior")
    if args.worker:
        run_worker(args)

    args.output_dir.mkdir(parents=True, exist_ok=False)
    trials = []
    estimated_pinned = sum(sample_count(size, args) * (1 << (size - 1).bit_length()) for size in args.sizes)
    print(f"Estimated retained pinned payload per worker: {estimated_pinned / MIB:.1f} MiB", flush=True)
    for repeat in range(args.repeats):
        # Alternate mode order; both modes use the same shuffled size order.
        for mode in ("cpu", "pinned") if repeat % 2 == 0 else ("pinned", "cpu"):
            path = args.output_dir / f"{mode}_{repeat}.json"
            command = [
                sys.executable,
                str(Path(__file__).resolve()),
                "--worker",
                "--mode",
                mode,
                "--repeat",
                str(repeat),
                "--worker-output",
                str(path),
                "--case-mib",
                str(args.case_mib),
                "--min-count",
                str(args.min_count),
                "--max-count",
                str(args.max_count),
                "--sizes",
                *map(str, args.sizes),
            ]
            if args.cpu is not None:
                command.extend(["--cpu", str(args.cpu)])
            if args.malloc_usable_size:
                command.append("--malloc-usable-size")
            subprocess.run(command, check=True)
            trial = json.loads(path.read_text())
            trials.append(trial)
            print(
                f'{mode} repeat {repeat + 1}: retained {trial["metadata"]["retained_measured_tensors"]} tensors; '
                f'RSS {trial["metadata"]["final_rss_bytes"] / MIB:.1f} MiB',
                flush=True,
            )

    summary = summarize(trials, args.sizes)
    with (args.output_dir / "summary.csv").open("w", newline="") as output:
        writer = csv.DictWriter(output, fieldnames=list(summary[0]))
        writer.writeheader()
        writer.writerows(summary)
    (args.output_dir / "results.json").write_text(
        json.dumps(
            {
                "parameters": {
                    key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()
                },
                "method": "torch.empty(uint8); no fill/copy; "
                "all warmup and measured tensors retained until process exit; "
                "CUDA initialization excluded; GC disabled; per-allocation timings; fresh process per mode/repeat",
                "estimated_retained_pinned_bytes": estimated_pinned,
                "summary": summary,
                "trials": trials,
            },
            indent=2,
        )
        + "\n"
    )
    print("bytes,count/repeat,cpu median us,pinned median us,ratio,pinned block bytes,cpu malloc min/max")
    for row in summary:
        print(
            f'{row["requested_bytes"]},{row["cpu_count_per_repeat"]},{row["cpu_median_us"]:.3f},'
            f'{row["pinned_median_us"]:.3f},{row["pinned_over_cpu_median"]:.2f},'
            f'{row["pinned_block_bytes"]},{row["cpu_malloc_usable_min"]}/{row["cpu_malloc_usable_max"]}'
        )
    print(f"Results: {args.output_dir}")


if __name__ == "__main__":
    main()
