"""Compare gather wrappers from two source files; no model or server required.

Example:
  git show BASE:lightllm/models/glm5_next/triton_kernel/kpool.py >/tmp/kpool_before.py
  python test/benchmark_glm53_kpool.py --baseline /tmp/kpool_before.py

The result includes allocation/page-table initialization, excludes compilation,
DeepGEMM, model computation and request scheduling, and is not an end-to-end TPS.
"""

import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import statistics
import sys

import torch
import triton.testing


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def measure(before, after, rounds):
    # Compile/warm both implementations before measuring; alternate AB/BA.
    before()
    after()
    torch.cuda.synchronize()
    values = {"before": [], "after": []}
    for index in range(rounds):
        order = (("before", before), ("after", after))
        for key, function in order if index % 2 == 0 else reversed(order):
            values[key].append(triton.testing.do_bench(function, warmup=10, rep=40, quantiles=[0.5, 0.2, 0.8])[0])
    old, new = (statistics.median(values[key]) for key in ("before", "after"))
    return {
        "before_us": old * 1000,
        "after_us": new * 1000,
        "speedup": old / new,
        "latency_change_percent": (new / old - 1) * 100,
        "before_rounds_ms": values["before"],
        "after_rounds_ms": values["after"],
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument(
        "--candidate",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "lightllm/models/glm5_next/triton_kernel/kpool.py",
    )
    parser.add_argument("--rounds", type=int, default=3)
    args = parser.parse_args()
    if args.rounds < 1:
        parser.error("rounds must be positive")
    before = load("kpool_baseline_benchmark", args.baseline)
    after = load("kpool_candidate_benchmark", args.candidate)
    torch.manual_seed(31)
    storage = torch.empty((262144, 1, 584), device="cuda", dtype=torch.bfloat16)
    packed = storage.view(torch.uint8)[:, :, -132:]
    packed[:, 0, :128] = torch.randn(262144, 128, device="cuda").to(torch.float8_e4m3fn).view(torch.uint8)
    packed[:, 0, 128:] = torch.ones(262144, device="cuda").view(torch.uint8).view(-1, 4)
    rows = []
    for pools in (640, 32768, 262144):
        table = torch.full((8, pools * 4), -1, device="cuda", dtype=torch.int32)
        table[:, 3::4] = torch.randint(0, 262144, (8, pools), device="cuda", dtype=torch.int32)
        for batch in (1, 4):
            req = torch.arange(batch, device="cuda", dtype=torch.int32)
            lengths = torch.full((batch,), pools * 4, device="cuda", dtype=torch.int32)
            rows.append(
                dict(
                    operation="prefill_gather",
                    max_pools=pools,
                    batch=batch,
                    **measure(
                        lambda: before.gather_pools(packed, table, req, lengths, pools),
                        lambda: after.gather_pools(packed, table, req, lengths, pools),
                        args.rounds,
                    ),
                )
            )
        for layout in ("shared_long", "mixed_long", "short_at_capacity", "empty_at_capacity"):
            batch = 8
            ids = [1] * 4 + [2] * 4 if layout == "shared_long" else list(range(batch))
            req = torch.tensor(ids, device="cuda", dtype=torch.int32)
            counts = [max(0, pools - i % 4) for i in range(batch)]
            if layout == "short_at_capacity":
                counts = [min(pools, 64 + i) for i in range(batch)]
            elif layout == "empty_at_capacity":
                counts = [0] * batch
            lengths = torch.tensor(counts, device="cuda", dtype=torch.int32)
            rows.append(
                dict(
                    operation="decode_paged_gather",
                    layout=layout,
                    max_pools=pools,
                    batch=batch,
                    **measure(
                        lambda: before.gather_paged_pools(packed, table, req, lengths, pools),
                        lambda: after.gather_paged_pools(packed, table, req, lengths, pools),
                        args.rounds,
                    ),
                )
            )
    report = {
        "device": torch.cuda.get_device_name(),
        "torch": torch.__version__,
        "rounds": args.rounds,
        "baseline_sha256": hashlib.sha256(args.baseline.read_bytes()).hexdigest(),
        "candidate_sha256": hashlib.sha256(args.candidate.read_bytes()).hexdigest(),
        "results": rows,
        "scope": (
            "Synthetic gather wrapper latency including allocations/page-table initialization; "
            "compilation excluded. Not model throughput or TTFT."
        ),
    }
    print(json.dumps(report))


if __name__ == "__main__":
    main()
