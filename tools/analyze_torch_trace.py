#!/usr/bin/env python3

"""Aggregate CUDA kernel time from one or more Torch profiler traces."""

from __future__ import annotations

import argparse
from collections import defaultdict
import gzip
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("traces", nargs="+", type=Path)
    parser.add_argument("--graph-only", action="store_true")
    parser.add_argument("--graph-id", type=int)
    parser.add_argument("--top", type=int, default=40)
    args = parser.parse_args()

    totals: dict[str, list[float]] = defaultdict(lambda: [0.0, 0.0, 0.0])
    graph_totals: dict[int, list[float]] = defaultdict(lambda: [0.0, 0.0])
    for path in args.traces:
        opener = gzip.open if path.suffix == ".gz" else open
        with opener(path, "rt", encoding="utf-8") as trace_file:
            events = json.load(trace_file)["traceEvents"]
        for event in events:
            if event.get("cat") != "kernel" or "dur" not in event:
                continue
            graph_id = int(event.get("args", {}).get("graph id", 0))
            if args.graph_only and graph_id == 0:
                continue
            if args.graph_id is not None and graph_id != args.graph_id:
                continue
            duration = float(event["dur"])
            name = event.get("name", "<unnamed>")
            totals[name][0] += duration
            totals[name][1] += 1
            totals[name][2] = max(totals[name][2], duration)
            graph_totals[graph_id][0] += duration
            graph_totals[graph_id][1] += 1

    total_us = sum(values[0] for values in totals.values())
    total_calls = int(sum(values[1] for values in totals.values()))
    print(f"CUDA total: {total_us / 1000:.3f} ms across {total_calls} kernels")
    print("Graph totals:")
    for graph_id, (duration, count) in sorted(graph_totals.items(), key=lambda item: item[1][0], reverse=True):
        print(
            f"  graph={graph_id:<5} total_ms={duration / 1000:10.3f} "
            f"share={duration / total_us:7.2%} calls={int(count)}"
        )
    print("Kernel totals:")
    for name, (duration, count, maximum) in sorted(totals.items(), key=lambda item: item[1][0], reverse=True)[
        : args.top
    ]:
        print(
            f"  total_ms={duration / 1000:10.3f} share={duration / total_us:7.2%} "
            f"calls={int(count):7d} avg_us={duration / count:9.3f} "
            f"max_us={maximum:9.3f}  {name}"
        )


if __name__ == "__main__":
    main()
