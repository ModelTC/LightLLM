"""Plot already audited aggregate metrics; never recompute or pool trials.

Input is report.py's numerical public export. One figure per model preserves
every group, including neutral and negative results and their seed-level CIs.
"""

import argparse
import gzip
import json
from pathlib import Path
import re

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter, MaxNLocator


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("summary", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--title-prefix", default="Paired cache-retention comparisons")
    args = parser.parse_args()
    payload = args.summary.read_bytes()
    if args.summary.suffix == ".gz":
        payload = gzip.decompress(payload)
    data = json.loads(payload)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    columns = [
        ("wall_speedup", "Replay wall-time speedup", 1),
        ("p95_ttft_speedup", "Server p95 TTFT speedup", 1),
        ("prefill_reduction_percent", "Uncached input reduction (%)", 0),
    ]
    for model in sorted({group["model"] for group in data["groups"]}):
        groups = sorted(
            (group for group in data["groups"] if group["model"] == model),
            key=lambda group: (group["case"], group["retention_seconds"], group["workload_sha256"]),
        )
        labels = [
            f"{g['case']} | TTL {g['retention_seconds']:g}s | {g['pairs']} pairs / {g['independent_seeds']} seeds"
            for g in groups
        ]
        fig, axes = plt.subplots(1, 3, figsize=(15, max(4.5, 0.42 * len(groups) + 2.2)), sharey=True)
        for axis, (key, title, neutral) in zip(axes, columns):
            axis.axvline(neutral, color="#777777", linestyle="--", linewidth=0.8)
            for position, group in enumerate(groups):
                metric = group["metrics"].get(key)
                if metric is None:
                    axis.text(0.5, position, "not available", ha="center", transform=axis.get_yaxis_transform())
                    continue
                value = metric["geomean"] if neutral == 1 else metric["mean"]
                interval = metric["ci95"]
                color = "#555555"
                if interval:
                    axis.hlines(position, *interval, color=color, linewidth=1.6)
                axis.scatter(value, position, color=color, s=24, zorder=3)
            if neutral == 1:
                axis.xaxis.set_major_locator(MaxNLocator(nbins=5, min_n_ticks=3))
                axis.xaxis.set_major_formatter(FuncFormatter(lambda value, _: f"{value:g}x"))
            axis.set_title(title, fontsize=10)
            axis.grid(axis="x", alpha=0.2)
            axis.tick_params(axis="x", labelsize=8, rotation=25)
            axis.set_xlabel("Larger values favor candidate", fontsize=9)
        axes[0].set_yticks(range(len(groups)), labels=labels, fontsize=9)
        axes[0].invert_yaxis()
        fig.suptitle(f"{args.title_prefix}\n{Path(model).name}", fontsize=12)
        fig.text(
            0.5,
            0.015,
            "Points: geometric-mean ratios / mean token reduction. Bars: descriptive 95% bootstrap intervals.\n"
            "Independent unit: workload seed; GPU crossover repetitions share a seed cluster. "
            "Uncached input = API prompt tokens minus cached tokens.",
            ha="center",
            fontsize=8,
        )
        fig.tight_layout(rect=(0, 0.09, 1, 0.99))
        stem = re.sub(r"[^a-zA-Z0-9._-]+", "-", Path(model).name)
        for suffix in ["svg", "png"]:
            output = args.output_dir / f"{stem}.{suffix}"
            fig.savefig(output, dpi=160, bbox_inches="tight")
            print(output)
        plt.close(fig)


if __name__ == "__main__":
    main()
