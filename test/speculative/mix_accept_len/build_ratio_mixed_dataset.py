#!/usr/bin/env python3
"""
Build a fixed-size dataset by mixing GSM8K with a low-accept dataset at a
specified ratio.
"""

from __future__ import annotations

import argparse
import json
import math
import random
from copy import deepcopy
from pathlib import Path
from typing import List


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Mix GSM8K with a low-accept dataset at a fixed ratio.")
    parser.add_argument("--gsm8k", required=True, help="Path to GSM8K JSON dataset.")
    parser.add_argument("--low-accept", required=True, help="Path to low-accept JSON dataset.")
    parser.add_argument("--ratio", type=float, required=True, help="Low-accept ratio in [0, 1].")
    parser.add_argument("--target-size", type=int, default=1000, help="Total sample count in output dataset.")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output", required=True)
    return parser.parse_args()


def load_json(path: Path) -> List[dict]:
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError(f"{path} must be a JSON list.")
    return data


def sample_with_wrap(data: List[dict], count: int, rng: random.Random) -> List[dict]:
    if count <= 0:
        return []
    if not data:
        raise ValueError("Cannot sample from an empty dataset.")

    shuffled = list(data)
    rng.shuffle(shuffled)
    out = []
    while len(out) < count:
        take = min(len(shuffled), count - len(out))
        out.extend(deepcopy(shuffled[:take]))
        rng.shuffle(shuffled)
    return out


def main() -> None:
    args = parse_args()
    if args.ratio < 0.0 or args.ratio > 1.0:
        raise ValueError("--ratio must be in [0, 1].")

    rng = random.Random(args.seed)
    gsm8k = load_json(Path(args.gsm8k))
    low_accept = load_json(Path(args.low_accept))

    low_count = int(round(args.target_size * args.ratio))
    gsm8k_count = args.target_size - low_count

    mixed = sample_with_wrap(gsm8k, gsm8k_count, rng) + sample_with_wrap(low_accept, low_count, rng)
    rng.shuffle(mixed)

    output_path = Path(args.output)
    meta_path = output_path.with_suffix(output_path.suffix + ".meta.json")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(mixed, f, ensure_ascii=False, indent=2)
    with meta_path.open("w", encoding="utf-8") as f:
        json.dump(
            {
                "gsm8k": args.gsm8k,
                "low_accept": args.low_accept,
                "ratio": args.ratio,
                "target_size": args.target_size,
                "gsm8k_count": gsm8k_count,
                "low_accept_count": low_count,
                "seed": args.seed,
                "output": str(output_path),
            },
            f,
            ensure_ascii=False,
            indent=2,
        )
    print(str(output_path))


if __name__ == "__main__":
    main()
