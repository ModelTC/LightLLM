#!/usr/bin/env python3
"""Benchmark GLM-5.3 KDA chunk-state kernel configs on a real packed shape."""

from __future__ import annotations

import argparse
import itertools
import json
import statistics

import torch

from lightllm.common.basemodel.triton_kernel.linear_att.fla.ops.chunk_delta_h import (
    chunk_gated_delta_rule_fwd_h,
)
from lightllm.common.basemodel.triton_kernel.linear_att.fla.ops.index import prepare_chunk_indices


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sequences", type=int, default=64)
    parser.add_argument("--sequence-length", type=int, default=268)
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--repeats", type=int, default=7)
    return parser.parse_args()


@torch.inference_mode()
def main() -> None:
    args = parse_args()
    device = torch.device("cuda")
    dtype = torch.bfloat16
    batch = 1
    heads = 8
    key_dim = 128
    value_dim = 128
    total_tokens = args.sequences * args.sequence_length

    shape = (batch, total_tokens, heads, key_dim)
    k = torch.zeros(shape, device=device, dtype=dtype)
    w = torch.zeros(shape, device=device, dtype=dtype)
    u = torch.zeros((batch, total_tokens, heads, value_dim), device=device, dtype=dtype)
    # The fused safe-gate+cumsum stage keeps cumulative decay in fp32; the
    # exp2 path in the state kernel expects that exact dtype.
    gk = torch.zeros(shape, device=device, dtype=torch.float32)
    initial_state = torch.zeros((args.sequences, heads, key_dim, value_dim), device=device, dtype=dtype)
    cu_seqlens = torch.arange(
        0,
        total_tokens + 1,
        args.sequence_length,
        device=device,
        dtype=torch.int32,
    )
    chunk_indices = prepare_chunk_indices(cu_seqlens, 64)

    def run(config: dict[str, int]) -> None:
        output = chunk_gated_delta_rule_fwd_h(
            k=k,
            w=w,
            u=u,
            gk=gk,
            initial_state=initial_state,
            output_final_state=True,
            save_new_value=True,
            cu_seqlens=cu_seqlens,
            chunk_indices=chunk_indices,
            chunk_size=64,
            use_exp2=True,
            run_config=config,
        )
        del output

    results = []
    for value_tile, num_warps, num_stages in itertools.product((32, 64), (2, 4), (2, 3, 4)):
        config = {
            "BV": value_tile,
            "num_warps": num_warps,
            "num_stages": num_stages,
        }
        for _ in range(args.warmup):
            run(config)
        torch.cuda.synchronize()

        samples_ms = []
        for _ in range(args.repeats):
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
            run(config)
            end.record()
            end.synchronize()
            samples_ms.append(start.elapsed_time(end))
        results.append(
            {
                **config,
                "median_ms": statistics.median(samples_ms),
                "min_ms": min(samples_ms),
            }
        )

    results.sort(key=lambda item: item["median_ms"])
    print(
        json.dumps(
            {
                "shape": {
                    "sequences": args.sequences,
                    "sequence_length": args.sequence_length,
                    "total_tokens": total_tokens,
                    "heads": heads,
                    "key_dim": key_dim,
                    "value_dim": value_dim,
                },
                "results": results,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
