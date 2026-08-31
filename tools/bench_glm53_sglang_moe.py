#!/usr/bin/env python3
"""Tune SGLang TritonMoE tiles for GLM-5.3-Flash TP8 on H100."""

import argparse
import json
import traceback
from contextlib import contextmanager
from types import SimpleNamespace

import torch


SMALL_M_CONFIGS = [
    # Decode runs with 24 physical rows for c8 + MTP2.  The original sweep
    # barely sampled this regime, so cover the useful tile/scheduling axes.
    (8, 32, 8, 4, 3),
    (8, 64, 8, 4, 3),
    (8, 128, 8, 4, 3),
    (8, 64, 4, 4, 3),
    (8, 64, 16, 4, 3),
    (8, 64, 32, 4, 3),
    (8, 128, 4, 4, 3),
    (8, 128, 16, 4, 3),
    (8, 128, 32, 4, 3),
    (8, 64, 8, 4, 2),
    (8, 64, 8, 4, 4),
    (8, 128, 8, 4, 2),
    (8, 128, 8, 4, 4),
    (8, 64, 8, 8, 3),
    (8, 128, 8, 8, 3),
    (16, 32, 16, 4, 3),
    (16, 64, 4, 4, 3),
    (16, 64, 8, 4, 3),
    (16, 64, 16, 4, 3),
    (16, 64, 32, 4, 3),
    (16, 128, 4, 4, 3),
    (16, 128, 8, 4, 3),
    (16, 128, 16, 4, 3),
    (16, 128, 32, 4, 3),
    (16, 64, 16, 4, 2),
    (16, 64, 16, 4, 4),
    (16, 128, 16, 4, 2),
    (16, 128, 16, 4, 4),
    (16, 64, 16, 8, 3),
    (16, 128, 16, 8, 3),
    (32, 64, 16, 4, 3),
    (32, 128, 16, 4, 3),
]


LARGE_M_CONFIGS = [
    (64, 128, 32, 4, 3),
    (64, 128, 8, 4, 3),
    (64, 128, 16, 4, 3),
    (64, 128, 64, 4, 3),
    (32, 128, 16, 4, 3),
    (128, 128, 32, 4, 3),
    (128, 128, 32, 8, 3),
    (64, 64, 32, 4, 3),
    (64, 256, 32, 8, 3),
    (128, 256, 32, 8, 3),
    (64, 128, 32, 4, 2),
    (64, 128, 32, 4, 4),
]


CONFIGS = SMALL_M_CONFIGS + LARGE_M_CONFIGS


def make_config(values):
    block_m, block_n, group_m, num_warps, num_stages = values
    return {
        "BLOCK_SIZE_M": block_m,
        "BLOCK_SIZE_N": block_n,
        "BLOCK_SIZE_K": 128,
        "GROUP_SIZE_M": group_m,
        "num_warps": num_warps,
        "num_stages": num_stages,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tokens", type=int, default=17152)
    parser.add_argument(
        "--tp-size",
        type=int,
        default=8,
        choices=(4, 8),
        help="Tensor-parallel size; GLM-5 has 2048 total expert intermediate rows.",
    )
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--iters", type=int, default=10)
    parser.add_argument("--max-configs", type=int, default=len(CONFIGS))
    parser.add_argument(
        "--config-set",
        choices=("all", "small", "large"),
        default="all",
        help="Restrict the sweep to decode-sized or prefill-sized tile configs.",
    )
    parser.add_argument(
        "--fuse-sum",
        action="store_true",
        help="Fuse the top-k expert sum into SGLang's down-projection kernel.",
    )
    parser.add_argument(
        "--tune-down",
        action="store_true",
        help="Tune a separate down-projection config with the best measured up config.",
    )
    parser.add_argument(
        "--fixed-up-config",
        choices=("decode", "prefill"),
        default="decode",
        help="Fixed up-projection config used by --tune-down.",
    )
    args = parser.parse_args()

    from sglang.srt.layers.moe.moe_runner.triton_utils import override_config
    from sglang.srt.layers.moe.moe_runner.triton_utils import fused_moe
    from sglang.srt.layers.moe.moe_runner.triton_utils import (
        fused_moe_triton_config,
    )

    standalone_exec = SimpleNamespace(
        moe=SimpleNamespace(enable_fused_moe_sum_all_reduce=args.fuse_sum),
        deterministic=SimpleNamespace(enable_deterministic_inference=False),
    )
    fused_moe.get_exec = lambda: standalone_exec
    fused_moe_triton_config.get_exec = lambda: standalone_exec

    device = torch.device("cuda:0")
    experts, hidden, intermediate, topk = 289, 4096, 2048 // args.tp_size, 9
    x = torch.zeros((args.tokens, hidden), dtype=torch.bfloat16, device=device)
    w1 = torch.zeros((experts, intermediate * 2, hidden), dtype=torch.float8_e4m3fn, device=device)
    w2 = torch.zeros((experts, hidden, intermediate), dtype=torch.float8_e4m3fn, device=device)
    w1_scale = torch.ones((experts, 4, 32), dtype=torch.float32, device=device)
    w2_scale = torch.ones((experts, 32, 2), dtype=torch.float32, device=device)
    rows = torch.arange(args.tokens, dtype=torch.int64, device=device)[:, None]
    cols = torch.arange(topk, dtype=torch.int64, device=device)[None, :]
    topk_ids = (rows * topk + cols) % experts
    topk_weights = torch.full((args.tokens, topk), 1.0 / topk, dtype=torch.float32, device=device)

    fixed_up_config = make_config(
        {
            "decode": (16, 64, 16, 4, 3),
            "prefill": (64, 128, 64, 4, 3),
        }[args.fixed_up_config]
    )

    @contextmanager
    def config_context(config):
        if not args.tune_down:
            with override_config(config):
                yield
            return
        original = fused_moe.try_get_optimal_moe_config

        def resolve_config(*resolve_args, return_down_config=False, **resolve_kwargs):
            if return_down_config:
                return fixed_up_config, (config, None)
            return fixed_up_config

        fused_moe.try_get_optimal_moe_config = resolve_config
        try:
            yield
        finally:
            fused_moe.try_get_optimal_moe_config = original

    def run(config):
        with config_context(config):
            fused_moe.fused_experts_impl(
                hidden_states=x,
                w1=w1,
                w2=w2,
                topk_weights=topk_weights,
                topk_ids=topk_ids,
                inplace=True,
                use_fp8_w8a8=True,
                w1_scale=w1_scale,
                w2_scale=w2_scale,
                block_shape=[128, 128],
                routed_scaling_factor=1.0,
                filter_expert=False,
                swiglu_limit=10.0,
                gate_up_interleaved=False,
            )

    def component_times(config):
        original = fused_moe.invoke_fused_moe_kernel
        events = []

        def timed_kernel(*kernel_args, **kernel_kwargs):
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
            result = original(*kernel_args, **kernel_kwargs)
            end.record()
            events.append((start, end))
            return result

        fused_moe.invoke_fused_moe_kernel = timed_kernel
        try:
            run(config)
            torch.cuda.synchronize()
        finally:
            fused_moe.invoke_fused_moe_kernel = original
        if len(events) != 2:
            raise RuntimeError(f"expected two MoE GEMM calls, got {len(events)}")
        return events[0][0].elapsed_time(events[0][1]), events[1][0].elapsed_time(events[1][1])

    results = []
    selected_configs = {
        "all": CONFIGS,
        "small": SMALL_M_CONFIGS,
        "large": LARGE_M_CONFIGS,
    }[args.config_set]
    if args.tune_down:
        # Both GEMMs share the same token alignment, so the down projection's
        # BLOCK_SIZE_M must match the fixed up projection.
        selected_configs = [values for values in CONFIGS if values[0] == fixed_up_config["BLOCK_SIZE_M"]]
    for values in selected_configs[: args.max_configs]:
        config = make_config(values)
        try:
            for _ in range(args.warmup):
                run(config)
            torch.cuda.synchronize()
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
            for _ in range(args.iters):
                run(config)
            end.record()
            end.synchronize()
            elapsed_ms = start.elapsed_time(end) / args.iters
            up_ms, down_ms = component_times(config)
            result = {"ms": elapsed_ms, "up_ms": up_ms, "down_ms": down_ms, **config}
            if args.tune_down:
                result["fixed_up_config"] = fixed_up_config
        except Exception as exc:
            result = {"error": f"{type(exc).__name__}: {exc}", **config}
            if not results:
                traceback.print_exc()
            torch.cuda.synchronize()
        results.append(result)
        print(json.dumps(result, sort_keys=True), flush=True)

    valid = [result for result in results if "ms" in result]
    if valid:
        print("BEST", json.dumps(min(valid, key=lambda result: result["ms"]), sort_keys=True))


if __name__ == "__main__":
    main()
