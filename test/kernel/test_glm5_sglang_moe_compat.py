# SPDX-License-Identifier: Apache-2.0

"""Numerically compare LightLLM and SGLang's GLM-5 FP8 MoE paths.

This is an integration probe for development images that provide both
packages.  It intentionally uses GLM-5's TP8 decode shapes, including the
fused shared expert, block-wise FP8 scales, and clamped SwiGLU.
"""

import argparse
import itertools
import json
from types import SimpleNamespace

import torch

from lightllm.common.basemodel.triton_kernel.fused_moe.grouped_fused_moe import (
    fused_experts as lightllm_fused_experts,
)
from sglang.srt.layers.moe.moe_runner.triton_utils import fused_moe as sglang_fused_moe
from sglang.srt.layers.moe.moe_runner.triton_utils import (
    fused_moe_triton_config as sglang_fused_moe_config,
)
from sglang.srt.layers.moe.moe_runner.triton_utils import override_config


def _fp8_randn(shape, *, scale=0.02):
    return (torch.randn(shape, device="cuda", dtype=torch.bfloat16) * scale).to(torch.float8_e4m3fn)


def _graph_ms(fn, source, iterations):
    static_input = source.clone()
    for _ in range(3):
        static_input.copy_(source)
        fn(static_input)
    torch.cuda.synchronize()

    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        static_input.copy_(source)
        fn(static_input)

    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(iterations):
        graph.replay()
    end.record()
    end.synchronize()
    elapsed_ms = start.elapsed_time(end) / iterations
    graph.reset()
    return elapsed_ms


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--benchmark", action="store_true")
    parser.add_argument(
        "--fuse-sum",
        action="store_true",
        help="Fuse the top-k sum into SGLang's down-projection kernel.",
    )
    parser.add_argument(
        "--tune-configs",
        action="store_true",
        help="Search a compact H100 config set for GLM-5's TP8 decode shape.",
    )
    parser.add_argument(
        "--tune-tma-configs",
        action="store_true",
        help="Search the compact small-M set with SGLang's up-projection TMA path.",
    )
    parser.add_argument(
        "--num-tokens",
        type=int,
        default=48,
        help="Physical token count used by the MoE probe (48 main, 8 MTP draft).",
    )
    parser.add_argument("--iterations", type=int, default=100)
    args = parser.parse_args()

    torch.manual_seed(123)

    num_tokens = args.num_tokens
    num_experts = 289
    topk = 9
    hidden_size = 4096
    tp_intermediate_size = 256

    hidden_states = torch.randn((num_tokens, hidden_size), device="cuda", dtype=torch.bfloat16)
    w13 = _fp8_randn((num_experts, 2 * tp_intermediate_size, hidden_size))
    w2 = _fp8_randn((num_experts, hidden_size, tp_intermediate_size))
    w13_scale = torch.ones(
        (num_experts, 2 * tp_intermediate_size // 128, hidden_size // 128),
        device="cuda",
        dtype=torch.float32,
    )
    w2_scale = torch.ones(
        (num_experts, hidden_size // 128, tp_intermediate_size // 128),
        device="cuda",
        dtype=torch.float32,
    )
    topk_ids = torch.randint(0, num_experts, (num_tokens, topk), device="cuda", dtype=torch.int64)
    topk_weights = torch.rand((num_tokens, topk), device="cuda", dtype=torch.float32)
    topk_weights.mul_(2.5 / topk_weights.sum(dim=-1, keepdim=True))

    def run_lightllm(output):
        lightllm_fused_experts(
            hidden_states=output,
            w1=w13,
            w2=w2,
            topk_weights=topk_weights,
            topk_ids=topk_ids,
            inplace=True,
            use_fp8_w8a8=True,
            w1_scale=w13_scale,
            w2_scale=w2_scale,
            limit=10.0,
            alpha=1.0,
            clamp_up_add_one=False,
        )

    lightllm_output = hidden_states.clone()
    run_lightllm(lightllm_output)

    # SGLang's fused sequence consults this runtime flag.  A standalone
    # LightLLM process has no SGLang RuntimeContext, so provide the selected
    # fused top-k sum behavior explicitly for this comparison.
    standalone_exec = SimpleNamespace(
        moe=SimpleNamespace(enable_fused_moe_sum_all_reduce=args.fuse_sum),
        deterministic=SimpleNamespace(enable_deterministic_inference=False),
    )
    sglang_fused_moe.get_exec = lambda: standalone_exec
    sglang_fused_moe_config.get_exec = lambda: standalone_exec

    def run_sglang(output):
        sglang_fused_moe.fused_experts_impl(
            hidden_states=output,
            w1=w13,
            w2=w2,
            topk_weights=topk_weights,
            topk_ids=topk_ids,
            inplace=True,
            use_fp8_w8a8=True,
            w1_scale=w13_scale,
            w2_scale=w2_scale,
            block_shape=[128, 128],
            routed_scaling_factor=1.0,
            filter_expert=False,
            swiglu_limit=10.0,
            gate_up_interleaved=False,
        )

    sglang_output = hidden_states.clone()
    run_sglang(sglang_output)
    torch.cuda.synchronize()

    diff = (lightllm_output.float() - sglang_output.float()).abs()
    reference = lightllm_output.float().abs()
    print(
        "max_abs=%.6f mean_abs=%.6f max_ref=%.6f mean_ref=%.6f"
        % (
            diff.max().item(),
            diff.mean().item(),
            reference.max().item(),
            reference.mean().item(),
        )
    )
    torch.testing.assert_close(
        sglang_output.float(),
        lightllm_output.float(),
        rtol=0.08,
        atol=0.08,
    )
    print("GLM-5 SGLang MoE compatibility: PASS")

    if args.benchmark:
        lightllm_ms = _graph_ms(run_lightllm, hidden_states, args.iterations)
        sglang_ms = _graph_ms(run_sglang, hidden_states, args.iterations)
        print("graph_ms lightllm=%.6f sglang=%.6f speedup=%.3fx" % (lightllm_ms, sglang_ms, lightllm_ms / sglang_ms))

    if args.tune_configs or args.tune_tma_configs:
        # GLM-5 decode has few physical tokens (48 for the main model and 8
        # for each draft) spread over 289 experts.  SGLang's generic block-FP8
        # fallback uses BLOCK_SIZE_M=64, which can pad this sparse workload
        # excessively.  Search the small-M region instead of the generic
        # 1920-config tuning space.
        search_space = itertools.product(
            (16, 32),
            (64, 128),
            (128,) if args.tune_tma_configs else (64, 128),
            (1, 16),
            (4, 8),
            (2, 3),
        )
        candidates = [
            {
                "BLOCK_SIZE_M": block_m,
                "BLOCK_SIZE_N": block_n,
                "BLOCK_SIZE_K": block_k,
                "GROUP_SIZE_M": group_m,
                "num_warps": num_warps,
                "num_stages": num_stages,
                **({"USE_TMA": True} if args.tune_tma_configs else {}),
            }
            for block_m, block_n, block_k, group_m, num_warps, num_stages in search_space
        ]
        default_config = {
            "BLOCK_SIZE_M": 64,
            "BLOCK_SIZE_N": 128,
            "BLOCK_SIZE_K": 128,
            "GROUP_SIZE_M": 32,
            "num_warps": 4,
            "num_stages": 3,
        }
        candidates.append(default_config)

        results = []
        for config in candidates:
            try:
                with override_config(config):
                    graph_ms = _graph_ms(run_sglang, hidden_states, args.iterations)
            except Exception as exc:
                print("config_failed=%s error=%r" % (json.dumps(config, sort_keys=True), exc))
                continue
            results.append((graph_ms, config))
            print("config_ms=%.6f config=%s" % (graph_ms, json.dumps(config, sort_keys=True)))

        results.sort(key=lambda item: item[0])
        if not results:
            raise RuntimeError("all SGLang MoE tuning candidates failed")
        print("top_configs=%s" % json.dumps(results[:10], sort_keys=True))
        print("best_config=%s" % json.dumps(results[0][1], sort_keys=True))
        print("best_graph_ms=%.6f" % results[0][0])


if __name__ == "__main__":
    main()
