#!/usr/bin/env python3

import argparse

import torch
import triton

from lightllm.common.basemodel.triton_kernel.fused_moe.grouped_topk import (
    single_group_sigmoid_topk_kernel,
    single_group_sigmoid_topk_bitonic_kernel,
    triton_grouped_topk,
)


def torch_reference(gating_output, correction_bias, topk):
    scores = gating_output.float().sigmoid()
    choice_scores = scores + correction_bias
    topk_ids = torch.topk(choice_scores, k=topk, dim=-1, largest=True, sorted=True).indices
    topk_weights = torch.gather(scores, 1, topk_ids)
    topk_weights /= topk_weights.sum(dim=-1, keepdim=True)
    return topk_weights, topk_ids


def run_topk(gating_output, correction_bias, *, fast):
    return triton_grouped_topk(
        hidden_states=None,
        gating_output=gating_output,
        correction_bias=correction_bias,
        topk=8,
        renormalize=True,
        num_expert_group=1,
        topk_group=1,
        scoring_func="sigmoid",
        group_score_used_topk_num=1,
        use_single_group_fast_path=fast,
    )


def run_fast_with_warps(gating_output, correction_bias, num_warps):
    tokens, experts = gating_output.shape
    topk_weights = torch.empty((tokens, 8), dtype=torch.float32, device="cuda")
    topk_ids = torch.empty((tokens, 8), dtype=torch.long, device="cuda")
    single_group_sigmoid_topk_kernel[(tokens,)](
        gating_output,
        gating_output.stride(0),
        correction_bias,
        topk_weights,
        topk_weights.stride(0),
        topk_ids,
        topk_ids.stride(0),
        experts,
        HAS_CORRECTION_BIAS=True,
        EXPERT_BLOCK_SIZE=triton.next_power_of_2(experts),
        TOPK_BLOCK_SIZE=8,
        TOPK_NUM=8,
        RENORMALIZE=True,
        num_warps=num_warps,
        num_stages=1,
    )
    return topk_weights, topk_ids


def run_scratch_free_bitonic(gating_output, correction_bias):
    tokens, experts = gating_output.shape
    topk_weights = torch.empty((tokens, 8), dtype=torch.float32, device="cuda")
    topk_ids = torch.empty((tokens, 8), dtype=torch.long, device="cuda")
    single_group_sigmoid_topk_bitonic_kernel[(tokens,)](
        gating_output,
        gating_output.stride(0),
        correction_bias,
        topk_weights,
        topk_weights.stride(0),
        topk_ids,
        topk_ids.stride(0),
        experts,
        HAS_CORRECTION_BIAS=True,
        EXPERT_BLOCK_SIZE=triton.next_power_of_2(experts),
        TOPK_NUM=8,
        RENORMALIZE=True,
        num_warps=4,
        num_stages=1,
    )
    return topk_weights, topk_ids


def assert_correct(tokens):
    generator = torch.Generator(device="cuda").manual_seed(20260828 + tokens)
    gating_output = torch.randn((tokens, 288), generator=generator, dtype=torch.float32, device="cuda")
    correction_bias = torch.randn((288,), generator=generator, dtype=torch.float32, device="cuda")

    ref_weights, ref_ids = torch_reference(gating_output, correction_bias, 8)
    fast_weights, fast_ids = run_topk(gating_output, correction_bias, fast=True)
    generic_weights, generic_ids = run_topk(gating_output, correction_bias, fast=False)

    torch.testing.assert_close(fast_ids, ref_ids, rtol=0, atol=0)
    torch.testing.assert_close(generic_ids, ref_ids, rtol=0, atol=0)
    torch.testing.assert_close(fast_weights, ref_weights, rtol=1e-5, atol=1e-6)
    torch.testing.assert_close(generic_weights, ref_weights, rtol=1e-5, atol=1e-6)
    weight_delta = (fast_weights - generic_weights).abs()
    print(
        f"tokens={tokens}: exact expert ids, weights match reference; "
        f"fast/generic bitwise={torch.equal(fast_weights, generic_weights)} "
        f"max_delta={weight_delta.max().item():.9g}"
    )
    return gating_output, correction_bias


def capture_graph(fn):
    for _ in range(3):
        outputs = fn()
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        outputs = fn()
    return graph, outputs


def graph_ms(graph, iterations):
    for _ in range(20):
        graph.replay()
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(iterations):
        graph.replay()
    end.record()
    end.synchronize()
    return start.elapsed_time(end) / iterations


def benchmark(tokens, iterations):
    gating_output, correction_bias = assert_correct(tokens)
    fast_graph, fast_outputs = capture_graph(lambda: run_topk(gating_output, correction_bias, fast=True))
    generic_graph, generic_outputs = capture_graph(lambda: run_topk(gating_output, correction_bias, fast=False))
    bitonic_graph, bitonic_outputs = capture_graph(lambda: run_scratch_free_bitonic(gating_output, correction_bias))
    fast_ms = graph_ms(fast_graph, iterations)
    generic_ms = graph_ms(generic_graph, iterations)
    bitonic_ms = graph_ms(bitonic_graph, iterations)
    torch.testing.assert_close(fast_outputs[1], generic_outputs[1], rtol=0, atol=0)
    torch.testing.assert_close(bitonic_outputs[0], generic_outputs[0], rtol=0, atol=0)
    torch.testing.assert_close(bitonic_outputs[1], generic_outputs[1], rtol=0, atol=0)
    speedup = generic_ms / fast_ms
    print(
        f"tokens={tokens}: graph fast={fast_ms:.6f} ms "
        f"bitonic={bitonic_ms:.6f} ms generic={generic_ms:.6f} ms "
        f"speedup={speedup:.2f}x bitonic_speedup={generic_ms / bitonic_ms:.2f}x"
    )
    return speedup


def tune_warps(tokens, iterations):
    gating_output, correction_bias = assert_correct(tokens)
    ref_weights, ref_ids = torch_reference(gating_output, correction_bias, 8)
    generic_weights, _ = run_topk(gating_output, correction_bias, fast=False)
    results = []
    for num_warps in (1, 2, 4, 8, 16):
        graph, outputs = capture_graph(
            lambda num_warps=num_warps: run_fast_with_warps(gating_output, correction_bias, num_warps)
        )
        graph.replay()
        torch.cuda.synchronize()
        if not torch.equal(outputs[1], ref_ids) or not torch.allclose(outputs[0], ref_weights, rtol=1e-5, atol=1e-6):
            print(f"tokens={tokens}: num_warps={num_warps} INVALID")
            continue
        elapsed_ms = graph_ms(graph, iterations)
        results.append((elapsed_ms, num_warps))
        max_delta = (outputs[0] - generic_weights).abs().max().item()
        print(
            f"tokens={tokens}: num_warps={num_warps} "
            f"graph={elapsed_ms:.6f} ms "
            f"generic_bitwise={torch.equal(outputs[0], generic_weights)} "
            f"max_delta={max_delta:.9g}"
        )
    best_ms, best_warps = min(results)
    print(f"tokens={tokens}: best num_warps={best_warps} graph={best_ms:.6f} ms")


def test_glm5_single_group_topk():
    if not torch.cuda.is_available():
        return
    for tokens in (1, 8, 48, 256):
        assert_correct(tokens)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--benchmark", action="store_true")
    parser.add_argument("--tune-warps", action="store_true")
    parser.add_argument("--iterations", type=int, default=5000)
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise SystemExit("CUDA is required")
    if args.tune_warps:
        for tokens in (1, 8, 48, 256):
            tune_warps(tokens, args.iterations)
    elif args.benchmark:
        for tokens in (1, 8, 48, 256):
            benchmark(tokens, args.iterations)
    else:
        test_glm5_single_group_topk()


if __name__ == "__main__":
    main()
