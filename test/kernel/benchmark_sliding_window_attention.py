"""Compare paged lookup with direct sliding-window addressing on identical KV.

This is a warm-cache, single-layer microbenchmark, not a serving benchmark.
Preparation runs once per model forward, NOT once per attention layer; its
separate timing must not be multiplied by the model's sliding-layer count.
KV writes and window commits are excluded from both paths.

Example (run only on a GPU approved for benchmarking):
    python test/kernel/benchmark_sliding_window_attention.py --family both --output /tmp/sliding-attention.json
"""

import argparse
import gc
import json
import os
import statistics
import time
from pathlib import Path
from types import SimpleNamespace

import torch
import triton
import triton.language as tl

from lightllm.common.basemodel.triton_kernel.att.decode_att.gqa.flash_decoding.gqa_flash_decoding import (
    gqa_token_decode_attention_flash_decoding,
)
from lightllm.models.gemma4.triton_kernel.context_attention_fwd_gemma4_mm import context_attention_fwd_gemma4_mm


# Verified from the text_config of gemma-4-E4B-it (TP2) and gemma-4-31B-it (TP4).
MODEL_SHAPES = {
    "e4b": {"tp": 2, "q_heads": 4, "kv_heads": 1, "head_dim": 256, "window": 512},
    "31b": {"tp": 4, "q_heads": 8, "kv_heads": 4, "head_dim": 256, "window": 1024},
}


# Frozen pre-change preparation kernels. Keeping them here lets the benchmark
# remain usable after the second request-token table is removed from serving.
@triton.jit
def _legacy_prepare_history(
    mapping, req_ids, seq_lens, q_lens, stride_req, stride_seq, WINDOW: tl.constexpr, BLOCK: tl.constexpr
):
    batch, block = tl.program_id(0), tl.program_id(1)
    req_idx = tl.load(req_ids + batch)
    history_end = tl.load(seq_lens + batch) - tl.load(q_lens + batch)
    history_start = tl.maximum(0, history_end - WINDOW)
    positions = history_start + block * BLOCK + tl.arange(0, BLOCK)
    tl.store(
        mapping + req_idx * stride_req + positions * stride_seq,
        req_idx * WINDOW + positions % WINDOW,
        positions < history_end,
    )


@triton.jit
def _legacy_prepare_current(
    mapping,
    req_ids,
    seq_lens,
    q_lens,
    q_starts,
    stride_req,
    stride_seq,
    SCRATCH_START: tl.constexpr,
    BLOCK: tl.constexpr,
):
    batch, block = tl.program_id(0), tl.program_id(1)
    req_idx = tl.load(req_ids + batch)
    q_len = tl.load(q_lens + batch)
    history_end = tl.load(seq_lens + batch) - q_len
    q_start = tl.load(q_starts + batch)
    offsets = block * BLOCK + tl.arange(0, BLOCK)
    tl.store(
        mapping + req_idx * stride_req + (history_end + offsets) * stride_seq,
        SCRATCH_START + q_start + offsets,
        offsets < q_len,
    )


def _graph_timing(fn, unroll, samples):
    """Capture repeated calls so Python launch overhead is outside GPU timing."""
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        fn()
        fn()
        stream.synchronize()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph, stream=stream):
            for _ in range(unroll):
                fn()
        graph.replay()
        stream.synchronize()
        timings = []
        for _ in range(samples):
            start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            start.record()
            graph.replay()
            end.record()
            end.synchronize()
            timings.append(start.elapsed_time(end) * 1000 / unroll)
    return {"median_us": statistics.median(timings), "min_us": min(timings), "max_us": max(timings)}


def _eager_timing(fn, iterations, samples):
    """Report enqueue time separately from synchronized whole-call latency.

    Enqueue time includes Python, allocation and driver calls, and can include
    queue backpressure. It is not an isolated measurement of CPU computation.
    """
    enqueue, wall = [], []
    for _ in range(samples):
        torch.cuda.synchronize()
        start = time.perf_counter_ns()
        for _ in range(iterations):
            fn()
        submitted = time.perf_counter_ns()
        torch.cuda.synchronize()
        completed = time.perf_counter_ns()
        enqueue.append((submitted - start) / iterations / 1000)
        wall.append((completed - start) / iterations / 1000)
    return {"enqueue_median_us": statistics.median(enqueue), "wall_median_us": statistics.median(wall)}


def _accuracy(actual, expected, atol, rtol):
    # Limit validation workspace so the 8192-token cases do not temporarily
    # allocate several additional full-sized FP32 attention outputs.
    max_abs, squared_error, bitwise_equal = 0.0, 0.0, True
    for actual_chunk, expected_chunk in zip(actual.flatten().split(1 << 20), expected.flatten().split(1 << 20)):
        torch.testing.assert_close(actual_chunk, expected_chunk, atol=atol, rtol=rtol)
        difference = actual_chunk.float() - expected_chunk.float()
        max_abs = max(max_abs, difference.abs().max().item())
        squared_error += difference.square().sum().item()
        bitwise_equal = bitwise_equal and torch.equal(actual_chunk, expected_chunk)
    return {"max_abs": max_abs, "rms": (squared_error / actual.numel()) ** 0.5, "bitwise_equal": bitwise_equal}


@torch.inference_mode()
def _benchmark_case(family, phase, batch, q_len, args):
    shape = MODEL_SHAPES[family]
    q_heads, kv_heads, dim, window = (shape[name] for name in ["q_heads", "kv_heads", "head_dim", "window"])
    request_slots = batch + 1
    scratch_start = request_slots * window
    token_num = batch * q_len
    max_seq_len = args.history + 7 * (batch - 1) + q_len
    backing_bytes = (scratch_start + token_num) * 2 * kv_heads * dim * 2
    mapping_bytes = request_slots * max_seq_len * 4
    q_and_outputs_bytes = 3 * token_num * q_heads * dim * 2
    legacy_blocks = 128 if batch <= 16 else 64 if batch <= 64 else 32
    decode_workspace_bytes = batch * q_heads * legacy_blocks * (dim * 2 + 4) if phase == "decode" else 0
    validation_bytes = min(token_num * q_heads * dim, 1 << 20) * 16
    estimated_bytes = backing_bytes + mapping_bytes + q_and_outputs_bytes + decode_workspace_bytes + validation_bytes
    if estimated_bytes > args.max_case_mib * 1024 ** 2:
        raise ValueError(
            f"case needs at least {estimated_bytes / 1024 ** 2:.1f} MiB; increase --max-case-mib explicitly"
        )

    req_ids = torch.arange(batch, 0, -1, dtype=torch.int32, device="cuda")
    history_lens = args.history + torch.arange(batch, dtype=torch.int32, device="cuda") * 7
    seq_lens = history_lens + q_len
    q_lens = torch.full((batch,), q_len, dtype=torch.int32, device="cuda")
    q_starts = torch.arange(batch, dtype=torch.int32, device="cuda") * q_len
    mapping = torch.full((request_slots, max_seq_len), -1, dtype=torch.int32, device="cuda")
    # Both paths read precisely this tensor: only the address computation changes.
    backing = torch.randn((scratch_start + token_num, 2 * kv_heads, dim), dtype=torch.bfloat16, device="cuda")
    k, v = backing[:, :kv_heads], backing[:, kv_heads:]
    q = torch.randn((token_num, q_heads, dim), dtype=torch.bfloat16, device="cuda")
    old_out, new_out = torch.empty_like(q), torch.empty_like(q)
    image_ends = torch.zeros((token_num,), dtype=torch.int32, device="cuda")

    def prepare():
        _legacy_prepare_history[(batch, triton.cdiv(window, 256))](
            mapping, req_ids, seq_lens, q_lens, *mapping.stride(), WINDOW=window, BLOCK=256
        )
        _legacy_prepare_current[(batch, triton.cdiv(q_len, 256))](
            mapping, req_ids, seq_lens, q_lens, q_starts, *mapping.stride(), SCRATCH_START=scratch_start, BLOCK=256
        )

    if phase == "prefill":

        def legacy_attention():
            context_attention_fwd_gemma4_mm(
                q,
                k,
                v,
                old_out,
                req_ids,
                q_starts,
                seq_lens,
                history_lens,
                q_len,
                mapping,
                image_ends,
                sliding_window=(window - 1, 0),
            )

        def direct_attention():
            context_attention_fwd_gemma4_mm(
                q,
                k,
                v,
                new_out,
                req_ids,
                q_starts,
                seq_lens,
                history_lens,
                q_len,
                None,
                image_ends,
                sliding_window=(window - 1, 0),
                scratch_start=scratch_start,
            )

    else:
        from lightllm.models.gemma4.triton_kernel.sliding_window_decode import sliding_window_decode_attention

        infer_state = SimpleNamespace(
            batch_size=batch,
            req_manager=SimpleNamespace(req_to_token_indexs=mapping),
            b_req_idx=req_ids,
            b_seq_len=seq_lens,
            max_kv_seq_len=max_seq_len,
        )

        def legacy_attention():
            gqa_token_decode_attention_flash_decoding(q, infer_state, k, v, out=old_out, sliding_window=(window - 1, 0))

        def direct_attention():
            sliding_window_decode_attention(
                q,
                k,
                v,
                req_ids,
                seq_lens,
                q_starts,
                sliding_window=window,
                scratch_start=scratch_start,
                out=new_out,
            )

    def legacy_step():
        prepare()
        legacy_attention()

    legacy_step()
    direct_attention()
    torch.cuda.synchronize()
    accuracy = _accuracy(new_out, old_out, atol=args.atol, rtol=args.rtol)
    functions = {
        "legacy_prepare_once": prepare,
        "legacy_attention_only": legacy_attention,
        "legacy_prepare_plus_one_attention": legacy_step,
        "direct_attention_only": direct_attention,
    }
    graph = {name: _graph_timing(fn, args.graph_unroll, args.samples) for name, fn in functions.items()}
    eager = {name: _eager_timing(fn, args.eager_iterations, args.samples) for name, fn in functions.items()}
    return {
        "family": family,
        "phase": phase,
        "batch_size": batch,
        "q_len": q_len,
        "history_min": args.history,
        "history_max": max_seq_len - q_len,
        **shape,
        "backing_mib": backing_bytes / 1024 ** 2,
        "removed_index_table_mib": mapping_bytes / 1024 ** 2,
        "accuracy": accuracy,
        "graph": graph,
        "eager": eager,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--family", choices=["e4b", "31b", "both"], default="both")
    parser.add_argument("--phase", choices=["prefill", "decode", "both"], default="both")
    parser.add_argument("--device", type=int, default=0, help="logical CUDA device within CUDA_VISIBLE_DEVICES")
    parser.add_argument("--history", type=int, default=32768)
    parser.add_argument("--prefill-lengths", type=int, nargs="+", default=[512, 4096, 8192])
    parser.add_argument("--decode-batches", type=int, nargs="+", default=[1, 8, 32])
    parser.add_argument("--graph-unroll", type=int, default=16)
    parser.add_argument("--eager-iterations", type=int, default=20)
    parser.add_argument("--samples", type=int, default=5)
    parser.add_argument("--max-case-mib", type=float, default=256)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--atol", type=float, default=0.005)
    parser.add_argument("--rtol", type=float, default=0.02)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if (
        args.history < 0
        or min(args.prefill_lengths + args.decode_batches + [args.graph_unroll, args.eager_iterations, args.samples])
        < 1
    ):
        parser.error("history must be nonnegative; lengths, batches, unroll, iterations and samples must be positive")
    torch.cuda.set_device(args.device)
    torch.manual_seed(args.seed)
    report = {
        "scope": "Shared-GPU-capable warm-cache attention microbenchmark, not end-to-end serving throughput.",
        "notes": [
            "Both paths use identical ring/scratch KV, Q and metadata; only attention addressing differs.",
            "Legacy preparation is once per model forward, not once per attention layer.",
            "KV writes, window commit, linear/full layers, scheduler and CPU-cache operations are excluded.",
            "Eager enqueue time includes Python/allocator/driver calls and possible queue backpressure.",
            "CUDA graph timing removes host enqueue overhead; shared GPU contention can still affect results.",
        ],
        "gpu": torch.cuda.get_device_name(args.device),
        "torch": torch.__version__,
        "triton": triton.__version__,
        "cuda_visible_devices": os.getenv("CUDA_VISIBLE_DEVICES"),
        "autotune_level": os.getenv("LIGHTLLM_TRITON_AUTOTUNE_LEVEL", "0"),
        "arguments": {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()},
        "cases": [],
    }
    print(
        "family phase   batch     q   old-att graph-us  new-att graph-us  prepare graph-us  old-step graph-us",
        flush=True,
    )
    for family in MODEL_SHAPES if args.family == "both" else [args.family]:
        cases = []
        if args.phase in ["prefill", "both"]:
            cases.extend(("prefill", 1, length) for length in args.prefill_lengths)
        if args.phase in ["decode", "both"]:
            cases.extend(("decode", batch, 1) for batch in args.decode_batches)
        for phase, batch, q_len in cases:
            result = _benchmark_case(family, phase, batch, q_len, args)
            report["cases"].append(result)
            timings = result["graph"]
            print(
                f"{family:6} {phase:7} {batch:5} {q_len:5} "
                f"{timings['legacy_attention_only']['median_us']:18.3f} "
                f"{timings['direct_attention_only']['median_us']:17.3f} "
                f"{timings['legacy_prepare_once']['median_us']:17.3f} "
                f"{timings['legacy_prepare_plus_one_attention']['median_us']:18.3f}",
                flush=True,
            )
            gc.collect()
            torch.cuda.empty_cache()
    if args.output:
        args.output.write_text(json.dumps(report, indent=2) + "\n")
        print(f"JSON report: {args.output}", flush=True)
    else:
        print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
