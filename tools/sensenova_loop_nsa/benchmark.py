"""Compare DSA kernels against the exact source assembled by the step519 launcher.

Run on one GPU; this script never starts/stops a service or changes weights.
The JSON explicitly distinguishes kernel timing from full-model performance.
"""

import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import statistics
import sys
import types

import torch


def load_baseline(root):
    folder = Path(root) / "lightllm/models/sensenova_loop"
    package = types.ModuleType("_loop_dsa_baseline")
    package.__path__ = [str(folder)]
    sys.modules[package.__name__] = package
    kernels = types.ModuleType(package.__name__ + ".triton_kernel")
    kernels.__path__ = [str(folder / "triton_kernel")]
    sys.modules[kernels.__name__] = kernels
    from importlib import import_module

    return import_module(kernels.__name__ + ".dsa_perf")


def unpack_scores(keys):
    ordered = keys >> 20
    bits = torch.where(ordered < 0, -2147483648 - ordered, ordered).to(torch.int32)
    return bits.view(torch.float32).masked_fill(keys == torch.iinfo(torch.int64).min, -torch.inf)


def time_ms(fn, samples=3, repeat=20):
    # Graph replay excludes Python launch time; allocations are captured too.
    for _ in range(2):
        fn()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        output = fn()
    measurements = []
    for _ in range(samples):
        start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        start.record()
        for _ in range(repeat):
            graph.replay()
        end.record()
        end.synchronize()
        measurements.append(start.elapsed_time(end) / repeat)
    del output, graph
    return statistics.median(measurements)


def run_case(baseline, rows, keys, mixed, *, measure=True, query_tile=4):
    from lightllm.models.sensenova_loop.triton_kernel.dsa_perf import (
        ranked_scores,
        packed_topk,
        fa3_selected_attention,
    )
    from lightllm.models.sensenova_loop.triton_kernel.dsa_select import exact_float_topk

    torch.manual_seed(123 + rows + keys)
    # TP4: one K head, one V head, with the original padded 384-wide cache.
    kv = torch.randn(keys, 2, 384, device="cuda", dtype=torch.bfloat16)
    table = torch.stack((torch.arange(keys, device="cuda"), torch.arange(keys - 1, -1, -1, device="cuda"))).int()
    requests = (torch.arange(rows, device="cuda") % (2 if mixed else 1)).int()
    positions = torch.full((rows,), keys - 1, device="cuda", dtype=torch.int32)
    if mixed:
        positions[::4] = -1
        positions[1::4] = min(1234, keys - 1)
    lengths = (positions + 1).clamp_min(0).contiguous()
    iq = torch.randn(rows, 8, 128, device="cuda", dtype=torch.bfloat16)
    iw = torch.randn(rows, 8, device="cuda", dtype=torch.bfloat16)
    q = torch.randn(rows, 4, 256, device="cuda", dtype=torch.bfloat16)
    ik = kv[:, :1, 256:].contiguous()
    k, v = kv[:, :1, :256].contiguous(), kv[:, 1:, :256].contiguous()
    ref_args = (iq, iw, kv, table, requests, positions, keys, 256)
    new_args = (iq, iw, ik, table, requests, positions, keys, 0)
    rank = baseline.ranked_scores(*ref_args)
    scores = ranked_scores(*new_args, packed=False, query_tile=query_tile)
    expected_scores = unpack_scores(rank)
    torch.testing.assert_close(scores, expected_scores, rtol=0, atol=0)
    reference = baseline.packed_topk(rank, 2048)
    new_packed = packed_topk(ranked_scores(*new_args), 2048)
    torch.testing.assert_close(new_packed, reference, rtol=0, atol=0)
    result = {
        "rows": rows,
        "keys": keys,
        "mixed_requests": mixed,
        "query_tile": query_tile,
        "scores_exact": True,
        "selectors": {},
    }
    for backend in ("torch", "radix", "auto"):
        selected = exact_float_topk(scores, lengths, 2048, backend=backend)
        torch.testing.assert_close(selected, reference, rtol=0, atol=0)
        entry = {"ids_exact": True}
        if measure:
            entry["topk_graph_ms"] = time_ms(lambda: exact_float_topk(scores, lengths, 2048, backend=backend))
            entry["score_topk_graph_ms"] = time_ms(
                lambda: exact_float_topk(
                    ranked_scores(*new_args, packed=False, query_tile=query_tile), lengths, 2048, backend=backend
                )
            )
        result["selectors"][backend] = entry
    # Use the same FA3 binding on both sides, retaining the baseline's 0-row behavior.
    from lightllm.utils.sgl_utils import flash_attn_with_kvcache

    ref_out = baseline.fa3_selected_attention(q, kv, table, requests, reference, flash_func=flash_attn_with_kvcache)
    cu_q = torch.arange(rows + 1, dtype=torch.int32, device=q.device)
    new_out = fa3_selected_attention(q, k, v, table, requests, reference, flash_func=flash_attn_with_kvcache, cu_q=cu_q)
    torch.testing.assert_close(new_out, ref_out, rtol=0, atol=0)
    result["fa3_output_exact"] = True
    if measure:
        result["baseline_topk_graph_ms"] = time_ms(lambda: baseline.packed_topk(rank, 2048))
        result["baseline_score_topk_graph_ms"] = time_ms(
            lambda: baseline.packed_topk(baseline.ranked_scores(*ref_args), 2048)
        )
        result["baseline_score_workspace_bytes"] = rank.numel() * rank.element_size()
        result["fp32_score_workspace_bytes"] = scores.numel() * scores.element_size()
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline-code", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--query-tile", type=int, choices=(4, 8, 16), default=4)
    args = parser.parse_args()
    baseline = load_baseline(args.baseline_code)
    report = {
        "scope": "single-GPU kernels; CUDA Graph replay; shared machine; no end-to-end speed claim",
        "torch": torch.__version__,
        "gpu": torch.cuda.get_device_name(),
        "baseline_files": {},
        "cases": [],
    }
    for file in (args.baseline_code / "lightllm/models/sensenova_loop").rglob("*.py"):
        report["baseline_files"][str(file.relative_to(args.baseline_code))] = hashlib.sha256(
            file.read_bytes()
        ).hexdigest()
    cases = (
        [(4, 8192, False), (5, 8192, True)]
        if args.quick
        else [
            (1, 4096, False),
            (4, 65536, False),
            (5, 8192, True),
            (16, 32768, True),
            (64, 32768, False),
            (512, 65536, False),
            (4, 131072, False),
            (4, 524288, False),
        ]
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    for case in cases:
        result = run_case(baseline, *case, query_tile=args.query_tile if case[0] >= 32 else 4)
        report["cases"].append(result)
        args.output.write_text(json.dumps(report, indent=2) + "\n")
        print(json.dumps(result), flush=True)
        torch.cuda.empty_cache()
    report["complete"] = True
    args.output.write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
