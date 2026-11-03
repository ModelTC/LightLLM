"""Compare dense-mask and candidate-only FP8 indexers on the same paged cache."""

import argparse
import json
import importlib.util
import statistics

import deep_gemm
import torch
from triton.testing import do_bench_cudagraph

from lightllm.models.deepseek_v41.triton_kernel.attention import candidate_indexer_logits, select_indexer_topk


def benchmark(context, rows, graph_capacity=None, mixed=False, score_fn=candidate_indexer_logits):
    capacity = ((graph_capacity or context) + 63) // 64 * 64
    device = "cuda"
    count = context // 64
    storage = torch.empty((count, 64 * 132), dtype=torch.uint8, device=device)
    keys = torch.randn(count, 64, 128, device=device).to(torch.float8_e4m3fn)
    storage[:, : 64 * 128].copy_(keys.view(torch.uint8).flatten(1))
    storage[:, 64 * 128 :].view(torch.float32).fill_(1)
    cache = storage.view(count, 64, 1, 132)
    pages = torch.randperm(count, dtype=torch.int32, device=device).expand(rows, -1).contiguous()
    query = torch.randn(rows, 32, 128, device=device).to(torch.float8_e4m3fn)
    weights = torch.rand(rows, 32, device=device)
    lengths = torch.tensor(
        [([0, min(1024, context), min(8192, context), context][row % 4] if mixed else context) for row in range(rows)],
        dtype=torch.int32,
        device=device,
    )
    width = min(2048, (capacity + 7) // 8)
    candidates = torch.full((rows, width), -1, dtype=torch.int32, device=device)
    for row, length in enumerate(lengths.tolist()):
        chosen = torch.randperm((length + 7) // 8, device=device)[:width].sort().values
        candidates[row, : chosen.numel()] = chosen
    columns = (candidates[:, :, None] * 8 + torch.arange(8, device=device)).flatten(1).long()
    valid = (candidates[:, :, None] >= 0).expand(-1, -1, 8).flatten(1) & (columns < lengths[:, None])
    columns = columns.clamp_min(0)
    mask = torch.zeros(rows, capacity, dtype=torch.int32, device=device)
    mask.scatter_add_(1, columns, valid.int())
    mask = mask.bool()
    schedule = deep_gemm.get_paged_mqa_logits_metadata(lengths[:, None], 64, deep_gemm.get_num_sms())
    slots = torch.empty(rows, 512, dtype=torch.int32, device=device)
    selected = torch.empty_like(lengths)

    def dense_scores():
        return deep_gemm.fp8_paged_mqa_logits(
            query[:, None],
            cache,
            weights,
            lengths[:, None],
            pages,
            schedule,
            capacity,
            False,
        )

    def sparse_scores():
        return score_fn(query, cache, weights, pages, candidates, lengths)

    reference = dense_scores().gather(1, columns)
    reference.masked_fill_(~valid, -float("inf"))
    torch.testing.assert_close(sparse_scores(), reference, atol=0.03, rtol=0.01)

    def dense():
        logits = dense_scores()
        logits.masked_fill_(~mask, -float("inf"))
        select_indexer_topk(logits, lengths, pages, slots, selected)

    def sparse():
        logits = sparse_scores()
        select_indexer_topk(logits, lengths, pages, slots, selected, candidate_blocks=candidates)

    result = {
        "context": context,
        "rows": rows,
        "graph_capacity": graph_capacity or context,
        "logits_capacity": capacity,
        "mixed": mixed,
        "dense_logits_bytes": rows * capacity * 4,
    }
    result["candidate_logits_bytes"] = rows * candidates.shape[1] * 8 * 4
    for name, fn in (
        ("dense_score", dense_scores),
        ("candidate_score", sparse_scores),
        ("dense_total", dense),
        ("candidate_total", sparse),
    ):
        fn()
        result[name + "_ms"] = statistics.median(do_bench_cudagraph(fn, rep=50) for _ in range(3))
    result["speedup"] = result["dense_total_ms"] / result["candidate_total_ms"]
    if capacity <= 16384:
        # Production keeps DeepGEMM when the candidate set contains every key.
        def selected_path():
            select_indexer_topk(dense_scores(), lengths, pages, slots, selected)

        result["deployed_total_ms"] = statistics.median(do_bench_cudagraph(selected_path, rep=50) for _ in range(3))
    else:
        result["deployed_total_ms"] = result["candidate_total_ms"]
    result["deployed_speedup"] = result["dense_total_ms"] / result["deployed_total_ms"]
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--contexts", type=int, nargs="+", default=[8192, 65536, 262144, 1048576])
    parser.add_argument("--rows", type=int, nargs="+", default=[1, 8, 64, 128, 384])
    parser.add_argument("--graph-capacity", type=int)
    parser.add_argument("--mixed", action="store_true")
    parser.add_argument(
        "--baseline-module",
        help="Saved attention.py for an identical-shape before/after comparison",
    )
    args = parser.parse_args()
    score_fn = candidate_indexer_logits
    if args.baseline_module:
        spec = importlib.util.spec_from_file_location("candidate_baseline", args.baseline_module)
        baseline = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(baseline)
        score_fn = baseline.candidate_indexer_logits
    torch.manual_seed(410)
    print(json.dumps({"gpu": torch.cuda.get_device_name(), "torch": torch.__version__}), flush=True)
    for context in args.contexts:
        for rows in args.rows:
            print(
                json.dumps(benchmark(context, rows, args.graph_capacity, args.mixed, score_fn)),
                flush=True,
            )
