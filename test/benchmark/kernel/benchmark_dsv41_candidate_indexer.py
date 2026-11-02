"""Compare dense-mask and candidate-only FP8 indexers on the same paged cache."""

import argparse
import json
import statistics

import deep_gemm
import torch
from triton.testing import do_bench_cudagraph

from lightllm.models.deepseek_v41.triton_kernel.attention import candidate_indexer_logits, select_indexer_topk


def benchmark(context, rows):
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
    lengths = torch.full((rows,), context, dtype=torch.int32, device=device)
    candidates = torch.stack(
        [torch.randperm(context // 8, device=device)[: min(2048, context // 8)].sort().values for _ in range(rows)]
    ).int()
    columns = (candidates[:, :, None] * 8 + torch.arange(8, device=device)).flatten(1).long()
    mask = torch.zeros(rows, context, dtype=torch.bool, device=device).scatter_(1, columns, True)
    schedule = deep_gemm.get_paged_mqa_logits_metadata(lengths[:, None], 64, deep_gemm.get_num_sms())
    slots = torch.empty(rows, 512, dtype=torch.int32, device=device)
    selected = torch.empty_like(lengths)

    def dense_scores():
        return deep_gemm.fp8_paged_mqa_logits(
            query[:, None], cache, weights, lengths[:, None], pages, schedule, context, False
        )

    def sparse_scores():
        return candidate_indexer_logits(query, cache, weights, pages, candidates, lengths)

    reference = dense_scores().gather(1, columns)
    torch.testing.assert_close(sparse_scores(), reference, atol=0.03, rtol=0.01)

    def dense():
        logits = dense_scores()
        logits.masked_fill_(~mask, -float("inf"))
        select_indexer_topk(logits, lengths, pages, slots, selected)

    def sparse():
        logits = sparse_scores()
        select_indexer_topk(logits, lengths, pages, slots, selected, candidate_blocks=candidates)

    result = {"context": context, "rows": rows, "dense_logits_bytes": rows * context * 4}
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
    if context <= 16384:
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
    args = parser.parse_args()
    torch.manual_seed(410)
    print(json.dumps({"gpu": torch.cuda.get_device_name(), "torch": torch.__version__}), flush=True)
    for context in args.contexts:
        for rows in args.rows:
            print(json.dumps(benchmark(context, rows)), flush=True)
