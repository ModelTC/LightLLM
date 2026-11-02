import argparse
import json
import os

import torch
import torch.nn.functional as F
import triton
from triton.testing import do_bench_cudagraph

from lightllm.common.qsa.indexer import QSAIndexer
from lightllm.common.qsa.attention import indexed_attention


def eager_select(indexer, query, cache, requests, lengths, table):
    blocks = cache.shape[1]
    count = lengths // indexer.compress_ratio

    scores = torch.matmul(query.float(), cache[0].float().transpose(-1, -2))
    scores = scores.relu().sum(1) / indexer.head_dim**0.5
    scores.masked_fill_(torch.arange(blocks, device=query.device)[None, :] >= count[:, None], -float("inf"))
    topk = min(blocks, indexer.token_budget // indexer.compress_ratio)
    selected = scores.topk(topk, sorted=False).indices
    width = indexer.token_budget + indexer.compress_ratio - 1
    col = torch.arange(width, device=query.device)[None, :].expand(query.shape[0], -1)
    chosen = count.clamp_max(indexer.token_budget // indexer.compress_ratio)
    full = col < chosen[:, None] * indexer.compress_ratio
    rank = (col // indexer.compress_ratio).clamp_max(topk - 1)
    position = selected.gather(1, rank) * indexer.compress_ratio + col % indexer.compress_ratio
    tail_col = col - chosen[:, None] * indexer.compress_ratio
    tail = (tail_col >= 0) & (tail_col < (lengths % indexer.compress_ratio)[:, None])
    position = torch.where(full, position, count[:, None] * indexer.compress_ratio + tail_col)
    slots = table[requests.long()[:, None], position.clamp(0, table.shape[1] - 1)]
    return slots.masked_fill(~(full | tail), -1)


def eager_attention(q, k, v, selected, valid_tokens=2048):
    from torch.nn.attention import SDPBackend, sdpa_kernel

    valid_slots = selected[:, :valid_tokens]
    gathered_k = k[valid_slots].transpose(1, 2)
    gathered_v = v[valid_slots].transpose(1, 2)
    with sdpa_kernel(SDPBackend.FLASH_ATTENTION):
        return F.scaled_dot_product_attention(q.unsqueeze(2), gathered_k, gathered_v, enable_gqa=True).squeeze(2)


def measure(reference, fused, repetitions):
    samples = {"torch": [], "kernel": []}
    for repeat in range(repetitions):
        functions = [("torch", reference), ("kernel", fused)]
        for name, function in functions[:: 1 if repeat % 2 == 0 else -1]:
            samples[name].append(do_bench_cudagraph(function, rep=100) * 1000)
    result = {name + "_us": sorted(values)[len(values) // 2] for name, values in samples.items()}
    return {**result, "speedup": result["torch_us"] / result["kernel_us"], "samples_us": samples}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--contexts", nargs="+", type=int, default=[4099, 65539])
    parser.add_argument("--index-rows", nargs="+", type=int, default=[1, 16])
    parser.add_argument("--attention-rows", nargs="+", type=int, default=[1, 16, 128])
    parser.add_argument("--valid-selections", nargs="+", type=int, default=[2048])
    parser.add_argument("--selection-bounds", action="store_true", help="Use native request visibility bounds")
    parser.add_argument("--repetitions", type=int, default=3)
    args = parser.parse_args()
    if not os.environ.get("EXP_ACTIVE"):
        parser.error('Wrap this benchmark with exp -m "purpose"')
    if any(count < 1 or count > 2048 for count in args.valid_selections):
        parser.error("Valid selection counts must lie in [1, 2048]")
    torch.manual_seed(537)

    def rand(*shape):
        return torch.randn(*shape, device="cuda", dtype=torch.bfloat16)

    indexer = QSAIndexer(rand(640, 2560), rand(128), rand(128))
    print(
        json.dumps(
            {
                "torch": torch.__version__,
                "triton": triton.__version__,
                "cuda": torch.version.cuda,
                "gpu": torch.cuda.get_device_name(),
                "dtype": "bfloat16",
                "index_heads": 4,
                "index_head_dim": 128,
                "token_budget": 2048,
                "compress_ratio": 4,
                "query_heads": 24,
                "kv_heads": 2,
                "attention_head_dim": 256,
                "timing": "CUDA graph, median of interleaved repetitions",
                "attention_baseline": "KV gather plus forced Torch FlashAttention, no padding mask",
                "selection_baseline": (
                    "Torch matmul with shared cached keys, ReLU head reduction, top-k and slot gather"
                ),
                "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
                "selection_bounds": args.selection_bounds,
            }
        ),
        flush=True,
    )
    for context in args.contexts:
        cache = rand(1, context // 4, 128)
        table = torch.arange(context, device="cuda", dtype=torch.int32).view(1, -1)
        for rows in args.index_rows:
            query = rand(rows, 4, 128)
            requests = torch.zeros(rows, device="cuda", dtype=torch.int32)
            lengths = (
                torch.full((rows,), context, device="cuda", dtype=torch.int32)
                - torch.arange(rows, device="cuda", dtype=torch.int32) % 7
            )

            def reference():
                return eager_select(indexer, query, cache, requests, lengths, table)

            def fused():
                return indexer.select(query, cache, requests, lengths, table)

            torch.testing.assert_close(fused().sort(-1).values, reference().sort(-1).values, atol=0, rtol=0)
            print(
                json.dumps(
                    {
                        "component": "cached_block_selection",
                        "queries": rows,
                        "context": context,
                        **measure(reference, fused, args.repetitions),
                    }
                ),
                flush=True,
            )
    slots = 65539
    key, value = rand(slots, 2, 256), rand(slots, 2, 256)
    for rows in args.attention_rows:
        query = rand(rows, 24, 256)
        choices = torch.stack([torch.randperm(slots, device="cuda")[:2051] for _ in range(rows)]).int()
        for count in args.valid_selections:
            selected = choices.clone()
            selected[:, count:] = -1

            bound = slots if count == 2048 else count
            lengths = torch.full((rows,), bound, device="cuda", dtype=torch.int32) if args.selection_bounds else None

            def reference():
                return eager_attention(query, key, value, selected, count)

            def fused():
                return indexed_attention(query, key, value, selected, selection_lengths=lengths)

            torch.testing.assert_close(fused(), reference(), atol=0.016, rtol=0.016)
            print(
                json.dumps(
                    {
                        "component": "indexed_gqa",
                        "queries": rows,
                        "physical_slots": slots,
                        "selected": count,
                        "selection_capacity": selected.shape[1],
                        **measure(reference, fused, args.repetitions),
                    }
                ),
                flush=True,
            )


if __name__ == "__main__":
    main()
