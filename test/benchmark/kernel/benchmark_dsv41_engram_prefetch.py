"""Synthetic overlap: a preceding BF16 projection plus an FP8 host/UVA Engram lookup.

This is a component benchmark, not a model TPOT benchmark. Hashes rotate on
every graph replay to avoid repeatedly reading the same small set of host rows.
"""

import json
import os
import statistics
import uuid
from types import SimpleNamespace

import torch
from triton.testing import do_bench_cudagraph

from lightllm.models.deepseek_v41.engram import EngramEmbeddingWeight
from lightllm.models.deepseek_v41.infer_struct import DeepseekV41InferStateInfo


if __name__ == "__main__":
    for name in (
        "GLOBAL_RANK",
        "CURRENT_RANK_IN_DP",
        "CURRENT_DEVICE_ID",
        "CURRENT_RANK_IN_NODE",
        "GLOBAL_DP_RANK",
        "DP_RANK_IN_NODE",
    ):
        os.environ["LIGHTLLM_" + name] = "0"
    for name in ("GLOBAL_WORLD_SIZE", "DP_WORLD_SIZE", "NODE_WORLD_SIZE", "DP_SIZE"):
        os.environ["LIGHTLLM_" + name] = "1"
    os.environ["LIGHTLLM_UNIQUE_SERVICE_NAME_ID"] = "bench_v41_prefetch_" + uuid.uuid4().hex
    torch.manual_seed(410)
    table_rows, dim, heads, hidden = 1 << 20, 256, 24, 5120
    embedding = EngramEmbeddingWeight("layers.1.engram.embed", table_rows, dim)
    embedding.weight.copy_(torch.randn(table_rows, dim).to(torch.float8_e4m3fn))
    embedding.scale.fill_(0.5)
    projection = torch.randn(hidden, hidden, dtype=torch.bfloat16, device="cuda")
    print(
        json.dumps(
            dict(
                gpu=torch.cuda.get_device_name(),
                table_bytes=table_rows * (dim + dim // 32),
                heads=heads,
                dim=dim,
                hidden=hidden,
                workload="synthetic projection + UVA lookup",
            )
        ),
        flush=True,
    )
    for tokens in (1, 6, 64, 384):
        inputs = torch.randn(tokens, hidden, dtype=torch.bfloat16, device="cuda")
        output = torch.empty_like(inputs)
        hashes = torch.randint(table_rows, (tokens, heads), device="cuda")
        state = DeepseekV41InferStateInfo()
        state.engram_hashes = hashes[:, None, :]
        state.engram_stream = torch.cuda.Stream()
        weight = SimpleNamespace(engram_embed_=embedding)

        def advance():
            hashes.add_(65537).remainder_(table_rows)

        def serial():
            advance()
            torch.mm(inputs, projection, out=output)
            return embedding.lookup(hashes)

        def prefetch():
            advance()
            state.prefetch_engram(weight, 0)
            torch.mm(inputs, projection, out=output)
            local, ready = state.engram_prefetch
            state.engram_prefetch = None
            current = torch.cuda.current_stream()
            current.wait_event(ready)
            local.record_stream(current)
            return embedding.lookup(hashes, local_embedding=local)

        actual = prefetch()
        torch.testing.assert_close(actual, embedding.lookup(hashes), atol=0, rtol=0)
        timings = {}
        for name, func in (("serial", serial), ("prefetch", prefetch)):
            func()
            timings[name] = statistics.median(do_bench_cudagraph(func, rep=100) for _ in range(3))
        print(
            json.dumps(dict(tokens=tokens, milliseconds=timings, speedup=timings["serial"] / timings["prefetch"])),
            flush=True,
        )
