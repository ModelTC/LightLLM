"""Synthetic overlap: a preceding BF16 projection plus an FP8 host/UVA Engram lookup.

This is a component benchmark, not a model TPOT benchmark. Hashes rotate on
every graph replay to avoid repeatedly reading the same small set of host rows.
"""

import json
import os
import statistics
import time
import uuid
from types import SimpleNamespace

import torch
from triton.testing import do_bench_cudagraph

from lightllm.models.deepseek_v41.engram import EngramEmbeddingWeight, EngramHasher
from lightllm.models.deepseek_v41.infer_struct import DeepseekV41InferStateInfo


def torch_hash(hasher, state):
    """The previous eager implementation, retained as a benchmark baseline."""
    positions, reqs = state.position_ids.long(), state.dsv4_sparse_req_idx.long()
    hasher.history[state.mem_index.long()] = hasher._map_token_ids(state.input_ids.long())
    blocked, tokens = torch.zeros_like(positions, dtype=torch.bool), []
    for shift in range(hasher.order):
        slot = state.req_manager.req_to_token_indexs[reqs, (positions - shift).clamp_min(0)].long()
        source = hasher.history[slot]
        blocked = blocked | (positions < shift) | (source == -1)
        tokens.append(torch.where(blocked, hasher.pad, source))
    products = torch.stack(tokens, -1)[:, None, :] * hasher.multipliers[None, :, :]
    rolling, hashes = products[..., 0], []
    for shift in range(1, hasher.order):
        rolling = torch.bitwise_xor(rolling, products[..., shift])
        hashes.append(rolling[..., None] % hasher.primes[:, (shift - 1) * hasher.heads : shift * hasher.heads])
    return torch.cat(hashes, -1) + hasher.offsets


def eager_ms(func):
    timings = []
    for _ in range(3):
        torch.cuda.synchronize()
        start = time.perf_counter()
        for _ in range(100):
            func()
        torch.cuda.synchronize()
        timings.append((time.perf_counter() - start) * 10)
    return statistics.median(timings)


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
    hasher = EngramHasher.__new__(EngramHasher)
    hasher.token_map = torch.arange(32768, device="cuda")
    hasher.has_vision, hasher.pad, hasher.order, hasher.heads = False, 2, 4, 8
    hasher.multipliers = torch.randint(1, (2 ** 63 - 1) // 32768 // 2, (2, 4), device="cuda") * 2 + 1
    hasher.primes = (1000003 + 2 * torch.arange(48, device="cuda")).view(2, 24)
    hasher.offsets = hasher.primes.cumsum(-1) - hasher.primes
    for tokens in (1, 6, 64, 384, 8192):
        hasher.history = torch.randint(32768, (tokens + 4,), device="cuda")
        table = torch.arange(tokens + 4, dtype=torch.int32, device="cuda")[None, :]
        position = torch.arange(4, tokens + 4, dtype=torch.int32, device="cuda")
        state = SimpleNamespace(
            input_ids=torch.randint(32768, (tokens,), device="cuda"),
            position_ids=position,
            dsv4_sparse_req_idx=torch.zeros(tokens, dtype=torch.int32, device="cuda"),
            mem_index=position,
            req_manager=SimpleNamespace(req_to_token_indexs=table),
        )
        torch.testing.assert_close(hasher(state), torch_hash(hasher, state), atol=0, rtol=0)
        graph_times, eager_times = {}, {}
        for name, func in (("previous", lambda: torch_hash(hasher, state)), ("fused", lambda: hasher(state))):
            func()
            graph_times[name] = statistics.median(do_bench_cudagraph(func, rep=100) for _ in range(3))
            eager_times[name] = eager_ms(func)
        print(json.dumps(dict(component="hash", tokens=tokens, graph_ms=graph_times, eager_ms=eager_times)), flush=True)

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
        all_hashes = torch.randint(table_rows, (tokens, 2, heads), device="cuda")
        hashes = all_hashes[:, 0]
        state = DeepseekV41InferStateInfo()
        state.engram_hashes = all_hashes
        state.engram_stream = torch.cuda.Stream()
        weight = SimpleNamespace(engram_embed_=embedding)

        def advance():
            all_hashes.add_(65537).remainder_(table_rows)

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
            json.dumps(
                dict(
                    component="lookup",
                    tokens=tokens,
                    milliseconds=timings,
                    speedup=timings["serial"] / timings["prefetch"],
                )
            ),
            flush=True,
        )
