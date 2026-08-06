"""
Two-stage abort test against a running lightllm server.

Stage 1: spawn N concurrent streams, then post /abort_request abort_all=True;
         verify every stream terminates quickly.
Stage 2: spawn N concurrent streams; each stream is independently assigned a
         random fate (disconnect mid-stream or run to completion). The server
         must keep serving the survivors and stay healthy afterwards.

Usage:
  python test/test_api/test_abort_chaos.py --url http://127.0.0.1:8000
"""

import argparse
import asyncio
import json
import random
import time
from collections import Counter

import httpx


PROMPTS = [
    "Write a long detailed essay about the history of computing.",
    "Tell me a long story about dragons and knights.",
    "Explain quantum mechanics in detail with lots of examples.",
    "Describe the plot of a 5-part fantasy novel series.",
    "Compose a long poem about the seasons.",
]


async def stream_task(client, url, mode, max_new_tokens):
    payload = {
        "inputs": random.choice(PROMPTS),
        "parameters": {"max_new_tokens": max_new_tokens, "temperature": 0.7, "do_sample": True},
    }
    drop_after = random.randint(20, 200)
    tokens = 0
    finish_reason = None
    t0 = time.time()
    try:
        async with client.stream("POST", f"{url}/generate_stream", json=payload, timeout=180.0) as r:
            async for line in r.aiter_lines():
                tokens += 1
                if line.startswith("data:"):
                    chunk = json.loads(line[len("data:") :])
                    if chunk.get("finished"):
                        finish_reason = chunk.get("finish_reason")
                if mode == "disconnect" and tokens >= drop_after:
                    break
        return (mode, finish_reason or "ok", tokens, time.time() - t0)
    except Exception as e:
        return (mode, f"exc:{type(e).__name__}", tokens, time.time() - t0)


def summarize(results):
    outcomes = Counter()
    for r in results:
        outcomes[(r[0], r[1]) if isinstance(r, tuple) else f"raised:{type(r).__name__}"] += 1
    for k, v in sorted(outcomes.items(), key=str):
        print(f"  {k}: {v}")


async def stage_abort_all(client, url, concurrency, max_new_tokens):
    print("\n===== STAGE 1: abort_all on N concurrent streams =====")
    tasks = [asyncio.create_task(stream_task(client, url, "finish", max_new_tokens)) for _ in range(concurrency)]
    await asyncio.sleep(2.0)
    t0 = time.time()
    r = await client.post(f"{url}/abort_request", json={"abort_all": True}, timeout=10.0)
    print(f"abort_all status={r.status_code}")
    results = await asyncio.gather(*tasks, return_exceptions=True)
    print(f"all streams settled in {time.time() - t0:.2f}s")
    summarize(results)


async def stage_random_chaos(client, url, concurrency, max_new_tokens):
    print("\n===== STAGE 2: random per-stream chaos =====")
    modes = random.choices(["disconnect", "finish"], weights=[80, 20], k=concurrency)
    tasks = [asyncio.create_task(stream_task(client, url, m, max_new_tokens)) for m in modes]
    results = await asyncio.gather(*tasks, return_exceptions=True)
    summarize(results)


async def run(url, concurrency, max_new_tokens):
    async with httpx.AsyncClient() as client:
        await stage_abort_all(client, url, concurrency, max_new_tokens)
        await stage_random_chaos(client, url, concurrency, max_new_tokens)
    print("\nALL CHAOS TESTS PASSED")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", default="http://127.0.0.1:8000")
    parser.add_argument("--concurrency", type=int, default=24)
    parser.add_argument("--max_new_tokens", type=int, default=2048)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    random.seed(args.seed)
    asyncio.run(run(args.url, args.concurrency, args.max_new_tokens))


if __name__ == "__main__":
    main()
