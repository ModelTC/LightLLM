"""Fixed-token GLM-5.3 Flash serving benchmark (TTFT and decode measured separately)."""

import argparse
import concurrent.futures
import json
import random
import statistics
import time
from pathlib import Path

import requests
from lightllm.server.tokenizer import get_tokenizer


def generate(url, prompt, output_tokens):
    start = time.perf_counter()
    arrivals = []
    ids = []
    with requests.post(
        url.rstrip("/") + "/generate_stream",
        json={
            "inputs": prompt,
            "parameters": {
                "do_sample": False,
                "ignore_eos": True,
                "max_new_tokens": output_tokens,
            },
        },
        stream=True,
        timeout=600,
    ) as response:
        response.raise_for_status()
        for line in response.iter_lines(chunk_size=1):
            if not line.startswith(b"data:"):
                continue
            event = json.loads(line[5:])
            if "token" not in event:
                raise RuntimeError(event)
            arrivals.append(time.perf_counter())
            ids.append(event["token"]["id"])
    if len(ids) != output_tokens:
        raise RuntimeError(f"Expected {output_tokens} output tokens, got {len(ids)}")
    decode_seconds = arrivals[-1] - arrivals[0]
    return {
        "input_tokens": len(prompt),
        "output_tokens": len(ids),
        "ttft_ms": (arrivals[0] - start) * 1000,
        "tpot_ms": decode_seconds / (len(ids) - 1) * 1000,
        "decode_tokens_per_second": (len(ids) - 1) / decode_seconds,
        "latency_seconds": arrivals[-1] - start,
        "token_ids": ids,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", default="http://127.0.0.1:18153")
    parser.add_argument("--model-dir", default="/mtc/models/GLM-5.3-Flash")
    parser.add_argument("--input-tokens", type=int, nargs="+", default=[1024, 4096])
    parser.add_argument("--concurrency", type=int, nargs="+", default=[1, 4, 8])
    parser.add_argument("--output-tokens", type=int, default=128)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--warmup-rounds", type=int, default=2)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output_tokens < 2:
        parser.error("--output-tokens must be at least 2 to measure decode")
    if args.seed is None:
        args.seed = time.time_ns()
    tokenizer = get_tokenizer(args.model_dir)
    content = tokenizer.encode(
        "The following document describes a language model inference service. "
        "Explain its performance clearly and continue the discussion. ",
        add_special_tokens=False,
    )
    rng = random.Random(args.seed)

    def prompt(length):
        # Vary the FIRST tokens, including warmups, so radix hits cannot
        # inflate prefill throughput while prompt caching remains enabled.
        return [rng.randrange(1000, 100000) for _ in range(min(length, 16))] + (content * (length // len(content) + 1))[
            : max(0, length - 16)
        ]

    report = {"settings": {**vars(args), "output": str(args.output)}, "results": []}
    for length in args.input_tokens:
        for concurrency in args.concurrency:
            with concurrent.futures.ThreadPoolExecutor(max_workers=concurrency) as pool:
                for _ in range(args.warmup_rounds):
                    list(pool.map(lambda p: generate(args.url, p, 32), [prompt(length) for _ in range(concurrency)]))
                for repeat in range(args.repeats):
                    prompts = [prompt(length) for _ in range(concurrency)]
                    start = time.perf_counter()
                    rows = list(pool.map(lambda p: generate(args.url, p, args.output_tokens), prompts))
                    elapsed = time.perf_counter() - start
                    result = {
                        "input_tokens": length,
                        "concurrency": concurrency,
                        "repeat": repeat,
                        "ttft_ms_median": statistics.median(r["ttft_ms"] for r in rows),
                        "tpot_ms_median": statistics.median(r["tpot_ms"] for r in rows),
                        "per_request_decode_tps_median": statistics.median(r["decode_tokens_per_second"] for r in rows),
                        "output_tps_including_prefill": concurrency * args.output_tokens / elapsed,
                        "elapsed_seconds": elapsed,
                        "requests": rows,
                    }
                    report["results"].append(result)
                    args.output.parent.mkdir(parents=True, exist_ok=True)
                    args.output.write_text(json.dumps(report, indent=2))
                    print(json.dumps({k: v for k, v in result.items() if k != "requests"}), flush=True)


if __name__ == "__main__":
    main()
