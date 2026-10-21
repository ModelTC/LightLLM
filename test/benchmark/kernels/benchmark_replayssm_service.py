"""Compare streamed requests on baseline and optimized services with identical warmup.

Reports request timings and token IDs for an exact cross-revision comparison.
"""

import argparse
import concurrent.futures
import json
import statistics
import time
import threading
from pathlib import Path

import requests
from transformers import AutoTokenizer

p = argparse.ArgumentParser()
p.add_argument("--output", required=True)
p.add_argument("--batch", type=int, nargs="+", default=[1, 8, 16])
p.add_argument("--repeats", type=int, default=3)
p.add_argument("--tokens", type=int, default=64)
p.add_argument("--warmups", type=int, default=1)
p.add_argument("--model-dir", required=True)
p.add_argument("--url", default="http://127.0.0.1:18785/generate_stream")
args = p.parse_args()
url = args.url
tok = AutoTokenizer.from_pretrained(args.model_dir, trust_remote_code=True)
prefix = tok.encode(
    "Describe how a computer executes a program, from source code to machine instructions. "
    "Explain the role of memory, processors, and operating systems. " * 100
)[:512]


def prepare_prompt(index, rep):
    suffix = tok.encode(
        f"Example number {rep * 1000 + index}. Discuss this scenario in detail. "
        + ("An engineer measures performance and validates numerical correctness. " * 100)
    )[:512]
    return prefix + suffix


def stream_request(index, rep, ids, barrier, cohort_start):
    payload = {
        "inputs": ids,
        "parameters": {
            "do_sample": False,
            "disable_prompt_cache": True,
            "ignore_eos": True,
            "max_new_tokens": args.tokens,
        },
    }
    barrier.wait()
    start = time.perf_counter()
    times = []
    tokens = []
    texts = []
    finished = False
    with requests.post(url, json=payload, stream=True, timeout=300) as response:
        response.raise_for_status()
        # Consume each HTTP chunk; byte-sized iteration can saturate the client CPU.
        for line in response.iter_lines(chunk_size=None):
            if not line.startswith(b"data:"):
                continue
            item = json.loads(line[5:])
            assert "error" not in item, item
            token = item.get("token", {})
            if token.get("id") is not None:
                times.append(time.perf_counter() - cohort_start)
                tokens.append(token["id"])
                texts.append(token.get("text", ""))
            finished = item.get("finished", finished)
    assert finished and len(tokens) == args.tokens, (finished, len(tokens))
    return dict(
        index=index,
        rep=rep,
        prompt_len=len(ids),
        token_ids=tokens,
        text="".join(texts),
        token_times_s=times,
        ttft_ms=(times[0] - (start - cohort_start)) * 1000,
        tpot_ms=(times[-1] - times[0]) * 1000 / max(1, len(times) - 1),
        elapsed_s=time.perf_counter() - start,
    )


def run_concurrent_requests(batch, rep):
    prompts = [prepare_prompt(i, rep) for i in range(batch)]
    barrier = threading.Barrier(batch)
    start = time.perf_counter()
    with concurrent.futures.ThreadPoolExecutor(max_workers=batch) as pool:
        samples = list(pool.map(lambda i: stream_request(i, rep, prompts[i], barrier, start), range(batch)))
    elapsed = time.perf_counter() - start
    begin = max(r["token_times_s"][0] for r in samples) + 0.5
    end = min(r["token_times_s"][-1] for r in samples) - 0.5
    count = sum(sum(begin <= t <= end for t in r["token_times_s"]) for r in samples)
    return dict(
        batch=batch,
        rep=rep,
        elapsed_s=elapsed,
        tokens_per_s=sum(len(r["token_ids"]) for r in samples) / elapsed,
        steady_tokens_per_s=count / (end - begin) if end > begin else None,
        steady_window_s=max(0.0, end - begin),
        ttft_ms=statistics.median(r["ttft_ms"] for r in samples),
        tpot_ms=statistics.median(r["tpot_ms"] for r in samples),
        requests=samples,
    )


# Warm every measured graph bucket before timing, with the same requests on both revisions.
for batch in args.batch:
    for warm in range(args.warmups):
        run_concurrent_requests(batch, 99 + warm)
rows = []
for batch in args.batch:
    for rep in range(args.repeats):
        row = run_concurrent_requests(batch, rep)
        rows.append(row)
        Path(args.output).write_text(json.dumps(rows, indent=2))
        print(json.dumps({k: v for k, v in row.items() if k != "requests"}), flush=True)
