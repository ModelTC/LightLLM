"""Small deterministic serving comparison; run through exp -m.

This checks generation and prefix-cache behavior, not a task-accuracy benchmark.
Use identical model, prefill backend and sampling settings on both endpoints.
"""
import argparse
import json
import urllib.request
import concurrent.futures
import time
import pathlib

parser = argparse.ArgumentParser()
parser.add_argument("--baseline-port", type=int, default=18932)
parser.add_argument("--replay-port", type=int, default=18934)
args = parser.parse_args()
prompts = [
    "The capital of France is",
    "Write a Python function to add two numbers.",
    "Count from one to twenty:",
    "Explain why the sky is blue in one paragraph.",
    "Here is some background. "
    + ("The quick brown fox jumps over the lazy dog. " * 90)
    + "\nSummarize this in a sentence:",
]


def run(arg):
    port, prompt = arg
    data = json.dumps(
        {"inputs": prompt, "parameters": {"max_new_tokens": 128, "do_sample": False, "return_details": True}}
    ).encode()
    start = time.monotonic()
    with urllib.request.urlopen(
        urllib.request.Request(
            f"http://127.0.0.1:{port}/generate", data=data, headers={"Content-Type": "application/json"}
        ),
        timeout=120,
    ) as r:
        res = json.load(r)
    return {
        "port": port,
        "tokens": [t["id"] for t in res["tokens"]],
        "text": res["generated_text"][0],
        "prompt_tokens": res["prompt_tokens"],
        "cache_len": res["tokens"][0].get("prompt_cache_len"),
        "mtp_accepted": res["tokens"][-1].get("mtp_accepted_token_num"),
        "elapsed": time.monotonic() - start,
    }


results = []
for round in range(2):
    with concurrent.futures.ThreadPoolExecutor(max_workers=5) as pool:
        baseline = list(pool.map(run, [(args.baseline_port, p) for p in prompts]))
        replay = list(pool.map(run, [(args.replay_port, p) for p in prompts]))
    for i, (a, b) in enumerate(zip(baseline, replay)):
        results.append({"round": round, "prompt": i, "baseline": a, "replay": b})
        prefix = next(
            (j for j, (x, y) in enumerate(zip(a["tokens"], b["tokens"])) if x != y),
            min(len(a["tokens"]), len(b["tokens"])),
        )
        print(
            json.dumps(
                {
                    "round": round,
                    "prompt": i,
                    "tokens_equal": a["tokens"] == b["tokens"],
                    "common_prefix": prefix,
                    "replay_cache": b["cache_len"],
                    "prompt_tokens": b["prompt_tokens"],
                    "mtp_accepted": b["mtp_accepted"],
                }
            ),
            flush=True,
        )
pathlib.Path("/tmp/replay-comparison-results.json").write_text(json.dumps(results))
