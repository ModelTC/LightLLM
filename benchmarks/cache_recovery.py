import argparse
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import re
import time

import requests
from transformers import AutoTokenizer


def replay(url, prompts, settle_seconds, request_tag):
    results = []
    for phase, order in (
        ("cold", range(len(prompts))),
        ("warm", reversed(range(len(prompts)))),
        ("revisit", range(min(4, len(prompts)))),
    ):
        time.sleep(settle_seconds)
        for index in order:
            response = requests.post(
                url.rstrip("/") + "/generate",
                json={
                    "inputs": prompts[index],
                    "parameters": {
                        "max_new_tokens": 64,
                        "do_sample": False,
                        "temperature": 0,
                        "top_k": 1,
                        "return_details": True,
                    },
                },
                headers={"X-Request-Id": f"{request_tag}-{phase}-{index}"},
                timeout=120,
            )
            response.raise_for_status()
            results.append(dict(phase=phase, prompt_index=index, result=response.json()))
    return results


def cache_hits(path, request_tag):
    if path is None:
        return None
    rows = []
    for line in Path(path).read_text().splitlines():
        if f"X-Request-Id:{request_tag}-" not in line or "first_token_cost:" not in line:
            continue
        rows.append(
            dict(
                request=re.search(r"X-Request-Id:(\S+)", line).group(1),
                **{
                    tier: int(re.search(tier + r"_prompt_cache_len:(\d+)", line).group(1))
                    for tier in ("gpu", "cpu", "disk")
                },
            )
        )
    return rows


def token_ids(row):
    return [token["id"] for token in row["result"]["tokens"]]


def main():
    parser = argparse.ArgumentParser(
        description="Compare generation before/after cache eviction on two running LightLLM servers."
    )
    parser.add_argument("--baseline-url", required=True)
    parser.add_argument("--candidate-url", required=True)
    parser.add_argument("--tokenizer", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--baseline-log")
    parser.add_argument("--candidate-log")
    parser.add_argument("--prefix-tokens", type=int, default=768)
    parser.add_argument("--prompts", type=int, default=12)
    parser.add_argument("--settle-seconds", type=float, default=2)
    parser.add_argument("--request-tag", default="cache-recovery")
    args = parser.parse_args()
    if args.prompts <= 0 or args.prefix_tokens <= 0 or args.settle_seconds < 0:
        parser.error("positive prompt count/length and nonnegative settle time are required")
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer, local_files_only=True)
    prompts = []
    for index in range(args.prompts):
        text = f"Record {index}. " + " ".join(
            f"The sensor recorded {index * 97 + j} units during interval {j}." for j in range(args.prefix_tokens)
        )
        ids = tokenizer.encode(text)[: args.prefix_tokens]
        prompts.append(tokenizer.decode(ids) + "\nSummarize the report in two sentences:\n")
    with ThreadPoolExecutor(2) as pool:
        futures = {
            name: pool.submit(replay, url, prompts, args.settle_seconds, args.request_tag)
            for name, url in (("baseline", args.baseline_url), ("candidate", args.candidate_url))
        }
        results = {name: future.result() for name, future in futures.items()}
    summary = {}
    for name, rows in results.items():
        cold = {row["prompt_index"]: row for row in rows if row["phase"] == "cold"}
        repeats = [row for row in rows if row["phase"] != "cold"]
        hits = cache_hits(getattr(args, name + "_log"), args.request_tag)
        summary[name] = dict(
            repeated=len(repeats),
            identical_repeated_tokens=sum(token_ids(row) == token_ids(cold[row["prompt_index"]]) for row in repeats),
            cache_hits=hits,
        )
    summary["cross_version"] = dict(
        requests=len(results["baseline"]),
        identical_tokens=sum(token_ids(a) == token_ids(b) for a, b in zip(results["baseline"], results["candidate"])),
    )
    args.output.write_text(json.dumps(dict(summary=summary, results=results), indent=2, ensure_ascii=False))
    print(json.dumps(summary, indent=2))
    if summary["cross_version"]["identical_tokens"] != summary["cross_version"]["requests"] or any(
        summary[name]["identical_repeated_tokens"] != summary[name]["repeated"] for name in results
    ):
        raise SystemExit("token mismatch; inspect the saved responses")


if __name__ == "__main__":
    main()
