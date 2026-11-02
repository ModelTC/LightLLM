import argparse
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import time

import requests
from transformers import AutoTokenizer

from .cache_recovery import cache_hits, token_ids


def main():
    parser = argparse.ArgumentParser(
        description="Compare forced DP prefix recovery, concurrent generation and optional stream cancellation."
    )
    parser.add_argument("--baseline-url", required=True)
    parser.add_argument("--candidate-url", required=True, help="DP=2 or more with shared KV enabled")
    parser.add_argument("--tokenizer", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--candidate-log")
    parser.add_argument("--request-tag", default="dp-recovery")
    parser.add_argument("--prompts", type=int, default=4)
    parser.add_argument("--prefix-tokens", type=int, default=768)
    parser.add_argument("--cancel-streams", action="store_true")
    args = parser.parse_args()
    if args.prompts <= 0 or args.prefix_tokens <= 0:
        parser.error("prompt count and prefix length must be positive")
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer, local_files_only=True)
    prompts = []
    for index in range(args.prompts):
        text = f"DP record {index}. " + " ".join(
            f"The instrument recorded {index * 97 + j} units at interval {j}." for j in range(args.prefix_tokens)
        )
        prompts.append(
            tokenizer.decode(tokenizer.encode(text)[: args.prefix_tokens])
            + "\nSummarize the report in two sentences:\n"
        )
    results, cancellations = [], []
    evidence = dict(configuration=vars(args) | {"output": str(args.output)}, prompts=prompts, results=results)

    def generate(url, index, dp, phase, cancel=False):
        endpoint = "/generate_stream" if cancel else "/generate"
        with requests.post(
            url.rstrip("/") + endpoint,
            json=dict(
                inputs=prompts[index],
                parameters=dict(
                    max_new_tokens=192 if cancel else 64,
                    do_sample=False,
                    temperature=0,
                    top_k=1,
                    return_details=True,
                    suggested_dp_index=dp,
                ),
            ),
            headers={"X-Request-Id": f"{args.request_tag}-{phase}-{index}"},
            stream=cancel,
            timeout=120,
        ) as response:
            response.raise_for_status()
            if cancel:
                event = next(line for line in response.iter_lines(chunk_size=1) if line)
                return dict(index=index, first_event=event.decode())
            return dict(phase=phase, prompt_index=index, dp=dp, result=response.json())

    try:
        for index in range(args.prompts):
            results.append(generate(args.baseline_url, index, -1, "baseline"))
            results.append(generate(args.candidate_url, index, 0, "cold-dp0"))
            results.append(generate(args.candidate_url, index, 1, "restore-dp1"))
        with ThreadPoolExecutor(min(args.prompts, 4)) as pool:
            for phase in ("concurrent-a", "concurrent-b"):
                results.extend(pool.map(lambda i: generate(args.candidate_url, i, i % 2, phase), range(args.prompts)))
            if args.cancel_streams:
                cancellations.extend(
                    pool.map(lambda i: generate(args.candidate_url, i, i % 2, "cancel", True), range(args.prompts))
                )
                time.sleep(1)
                for phase in ("after-cancel-a", "after-cancel-b"):
                    results.extend(
                        pool.map(lambda i: generate(args.candidate_url, i, 1 - i % 2, phase), range(args.prompts))
                    )
        reference = {row["prompt_index"]: token_ids(row) for row in results if row["phase"] == "baseline"}
        checks = [
            dict(
                phase=row["phase"],
                index=row["prompt_index"],
                identical=token_ids(row) == reference[row["prompt_index"]],
            )
            for row in results
            if row["phase"] != "baseline"
        ]
        evidence.update(checks=checks, cache_hits=cache_hits(args.candidate_log, args.request_tag))
        print(f"identical tokens: {sum(row['identical'] for row in checks)}/{len(checks)}", flush=True)
        if not all(row["identical"] for row in checks):
            raise RuntimeError("token mismatch; inspect the saved responses")
    except Exception as error:
        evidence["error"] = f"{type(error).__name__}: {error}"
        raise
    finally:
        evidence["cancellations"] = cancellations
        args.output.write_text(json.dumps(evidence, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
