"""Untimed cache isolation without enabling RL-mode inference overhead.

After all leases expire, independent ordinary completions exceeding KV capacity
replace prior trial prefixes. Their first token is outside chat-control tokens.
Run under exp. This prelude is excluded from the measured replay interval.
"""

import argparse
import json
import math
import random
import time
import urllib.request


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--cache-tokens", type=int, required=True)
    parser.add_argument("--max-req-tokens", type=int, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--sleep-seconds", type=float, default=0)
    args = parser.parse_args()
    if args.sleep_seconds:
        time.sleep(args.sleep_seconds)
    length = min(8192, args.max_req_tokens // 2, args.cache_tokens // 2)
    count = math.ceil(args.cache_tokens / length) + 2
    rng = random.Random(args.seed)
    for index in range(count):
        payload = {
            "model": args.model,
            "prompt": [rng.randrange(100, 140000) for _ in range(length)],
            "max_tokens": 1,
            "temperature": 0,
            "ignore_eos": True,
        }
        request = urllib.request.Request(
            args.url + "/v1/completions",
            data=json.dumps(payload).encode(),
            headers={"Content-Type": "application/json"},
        )
        with urllib.request.urlopen(request, timeout=300) as response:
            result = json.load(response)
        if result["usage"]["prompt_tokens"] != length:
            raise RuntimeError("cache reset prompt length mismatch")
        print(json.dumps({"reset_request": index, "usage": result["usage"]}), flush=True)
    print(
        json.dumps(
            {
                "reset_requests": count,
                "total_input_tokens": count * length,
                "cache_capacity_tokens": args.cache_tokens,
                "sleep_seconds": args.sleep_seconds,
            }
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
