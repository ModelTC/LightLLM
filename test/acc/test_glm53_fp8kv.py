"""GLM-5.3 Flash HTTP regression for sparse prefix reuse and CPU/P-D cache transport."""

import argparse
import json
import re
import time
import uuid
from pathlib import Path

import requests


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", required=True)
    parser.add_argument("--model-dir", required=True)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--server-log", type=Path)
    parser.add_argument("--reference", type=Path, help="Compare cold outputs against a normal-mode run")
    parser.add_argument("--cpu-cache", action="store_true", help="Require a CPU hit after GPU eviction")
    parser.add_argument("--evict-prompts", type=int, default=4)
    args = parser.parse_args()
    if args.cpu_cache and args.server_log is None:
        parser.error("--cpu-cache requires --server-log")
    from lightllm.server.tokenizer import get_tokenizer

    tokenizer = get_tokenizer(args.model_dir)
    content = "Question: What is 2 + 3?\nAnswer: 5.\n"
    suffix = "Question: What is 2 + 3?\nAnswer:"
    run_id = uuid.uuid4().hex

    def prompt(length, tag):
        prefix = f"Document {tag}.\n"
        repeat = (length - len(tokenizer.encode(prefix + suffix))) // len(tokenizer.encode(content)) - 1
        body = (prefix + content * repeat).rstrip() + " "
        padding = length - len(tokenizer.encode(body + suffix))
        text = body + "a " * padding + suffix
        assert len(tokenizer.encode(text)) == length
        return text

    def generate(inputs, label, output_tokens=32):
        input_ids = tokenizer.encode(inputs)
        request_id = f"{run_id}-{label}"
        response = requests.post(
            args.url.rstrip("/") + "/generate",
            headers={"X-Request-Id": request_id},
            json={
                "inputs": inputs,
                "parameters": {
                    "do_sample": False,
                    "ignore_eos": True,
                    "max_new_tokens": output_tokens,
                    "return_details": True,
                },
            },
            timeout=600,
        )
        response.raise_for_status()
        result = response.json()
        assert result["count_output_tokens"] == output_tokens, result
        assert result["finish_reason"] == "length", result
        assert len(result["tokens"]) == output_tokens, result
        assert result["prompt_tokens"] == len(input_ids), result
        return {
            "request_id": request_id,
            "input_tokens": len(input_ids),
            "input_ids": input_ids,
            "token_ids": [t["id"] for t in result["tokens"]],
            "text": result["generated_text"][0],
            "cache_hit_tokens": result["tokens"][0].get("prompt_cache_len", 0),
        }

    report = {"url": args.url, "cpu_cache": args.cpu_cache, "results": []}
    reference = json.loads(args.reference.read_text()) if args.reference else None
    try:
        for remainder in range(4):
            inputs = prompt(2304 + remainder, f"target-{remainder}")
            row = {"remainder": remainder}
            report["results"].append(row)
            row["cold"] = generate(inputs, f"{remainder}-cold")
            row["warm"] = generate(inputs, f"{remainder}-warm")
            assert row["warm"]["cache_hit_tokens"] >= 2048, row
            assert row["warm"]["token_ids"] == row["cold"]["token_ids"], row
            if reference:
                assert row["cold"]["input_ids"] == reference["results"][remainder]["cold"]["input_ids"], row
                assert row["cold"]["token_ids"] == reference["results"][remainder]["cold"]["token_ids"], row
            if args.cpu_cache:
                for index in range(args.evict_prompts):
                    generate(prompt(2304, f"evict-{run_id}-{remainder}-{index}"), f"evict-{remainder}-{index}", 1)
                row["reloaded"] = generate(inputs, f"{remainder}-reloaded")
                deadline = time.monotonic() + 10
                while True:
                    lines = args.server_log.read_text().splitlines()
                    completed = [line for line in lines if f"X-Request-Id:{row['reloaded']['request_id']} " in line]
                    if completed:
                        break
                    if time.monotonic() >= deadline:
                        raise AssertionError("Missing CPU reload completion log")
                    time.sleep(0.2)
                match = re.search(r"cpu_prompt_cache_len:(\d+)", completed[-1])
                assert match and int(match[1]) > 0, completed[-1]
                row["cpu_hit_tokens"] = int(match[1])
                assert row["reloaded"]["token_ids"] == row["warm"]["token_ids"], row
        report["status"] = "passed"
    except Exception as exc:
        report["status"] = "failed"
        report["error"] = repr(exc)
        raise
    finally:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2))
        print(json.dumps(report), flush=True)


if __name__ == "__main__":
    main()
