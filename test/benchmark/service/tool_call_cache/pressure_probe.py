"""Controlled eviction probe, not a production-traffic throughput benchmark.

Uses real tool generation and independent length-limited requests to create KV
pressure. The return request repeats the initial prompt exactly, isolating its
residency from chat-template/history rewriting. Use smoke.py for actual tool
result continuation correctness. Run both policies with identical arguments.
"""

import argparse
import copy
import hashlib
import json
from pathlib import Path
import random
import statistics
import time
import urllib.request


def post(base, endpoint, body):
    request = urllib.request.Request(
        base + endpoint, data=json.dumps(body).encode(), headers={"Content-Type": "application/json"}
    )
    started = time.perf_counter()
    with urllib.request.urlopen(request, timeout=180) as response:
        result = json.load(response)
    return result, time.perf_counter() - started


def signature(response):
    message = response["choices"][0]["message"]
    return {"content": message.get("content"), "calls": [call["function"] for call in message.get("tool_calls") or []]}


def continuation_body(body, response):
    """Use actual tool results, keeping opaque call IDs deterministic in pairs."""
    result = copy.deepcopy(body)
    message = response["choices"][0]["message"]
    calls = copy.deepcopy(message.get("tool_calls") or [])
    if not calls:
        raise ValueError("a tool-result continuation requires an actual tool call")
    tool_results = []
    for index, call in enumerate(calls):
        arguments = json.loads(call["function"]["arguments"])
        if call["function"]["name"] != "echo" or not isinstance(arguments.get("text"), str):
            raise ValueError("unexpected echo invocation")
        call["id"] = f"pressure_call_{index}"
        tool_results.append({"role": "tool", "tool_call_id": call["id"], "content": arguments["text"]})
    result["messages"].append({"role": "assistant", "content": message.get("content") or "", "tool_calls": calls})
    result["messages"].extend(tool_results)
    result["messages"].append({"role": "user", "content": "Call echo with text hello. Use the tool now."})
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", required=True)
    parser.add_argument("--model", default="tool-cache-qwen4b")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--repeats", type=int, default=6)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--prefix-words", type=int, default=1024)
    parser.add_argument("--churn-requests", type=int, default=6)
    parser.add_argument("--churn-tokens", type=int, default=8192)
    parser.add_argument("--wait-seconds", type=float, default=0)
    parser.add_argument("--kind", choices=["tool", "normal", "length"], default="tool")
    parser.add_argument("--return-mode", choices=["repeat", "continuation"], default="repeat")
    args = parser.parse_args()
    if args.return_mode == "continuation" and args.kind != "tool":
        parser.error("tool-result continuation requires --kind tool")
    tools = [
        {
            "type": "function",
            "function": {
                "name": "echo",
                "description": "Echo text.",
                "parameters": {"type": "object", "properties": {"text": {"type": "string"}}, "required": ["text"]},
            },
        }
    ]
    rng = random.Random(args.seed)
    results = []
    for repeat in range(args.repeats):
        filler = " ".join(str(rng.randrange(1000, 9999)) for _ in range(args.prefix_words))
        instruction = (
            "Call echo with text hello. Use the tool now."
            if args.kind != "normal"
            else "Say OK. Do not call any tools."
        )
        body = {
            "model": args.model,
            "messages": [{"role": "user", "content": filler + "\n" + instruction}],
            "tools": tools,
            "temperature": 0,
            "max_tokens": 1 if args.kind == "length" else 128,
            "chat_template_kwargs": {"enable_thinking": False},
        }
        initial, initial_s = post(args.url, "/v1/chat/completions", body)
        expected = {"tool": "tool_calls", "normal": "stop", "length": "length"}[args.kind]
        if initial["choices"][0]["finish_reason"] != expected:
            raise RuntimeError(f"unexpected end state: {initial}")
        if args.wait_seconds:
            time.sleep(args.wait_seconds)
        churn = []
        for _ in range(args.churn_requests):
            prompt = [rng.randrange(100, 140000) for _ in range(args.churn_tokens)]
            response, elapsed = post(
                args.url,
                "/v1/completions",
                {
                    "model": args.model,
                    "prompt": prompt,
                    "max_tokens": 1,
                    "temperature": 0,
                },
            )
            churn.append({"seconds": elapsed, "usage": response["usage"]})
        return_body = continuation_body(body, initial) if args.return_mode == "continuation" else body
        returned, return_s = post(args.url, "/v1/chat/completions", return_body)
        if returned["choices"][0]["finish_reason"] != expected:
            raise RuntimeError(f"unexpected return end state: {returned}")
        results.append(
            {
                "repeat": repeat,
                "request_sha256": hashlib.sha256(json.dumps(body, sort_keys=True).encode()).hexdigest(),
                "return_request_sha256": hashlib.sha256(json.dumps(return_body, sort_keys=True).encode()).hexdigest(),
                "initial_seconds": initial_s,
                "initial_usage": initial["usage"],
                "initial_finish_reason": initial["choices"][0]["finish_reason"],
                "initial_output_signature": signature(initial),
                "return_seconds": return_s,
                "return_usage": returned["usage"],
                "return_finish_reason": returned["choices"][0]["finish_reason"],
                "return_output_signature": signature(returned),
                "churn": churn,
            }
        )
        print(json.dumps(results[-1]), flush=True)
    result = {
        "configuration": {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
        "results": results,
        "mean_return_seconds": statistics.mean(r["return_seconds"] for r in results),
        "cached_return_tokens": sum(r["return_usage"]["prompt_tokens_details"]["cached_tokens"] for r in results),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({k: v for k, v in result.items() if k != "results"}), flush=True)


if __name__ == "__main__":
    main()
