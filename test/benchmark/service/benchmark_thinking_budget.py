"""Measure fixed-length HTTP inference and check cancellation/request isolation.

PYTHONPATH=. python test/benchmark/service/benchmark_thinking_budget.py --url ...
The optional baseline must use the same model and capacity with budgets disabled.
"""

import argparse
import asyncio
from collections import Counter
import json
from pathlib import Path
import statistics
import sys
import time

import httpx

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "test_api"))
from test_thinking_budget_service import Scenario, chat, run_case, schema_format  # noqa: E402


async def cancellation(client, url, budget, cutoff):
    """Disconnect during reasoning, a multi-token delimiter, or answer generation."""
    marker = [100, 101, 100]
    params = {
        "max_new_tokens": 1024,
        "min_new_tokens": 1024,
        "ignore_eos": True,
        "do_sample": False,
        "thinking_token_budget": budget,
        "guided_reasoning_end": marker,
    }
    tokens = []
    async with client.stream(
        "POST",
        url + "/generate_stream",
        headers={"X-Request-Id": f"thinking-review-cancel-{budget}-{cutoff}"},
        json={"inputs": "Explain the proof in detail. <think>", "parameters": params},
    ) as response:
        response.raise_for_status()
        async for line in response.aiter_lines():
            if line.startswith("data:"):
                event = json.loads(line[5:])
                assert "token" in event, event
                tokens.append(event["token"]["id"])
                if len(tokens) >= cutoff:
                    break
    probe = await run_case(
        client, url, Scenario("after_cancel", chat(0, response_format=schema_format()), check="schema")
    )
    assert probe["passed"], probe.get("error")
    assert (await client.get(url + "/readiness")).status_code == 200
    return {"name": f"cancel/b{budget}/cut{cutoff}", "passed": True, "observed_token_ids": tokens}


async def fixed_request(client, url, budget, request_id):
    params = {"max_new_tokens": 64, "min_new_tokens": 64, "ignore_eos": True, "do_sample": False}
    if budget is not None:
        params.update(thinking_token_budget=budget, guided_reasoning_end=[151668])
    started = time.perf_counter()
    first = None
    last = None
    count = 0
    token_ids = []
    async with client.stream(
        "POST",
        url + "/generate_stream",
        headers={"X-Request-Id": request_id},
        json={"inputs": "Give a detailed step by step explanation of prime numbers. <think>", "parameters": params},
    ) as response:
        response.raise_for_status()
        async for line in response.aiter_lines():
            if line.startswith("data:"):
                last = json.loads(line[5:])
                assert "token" in last, last
                first = first or time.perf_counter()
                count += 1
                token_ids.append(last["token"]["id"])
    elapsed = time.perf_counter() - started
    assert count == 64 and last["finished"] and last["finish_reason"] in ("length", "stop"), (count, last)
    return {"elapsed_s": elapsed, "ttft_s": first - started, "tokens": count, "token_ids": token_ids}


async def main(args):
    args.output.parent.mkdir(parents=True, exist_ok=True)
    report = {
        "url": args.url,
        "baseline_url": args.baseline_url,
        "cancellation": [],
        "concurrency": [],
        "performance": [],
    }
    async with httpx.AsyncClient(trust_env=False, timeout=120, limits=httpx.Limits(max_connections=80)) as client:
        try:
            for budget in (0, 7, 31):
                for cutoff in (1, 2, 8, 40):
                    try:
                        result = await cancellation(client, args.url, budget, cutoff)
                    except Exception as exc:
                        result = {"name": f"cancel/b{budget}/cut{cutoff}", "passed": False, "error": repr(exc)}
                    report["cancellation"].append(result)
                    print(json.dumps(result), flush=True)

            for concurrency in (1, 8, 32, 64):
                cases = []
                for i in range(concurrency):
                    budget = (0, 1, 7, 31)[i % 4]
                    payload = chat(budget, stream=bool(i % 2), max_completion_tokens=96)
                    check = "chat"
                    if i % 3 == 0:
                        payload["response_format"] = schema_format()
                        check = "schema"
                    cases.append(Scenario(f"concurrency{concurrency}/request{i}/b{budget}", payload, check=check))
                results = await asyncio.gather(*(run_case(client, args.url, c) for c in cases))
                group = {"concurrency": concurrency, "results": results, "passed": all(r["passed"] for r in results)}
                report["concurrency"].append(group)
                print(f"concurrency {concurrency}: {dict(Counter(r['passed'] for r in results))}", flush=True)

            for concurrency in (1, 8, 16):
                modes = [
                    ("enabled_no_budget", args.url, None),
                    ("enabled_budget64", args.url, 64),
                    ("enabled_budget8", args.url, 8),
                    ("enabled_budget0", args.url, 0),
                ]
                if args.baseline_url:
                    modes.insert(0, ("disabled_no_budget", args.baseline_url, None))
                # Warm up each workload before collecting interleaved repetitions.
                for name, url, budget in modes:
                    await fixed_request(client, url, budget, "thinking-review-warmup-" + name)
                for repeat in range(3):
                    for name, url, budget in modes[repeat:] + modes[:repeat]:
                        started = time.perf_counter()
                        results = await asyncio.gather(
                            *(
                                fixed_request(
                                    client, url, budget, f"thinking-review-perf-{name}-{concurrency}-{repeat}-{i}"
                                )
                                for i in range(concurrency)
                            )
                        )
                        wall = time.perf_counter() - started
                        group = {
                            "mode": name,
                            "concurrency": concurrency,
                            "repeat": repeat,
                            "tokens_per_second": sum(r["tokens"] for r in results) / wall,
                            "ttft_median_s": statistics.median(r["ttft_s"] for r in results),
                            "latency_median_s": statistics.median(r["elapsed_s"] for r in results),
                            "requests": results,
                        }
                        report["performance"].append(group)
                        print(json.dumps({k: v for k, v in group.items() if k != "requests"}), flush=True)
        except Exception as exc:
            report["error"] = repr(exc)
            raise
        finally:
            args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2))
    return int(any(not r["passed"] for category in ("cancellation", "concurrency") for r in report[category]))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", default="http://127.0.0.1:22100")
    parser.add_argument("--baseline-url")
    parser.add_argument("--output", type=Path, default=Path("artifacts/thinking_budget_stress.json"))
    raise SystemExit(asyncio.run(main(parser.parse_args())))
