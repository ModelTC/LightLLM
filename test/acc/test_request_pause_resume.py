"""Exercise real scheduler preemption through the streaming HTTP API."""

import argparse
import concurrent.futures
import json
import re
import threading
import time
import uuid
from pathlib import Path

import requests


def generate(url, prompt, output_tokens, request_id, timeout):
    token_ids = []
    finish_reason = None
    with requests.post(
        url.rstrip("/") + "/generate_stream",
        headers={"X-Request-Id": request_id},
        json={
            "inputs": prompt,
            "parameters": {
                "do_sample": False,
                # Keep EOS enabled so admission uses its normal length estimate.
                # min_new_tokens ensures the actual workload exceeds that estimate.
                "ignore_eos": False,
                "min_new_tokens": output_tokens,
                "max_new_tokens": output_tokens,
            },
        },
        stream=True,
        timeout=timeout,
    ) as response:
        response.raise_for_status()
        for line in response.iter_lines(chunk_size=1):
            if not line.startswith(b"data:"):
                continue
            event = json.loads(line[5:])
            if "token" not in event:
                raise RuntimeError(f"Unexpected stream event: {event}")
            token_ids.append(event["token"]["id"])
            if event.get("finished"):
                finish_reason = event["finish_reason"]
    if len(token_ids) != output_tokens or finish_reason not in ("length", "stop"):
        raise AssertionError(f"{request_id}: received {len(token_ids)} tokens, finish_reason={finish_reason}")
    return {"request_id": request_id, "token_ids": token_ids, "finish_reason": finish_reason}


def check_pause_events(log, request_ids):
    completed = {}
    for line in log.splitlines():
        match = re.search(r"X-Request-Id:(\S+).*lightllm_req_id:(\d+)", line)
        if match and match[1] in request_ids:
            completed[match[2]] = match[1]
    paused, recovered, pending = set(), set(), set()
    events = []
    for match in re.finditer(r"infer (paused|recover paused) req id (\d+)", log):
        action, req_id = match.groups()
        if req_id not in completed:
            continue
        events.append({"action": action, "req_id": req_id, "request_id": completed[req_id]})
        if action == "paused":
            paused.add(req_id)
            pending.add(req_id)
        elif req_id in pending:
            recovered.add(req_id)
            pending.remove(req_id)
    if len(completed) != len(request_ids):
        raise AssertionError("Missing completion log for a pressure request")
    if not paused or paused != recovered or pending:
        raise AssertionError(f"Actual pause/resume not proven: paused={sorted(paused)}, recovered={sorted(recovered)}")
    return events


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", required=True)
    parser.add_argument("--model-dir", required=True)
    parser.add_argument("--server-log", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--input-tokens", type=int, default=512)
    parser.add_argument("--output-tokens", type=int, default=512)
    parser.add_argument("--concurrency", type=int, default=8)
    parser.add_argument("--warmup-requests", type=int, default=64)
    parser.add_argument("--timeout", type=float, default=600)
    args = parser.parse_args()
    from lightllm.server.tokenizer import get_tokenizer

    tokenizer = get_tokenizer(args.model_dir)
    prefix = uuid.uuid4().hex
    content = tokenizer.encode("Continue the numbered explanation of language model inference and caching. ")

    def prompt(index):
        start = tokenizer.encode(f"{prefix}-{index}: ", add_special_tokens=False)
        return (start + content * (args.input_tokens // len(content) + 1))[: args.input_tokens]

    report = {"settings": {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()}}
    try:
        # Obtain the reference before preemption, one request at a time.
        prompts = [prompt(i) for i in range(args.concurrency)]
        reference = [
            generate(args.url, p, args.output_tokens, f"{prefix}-reference-{i}", args.timeout)
            for i, p in enumerate(prompts)
        ]
        report["reference"] = reference
        # Short completions lower the scheduler's EMA; distinct prompts evict
        # reference KV from the deliberately small GPU cache.
        for i in range(args.warmup_requests):
            generate(args.url, prompt(i + args.concurrency), 1, f"{prefix}-warmup-{i}", args.timeout)
        offset = args.server_log.stat().st_size
        barrier = threading.Barrier(args.concurrency)

        def run(index):
            barrier.wait(timeout=args.timeout)
            return generate(args.url, prompts[index], args.output_tokens, f"{prefix}-pressure-{index}", args.timeout)

        with concurrent.futures.ThreadPoolExecutor(max_workers=args.concurrency) as pool:
            report["pressure"] = list(pool.map(run, range(args.concurrency)))
        # HTTP completion is emitted just before its server-side summary log.
        deadline = time.monotonic() + 10
        while True:
            with args.server_log.open() as f:
                f.seek(offset)
                log = f.read()
            try:
                report["pause_events"] = check_pause_events(log, {r["request_id"] for r in report["pressure"]})
                break
            except AssertionError:
                if time.monotonic() >= deadline:
                    raise
                time.sleep(0.2)
        mismatches = []
        for expected, actual in zip(reference, report["pressure"]):
            if actual["token_ids"] != expected["token_ids"]:
                first = next(i for i, (a, b) in enumerate(zip(actual["token_ids"], expected["token_ids"])) if a != b)
                mismatches.append({"request_id": actual["request_id"], "first_mismatch": first})
        report["mismatches"] = mismatches
        if mismatches:
            raise AssertionError(f"Paused workload differs from serial reference: {mismatches}")
        report["status"] = "passed"
    except Exception as exc:
        report["status"] = "failed"
        report["error"] = repr(exc)
        raise
    finally:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2))
        print(json.dumps({k: v for k, v in report.items() if k not in ("reference", "pressure")}), flush=True)


if __name__ == "__main__":
    main()
