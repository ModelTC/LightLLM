import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
import json
import os
from pathlib import Path
import threading
import time

import requests
from transformers import AutoTokenizer

from check_generation import REVISION, TOKENIZER_SHA256, cases_for, verified_checkpoint


def scheduler_cases(root):
    raw = cases_for(root)
    tokenizer = AutoTokenizer.from_pretrained(root, local_files_only=True)
    result = {
        name: {"inputs": raw[name], "max_new_tokens": length, "ignore_eos": True}
        for name, length in (
            ("english", 32),
            ("chinese", 31),
            ("code_unicode", 17),
            ("long_2049", 32),
            ("long_4063", 32),
        )
    }
    result["eos_history"] = {
        "inputs": [248044] + raw["english"] + [248046] + raw["chinese"],
        "max_new_tokens": 19,
        "ignore_eos": True,
    }
    for name, question in (
        ("chat_ok", "Reply with exactly OK."),
        ("chat_number", "What is 2+2? Reply only with the number."),
    ):
        result[name] = {
            "inputs": tokenizer.apply_chat_template(
                [{"role": "user", "content": question}],
                tokenize=True,
                add_generation_prompt=True,
                enable_thinking=False,
            ),
            "max_new_tokens": 96,
            "ignore_eos": False,
        }
    generation_eos = json.loads((root / "generation_config.json").read_text())["eos_token_id"]
    eos = set(generation_eos if isinstance(generation_eos, list) else [generation_eos])
    eos.add(tokenizer.eos_token_id)
    return result, sorted(eos)


def generate(base_url, case, eos_ids, timeout, response_path, cold=False):
    parameters = {key: value for key, value in case.items() if key != "inputs"}
    parameters.update(do_sample=False, return_details=True, print_eos_token=True, disable_prompt_cache=cold)
    started = time.monotonic()
    with requests.Session() as client:
        client.trust_env = False
        response = client.post(
            base_url.rstrip("/") + "/generate",
            json={"inputs": case["inputs"], "parameters": parameters},
            timeout=timeout,
        )
    ended = time.monotonic()
    response_path.write_text(response.text)
    response.raise_for_status()
    body = response.json()
    ids = validate_generation(body, len(case["inputs"]), case["max_new_tokens"], case["ignore_eos"], eos_ids)
    return {"token_ids": ids, "response": body, "started": started, "ended": ended}


def validate_generation(body, prompt_tokens, limit, ignore_eos, eos_ids):
    ids = [token["id"] for token in body["tokens"]]
    if not ids or any(type(token) is not int or token < 0 for token in ids) or len(ids) > limit:
        raise ValueError("Invalid output token IDs or output length")
    if body["prompt_tokens"] != prompt_tokens or body["count_output_tokens"] != len(ids):
        raise ValueError("Service prompt/output counts do not match exact token IDs")
    if ignore_eos:
        if len(ids) != limit or body["finish_reason"] != "length":
            raise ValueError("Fixed-length request did not reach its requested length")
    elif ids[-1] not in eos_ids or any(token in eos_ids for token in ids[:-1]) or body["finish_reason"] != "stop":
        raise ValueError("EOS request must stop at its first declared EOS token")
    return ids


def monitor(base_url, stop, summary, path):
    with requests.Session() as client:
        client.trust_env = False
        while not stop.is_set():
            try:
                response = client.get(base_url.rstrip("/") + "/metrics", timeout=5)
                response.raise_for_status()
                path.write_text(response.text)
                for line in response.text.splitlines():
                    if line.startswith(("lightllm_batch_current_size", "lightllm_num_running_reqs")):
                        summary["maximum_router_batch"] = max(summary["maximum_router_batch"], float(line.split()[-1]))
                summary["samples"] += 1
            except Exception as error:
                summary["last_error"] = str(error)
            stop.wait(0.02)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", required=True)
    parser.add_argument("--model-directory", type=Path, required=True)
    parser.add_argument("--service-metadata", type=Path, required=True)
    parser.add_argument("--reference", type=Path)
    parser.add_argument("--concurrency", type=int, nargs="+", default=[2, 4, 8])
    parser.add_argument("--timeout", type=float, default=600)
    args = parser.parse_args()
    if not os.environ.get("EXP_ACTIVE"):
        parser.error('Run through exp -m "purpose"')
    if not args.concurrency or min(args.concurrency) < 2 or max(args.concurrency) > 8:
        parser.error("Concurrent waves must use between two and eight requests")
    run = Path(os.environ["EXP_RUN_DIR"])
    verified_checkpoint(args.model_directory)
    metadata = json.loads(args.service_metadata.read_text())
    if metadata.get("framework") != "lightllm" or any(
        not metadata.get(key) for key in ("version", "command", "hardware", "configuration")
    ):
        raise ValueError("Record the actual native service version, command, hardware and configuration")
    cases, eos_ids = scheduler_cases(args.model_directory)
    source = {
        name: hashlib.sha256((args.model_directory / name).read_bytes()).hexdigest()
        for name in ("tokenizer.json", "tokenizer_config.json", "generation_config.json")
    }
    reference = json.loads(args.reference.read_text()) if args.reference else None
    if reference and (
        reference.get("status") not in ("passed", "recorded")
        or reference["revision"] != REVISION
        or reference["cases"] != cases
        or reference["source"] != source
        or reference["eos_ids"] != eos_ids
    ):
        raise ValueError("Reference must pass with identical checkpoint, tokenizer, inputs and EOS contract")
    report = dict(
        status="running",
        scope="full trained scheduler/EOS/cache correctness; no performance claim",
        revision=REVISION,
        tokenizer_sha256=TOKENIZER_SHA256,
        source=source,
        service=metadata,
        cases=cases,
        eos_ids=eos_ids,
        reference=str(args.reference) if args.reference else None,
        serial={},
        waves=[],
        mismatches=[],
    )
    output = run / "scheduler.json"
    stop = threading.Event()
    metrics = {"maximum_router_batch": 0, "samples": 0}
    thread = threading.Thread(target=monitor, args=(args.base_url, stop, metrics, run / "metrics.txt"), daemon=True)
    thread.start()
    active = None
    try:
        for name, case in cases.items():
            active = {"wave": "serial", "case": name}
            result = generate(args.base_url, case, eos_ids, args.timeout, run / f"serial-{name}.txt")
            report["serial"][name] = result
            if reference and result["token_ids"] != reference["serial"][name]["token_ids"]:
                report["mismatches"].append({"wave": "serial_reference", "case": name})
            output.write_text(json.dumps(report, indent=2))
        names = list(cases)
        plans = [(count, False) for count in args.concurrency] + [(8, True), (8, False)]
        for index, (count, cold) in enumerate(plans):
            order = names[index:] + names[:index]
            if index == len(plans) - 1:
                order = list(reversed(order))
            wave = {"concurrency": count, "disable_prompt_cache": cold, "results": {}}
            report["waves"].append(wave)
            with ThreadPoolExecutor(max_workers=count) as executor:
                pending = {
                    executor.submit(
                        generate,
                        args.base_url,
                        cases[name],
                        eos_ids,
                        args.timeout,
                        run / f"wave-{index}-{name}.txt",
                        cold,
                    ): name
                    for name in order
                }
                for future in as_completed(pending):
                    name = pending[future]
                    active = {"wave": index, "case": name}
                    result = future.result()
                    wave["results"][name] = result
                    if result["token_ids"] != report["serial"][name]["token_ids"]:
                        report["mismatches"].append({"wave": index, "case": name})
                    output.write_text(json.dumps(report, indent=2))
            print(
                json.dumps({"wave": index, "concurrency": count, "cold": cold, "mismatches": report["mismatches"]}),
                flush=True,
            )
        if report["mismatches"]:
            raise ValueError("Serial/concurrent or ordinary/speculative token IDs differ")
        if metrics["maximum_router_batch"] < 2:
            raise ValueError("No actual multi-request router batch observed; packed serving remains unverified")
        report["status"] = "passed" if reference else "recorded"
    except Exception as error:
        report.update(status="failed", failed_request=active, error_type=type(error).__name__, error=str(error))
        raise
    finally:
        stop.set()
        thread.join(timeout=6)
        report["router_metrics"] = metrics
        output.write_text(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
