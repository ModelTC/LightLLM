import argparse
import hashlib
import json
import os
from pathlib import Path
import secrets
import shutil
import time

import requests

from lightllm.server.tokenizer import get_tokenizer
from test.benchmark.qwen38.check_generation import verified_checkpoint
from test.benchmark.qwen38.retrieval_quality import load_workload, score_records
from test.benchmark.service.completion import completion_request, completion_tokens


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--framework", choices=("lightllm", "vllm", "sglang"), required=True)
    parser.add_argument("--base-url", required=True)
    parser.add_argument("--model-directory", type=Path, required=True)
    parser.add_argument("--service-metadata", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--timeout", type=float, default=1200)
    args = parser.parse_args()
    if not os.environ.get("EXP_ACTIVE"):
        parser.error('Run through exp -m "purpose"')
    run = Path(os.environ["EXP_RUN_DIR"])
    shutil.copyfile(__file__, run / "client.py")
    verified_checkpoint(args.model_directory)
    manifest = load_workload(args.manifest, args.model_directory)
    service = json.loads(args.service_metadata.read_text())
    if service.get("framework") != args.framework or any(
        not service.get(key) for key in ("version", "command", "hardware", "configuration")
    ):
        raise ValueError("Record actual service framework, version, command, hardware and configuration")
    configuration = service["configuration"]
    context = configuration.get("context_limit", configuration.get("effective_token_budget", 0))
    if type(context) is not int or context < manifest["context_limit"]:
        raise ValueError("Service must declare the actual full context budget")
    tokenizer = get_tokenizer(str(args.model_directory)).tokenizer
    config = json.loads((args.model_directory / "generation_config.json").read_text())
    eos = config["eos_token_id"]
    eos_ids = sorted(set(eos if isinstance(eos, list) else [eos]) | {tokenizer.eos_token_id})
    report = dict(
        status="running",
        scope=__doc__,
        service=service,
        manifest_sha256=hashlib.sha256(json.dumps(manifest, sort_keys=True).encode()).hexdigest(),
        results={},
    )
    output = run / "long-retrieval.json"
    client = requests.Session()
    client.trust_env = False
    active_case = None
    try:
        for case in manifest["cases"]:
            active_case = case["name"]
            endpoint, payload = completion_request(
                args.framework, str(args.model_directory.resolve()), case["prompt_ids"], case["max_output_tokens"]
            )
            if args.framework == "lightllm":
                payload["parameters"]["disable_prompt_cache"] = True
            else:
                payload["cache_salt"] = secrets.token_urlsafe(32)
            started = time.monotonic()
            response = client.post(args.base_url.rstrip("/") + endpoint, json=payload, timeout=args.timeout)
            ended = time.monotonic()
            (run / f"{active_case}.json").write_text(response.text)
            response.raise_for_status()
            body = response.json()
            result = completion_tokens(body, args.framework, case["prompt_ids"], case["max_output_tokens"], eos_ids)
            if args.framework == "lightllm":
                cached = body["tokens"][0]["prompt_cache_len"]
            else:
                cached = (body["usage"].get("prompt_tokens_details") or {}).get("cached_tokens")
            if type(cached) is not int or cached != 0:
                raise ValueError("Every context case must report exactly zero cached prompt tokens")
            text = tokenizer.decode(result["token_ids"], skip_special_tokens=True)
            score = score_records(text, case["expected_records"])
            score["correct"] = score["correct"] and not result["truncated"]
            result.update(
                text=text,
                prompt_tokens=len(case["prompt_ids"]),
                cached_input_tokens=cached,
                observed_request_s=ended - started,
                **score,
            )
            report["results"][active_case] = result
            output.write_text(json.dumps(report, indent=2))
            print(
                json.dumps(
                    dict(
                        case=active_case,
                        **{key: result[key] for key in ("prompt_tokens", "correct", "records", "truncated")},
                    )
                ),
                flush=True,
            )
        report.update(cases=len(manifest["cases"]), correct=sum(row["correct"] for row in report["results"].values()))
        if report["cases"] != 4 or report["correct"] != report["cases"]:
            raise ValueError("Some declared-context retrieval answers differ from the pinned known records")
        report["status"] = "passed"
    except Exception as error:
        report.update(status="failed", failed_case=active_case, error_type=type(error).__name__, error=str(error))
        raise
    finally:
        output.write_text(json.dumps(report, indent=2))
        client.close()


if __name__ == "__main__":
    main()
