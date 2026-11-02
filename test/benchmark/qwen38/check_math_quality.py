import argparse
import hashlib
import json
import os
from pathlib import Path

import requests

from lightllm.server.tokenizer import get_tokenizer
from test.benchmark.qwen38.check_generation import REVISION, TOKENIZER_SHA256, verified_checkpoint
from test.benchmark.qwen38.math_quality import (
    DATASET_REVISION,
    DATASET_SHA256,
    load_examples,
    prepare_cases,
    score_answer,
    select_examples,
)
from test.benchmark.service.completion_runner import CompletionClient


def verified_workload(manifest, root, dataset_path, tokenizer):
    if (
        manifest["status"] != "prepared"
        or manifest["model_revision"] != REVISION
        or manifest["dataset"]["revision"] != DATASET_REVISION
        or manifest["dataset"]["sha256"] != DATASET_SHA256
        or manifest["dataset"]["split"] != "test"
        or manifest["decoding"] != dict(temperature=0, ignore_eos=False, enable_thinking=False)
        or manifest["tokenizer_files"].get("tokenizer.json") != TOKENIZER_SHA256
    ):
        raise ValueError("Workload must use pinned model, dataset and greedy non-thinking decoding")
    for name in ("tokenizer.json", "tokenizer_config.json", "chat_template.jinja"):
        if hashlib.sha256((root / name).read_bytes()).hexdigest() != manifest["tokenizer_files"].get(name):
            raise ValueError("Tokenizer files differ from workload preparation")
    examples = load_examples(dataset_path)
    if len(examples) != manifest["dataset"]["records"]:
        raise ValueError("Dataset record count differs")
    cases = prepare_cases(
        select_examples(examples, manifest["sample_count"], manifest["sample_seed"]),
        tokenizer,
        manifest["max_output_tokens"],
        manifest["context_limit"],
    )
    if cases != manifest["cases"]:
        raise ValueError("Workload prompts or gold labels differ from pinned preparation")
    return cases


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--framework", choices=("lightllm", "vllm", "sglang"), required=True)
    parser.add_argument("--base-url", required=True)
    parser.add_argument("--model-directory", type=Path, required=True)
    parser.add_argument("--service-metadata", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--timeout", type=float, default=600)
    parser.add_argument("--concurrency", type=int, default=1)
    args = parser.parse_args()
    if not 1 <= args.concurrency <= 8:
        parser.error("Concurrency must be between one and eight")
    if not os.environ.get("EXP_ACTIVE"):
        parser.error('Run through exp -m "purpose"')
    run = Path(os.environ["EXP_RUN_DIR"])
    verified_checkpoint(args.model_directory)
    metadata = json.loads(args.service_metadata.read_text())
    if metadata.get("framework") != args.framework or any(
        not metadata.get(key) for key in ("version", "command", "hardware", "configuration")
    ):
        raise ValueError("Record actual service framework, version, command, hardware and configuration")
    manifest = json.loads(args.manifest.read_text())
    tokenizer = get_tokenizer(str(args.model_directory)).tokenizer
    cases = verified_workload(manifest, args.model_directory, args.dataset, tokenizer)
    config = json.loads((args.model_directory / "generation_config.json").read_text())
    eos = config["eos_token_id"]
    eos_ids = sorted(set(eos if isinstance(eos, list) else [eos]) | {tokenizer.eos_token_id})
    report = dict(
        status="running",
        scope="Pinned non-thinking scored math sample; no performance or broad accuracy claim",
        service=metadata,
        manifest=manifest,
        manifest_sha256=hashlib.sha256(json.dumps(manifest, sort_keys=True).encode()).hexdigest(),
        eos_ids=eos_ids,
        concurrency=args.concurrency,
        results=[],
    )
    output = run / "math-quality.json"
    client = CompletionClient(
        args.framework,
        str(args.model_directory.resolve()),
        args.base_url,
        args.timeout,
        eos_ids,
        run,
        concurrency=args.concurrency,
        session_factory=requests.Session,
    )
    active_case = None
    try:
        responses = iter(client.results(cases))
        for case in cases:
            active_case = case["dataset_index"]
            result = next(responses)
            text = tokenizer.decode(result["token_ids"], skip_special_tokens=True)
            score = score_answer(text, case["expected_number"])
            score["correct"] = score["correct"] and not result["truncated"]
            result.update(dataset_index=active_case, text=text, **score)
            report["results"].append(result)
            output.write_text(json.dumps(report, indent=2))
            print(json.dumps({key: result[key] for key in ("dataset_index", "correct", "truncated")}), flush=True)
        report.update(
            status="recorded",
            scored=len(cases),
            correct=sum(row["correct"] for row in report["results"]),
            truncated=sum(row["truncated"] for row in report["results"]),
            invalid_final_answer=sum(row["predicted"] is None for row in report["results"]),
        )
        report["accuracy"] = report["correct"] / report["scored"]
    except Exception as error:
        report.update(status="failed", failed_case=active_case, error_type=type(error).__name__, error=str(error))
        raise
    finally:
        output.write_text(json.dumps(report, indent=2))
        client.close()


if __name__ == "__main__":
    main()
