import argparse
import hashlib
import json
import os
from pathlib import Path

from check_generation import REVISION, verified_checkpoint
from test.benchmark.service.token_stream import request_stream


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--framework", choices=("lightllm", "vllm", "sglang"), required=True)
    parser.add_argument("--base-url", required=True)
    parser.add_argument("--model-directory", type=Path, required=True)
    parser.add_argument("--service-metadata", type=Path, required=True)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--cases", nargs="+", default=["english", "chinese", "long_2049"])
    parser.add_argument("--timeout", type=float, default=600)
    args = parser.parse_args()
    if not os.environ.get("EXP_ACTIVE"):
        parser.error("Run through exp")
    verified_checkpoint(args.model_directory)
    reference = json.loads(args.reference.read_text())
    service = json.loads(args.service_metadata.read_text())
    if service.get("framework") != args.framework or any(
        not service.get(key) for key in ("version", "command", "hardware", "configuration")
    ):
        raise ValueError("Record actual service provenance")
    if reference["status"] not in ("passed", "recorded") or reference["revision"] != REVISION:
        raise ValueError("Require a passing pinned scheduler reference")
    for name, expected in reference["source"].items():
        if hashlib.sha256((args.model_directory / name).read_bytes()).hexdigest() != expected:
            raise ValueError(f"Checkpoint/tokenizer changed after reference: {name}")
    run = Path(os.environ["EXP_RUN_DIR"])
    report = dict(
        status="running",
        scope="real streaming protocol/greedy parity; no performance comparison",
        revision=REVISION,
        service=service,
        reference=str(args.reference),
        results={},
    )
    path = run / "streaming.json"
    active = None
    try:
        for name in args.cases:
            active = name
            case = reference["cases"][name]
            if not case["ignore_eos"]:
                raise ValueError("Streaming fixed-length gate requires EOS ignored")
            result = request_stream(
                args.base_url,
                args.framework,
                str(args.model_directory.resolve()),
                case["inputs"],
                case["max_new_tokens"],
                args.timeout,
                run / f"response-{name}.sse",
            )
            result["reference_matches"] = result["token_ids"] == reference["serial"][name]["token_ids"]
            report["results"][name] = result
            path.write_text(json.dumps(report, indent=2))
            print(json.dumps({"case": name, **result}), flush=True)
        if not all(result["reference_matches"] for result in report["results"].values()):
            raise ValueError("Streamed token IDs differ from the exact scheduler reference")
        report["status"] = "passed"
    except Exception as error:
        report.update(status="failed", failed_case=active, error_type=type(error).__name__, error=str(error))
        raise
    finally:
        path.write_text(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
