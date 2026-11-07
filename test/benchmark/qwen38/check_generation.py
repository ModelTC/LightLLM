import argparse
import hashlib
import json
import os
from pathlib import Path
import secrets

import requests
from tokenizers import Tokenizer

REVISION = "de4b8e4d43b917e7706784d8bb445c9af86a3540"
TOKENIZER_SHA256 = "0997f410c57a1f4e53b09e4be8f4a172d90edd9564368fb0847030937229b9f3"


def verified_checkpoint(root):
    acquisition = json.loads((root / "acquisition.json").read_text())
    verified = json.loads((root / "verified_files.json").read_text())
    if acquisition["status"] != "verified" or acquisition["revision"] != REVISION or verified["revision"] != REVISION:
        raise ValueError("The complete pinned checkpoint must finish verification before this check")
    index = json.loads((root / "model.safetensors.index.json").read_text())
    required = set(index["weight_map"].values()) | {"config.json", "model.safetensors.index.json"}
    for name in required:
        metadata = verified["files"][name]
        if (root / name).stat().st_size != metadata["size"]:
            raise ValueError(f"Checkpoint file size changed after verification: {name}")
    return verified


def cases_for(root, long_lengths=(2049, 4063)):
    if not long_lengths or any(type(length) is not int or length < 1 for length in long_lengths):
        raise ValueError("Long prompt lengths must be positive integers")
    if len(set(long_lengths)) != len(long_lengths):
        raise ValueError("Long prompt lengths must be distinct")
    path = root / "tokenizer.json"
    if hashlib.sha256(path.read_bytes()).hexdigest() != TOKENIZER_SHA256:
        raise ValueError("Tokenizer does not match the pinned checkpoint")
    tokenizer = Tokenizer.from_file(str(path))
    prompts = {
        "english": "The capital of France is",
        "chinese": "中国的首都是",
        "code_unicode": "def add(a, b):\n    # 求和 🐍\n    return",
    }
    cases = {name: tokenizer.encode(prompt, add_special_tokens=False).ids for name, prompt in prompts.items()}
    seed = tokenizer.encode("A short factual note about language models and their context. ").ids
    for length in long_lengths:
        cases[f"long_{length}"] = (seed * (length // len(seed) + 1))[:length]
        cases[f"repeat_long_{length}"] = list(cases[f"long_{length}"])
    return cases


def generate(
    client, base_url, framework, model, tokens, output_length, timeout, response_path=None, *, cold_cache=False
):
    if framework == "lightllm":
        endpoint = "/generate"
        payload = {
            "inputs": tokens,
            "parameters": {
                "do_sample": False,
                "ignore_eos": True,
                "max_new_tokens": output_length,
                "return_details": True,
            },
        }
        if cold_cache:
            payload["parameters"]["disable_prompt_cache"] = True
    else:
        endpoint = "/v1/completions"
        payload = {
            "model": model,
            "prompt": tokens,
            "max_tokens": output_length,
            "temperature": 0,
            "seed": 3107,
            "ignore_eos": True,
            "return_token_ids": True,
        }
        if cold_cache:
            payload["cache_salt"] = secrets.token_urlsafe(32)
    response = client.post(base_url.rstrip("/") + endpoint, json=payload, timeout=timeout)
    if response_path is not None:
        response_path.write_text(response.text)
    response.raise_for_status()
    body = response.json()
    if framework == "lightllm":
        ids = [token["id"] for token in body["tokens"]]
        prompt_length = body["prompt_tokens"]
    else:
        ids = body["choices"][0]["token_ids"]
        prompt_length = body["usage"]["prompt_tokens"]
    if not isinstance(ids, list) or len(ids) != output_length or not all(type(token) is int for token in ids):
        received = len(ids) if isinstance(ids, list) else type(ids).__name__
        raise ValueError(f"Expected {output_length} integer token IDs, received {received}")
    if prompt_length != len(tokens):
        raise ValueError("Service prompt token count does not match the shared tokenized input")
    if cold_cache:
        cached = (
            [token.get("prompt_cache_len") for token in body["tokens"]]
            if framework == "lightllm"
            else [(body["usage"].get("prompt_tokens_details") or {}).get("cached_tokens")]
        )
        if not cached or any(type(count) is not int or count != 0 for count in cached):
            raise ValueError("Cold generation must explicitly report zero cached input tokens")
    return {"token_ids": ids, "response": body}


def main():
    if not os.environ.get("EXP_ACTIVE"):
        raise RuntimeError("Run through exp")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--framework", choices=("lightllm", "vllm", "sglang"), required=True)
    parser.add_argument("--base-url", required=True)
    parser.add_argument("--model-directory", type=Path, required=True)
    parser.add_argument("--service-metadata", type=Path, required=True)
    parser.add_argument("--reference", type=Path)
    parser.add_argument("--output-length", type=int, default=32)
    parser.add_argument("--long-prompt-lengths", type=int, nargs="+", default=[2049, 4063])
    parser.add_argument("--timeout", type=float, default=600)
    parser.add_argument(
        "--cold-cache", action="store_true", help="Isolate each request and require zero reported cache hits"
    )
    parser.add_argument(
        "--record-observations",
        action="store_true",
        help="Retain baseline repeat failures as observations, without qualifying a reference",
    )
    args = parser.parse_args()
    if args.record_observations and (args.framework == "lightllm" or args.reference or not args.cold_cache):
        parser.error("Baseline observations require cold requests and no reference; native qualification stays strict")
    if args.output_length < 1:
        parser.error("Output length must be positive")
    if any(length < 1 for length in args.long_prompt_lengths) or len(set(args.long_prompt_lengths)) != len(
        args.long_prompt_lengths
    ):
        parser.error("Long prompt lengths must be distinct positive integers")
    run = Path(os.environ["EXP_RUN_DIR"])
    metadata = json.loads(args.service_metadata.read_text())
    for key in ("framework", "version", "command", "hardware", "configuration"):
        if not metadata.get(key):
            raise ValueError(f"Service metadata must record {key}")
    if metadata["framework"] != args.framework:
        raise ValueError("Service metadata names a different framework")
    (run / "verified-files.json").write_text(json.dumps(verified_checkpoint(args.model_directory), indent=2))
    cases = cases_for(args.model_directory, args.long_prompt_lengths)
    reference = json.loads(args.reference.read_text()) if args.reference else None
    if reference and (
        reference["status"] not in ("passed", "recorded")
        or reference["revision"] != REVISION
        or reference["cases"] != cases
        or reference["output_length"] != args.output_length
        or reference.get("cache_policy", "service_default") != ("cold" if args.cold_cache else "service_default")
    ):
        raise ValueError("Reference must use the same revision, exact inputs and output length")
    report = {
        "status": "running",
        "revision": REVISION,
        "tokenizer_sha256": TOKENIZER_SHA256,
        "service": metadata,
        "base_url": args.base_url,
        "cases": cases,
        "output_length": args.output_length,
        "long_prompt_lengths": args.long_prompt_lengths,
        "reference": str(args.reference) if args.reference else None,
        "cache_policy": "cold" if args.cold_cache else "service_default",
        "results": {},
    }
    output = run / "generation.json"
    client = requests.Session()
    client.trust_env = False
    active_case = None
    try:
        for name, tokens in cases.items():
            active_case = name
            result = generate(
                client,
                args.base_url,
                args.framework,
                str(args.model_directory.resolve()),
                tokens,
                args.output_length,
                args.timeout,
                response_path=run / f"response-{name}.txt",
                cold_cache=args.cold_cache,
            )
            report["results"][name] = result
            result["reference_matches"] = (
                result["token_ids"] == reference["results"][name]["token_ids"] if reference else None
            )
            output.write_text(json.dumps(report, indent=2))
            print(
                json.dumps(
                    {"case": name, "token_ids": result["token_ids"], "reference_matches": result["reference_matches"]}
                ),
                flush=True,
            )
        active_case = None
        results = report["results"]
        repeat_matches = all(
            results[f"long_{length}"]["token_ids"] == results[f"repeat_long_{length}"]["token_ids"]
            for length in args.long_prompt_lengths
        )
        report["repeat_consistency"] = dict(
            status="passed" if repeat_matches else "failed",
            mismatches=[
                length
                for length in args.long_prompt_lengths
                if results[f"long_{length}"]["token_ids"] != results[f"repeat_long_{length}"]["token_ids"]
            ],
        )
        if args.record_observations:
            report["status"] = "observations_recorded"
            report[
                "scope"
            ] = "Baseline token/count observations; repeat failures remain explicit, no exact-output qualification"
            return
        if not repeat_matches or any(item["reference_matches"] is False for item in results.values()):
            raise ValueError("Greedy token IDs differ across repeated prompts or reference services")
        report["status"] = "passed" if reference else "recorded"
    except Exception as error:
        report.update(status="failed", failed_case=active_case, error_type=type(error).__name__, error=str(error))
        raise
    finally:
        output.write_text(json.dumps(report, indent=2))
        client.close()


if __name__ == "__main__":
    main()
