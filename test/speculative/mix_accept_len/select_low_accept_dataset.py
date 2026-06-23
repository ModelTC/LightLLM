#!/usr/bin/env python3
"""
Probe per-sample MTP accept length from candidate datasets, then build:
1. A detailed metrics report.
2. A low-accept-length subset dataset.
3. A mixed GSM8K dataset with low-accept samples interleaved in.
"""

from __future__ import annotations

import argparse
import csv
import json
import random
import re
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple


REQUEST_ID_PREFIX = "low-accept-probe"
ROLE_MAP = {
    "human": "user",
    "user": "user",
    "assistant": "assistant",
    "system": "system",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Select low-accept-length samples and mix them into GSM8K.")
    parser.add_argument("--server-url", default="http://127.0.0.1:8088/v1/chat/completions", help="Chat completions endpoint.")
    parser.add_argument("--server-log", required=True, help="Readable LightLLM server log file.")
    parser.add_argument("--model", required=True, help="Model name sent to the OpenAI-compatible endpoint.")
    parser.add_argument(
        "--candidate-datasets",
        nargs="+",
        default=[
            "datasets/drop.json",
            "datasets/qasc.json",
            "datasets/quartz.json",
            "datasets/sciq.json",
            "datasets/openbookqa.json",
            "datasets/pubmedqa.json",
            "datasets/truthfulqa.json",
        ],
        help="Candidate datasets used to probe low accept length samples.",
    )
    parser.add_argument("--gsm8k-path", default="datasets/gsm8k.json", help="Base GSM8K dataset path.")
    parser.add_argument("--probe-samples-per-dataset", type=int, default=100, help="How many samples to probe from each candidate dataset.")
    parser.add_argument("--max-new-tokens", type=int, default=128, help="Max generated tokens for each probe request.")
    parser.add_argument("--request-timeout", type=int, default=1800, help="HTTP request timeout in seconds.")
    parser.add_argument("--log-timeout", type=float, default=30.0, help="How long to wait for the matching log line.")
    parser.add_argument(
        "--accept-threshold",
        type=float,
        default=None,
        help="Alias of --token-threshold for backward compatibility.",
    )
    parser.add_argument(
        "--token-threshold",
        type=float,
        default=None,
        help="Keep samples whose mtp_avg_token_per_step is <= this threshold.",
    )
    parser.add_argument(
        "--verify-threshold",
        type=float,
        default=None,
        help="Keep samples whose mtp_avg_verify_tokens_per_step is <= this threshold.",
    )
    parser.add_argument(
        "--low-accept-count",
        type=int,
        default=64,
        help="If threshold is not enough or not provided, keep the lowest N samples.",
    )
    parser.add_argument(
        "--gsm8k-limit",
        type=int,
        default=None,
        help="Optionally truncate GSM8K before mixing. Default keeps all rows.",
    )
    parser.add_argument(
        "--mix-mode",
        choices=["interleave", "append"],
        default="interleave",
        help="How to merge low-accept samples into GSM8K.",
    )
    parser.add_argument("--seed", type=int, default=0, help="Random seed used for stable ordering when needed.")
    parser.add_argument("--output-dir", required=True, help="Directory for reports and exported datasets.")
    return parser.parse_args()


def load_json_dataset(path: Path) -> List[dict]:
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError(f"Dataset {path} must be a JSON list.")
    return data


def normalize_messages(conversations: Sequence[dict]) -> Optional[Tuple[List[dict], str]]:
    if not conversations:
        return None

    last_assistant_idx = -1
    for idx in range(len(conversations) - 1, -1, -1):
        raw_role = str(conversations[idx].get("from") or conversations[idx].get("role") or "").lower()
        if ROLE_MAP.get(raw_role, "assistant") == "assistant":
            last_assistant_idx = idx
            break

    if last_assistant_idx <= 0:
        return None

    messages: List[dict] = []
    for turn in conversations[:last_assistant_idx]:
        raw_role = str(turn.get("from") or turn.get("role") or "user").lower()
        role = ROLE_MAP.get(raw_role, "assistant")
        content = turn.get("value") or turn.get("content") or ""
        if not content:
            continue
        messages.append({"role": role, "content": content})

    completion = conversations[last_assistant_idx].get("value") or conversations[last_assistant_idx].get("content") or ""
    if not messages or not completion:
        return None
    return messages, completion


def iter_probe_samples(dataset_path: Path, limit: int) -> Iterable[Tuple[int, dict, List[dict], str]]:
    dataset = load_json_dataset(dataset_path)
    emitted = 0
    for sample_index, item in enumerate(dataset):
        normalized = normalize_messages(item.get("conversations", []))
        if normalized is None:
            continue
        messages, reference_answer = normalized
        yield sample_index, item, messages, reference_answer
        emitted += 1
        if emitted >= limit:
            break


def send_probe_request(
    server_url: str,
    model: str,
    messages: List[dict],
    max_new_tokens: int,
    timeout: int,
    request_id: str,
) -> dict:
    payload = {
        "model": model,
        "messages": messages,
        "temperature": 0.0,
        "top_p": 1.0,
        "max_tokens": max_new_tokens,
        "stream": False,
        "ignore_eos": True,
    }
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        server_url,
        data=data,
        headers={
            "Content-Type": "application/json",
            "X-Request-Id": request_id,
            "User-Agent": "lightllm-low-accept-probe",
        },
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read().decode("utf-8"))


def wait_for_log_metrics(log_path: Path, request_id: str, start_offset: int, timeout_s: float) -> Dict[str, float]:
    patterns = {
        "mtp_avg_token_per_step": re.compile(r"mtp_avg_token_per_step:([0-9.]+)"),
        "mtp_avg_verify_tokens_per_step": re.compile(r"mtp_avg_verify_tokens_per_step:([0-9.]+)"),
        "out_token_counter": re.compile(r"out_token_counter:([0-9]+)"),
        "prompt_token_num": re.compile(r"prompt_token_num:([0-9]+)"),
        "first_token_cost_ms": re.compile(r"first_token_cost:([0-9.]+)ms"),
        "total_cost_time_ms": re.compile(r"total_cost_time:([0-9.]+)ms"),
        "pure_decode_time_ms": re.compile(r"pure_decode_time_ms:([0-9.]+)"),
        "pure_decode_token_num": re.compile(r"pure_decode_token_num:([0-9]+)"),
        "pure_decode_time_per_token_ms": re.compile(r"pure_decode_time_per_token_ms:([0-9.]+)"),
        "pure_decode_throughput": re.compile(r"pure_decode_throughput:([0-9.]+)"),
    }

    deadline = time.time() + timeout_s
    with log_path.open("r", encoding="utf-8", errors="ignore") as f:
        f.seek(start_offset)
        while time.time() < deadline:
            line = f.readline()
            if not line:
                time.sleep(0.1)
                continue
            if f"X-Request-Id:{request_id} " not in line:
                continue

            metrics: Dict[str, float] = {}
            for key, pattern in patterns.items():
                match = pattern.search(line)
                if match:
                    value = match.group(1)
                    metrics[key] = float(value) if "." in value else int(value)
            if "mtp_avg_token_per_step" in metrics and "mtp_avg_verify_tokens_per_step" in metrics:
                return metrics

    raise TimeoutError(f"Timed out waiting for metrics of request {request_id} in {log_path}.")


def probe_dataset(
    dataset_path: Path,
    args: argparse.Namespace,
    request_counter_start: int,
) -> Tuple[List[dict], int]:
    results: List[dict] = []
    request_counter = request_counter_start

    for sample_index, raw_item, messages, reference_answer in iter_probe_samples(dataset_path, args.probe_samples_per_dataset):
        request_id = f"{REQUEST_ID_PREFIX}-{request_counter}"
        request_counter += 1

        log_size_before = Path(args.server_log).stat().st_size
        try:
            response = send_probe_request(
                server_url=args.server_url,
                model=args.model,
                messages=messages,
                max_new_tokens=args.max_new_tokens,
                timeout=args.request_timeout,
                request_id=request_id,
            )
        except urllib.error.HTTPError as exc:
            body = exc.read().decode("utf-8", errors="ignore")
            raise RuntimeError(f"HTTP {exc.code} while probing {dataset_path} sample {sample_index}: {body}") from exc

        metrics = wait_for_log_metrics(
            log_path=Path(args.server_log),
            request_id=request_id,
            start_offset=log_size_before,
            timeout_s=args.log_timeout,
        )

        generated_text = ""
        choices = response.get("choices", [])
        if choices:
            message = choices[0].get("message", {})
            generated_text = message.get("content", "") or ""

        result = {
            "request_id": request_id,
            "dataset": dataset_path.stem,
            "dataset_path": str(dataset_path),
            "sample_index": sample_index,
            "mtp_avg_token_per_step": metrics["mtp_avg_token_per_step"],
            "accepted_draft_tokens_per_step": max(float(metrics["mtp_avg_token_per_step"]) - 1.0, 0.0),
            "mtp_avg_verify_tokens_per_step": metrics.get("mtp_avg_verify_tokens_per_step", 0.0),
            "prompt_token_num": metrics.get("prompt_token_num", 0),
            "out_token_counter": metrics.get("out_token_counter", 0),
            "first_token_cost_ms": metrics.get("first_token_cost_ms", 0.0),
            "total_cost_time_ms": metrics.get("total_cost_time_ms", 0.0),
            "pure_decode_time_ms": metrics.get("pure_decode_time_ms", 0.0),
            "pure_decode_token_num": metrics.get("pure_decode_token_num", 0),
            "pure_decode_time_per_token_ms": metrics.get("pure_decode_time_per_token_ms", 0.0),
            "pure_decode_throughput": metrics.get("pure_decode_throughput", 0.0),
            "messages": messages,
            "reference_answer": reference_answer,
            "generated_text": generated_text,
            "source_item": raw_item,
        }
        results.append(result)
        print(
            f"[probe] {dataset_path.stem} sample={sample_index} "
            f"mtp_avg_token_per_step={result['mtp_avg_token_per_step']:.4f} "
            f"verify={result['mtp_avg_verify_tokens_per_step']:.4f}"
        )

    return results, request_counter


def select_low_accept_samples(
    probe_results: List[dict],
    token_threshold: Optional[float],
    verify_threshold: Optional[float],
    low_accept_count: int,
) -> List[dict]:
    sorted_results = sorted(
        probe_results,
        key=lambda item: (
            item["mtp_avg_verify_tokens_per_step"],
            item["mtp_avg_token_per_step"],
            item["dataset"],
            item["sample_index"],
        ),
    )

    if token_threshold is not None or verify_threshold is not None:
        threshold_results = [
            item
            for item in sorted_results
            if (token_threshold is None or item["mtp_avg_token_per_step"] <= token_threshold)
            and (verify_threshold is None or item["mtp_avg_verify_tokens_per_step"] <= verify_threshold)
        ]
        if len(threshold_results) >= low_accept_count:
            return threshold_results[:low_accept_count]
        threshold_ids = {item["request_id"] for item in threshold_results}
        remainder = [item for item in sorted_results if item["request_id"] not in threshold_ids]
        return threshold_results + remainder[: max(low_accept_count - len(threshold_results), 0)]
    return sorted_results[:low_accept_count]


def interleave_gsm8k_with_low_accept(gsm8k_data: List[dict], low_accept_data: List[dict]) -> List[dict]:
    if not low_accept_data:
        return list(gsm8k_data)
    if not gsm8k_data:
        return list(low_accept_data)

    mixed: List[dict] = []
    slots = len(low_accept_data) + 1
    cursor = 0
    for slot in range(slots):
        next_cursor = round((slot + 1) * len(gsm8k_data) / slots)
        mixed.extend(gsm8k_data[cursor:next_cursor])
        cursor = next_cursor
        if slot < len(low_accept_data):
            mixed.append(low_accept_data[slot])
    return mixed


def export_probe_csv(rows: List[dict], path: Path) -> None:
    fieldnames = [
        "request_id",
        "dataset",
        "dataset_path",
        "sample_index",
        "mtp_avg_token_per_step",
        "accepted_draft_tokens_per_step",
        "mtp_avg_verify_tokens_per_step",
        "prompt_token_num",
        "out_token_counter",
        "first_token_cost_ms",
        "total_cost_time_ms",
        "pure_decode_time_ms",
        "pure_decode_token_num",
        "pure_decode_time_per_token_ms",
        "pure_decode_throughput",
    ]
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key) for key in fieldnames})


def export_json(path: Path, data: object) -> None:
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def build_per_dataset_summary(rows: Sequence[dict]) -> List[dict]:
    grouped: Dict[str, List[dict]] = {}
    for row in rows:
        grouped.setdefault(row["dataset"], []).append(row)

    summary_rows = []
    for dataset_name in sorted(grouped):
        items = grouped[dataset_name]
        token_values = [float(item["mtp_avg_token_per_step"]) for item in items]
        verify_values = [float(item["mtp_avg_verify_tokens_per_step"]) for item in items]
        accepted_values = [float(item["accepted_draft_tokens_per_step"]) for item in items]
        summary_rows.append(
            {
                "dataset": dataset_name,
                "count": len(items),
                "avg_mtp_avg_token_per_step": sum(token_values) / len(token_values),
                "avg_accepted_draft_tokens_per_step": sum(accepted_values) / len(accepted_values),
                "avg_mtp_avg_verify_tokens_per_step": sum(verify_values) / len(verify_values),
                "min_mtp_avg_token_per_step": min(token_values),
                "max_mtp_avg_token_per_step": max(token_values),
                "min_mtp_avg_verify_tokens_per_step": min(verify_values),
                "max_mtp_avg_verify_tokens_per_step": max(verify_values),
            }
        )
    return summary_rows


def export_per_dataset_summary_csv(rows: Sequence[dict], path: Path) -> None:
    fieldnames = [
        "dataset",
        "count",
        "avg_mtp_avg_token_per_step",
        "avg_accepted_draft_tokens_per_step",
        "avg_mtp_avg_verify_tokens_per_step",
        "min_mtp_avg_token_per_step",
        "max_mtp_avg_token_per_step",
        "min_mtp_avg_verify_tokens_per_step",
        "max_mtp_avg_verify_tokens_per_step",
    ]
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def build_low_accept_dataset(rows: Sequence[dict]) -> List[dict]:
    dataset_rows = []
    for row in rows:
        item = dict(row["source_item"])
        item["_low_accept_probe"] = {
            "source_dataset": row["dataset"],
            "source_sample_index": row["sample_index"],
            "mtp_avg_token_per_step": row["mtp_avg_token_per_step"],
            "accepted_draft_tokens_per_step": row["accepted_draft_tokens_per_step"],
            "mtp_avg_verify_tokens_per_step": row["mtp_avg_verify_tokens_per_step"],
        }
        dataset_rows.append(item)
    return dataset_rows


def main() -> None:
    args = parse_args()
    random.seed(args.seed)
    token_threshold = args.token_threshold if args.token_threshold is not None else args.accept_threshold

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    candidate_paths = [Path(p) for p in args.candidate_datasets]
    gsm8k_path = Path(args.gsm8k_path)
    server_log = Path(args.server_log)

    if not server_log.exists():
        raise FileNotFoundError(f"Server log does not exist: {server_log}")
    for path in candidate_paths + [gsm8k_path]:
        if not path.exists():
            raise FileNotFoundError(f"Dataset does not exist: {path}")

    all_probe_results: List[dict] = []
    request_counter = 0
    for dataset_path in candidate_paths:
        dataset_results, request_counter = probe_dataset(dataset_path, args, request_counter)
        all_probe_results.extend(dataset_results)

    if not all_probe_results:
        raise RuntimeError("No valid probe samples were collected.")

    low_accept_rows = select_low_accept_samples(
        probe_results=all_probe_results,
        token_threshold=token_threshold,
        verify_threshold=args.verify_threshold,
        low_accept_count=args.low_accept_count,
    )

    low_accept_dataset = build_low_accept_dataset(low_accept_rows)
    gsm8k_data = load_json_dataset(gsm8k_path)
    if args.gsm8k_limit is not None:
        gsm8k_data = gsm8k_data[: args.gsm8k_limit]

    if args.mix_mode == "append":
        mixed_dataset = list(gsm8k_data) + low_accept_dataset
    else:
        mixed_dataset = interleave_gsm8k_with_low_accept(gsm8k_data, low_accept_dataset)

    probe_report_path = output_dir / "probe_metrics.csv"
    probe_report_json_path = output_dir / "probe_metrics.json"
    per_dataset_summary_json_path = output_dir / "per_dataset_summary.json"
    per_dataset_summary_csv_path = output_dir / "per_dataset_summary.csv"
    low_accept_path = output_dir / "low_accept_subset.json"
    mixed_dataset_path = output_dir / "gsm8k_mixed_low_accept.json"
    summary_path = output_dir / "summary.json"
    per_dataset_summary = build_per_dataset_summary(all_probe_results)

    export_probe_csv(all_probe_results, probe_report_path)
    export_json(probe_report_json_path, all_probe_results)
    export_json(per_dataset_summary_json_path, per_dataset_summary)
    export_per_dataset_summary_csv(per_dataset_summary, per_dataset_summary_csv_path)
    export_json(low_accept_path, low_accept_dataset)
    export_json(mixed_dataset_path, mixed_dataset)

    summary = {
        "candidate_datasets": [str(p) for p in candidate_paths],
        "gsm8k_path": str(gsm8k_path),
        "server_url": args.server_url,
        "server_log": str(server_log),
        "probe_samples_per_dataset": args.probe_samples_per_dataset,
        "max_new_tokens": args.max_new_tokens,
        "accept_threshold": args.accept_threshold,
        "token_threshold": token_threshold,
        "verify_threshold": args.verify_threshold,
        "low_accept_count": args.low_accept_count,
        "mix_mode": args.mix_mode,
        "probed_sample_count": len(all_probe_results),
        "selected_low_accept_count": len(low_accept_rows),
        "mixed_dataset_count": len(mixed_dataset),
        "selected_low_accept_avg_verify_tokens_per_step": sum(
            row["mtp_avg_verify_tokens_per_step"] for row in low_accept_rows
        )
        / max(len(low_accept_rows), 1),
        "selected_low_accept_avg_token_per_step": sum(row["mtp_avg_token_per_step"] for row in low_accept_rows)
        / max(len(low_accept_rows), 1),
        "lowest_samples": [
            {
                "dataset": row["dataset"],
                "sample_index": row["sample_index"],
                "mtp_avg_token_per_step": row["mtp_avg_token_per_step"],
                "mtp_avg_verify_tokens_per_step": row["mtp_avg_verify_tokens_per_step"],
            }
            for row in low_accept_rows[: min(10, len(low_accept_rows))]
        ],
        "outputs": {
            "probe_metrics_csv": str(probe_report_path),
            "probe_metrics_json": str(probe_report_json_path),
            "per_dataset_summary_json": str(per_dataset_summary_json_path),
            "per_dataset_summary_csv": str(per_dataset_summary_csv_path),
            "low_accept_subset_json": str(low_accept_path),
            "gsm8k_mixed_low_accept_json": str(mixed_dataset_path),
        },
    }
    export_json(summary_path, summary)

    print("\n=== Done ===")
    print(f"Probed samples: {len(all_probe_results)}")
    print(f"Selected low-accept samples: {len(low_accept_rows)}")
    print(f"Mixed dataset size: {len(mixed_dataset)}")
    print(f"Probe metrics: {probe_report_path}")
    print(f"Low-accept subset: {low_accept_path}")
    print(f"Mixed GSM8K dataset: {mixed_dataset_path}")
    print(f"Summary: {summary_path}")


if __name__ == "__main__":
    main()
