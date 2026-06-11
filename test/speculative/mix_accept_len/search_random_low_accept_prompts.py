#!/usr/bin/env python3
"""
Randomly synthesize prompts that are likely to shorten speculative accept length,
probe them against a running LightLLM server, and export the best prompts as a
benchmark dataset.

This is intentionally different from selecting rows from existing datasets:
it creates synthetic, distribution-shifted prompts that often make draft and
target models disagree more quickly.

Example:
    python test/speculative/low_accept/search_random_low_accept_prompts.py \
        --server-log lightllm_qwen3_32b.log \
        --model qwen/qwen3-32b \
        --tokenizer /mtc/models/qwen3-32b \
        --num-candidates 500 \
        --keep-best 32 \
        --target-size 1000 \
        --output dataset/random_synth_low_accept_1k.json
"""

from __future__ import annotations

import argparse
import json
import random
import re
import string
import time
import urllib.request
from copy import deepcopy
from pathlib import Path
from typing import Dict, List, Sequence

REQUEST_ID_PREFIX = "random-low-accept"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Randomly synthesize prompts to minimize speculative accept length.")
    parser.add_argument("--server-url", default="http://127.0.0.1:8088/v1/chat/completions")
    parser.add_argument("--server-log", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--tokenizer", required=True, help="Tokenizer path used to sample unusual vocab tokens.")
    parser.add_argument("--num-candidates", type=int, default=500, help="How many random prompts to probe.")
    parser.add_argument("--keep-best", type=int, default=32, help="How many lowest-accept prompts to keep.")
    parser.add_argument("--target-size", type=int, default=1000, help="How many requests to export.")
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument("--request-timeout", type=int, default=1800)
    parser.add_argument("--log-timeout", type=float, default=30.0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output", required=True)
    return parser.parse_args()


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
            "User-Agent": "lightllm-random-low-accept",
        },
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read().decode("utf-8"))


def wait_for_log_metrics(log_path: Path, request_id: str, start_offset: int, timeout_s: float) -> Dict[str, float]:
    patterns = {
        "mtp_avg_token_per_step": re.compile(r"mtp_avg_token_per_step:([0-9.]+)"),
        "mtp_avg_verify_tokens_per_step": re.compile(r"mtp_avg_verify_tokens_per_step:([0-9.]+)"),
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
                    metrics[key] = float(match.group(1))
            if "mtp_avg_token_per_step" in metrics and "mtp_avg_verify_tokens_per_step" in metrics:
                return metrics

    raise TimeoutError(f"Timed out waiting for metrics of request {request_id} in {log_path}.")


def random_ascii_noise(rng: random.Random, min_len: int = 20, max_len: int = 80) -> str:
    alphabet = string.ascii_letters + string.digits + string.punctuation
    n = rng.randint(min_len, max_len)
    return "".join(rng.choice(alphabet) for _ in range(n))


def random_hex_block(rng: random.Random, blocks: int = 8) -> str:
    return " ".join("".join(rng.choice("0123456789abcdef") for _ in range(rng.randint(6, 16))) for _ in range(blocks))


def random_repetition_pattern(rng: random.Random) -> str:
    base = rng.choice(["alpha", "delta", "sigma", "void", "trace", "token", "phase", "error"])
    repeat = rng.randint(8, 24)
    sep = rng.choice([" ", "-", "_", " | ", " / "])
    return sep.join([f"{base}{rng.randint(0, 999)}" for _ in range(repeat)])


def random_vocab_fragments(tokenizer, rng: random.Random, count: int = 32) -> str:
    vocab = tokenizer.get_vocab()
    all_tokens = list(vocab.keys())
    chosen = []
    attempts = 0
    while len(chosen) < count and attempts < count * 50:
        attempts += 1
        token = rng.choice(all_tokens)
        # Prefer "weird" subword fragments over clean common words.
        if len(token.strip()) <= 1:
            continue
        if token.lower() in {"the", "and", "you", "for", "that"}:
            continue
        chosen.append(token)
    return " ".join(chosen[:count])


def random_multilingual_fragment(rng: random.Random) -> str:
    pieces = [
        "请忽略直觉只输出最短可能模式",
        "контрольный фрагмент без очевидного продолжения",
        "fragmento extrano con simbolos y cambios de tema",
        "مقطع غير معتاد مع انتقالات مفاجئة",
        "section inattendue avec nombres 731 442 908",
        "unordered state transition pseudo trace begin",
    ]
    rng.shuffle(pieces)
    return " || ".join(pieces[: rng.randint(2, 5)])


def build_prompt_templates(tokenizer, rng: random.Random) -> List[str]:
    weird_vocab = random_vocab_fragments(tokenizer, rng, count=rng.randint(16, 40))
    noise = random_ascii_noise(rng)
    hex_block = random_hex_block(rng, blocks=rng.randint(6, 12))
    repetition = random_repetition_pattern(rng)
    multilingual = random_multilingual_fragment(rng)

    templates = [
        f"Continue the following corrupted trace exactly:\n{noise}\n{hex_block}\n{repetition}",
        f"Do not explain. Predict the next line for this unstable sequence:\n{weird_vocab}\n{noise}",
        f"Merge these fragments into one stream without reasoning:\n{multilingual}\n{weird_vocab}",
        f"Repeat only the latent pattern hidden below:\n{hex_block}\n{multilingual}\n{repetition}",
        f"Given this malformed data, output the immediate continuation token:\n{noise}\n{weird_vocab}\n{hex_block}",
        f"Simulate a broken tokenizer dump and continue it:\n{weird_vocab}\n{repetition}\n{noise}",
    ]
    return templates


def synthesize_prompt(tokenizer, rng: random.Random) -> str:
    templates = build_prompt_templates(tokenizer, rng)
    prompt = rng.choice(templates)
    if rng.random() < 0.5:
        prompt += "\n\nConstraints: no explanation, no formatting cleanup, continue directly."
    if rng.random() < 0.35:
        prompt += f"\n\nMarker: {random_ascii_noise(rng, 8, 24)}"
    return prompt


def probe_prompt(
    prompt: str,
    args: argparse.Namespace,
    request_index: int,
) -> Dict[str, object]:
    request_id = f"{REQUEST_ID_PREFIX}-{request_index}"
    log_size_before = Path(args.server_log).stat().st_size
    response = send_probe_request(
        server_url=args.server_url,
        model=args.model,
        messages=[{"role": "user", "content": prompt}],
        max_new_tokens=args.max_new_tokens,
        timeout=args.request_timeout,
        request_id=request_id,
    )
    metrics = wait_for_log_metrics(
        log_path=Path(args.server_log),
        request_id=request_id,
        start_offset=log_size_before,
        timeout_s=args.log_timeout,
    )
    generated_text = ""
    choices = response.get("choices", [])
    if choices:
        generated_text = choices[0].get("message", {}).get("content", "") or ""
    return {
        "request_id": request_id,
        "prompt": prompt,
        "generated_text": generated_text,
        "mtp_avg_token_per_step": float(metrics["mtp_avg_token_per_step"]),
        "mtp_avg_verify_tokens_per_step": float(metrics["mtp_avg_verify_tokens_per_step"]),
        "accepted_draft_tokens_per_step": max(float(metrics["mtp_avg_token_per_step"]) - 1.0, 0.0),
        "source_item": {
            "conversations": [
                {"from": "human", "value": prompt},
                {"from": "assistant", "value": generated_text or "synthetic-target-placeholder"},
            ]
        },
    }


def expand_rows(rows: Sequence[dict], target_size: int, seed: int) -> List[dict]:
    rng = random.Random(seed)
    repeated = []
    per_row = target_size // len(rows)
    remainder = target_size % len(rows)
    for i, row in enumerate(rows):
        repeat_count = per_row + (1 if i < remainder else 0)
        for repeat_index in range(repeat_count):
            item = deepcopy(row["source_item"])
            item["_random_prompt_source"] = {
                "subset_index": i,
                "repeat_index": repeat_index,
                "request_id": row["request_id"],
                "mtp_avg_token_per_step": row["mtp_avg_token_per_step"],
                "mtp_avg_verify_tokens_per_step": row["mtp_avg_verify_tokens_per_step"],
                "accepted_draft_tokens_per_step": row["accepted_draft_tokens_per_step"],
            }
            repeated.append(item)
    rng.shuffle(repeated)
    return repeated


def main() -> None:
    args = parse_args()
    from transformers import AutoTokenizer

    rng = random.Random(args.seed)
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer, trust_remote_code=False)

    all_rows = []
    for request_index in range(args.num_candidates):
        prompt = synthesize_prompt(tokenizer, rng)
        row = probe_prompt(prompt, args, request_index)
        all_rows.append(row)
        print(
            f"[probe] idx={request_index} "
            f"token={row['mtp_avg_token_per_step']:.4f} "
            f"verify={row['mtp_avg_verify_tokens_per_step']:.4f}"
        )

    best_rows = sorted(
        all_rows,
        key=lambda row: (
            float(row["mtp_avg_token_per_step"]),
            float(row["mtp_avg_verify_tokens_per_step"]),
        ),
    )[: args.keep_best]

    dataset_rows = expand_rows(best_rows, args.target_size, args.seed)
    output_path = Path(args.output)
    meta_path = output_path.with_suffix(output_path.suffix + ".meta.json")
    probe_path = output_path.with_suffix(output_path.suffix + ".probe.json")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("w", encoding="utf-8") as f:
        json.dump(dataset_rows, f, ensure_ascii=False, indent=2)
    with probe_path.open("w", encoding="utf-8") as f:
        json.dump(all_rows, f, ensure_ascii=False, indent=2)

    best_avg = sum(float(row["mtp_avg_token_per_step"]) for row in best_rows) / len(best_rows)
    best_verify_avg = sum(float(row["mtp_avg_verify_tokens_per_step"]) for row in best_rows) / len(best_rows)
    with meta_path.open("w", encoding="utf-8") as f:
        json.dump(
            {
                "num_candidates": args.num_candidates,
                "keep_best": args.keep_best,
                "target_size": args.target_size,
                "seed": args.seed,
                "best_avg_mtp_avg_token_per_step": best_avg,
                "best_avg_mtp_avg_verify_tokens_per_step": best_verify_avg,
                "best_rows": [
                    {
                        "request_id": row["request_id"],
                        "prompt": row["prompt"],
                        "mtp_avg_token_per_step": row["mtp_avg_token_per_step"],
                        "mtp_avg_verify_tokens_per_step": row["mtp_avg_verify_tokens_per_step"],
                    }
                    for row in best_rows
                ],
                "output_dataset": str(output_path),
                "probe_output": str(probe_path),
            },
            f,
            ensure_ascii=False,
            indent=2,
        )

    print(f"Exported dataset: {output_path}")
    print(f"Exported metadata: {meta_path}")
    print(f"Exported probe results: {probe_path}")
    print(f"Best avg token={best_avg:.4f}, best avg verify={best_verify_avg:.4f}")


if __name__ == "__main__":
    main()
