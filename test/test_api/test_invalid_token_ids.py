"""
Smoke test for the invalid_token_ids feature (logit_bias path).

Hits the lightllm-native /generate endpoint, which forwards `logit_bias` keys
into the SamplingParams `invalid_token_ids` field. The kernel masks those
ids to -inf, so they must never appear in the output.

Run:
    python test/test_api/test_invalid_token_ids.py

Assumes the server is up on http://localhost:8000 and the model tokenizer
is Qwen3.5 (matches the launch command in the PR description).
"""


import json
import sys
from typing import Dict, List, Tuple

import requests
from transformers import AutoTokenizer


URL = "http://localhost:8000/generate"
HEADERS = {"Content-Type": "application/json"}
MODEL_DIR = "/nvme/models/Qwen3.5-35B-A3B"

# Stay under INVALID_TOKEN_IDS_MAX_LENGTH (default 10).
BLOCK_WORDS = ["the", " the", "The", " is", " a", " of", " and"]


def _post_generate(prompt: str, parameters: dict, timeout: int = 120) -> dict:
    payload = {"inputs": prompt, "parameters": parameters}
    resp = requests.post(URL, headers=HEADERS, data=json.dumps(payload), timeout=timeout)
    if resp.status_code != 200:
        raise RuntimeError(f"{resp.status_code} {resp.text}")
    return resp.json()


def _generated_text(resp: dict) -> str:
    text = resp["generated_text"]
    return text[0] if isinstance(text, list) else text


def _token_ids_from_details(resp: dict) -> List[int]:
    tokens = resp.get("tokens", [])
    if tokens and isinstance(tokens[0], list):
        tokens = tokens[0]
    out: List[int] = []
    for tok in tokens:
        tid = tok.get("id")
        if tid is not None:
            out.append(int(tid))
    return out


def _build_block_map(tokenizer) -> Tuple[Dict[int, float], Dict[int, str]]:
    """Map token id -> bias (-100 = block) and id -> source word."""
    bias: Dict[int, float] = {}
    source: Dict[int, str] = {}
    for w in BLOCK_WORDS:
        ids = tokenizer.encode(w, add_special_tokens=False)
        for tid in ids:
            bias.setdefault(tid, -100.0)
            source.setdefault(tid, w)
    return bias, source


def test_invalid_token_ids():
    print("[1/3] Loading tokenizer...", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR, trust_remote_code=True)

    bias_map, source_map = _build_block_map(tokenizer)
    blocked_ids = sorted(bias_map.keys())
    print(f"      Blocking {len(blocked_ids)} token ids: {blocked_ids}")
    for tid in blocked_ids:
        print(f"        {tid:6d} <- {source_map[tid]!r}")

    prompt = "Write three short English sentences about San Francisco. " "Mention the bay, the bridge and the weather."
    base_params = {
        "do_sample": False,
        "temperature": 1.0,
        "max_new_tokens": 80,
        "return_details": True,
    }

    print("[2/3] Baseline request (no logit_bias)...", flush=True)
    base_resp = _post_generate(prompt, dict(base_params))
    base_text = _generated_text(base_resp)
    base_ids = _token_ids_from_details(base_resp)
    print(f"      text: {base_text!r}")
    base_hits = [tid for tid in base_ids if tid in bias_map]
    print(f"      blocked-tokens that appeared in baseline: {len(base_hits)} ({base_hits[:10]})")

    print("[3/3] logit_bias request...", flush=True)
    bias_params = dict(base_params)
    bias_params["logit_bias"] = {str(k): v for k, v in bias_map.items()}
    biased_resp = _post_generate(prompt, bias_params)
    biased_text = _generated_text(biased_resp)
    biased_ids = _token_ids_from_details(biased_resp)
    print(f"      text: {biased_text!r}")
    biased_hits = [(tid, source_map[tid]) for tid in biased_ids if tid in bias_map]
    print(f"      blocked-tokens that appeared with bias: {len(biased_hits)} ({biased_hits[:10]})")

    failures = []
    if biased_hits:
        failures.append(f"Blocked token ids leaked into biased output: {biased_hits}")

    # Sanity check: the baseline should have produced at least one of the blocked tokens.
    # If it did not, the test is uninformative (but still passes the strict check above).
    if not base_hits:
        print(
            "      WARNING: baseline did not produce any of the target tokens; "
            "the assertion below is trivially satisfied."
        )

    if biased_text == base_text:
        failures.append("Biased output is identical to baseline; bias may not be applied.")

    if failures:
        for f in failures:
            print(f"FAIL: {f}", file=sys.stderr)
        sys.exit(1)

    print("PASS: invalid_token_ids correctly suppressed blocked tokens.")


if __name__ == "__main__":
    test_invalid_token_ids()
