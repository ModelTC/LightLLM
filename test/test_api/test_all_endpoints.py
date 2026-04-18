"""End-to-end smoke tests for every HTTP endpoint exposed by LightLLM.

Usage:
    python test_all_endpoints.py --url http://127.0.0.1:8089
    python test_all_endpoints.py --url http://127.0.0.1:8089 --model Qwen3-8B --skip get_score

Endpoints covered (see lightllm/server/api_http.py):
    GET/POST  /liveness
    GET/POST  /readiness
    GET/POST  /get_model_name
    GET/HEAD  /health, /healthz
    GET       /token_load
    GET       /v1/models
    GET       /metrics
    GET/POST  /tokens
    POST      /generate                (LightLLM native)
    POST      /generate_stream         (LightLLM SSE)
    POST      /                        (compat: stream flag routes to the two above)
    POST      /get_score               (reward-model endpoint; skipped by default)
    POST      /v1/completions          (OpenAI)
    POST      /v1/chat/completions     (OpenAI, non-stream + stream)
    POST      /v1/messages             (Anthropic Messages)

Websocket endpoints (/pd_register, /kv_move_status) are intentionally omitted —
they are only used internally by prefill/decode master workers.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple

import requests

GREEN = "\033[32m"
RED = "\033[31m"
YELLOW = "\033[33m"
CYAN = "\033[36m"
DIM = "\033[2m"
RESET = "\033[0m"


@dataclass
class TestResult:
    name: str
    ok: bool
    detail: str = ""
    elapsed_ms: float = 0.0
    skipped: bool = False


@dataclass
class Suite:
    base_url: str
    model: Optional[str]
    timeout: float
    prompt: str
    chat_prompt: str
    chat_max_tokens: int
    skip: set
    only: set
    results: List[TestResult] = field(default_factory=list)

    def url(self, path: str) -> str:
        return self.base_url.rstrip("/") + path

    def should_run(self, name: str) -> bool:
        if self.only:
            return name in self.only
        return name not in self.skip

    def record(self, result: TestResult) -> None:
        tag = f"{YELLOW}SKIP{RESET}" if result.skipped else f"{GREEN}PASS{RESET}" if result.ok else f"{RED}FAIL{RESET}"
        t = f"{DIM}({result.elapsed_ms:6.1f} ms){RESET}"
        print(f"  [{tag}] {result.name:<32s} {t}  {result.detail}")
        self.results.append(result)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _run(name: str, fn: Callable[[], Tuple[bool, str]]) -> TestResult:
    start = time.perf_counter()
    try:
        ok, detail = fn()
    except Exception as exc:  # noqa: BLE001
        ok, detail = False, f"exception: {exc!r}"
    elapsed = (time.perf_counter() - start) * 1000.0
    return TestResult(name=name, ok=ok, detail=detail, elapsed_ms=elapsed)


def _assert_json_key(data: Dict[str, Any], key: str) -> None:
    if key not in data:
        raise AssertionError(f"response missing key '{key}': {data!r}")


def _preview(text: str, limit: int = 80) -> str:
    text = text.replace("\n", " ").strip()
    return text if len(text) <= limit else text[: limit - 3] + "..."


# ---------------------------------------------------------------------------
# Individual endpoint tests
# ---------------------------------------------------------------------------


def test_liveness(s: Suite) -> Tuple[bool, str]:
    r = requests.get(s.url("/liveness"), timeout=s.timeout)
    r.raise_for_status()
    return r.json().get("status") == "ok", f"status={r.json().get('status')}"


def test_readiness(s: Suite) -> Tuple[bool, str]:
    r = requests.get(s.url("/readiness"), timeout=s.timeout)
    r.raise_for_status()
    return r.json().get("status") == "ok", f"status={r.json().get('status')}"


def test_health(s: Suite) -> Tuple[bool, str]:
    r = requests.get(s.url("/health"), timeout=s.timeout)
    # /health may return 503 if backend is busy; we still count HTTP reachable as OK
    if r.status_code not in (200, 503):
        return False, f"unexpected status {r.status_code}"
    return True, f"http {r.status_code} msg={r.json().get('message')}"


def test_healthz(s: Suite) -> Tuple[bool, str]:
    r = requests.get(s.url("/healthz"), timeout=s.timeout)
    if r.status_code not in (200, 503):
        return False, f"unexpected status {r.status_code}"
    return True, f"http {r.status_code}"


def test_health_head(s: Suite) -> Tuple[bool, str]:
    r = requests.head(s.url("/health"), timeout=s.timeout)
    if r.status_code not in (200, 503):
        return False, f"unexpected status {r.status_code}"
    return True, f"http {r.status_code}"


def test_get_model_name(s: Suite) -> Tuple[bool, str]:
    r = requests.get(s.url("/get_model_name"), timeout=s.timeout)
    r.raise_for_status()
    data = r.json()
    _assert_json_key(data, "model_name")
    # POST form should work identically
    r2 = requests.post(s.url("/get_model_name"), timeout=s.timeout)
    r2.raise_for_status()
    return data == r2.json(), f"model={data['model_name']}"


def test_token_load(s: Suite) -> Tuple[bool, str]:
    r = requests.get(s.url("/token_load"), timeout=s.timeout)
    r.raise_for_status()
    data = r.json()
    for key in ("current_load", "logical_max_load", "dynamic_max_load"):
        _assert_json_key(data, key)
    return True, _preview(json.dumps(data, ensure_ascii=False), 120)


def test_v1_models(s: Suite) -> Tuple[bool, str]:
    r = requests.get(s.url("/v1/models"), timeout=s.timeout)
    r.raise_for_status()
    data = r.json()
    if not data.get("data"):
        return False, "empty model list"
    card = data["data"][0]
    for key in ("id", "created", "max_model_len"):
        _assert_json_key(card, key)
    return True, f"id={card['id']} max_len={card['max_model_len']}"


def test_metrics(s: Suite) -> Tuple[bool, str]:
    r = requests.get(s.url("/metrics"), timeout=s.timeout)
    r.raise_for_status()
    body = r.text
    # Prometheus scrape format — at minimum contains HELP/TYPE markers when any metric exists.
    ok = ("# HELP" in body) or ("lightllm_" in body) or len(body) > 0
    return ok, f"bytes={len(body)}"


def test_tokens(s: Suite) -> Tuple[bool, str]:
    payload = {"text": s.prompt, "parameters": {"max_new_tokens": 4}}
    r = requests.post(s.url("/tokens"), json=payload, timeout=s.timeout)
    r.raise_for_status()
    data = r.json()
    _assert_json_key(data, "ntokens")
    return data["ntokens"] > 0, f"ntokens={data['ntokens']}"


def test_generate(s: Suite) -> Tuple[bool, str]:
    payload = {
        "inputs": s.prompt,
        "parameters": {"do_sample": False, "max_new_tokens": 16, "return_details": True},
    }
    r = requests.post(s.url("/generate"), json=payload, timeout=s.timeout)
    r.raise_for_status()
    data = r.json()
    _assert_json_key(data, "generated_text")
    text = data["generated_text"]
    if isinstance(text, list):
        text = text[0]
    ok = isinstance(text, str) and len(text) > 0
    return ok, f"text={_preview(text)}"


def test_generate_stream(s: Suite) -> Tuple[bool, str]:
    payload = {
        "inputs": s.prompt,
        "parameters": {"do_sample": False, "max_new_tokens": 16},
    }
    chunks: List[str] = []
    finished = False
    with requests.post(s.url("/generate_stream"), json=payload, timeout=s.timeout, stream=True) as r:
        r.raise_for_status()
        for raw in r.iter_lines(decode_unicode=True):
            if not raw or not raw.startswith("data:"):
                continue
            evt = json.loads(raw[len("data:") :])
            chunks.append(evt["token"]["text"])
            if evt.get("finished"):
                finished = True
                break
    text = "".join(chunks)
    return finished and len(text) > 0, f"chunks={len(chunks)} text={_preview(text)}"


def test_compat_generate(s: Suite) -> Tuple[bool, str]:
    # POST / with stream=false routes to /generate
    payload = {
        "inputs": s.prompt,
        "parameters": {"do_sample": False, "max_new_tokens": 8},
        "stream": False,
    }
    r = requests.post(s.url("/"), json=payload, timeout=s.timeout)
    r.raise_for_status()
    data = r.json()
    _assert_json_key(data, "generated_text")
    return True, f"text={_preview(str(data['generated_text']))}"


def test_get_score(s: Suite) -> Tuple[bool, str]:
    # Reward-model only — typically fails on chat/causal-LM deployments.
    payload = {
        "chat": s.prompt,
        "parameters": {},
    }
    r = requests.post(s.url("/get_score"), json=payload, timeout=s.timeout)
    if r.status_code != 200:
        return False, f"http {r.status_code} (expected if model is not a reward model)"
    data = r.json()
    return "score" in data, f"score={data.get('score')}"


def test_v1_completions(s: Suite) -> Tuple[bool, str]:
    payload = {
        "model": s.model or "default",
        "prompt": s.prompt,
        "max_tokens": 16,
        "temperature": 0.0,
    }
    r = requests.post(s.url("/v1/completions"), json=payload, timeout=s.timeout)
    r.raise_for_status()
    data = r.json()
    _assert_json_key(data, "choices")
    text = data["choices"][0].get("text", "")
    return len(text) > 0, f"text={_preview(text)} usage={data.get('usage')}"


def test_v1_chat_completions(s: Suite) -> Tuple[bool, str]:
    payload = {
        "model": s.model or "default",
        "messages": [{"role": "user", "content": s.chat_prompt}],
        "max_tokens": s.chat_max_tokens,
        "temperature": 0.0,
    }
    r = requests.post(s.url("/v1/chat/completions"), json=payload, timeout=s.timeout)
    r.raise_for_status()
    data = r.json()
    _assert_json_key(data, "choices")
    msg = data["choices"][0]["message"]
    content = msg.get("content") or ""
    reasoning = msg.get("reasoning") or msg.get("reasoning_content") or ""
    combined = content + reasoning
    tag = "content" if content else ("reasoning" if reasoning else "empty")
    return len(combined) > 0, f"{tag}={_preview(combined)}"


def test_v1_chat_completions_stream(s: Suite) -> Tuple[bool, str]:
    payload = {
        "model": s.model or "default",
        "messages": [{"role": "user", "content": s.chat_prompt}],
        "max_tokens": s.chat_max_tokens,
        "temperature": 0.0,
        "stream": True,
    }
    content_chunks: List[str] = []
    reasoning_chunks: List[str] = []
    done = False
    with requests.post(s.url("/v1/chat/completions"), json=payload, timeout=s.timeout, stream=True) as r:
        r.raise_for_status()
        for raw in r.iter_lines(decode_unicode=True):
            if not raw or not raw.startswith("data:"):
                continue
            body = raw[len("data:") :].strip()
            if body == "[DONE]":
                done = True
                break
            evt = json.loads(body)
            choices = evt.get("choices") or []
            if not choices:
                continue  # usage-only chunk, no delta
            delta = choices[0].get("delta") or {}
            if delta.get("content"):
                content_chunks.append(delta["content"])
            if delta.get("reasoning"):
                reasoning_chunks.append(delta["reasoning"])
            if delta.get("reasoning_content"):
                reasoning_chunks.append(delta["reasoning_content"])
    content = "".join(content_chunks)
    reasoning = "".join(reasoning_chunks)
    combined = content + reasoning
    tag = "content" if content else ("reasoning" if reasoning else "empty")
    total_chunks = len(content_chunks) + len(reasoning_chunks)
    return done and len(combined) > 0, f"chunks={total_chunks} {tag}={_preview(combined)}"


def test_v1_responses(s: Suite) -> Tuple[bool, str]:
    payload = {
        "model": s.model or "default",
        "input": s.chat_prompt,
        "max_output_tokens": s.chat_max_tokens,
    }
    r = requests.post(s.url("/v1/responses"), json=payload, timeout=s.timeout)
    r.raise_for_status()
    data = r.json()
    if data.get("object") != "response":
        return False, f"unexpected object={data.get('object')}"
    if data.get("status") not in ("completed", "incomplete"):
        return False, f"unexpected status={data.get('status')}"
    text = ""
    for item in data.get("output") or []:
        if item.get("type") != "message":
            continue
        for part in item.get("content") or []:
            if part.get("type") == "output_text":
                text += part.get("text", "")
    usage = data.get("usage") or {}
    ok = usage.get("output_tokens", 0) > 0
    return ok, f"status={data['status']} out_tokens={usage.get('output_tokens')} text={_preview(text)}"


def test_v1_responses_stream(s: Suite) -> Tuple[bool, str]:
    payload = {
        "model": s.model or "default",
        "input": s.chat_prompt,
        "max_output_tokens": s.chat_max_tokens,
        "stream": True,
    }
    event_types: List[str] = []
    text_deltas: List[str] = []
    completed_status: Optional[str] = None
    with requests.post(s.url("/v1/responses"), json=payload, timeout=s.timeout, stream=True) as r:
        r.raise_for_status()
        for raw in r.iter_lines(decode_unicode=True):
            if not raw or not raw.startswith("data:"):
                continue
            body = raw[len("data:") :].strip()
            if body == "[DONE]":
                break
            evt = json.loads(body)
            etype = evt.get("type") or ""
            event_types.append(etype)
            if etype == "response.output_text.delta":
                text_deltas.append(evt.get("delta", ""))
            elif etype == "response.completed":
                completed_status = (evt.get("response") or {}).get("status")
    first_ok = bool(event_types) and event_types[0] == "response.created"
    last_ok = bool(event_types) and event_types[-1] == "response.completed"
    ok = first_ok and last_ok and completed_status in ("completed", "incomplete")
    text = "".join(text_deltas)
    return ok, f"events={len(event_types)} status={completed_status} text={_preview(text)}"


def test_v1_messages(s: Suite) -> Tuple[bool, str]:
    payload = {
        "model": s.model or "default",
        "max_tokens": s.chat_max_tokens,
        "messages": [{"role": "user", "content": s.chat_prompt}],
    }
    r = requests.post(s.url("/v1/messages"), json=payload, timeout=s.timeout)
    if r.status_code == 500 and "requires the 'litellm' package" in r.text:
        return False, "server missing 'litellm' dependency (pip install 'lightllm[anthropic]')"
    r.raise_for_status()
    data = r.json()
    _assert_json_key(data, "content")
    blocks = data["content"]
    text = "".join(b.get("text", "") for b in blocks if b.get("type") == "text")
    thinking = "".join(b.get("thinking", "") for b in blocks if b.get("type") == "thinking")
    combined = text + thinking
    tag = "text" if text else ("thinking" if thinking else "empty")
    return len(combined) > 0, f"stop={data.get('stop_reason')} {tag}={_preview(combined)}"


def test_v1_messages_invalid_json(s: Suite) -> Tuple[bool, str]:
    """Malformed JSON body → 400 with Anthropic-shaped error envelope."""
    r = requests.post(
        s.url("/v1/messages"),
        data="not json at all",
        headers={"Content-Type": "application/json"},
        timeout=s.timeout,
    )
    if r.status_code != 400:
        return False, f"expected 400, got {r.status_code}"
    try:
        body = r.json()
    except Exception:
        return False, f"non-JSON error body: {r.text[:80]}"
    if body.get("type") != "error" or not isinstance(body.get("error"), dict):
        return False, f"not Anthropic-shaped: {body!r}"
    return True, "invalid JSON -> 400 anthropic error"


def test_v1_responses_invalid_json(s: Suite) -> Tuple[bool, str]:
    r = requests.post(
        s.url("/v1/responses"),
        data="not json at all",
        headers={"Content-Type": "application/json"},
        timeout=s.timeout,
    )
    if r.status_code != 400:
        return False, f"expected 400, got {r.status_code}"
    try:
        body = r.json()
    except Exception:
        return False, f"non-JSON error body: {r.text[:80]}"
    err = (body or {}).get("error")
    if not isinstance(err, dict) or not err.get("message"):
        return False, f"not Responses-shaped: {body!r}"
    return True, "invalid JSON -> 400 responses error"


def test_v1_responses_stateless_lifecycle_404(s: Suite) -> Tuple[bool, str]:
    r = requests.get(s.url("/v1/responses/resp_does_not_exist"), timeout=s.timeout)
    if r.status_code != 404:
        return False, f"expected 404, got {r.status_code}"
    err = ((r.json() or {}).get("error")) or {}
    if "stateless" not in (err.get("message") or "").lower():
        return False, f"message missing 'stateless' hint: {err}"
    return True, "retrieve -> 404 stateless"


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------


TESTS: List[Tuple[str, Callable[[Suite], Tuple[bool, str]]]] = [
    ("liveness", test_liveness),
    ("readiness", test_readiness),
    ("health", test_health),
    ("healthz", test_healthz),
    ("health_head", test_health_head),
    ("get_model_name", test_get_model_name),
    ("token_load", test_token_load),
    ("v1_models", test_v1_models),
    ("metrics", test_metrics),
    ("tokens", test_tokens),
    ("generate", test_generate),
    ("generate_stream", test_generate_stream),
    ("compat_generate", test_compat_generate),
    ("v1_completions", test_v1_completions),
    ("v1_chat_completions", test_v1_chat_completions),
    ("v1_chat_completions_stream", test_v1_chat_completions_stream),
    ("v1_messages", test_v1_messages),
    ("v1_responses", test_v1_responses),
    ("v1_responses_stream", test_v1_responses_stream),
    ("v1_messages_invalid_json", test_v1_messages_invalid_json),
    ("v1_responses_invalid_json", test_v1_responses_invalid_json),
    ("v1_responses_stateless_404", test_v1_responses_stateless_lifecycle_404),
    # get_score needs a reward model — opt-in only.
    ("get_score", test_get_score),
]

DEFAULT_SKIP = {"get_score"}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--url", default="http://127.0.0.1:8089", help="base URL of the LightLLM server")
    p.add_argument("--model", default=None, help="model id for OpenAI/Anthropic endpoints (defaults to /v1/models)")
    p.add_argument("--timeout", type=float, default=60.0, help="per-request timeout in seconds")
    p.add_argument("--prompt", default="San Francisco is a", help="raw completion prompt")
    p.add_argument("--chat-prompt", default="Say 'hello' in one word.", help="chat prompt")
    p.add_argument(
        "--chat-max-tokens",
        type=int,
        default=512,
        help="max_tokens for chat/messages endpoints (large enough to clear a thinking block)",
    )
    p.add_argument(
        "--skip",
        default=",".join(sorted(DEFAULT_SKIP)),
        help=f"comma-separated test names to skip (default: {','.join(sorted(DEFAULT_SKIP))})",
    )
    p.add_argument("--only", default="", help="comma-separated test names to run exclusively")
    p.add_argument("--list", action="store_true", help="print test names and exit")
    return p.parse_args()


def resolve_model(base_url: str, timeout: float) -> Optional[str]:
    try:
        r = requests.get(base_url.rstrip("/") + "/v1/models", timeout=timeout)
        r.raise_for_status()
        return r.json()["data"][0]["id"]
    except Exception:
        return None


def main() -> int:
    args = parse_args()

    if args.list:
        for name, _ in TESTS:
            print(name)
        return 0

    model = args.model or resolve_model(args.url, args.timeout)

    suite = Suite(
        base_url=args.url,
        model=model,
        timeout=args.timeout,
        prompt=args.prompt,
        chat_prompt=args.chat_prompt,
        chat_max_tokens=args.chat_max_tokens,
        skip={n.strip() for n in args.skip.split(",") if n.strip()},
        only={n.strip() for n in args.only.split(",") if n.strip()},
    )

    print(f"{CYAN}LightLLM endpoint smoke test{RESET}")
    print(f"  base_url = {suite.base_url}")
    print(f"  model    = {suite.model or '(unresolved)'}")
    print(f"  timeout  = {suite.timeout}s")
    print()

    for name, fn in TESTS:
        if not suite.should_run(name):
            suite.record(TestResult(name=name, ok=True, skipped=True, detail="skipped"))
            continue
        suite.record(_run(name, lambda fn=fn: fn(suite)))

    ran = [r for r in suite.results if not r.skipped]
    passed = [r for r in ran if r.ok]
    failed = [r for r in ran if not r.ok]

    skipped_count = len(suite.results) - len(ran)
    print()
    print(f"{CYAN}Summary{RESET}: ran={len(ran)} passed={len(passed)} " f"failed={len(failed)} skipped={skipped_count}")
    if failed:
        print(f"{RED}Failed tests:{RESET}")
        for r in failed:
            print(f"  - {r.name}: {r.detail}")

    return 0 if not failed else 1


if __name__ == "__main__":
    sys.exit(main())
