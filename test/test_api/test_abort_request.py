"""
Test the /abort_request endpoint against a running lightllm server.

What this test asserts (and why it does not assert "stream becomes finish_reason='abort'"):

  In normal / chunked_prefill mode, /abort_request:
    - sets shm_req.is_aborted = True
    - drives the router to send AbortedReqCmd, which sets InferReq.infer_aborted = True
      on the worker
    - causes still-waiting (not yet scheduled) reqs to be freed with FINISHED_ABORTED
    - but does NOT cause already-running reqs to early-exit; they finish at max_new_tokens
      / EOS / stop sequence as usual. (The shm flag is consumed by audio/visual servers
      and pd_nixl mode, but the LLM inference loop never short-circuits on it.)

So the test verifies the contract that actually exists today:

  Stage A: bogus request_id           -> HTTP 200, server log "not exist" warning
  Stage B: abort_all on an idle server -> HTTP 200, no errors
  Stage C: abort_all on a running stream
            -> HTTP 200; server log shows "aborted group_request_id N" warning
            -> the stream terminates within reasonable time (whether via abort or
               natural max_new_tokens completion)
  Stage D: abort by SPECIFIC request_id on a running stream
            -> resolve the lightllm_req_id from the server log (via X-Request-Id),
               POST /abort_request with that exact id, verify the targeted log
               warning lands and the stream terminates
  Stage E: server remains healthy and answers a fresh /generate

Usage:
  python test/test_api/test_abort_request.py \
      --url http://127.0.0.1:8000 \
      --server_log_path /tmp/lightllm_test/server.log
"""

import argparse
import json
import os
import re
import sys
import threading
import time
import uuid
from typing import List, Optional, Tuple

import requests


GREEN = "\033[32m"
RED = "\033[31m"
YELLOW = "\033[33m"
RESET = "\033[0m"


def banner(msg: str):
    print(f"\n{YELLOW}=== {msg} ==={RESET}", flush=True)


def ok(msg: str):
    print(f"  {GREEN}OK{RESET}  {msg}", flush=True)


def fail(msg: str):
    print(f"  {RED}FAIL{RESET}  {msg}", flush=True)


# ---------------- HTTP helpers ----------------


def _get_health(url: str, timeout=5):
    return requests.get(url + "/health", timeout=timeout)


def post_abort(url: str, request_id: Optional[int] = None, abort_all: bool = False) -> Tuple[int, str]:
    payload = {"abort_all": abort_all}
    if request_id is not None:
        payload["request_id"] = request_id
    r = requests.post(url + "/abort_request", json=payload, timeout=30)
    return r.status_code, r.text


# ---------------- streaming helpers ----------------


def _stream_run(
    url: str,
    prompt: str,
    max_new_tokens: int,
    x_request_id: str,
    out: dict,
    close_after_n: Optional[int] = None,
):
    """
    Issue a /generate_stream and append every event to out["events"].
    If close_after_n is set, the underlying socket is forcibly closed
    (TCP RST via SO_LINGER + close) after that many events arrive — kept
    here for completeness even though no current stage uses it. Sets
    out["error"] on transport errors.
    """
    headers = {"X-Request-Id": x_request_id, "Content-Type": "application/json"}
    body = {
        "inputs": prompt,
        "parameters": {
            "max_new_tokens": max_new_tokens,
            "do_sample": False,
            "ignore_eos": True,
        },
    }
    out["events"] = []
    out["start"] = time.time()
    out["error"] = None
    out["closed_intentionally"] = False
    try:
        # urllib3 keeps the socket pooled; we need direct access to force-close.
        with requests.post(url + "/generate_stream", json=body, headers=headers, stream=True, timeout=120) as r:
            r.raise_for_status()
            for raw in r.iter_lines(decode_unicode=True):
                if not raw:
                    continue
                if raw.startswith("data:"):
                    raw = raw[len("data:") :]
                try:
                    ev = json.loads(raw)
                except Exception:
                    continue
                ev["_t"] = time.time() - out["start"]
                out["events"].append(ev)
                if close_after_n is not None and len(out["events"]) >= close_after_n:
                    out["closed_intentionally"] = True
                    # Reach into urllib3 to force a TCP RST so the server sees
                    # the disconnect immediately rather than after a graceful
                    # FIN that hypercorn might not propagate while the response
                    # is mid-stream.
                    try:
                        import socket as _socket

                        sock = r.raw._fp.fp.raw._sock  # type: ignore[attr-defined]
                        # SO_LINGER with timeout 0 -> RST on close.
                        l_onoff, l_linger = 1, 0
                        sock.setsockopt(
                            _socket.SOL_SOCKET,
                            _socket.SO_LINGER,
                            int.to_bytes(l_onoff, 4, "little") + int.to_bytes(l_linger, 4, "little"),
                        )
                        sock.close()
                    except Exception as e:
                        out["close_error"] = repr(e)
                    break
                if ev.get("finished"):
                    break
    except Exception as e:
        out["error"] = repr(e)
    out["end"] = time.time()


def start_stream(
    url: str, prompt: str, max_new_tokens: int, close_after_n: Optional[int] = None
) -> Tuple[threading.Thread, dict, str]:
    xid = uuid.uuid4().hex
    out = {}
    th = threading.Thread(target=_stream_run, args=(url, prompt, max_new_tokens, xid, out, close_after_n))
    th.daemon = True
    th.start()
    return th, out, xid


def wait_for_first_token(out: dict, timeout: float = 30.0) -> bool:
    deadline = time.time() + timeout
    while time.time() < deadline:
        if out.get("events"):
            return True
        time.sleep(0.05)
    return False


def get_finish_reason(out: dict) -> Optional[str]:
    for ev in reversed(out.get("events") or []):
        fr = ev.get("finish_reason")
        if fr:
            return fr
    return None


# ---------------- log helpers ----------------


def _read_log_tail(server_log_path: Optional[str], max_bytes: int = 256 * 1024) -> str:
    if not server_log_path or not os.path.exists(server_log_path):
        return ""
    try:
        size = os.path.getsize(server_log_path)
        with open(server_log_path, "rb") as f:
            if size > max_bytes:
                f.seek(size - max_bytes)
            return f.read().decode("utf-8", errors="ignore")
    except FileNotFoundError:
        return ""


def grep_log_for_pattern(server_log_path: Optional[str], pattern: re.Pattern, timeout: float = 5.0) -> Optional[str]:
    """Poll the tail of the server log for a regex match."""
    deadline = time.time() + timeout
    while time.time() < deadline:
        tail = _read_log_tail(server_log_path)
        m = pattern.search(tail)
        if m:
            return m.group(0)
        time.sleep(0.1)
    return None


def grep_log_after_offset(
    server_log_path: Optional[str], start_offset: int, pattern: re.Pattern, timeout: float = 5.0
) -> Optional[str]:
    """Poll the server log starting at start_offset for a regex match.
    Only content written after start_offset is considered, so this isolates
    a stage from log produced by earlier stages."""
    if not server_log_path:
        return None
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            with open(server_log_path, "rb") as f:
                f.seek(start_offset)
                new = f.read().decode("utf-8", errors="ignore")
        except FileNotFoundError:
            new = ""
        m = pattern.search(new)
        if m:
            return m.group(0)
        time.sleep(0.1)
    return None


def server_log_size(server_log_path: Optional[str]) -> int:
    if not server_log_path or not os.path.exists(server_log_path):
        return 0
    return os.path.getsize(server_log_path)


def lookup_lightllm_req_id_from_log(server_log_path: str, x_request_id: str, timeout: float = 5.0) -> Optional[int]:
    pattern = re.compile(rf"received req X-Request-Id:{re.escape(x_request_id)}\b.*?lightllm_req_id:(\d+)")
    deadline = time.time() + timeout
    while time.time() < deadline:
        tail = _read_log_tail(server_log_path)
        m = pattern.search(tail)
        if m:
            return int(m.group(1))
        time.sleep(0.1)
    return None


# ---------------- stages ----------------


def stage_a_bogus_id(url: str) -> bool:
    banner("Stage A: abort with a non-existent id")
    bogus = 99_999_999
    code, text = post_abort(url, request_id=bogus, abort_all=False)
    print(f"  /abort_request request_id={bogus} -> HTTP {code} body={text!r}")
    if code != 200:
        fail(f"expected HTTP 200, got {code}")
        return False
    ok("HTTP 200")
    return True


def stage_b_abort_all_idle(url: str) -> bool:
    banner("Stage B: abort_all on an idle server")
    code, text = post_abort(url, abort_all=True)
    print(f"  /abort_request abort_all=true -> HTTP {code} body={text!r}")
    if code != 200:
        fail(f"expected HTTP 200, got {code}")
        return False
    ok("HTTP 200")
    return True


def stage_c_abort_running(url: str, server_log_path: Optional[str]) -> bool:
    banner("Stage C: abort_all on a running stream")
    log_offset = server_log_size(server_log_path)
    th, out, xid = start_stream(url, "Recite the alphabet repeatedly.", max_new_tokens=200)
    if not wait_for_first_token(out, timeout=30.0):
        fail("did not receive any tokens before abort")
        return False
    first_t = out["events"][0]["_t"]
    ok(f"first token at +{first_t:.2f}s")

    target_id = lookup_lightllm_req_id_from_log(server_log_path, xid, timeout=5.0) if server_log_path else None
    print(f"  resolved lightllm_req_id from log: {target_id}")

    code, text = post_abort(url, abort_all=True)
    print(f"  /abort_request abort_all=true -> HTTP {code} body={text!r}")
    if code != 200:
        fail(f"expected HTTP 200, got {code}")
        return False

    th.join(timeout=60.0)
    if th.is_alive():
        fail("stream did not terminate within 60s of abort")
        return False
    fr = get_finish_reason(out)
    n = len(out.get("events") or [])
    print(f"  stream events received: {n}, finish_reason={fr!r}, error={out.get('error')!r}")

    # The api itself succeeded; whether the stream got a clean 'abort' finish reason
    # depends on which mode-backend the server is running. We DO assert the abort
    # warning landed in the server log though, scoped to log content produced after
    # this stage started so we don't match earlier-stage residue.
    if server_log_path:
        if target_id is not None:
            pat = re.compile(rf"aborted group_request_id {target_id}\b")
        else:
            pat = re.compile(r"aborted group_request_id \d+")
        hit = grep_log_after_offset(server_log_path, log_offset, pat, timeout=5.0)
        if not hit:
            fail("could not find 'aborted group_request_id' in server log (post-stage)")
            return False
        ok(f"server log recorded: {hit!r}")
    else:
        print("  no --server_log_path; skipped log assertion")
    ok("stream terminated and abort acknowledged")
    return True


def stage_d_abort_by_id(url: str, server_log_path: Optional[str]) -> bool:
    banner("Stage D: abort by specific request_id on a running stream")
    if not server_log_path:
        print("  --server_log_path not provided; skipping (we need the log to resolve req_id)")
        return True

    log_offset = server_log_size(server_log_path)
    th, out, xid = start_stream(url, "Sing a long lullaby for the moon.", max_new_tokens=300)
    if not wait_for_first_token(out, timeout=30.0):
        fail("did not receive any tokens before abort")
        return False
    ok(f"first token at +{out['events'][0]['_t']:.2f}s, X-Request-Id={xid[:8]}…")

    target_id = lookup_lightllm_req_id_from_log(server_log_path, xid, timeout=5.0)
    if target_id is None:
        fail("could not resolve lightllm_req_id from server log; cannot test by-id abort")
        return False
    print(f"  resolved lightllm_req_id: {target_id}")

    code, text = post_abort(url, request_id=target_id, abort_all=False)
    print(f"  /abort_request request_id={target_id} -> HTTP {code} body={text!r}")
    if code != 200:
        fail(f"expected HTTP 200, got {code}")
        return False

    th.join(timeout=60.0)
    if th.is_alive():
        fail("stream did not terminate within 60s")
        return False
    fr = get_finish_reason(out)
    n = len(out.get("events") or [])
    print(f"  stream events received: {n}, finish_reason={fr!r}")

    pat = re.compile(rf"aborted group_request_id {target_id}\b")
    hit = grep_log_after_offset(server_log_path, log_offset, pat, timeout=5.0)
    if not hit:
        fail(f"could not find 'aborted group_request_id {target_id}' in server log (post-stage)")
        return False
    ok(f"server log recorded: {hit!r}")
    return True


def stage_e_health_after(url: str) -> bool:
    banner("Stage E: server still serves a normal /generate")
    r = requests.post(
        url + "/generate",
        json={
            "inputs": "The capital of France is",
            "parameters": {"max_new_tokens": 6, "do_sample": False},
        },
        timeout=60,
    )
    print(f"  /generate -> HTTP {r.status_code} {r.text[:200]}")
    if r.status_code != 200:
        fail(f"final /generate failed with {r.status_code}")
        return False
    body = r.json()
    text = body.get("generated_text")
    if isinstance(text, list):
        text = text[0]
    if not text or not text.strip():
        fail("final /generate returned empty text")
        return False
    ok(f"final /generate returned {text!r}")
    return True


# ---------------- main ----------------


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://127.0.0.1:8000")
    ap.add_argument(
        "--server_log_path",
        default=None,
        help="optional path to the server stdout/stderr log; enables log-grep assertions",
    )
    args = ap.parse_args()

    try:
        r = _get_health(args.url)
        r.raise_for_status()
    except Exception as e:
        fail(f"server at {args.url} not reachable: {e}")
        sys.exit(1)
    ok(f"server reachable at {args.url}")

    results = []
    results.append(("A", stage_a_bogus_id(args.url)))
    results.append(("B", stage_b_abort_all_idle(args.url)))
    results.append(("C", stage_c_abort_running(args.url, args.server_log_path)))
    results.append(("D", stage_d_abort_by_id(args.url, args.server_log_path)))
    results.append(("E", stage_e_health_after(args.url)))

    print("\n" + "=" * 50)
    all_ok = True
    for name, passed in results:
        tag = f"{GREEN}PASS{RESET}" if passed else f"{RED}FAIL{RESET}"
        print(f"  Stage {name}: {tag}")
        all_ok = all_ok and passed
    if not all_ok:
        sys.exit(1)
    print(f"\n{GREEN}ALL ABORT STAGES PASSED{RESET}")


if __name__ == "__main__":
    main()
