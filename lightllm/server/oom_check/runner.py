import asyncio
import json
import time
from typing import List, Optional

import aiohttp
import requests

from lightllm.utils.log_utils import init_logger

logger = init_logger(__name__)


_SAFETY_MARGIN_TOKENS = 8
_HEALTH_POLL_INTERVAL_S = 2.0
_HEALTH_POLL_TIMEOUT_S = 300.0
_REQUEST_TIMEOUT_S = 1800.0


def build_input_string(model_dir: str, trust_remote_code: bool, n_tokens: int) -> str:
    """Produce a string that tokenizes to exactly ``n_tokens`` tokens under the
    model's HF tokenizer. Uses `" the"` as a cheap repeatable unit and truncates
    at the id level before decoding so the resulting string round-trips stably.
    """
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_dir, trust_remote_code=trust_remote_code)
    seed = " the" * (n_tokens + 32)
    ids = tokenizer.encode(seed, add_special_tokens=False)
    if len(ids) < n_tokens:
        # Extremely unusual tokenizer (fewer ids than chars); pad by repeating.
        repeat = (n_tokens // max(len(ids), 1)) + 2
        ids = tokenizer.encode(seed * repeat, add_special_tokens=False)
    return tokenizer.decode(ids[:n_tokens])


def classify_outcome(status_code: Optional[int], exception: Optional[str]) -> str:
    if exception is not None:
        if "timeout" in exception.lower() or "timedout" in exception.lower():
            return "timeout"
        return "other"
    if status_code == 200:
        return "ok"
    if status_code is not None and 400 <= status_code < 500:
        return "http_4xx"
    if status_code is not None and 500 <= status_code < 600:
        return "http_5xx"
    return "other"


def summarize(outcomes: List[dict], duration_s: float) -> dict:
    by_class: dict = {}
    latencies: List[float] = []
    for o in outcomes:
        klass = o["class"]
        by_class[klass] = by_class.get(klass, 0) + 1
        if o.get("latency_s") is not None:
            latencies.append(o["latency_s"])
    ok_count = by_class.get("ok", 0)
    total = len(outcomes)
    latencies.sort()
    n = len(latencies)

    def pct(p: float) -> Optional[float]:
        if n == 0:
            return None
        idx = min(n - 1, int(p * n))
        return latencies[idx]

    return {
        "total": total,
        "ok": ok_count,
        "failed": total - ok_count,
        "by_class": by_class,
        "duration_s": round(duration_s, 3),
        "p50_s": None if pct(0.50) is None else round(pct(0.50), 3),
        "p95_s": None if pct(0.95) is None else round(pct(0.95), 3),
        "max_s": None if n == 0 else round(latencies[-1], 3),
    }


def wait_health(host: str, port: int, timeout_s: float = _HEALTH_POLL_TIMEOUT_S) -> bool:
    url = f"http://{host}:{port}/health"
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        try:
            resp = requests.get(url, timeout=5)
            if resp.status_code == 200:
                return True
        except Exception:
            pass
        time.sleep(_HEALTH_POLL_INTERVAL_S)
    return False


async def _fire_one(session: aiohttp.ClientSession, url: str, payload: dict, sem: asyncio.Semaphore) -> dict:
    async with sem:
        t0 = time.time()
        try:
            async with session.post(url, json=payload) as resp:
                status = resp.status
                await resp.read()
                latency = time.time() - t0
                return {"class": classify_outcome(status, None), "status": status, "latency_s": latency, "error": None}
        except asyncio.TimeoutError:
            return {"class": "timeout", "status": None, "latency_s": time.time() - t0, "error": "timeout"}
        except Exception as e:
            return {
                "class": classify_outcome(None, repr(e)),
                "status": None,
                "latency_s": time.time() - t0,
                "error": repr(e),
            }


async def _run_stress(
    host: str, port: int, input_str: str, max_new_tokens: int, concurrency: int, total: int
) -> List[dict]:
    url = f"http://{host}:{port}/generate"
    payload = {
        "inputs": input_str,
        "parameters": {
            "do_sample": False,
            "ignore_eos": True,
            "max_new_tokens": max_new_tokens,
        },
    }
    timeout = aiohttp.ClientTimeout(total=_REQUEST_TIMEOUT_S)
    sem = asyncio.Semaphore(concurrency)
    async with aiohttp.ClientSession(timeout=timeout) as session:
        tasks = [_fire_one(session, url, payload, sem) for _ in range(total)]
        return await asyncio.gather(*tasks)


def run_oom_check(
    host: str, port: int, model_dir: str, trust_remote_code: bool, running_max_req_size: int, max_req_total_len: int
) -> dict:
    if running_max_req_size <= 0:
        logger.warning(f"[OOM_CHECK] running_max_req_size={running_max_req_size}, skipping.")
        return {"skipped": True}

    total = 2 * running_max_req_size
    n_input = int(0.9 * max_req_total_len)
    max_new_tokens = max_req_total_len - n_input - _SAFETY_MARGIN_TOKENS
    logger.info(
        f"[OOM_CHECK] starting: total={total}, concurrency={running_max_req_size}, "
        f"input_tokens={n_input}, max_new_tokens={max_new_tokens}"
    )

    if not wait_health(host, port):
        logger.error(f"[OOM_CHECK] /health did not go green within {_HEALTH_POLL_TIMEOUT_S}s; aborting.")
        return {"skipped": True, "reason": "health_timeout"}

    input_str = build_input_string(model_dir, trust_remote_code, n_input)
    logger.info(f"[OOM_CHECK] input string built (len={len(input_str)} chars).")

    t0 = time.time()
    outcomes = asyncio.run(_run_stress(host, port, input_str, max_new_tokens, running_max_req_size, total))
    duration = time.time() - t0

    summary = summarize(outcomes, duration)
    logger.info(f"[OOM_CHECK_RESULT] {json.dumps(summary)}")
    return summary
