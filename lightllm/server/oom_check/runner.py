import asyncio
import base64
import io
import json
import os
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

# Profile: burst one wave at t=0 so ViT encode and LLM decode peak together.
#   * image is big (stress ViT)
#   * max_new_tokens is moderate (keep LLM decode busy, not trivial)
#   * total = 2 * running_max_req_size so a second wave of ViT work overlaps
#     the first wave's LLM decode
#   * concurrency = total means all requests fire simultaneously (client-side
#     semaphore is effectively a no-op; the server decides batch composition)
#   * hard deadline cancels everything at 60s so the probe never overruns
# total / concurrency default to ``2 * running_max_req_size`` (resolved at call
# time). Any knob can be overridden by its env var or an explicit kwarg.
_DEFAULT_IMAGE_SIZE = 4096
_DEFAULT_MAX_NEW_TOKENS = 256
_DEFAULT_DEADLINE_S = 60.0

# Qwen2-VL / Qwen2.5-VL / Qwen3-VL and Qwen3.5 all consume this exact span in
# raw prompts; the model's encode() strips the inner pad and splices in real
# image token ids. Non-Qwen multimodal families (InternVL, LLaVA, Gemma3, ...)
# use different placeholders and are not supported by this probe — see
# ``_is_qwen_family_vl`` below; we fall back to text mode for those.
_IMAGE_PLACEHOLDER = "<|vision_start|><|image_pad|><|vision_end|>"
_MULTIMODAL_PROMPT = _IMAGE_PLACEHOLDER + "Describe the image."
_QWEN_VL_MODEL_TYPES = {
    "qwen2_vl",
    "qwen2_5_vl",
    "qwen3_vl",
    "qwen3_vl_moe",
    "qwen3_5",
    "qwen3_5_moe",
}


def _env_int(name: str, default: int) -> int:
    raw = os.environ.get(name)
    if raw is None or raw.strip() == "":
        return default
    try:
        v = int(raw)
        return v if v > 0 else default
    except ValueError:
        return default


def _env_float(name: str, default: float) -> float:
    raw = os.environ.get(name)
    if raw is None or raw.strip() == "":
        return default
    try:
        v = float(raw)
        return v if v > 0 else default
    except ValueError:
        return default


def _is_qwen_family_vl(model_dir: str) -> bool:
    """Return True if the model's HF config identifies it as a Qwen-family VL
    model (the only family whose prompt placeholder this probe knows how to
    construct). Any other multimodal family will have a different placeholder
    and must fall back to text mode.
    """
    try:
        from transformers.configuration_utils import PretrainedConfig

        cfg, _ = PretrainedConfig.get_config_dict(model_dir)
        model_type = cfg.get("model_type")
        if model_type in _QWEN_VL_MODEL_TYPES:
            return True
        # Qwen3-Omni wraps the VL model in thinker_config.
        thinker_vision_type = (
            cfg.get("thinker_config", {}).get("vision_config", {}).get("model_type")
        )
        if thinker_vision_type == "qwen3_omni_moe_vision_encoder":
            return True
        return False
    except Exception as e:
        logger.warning(f"[OOM_CHECK] could not read config for {model_dir}: {e!r}")
        return False


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


def build_large_image_b64(size: int) -> str:
    """Generate a ``size x size`` RGB noise image and return it as a base64-encoded
    JPEG. Noise is used deliberately so JPEG can't collapse the payload to a tiny
    constant-colour blob, and the pixel dimensions are what the ViT path stresses.
    """
    import numpy as np
    from PIL import Image

    arr = np.random.randint(0, 256, (size, size, 3), dtype=np.uint8)
    img = Image.fromarray(arr)
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=90)
    return base64.b64encode(buf.getvalue()).decode("ascii")


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
    host: str, port: int, payload: dict, concurrency: int, total: int, deadline_s: float
) -> List[dict]:
    """Fire ``total`` /generate requests with at most ``concurrency`` in flight and
    a hard ``deadline_s`` wall clock. At the deadline, any task that hasn't produced
    a response is cancelled; closing the aiohttp session immediately afterwards
    makes the server observe client disconnects and abort the underlying requests,
    which releases KV / multimodal cache slots.

    Returns one outcome dict per original task, in order. Tasks that finished are
    reported as ``_fire_one`` returned them; cancelled tasks are reported with
    class ``aborted`` and latency equal to ``deadline_s``.
    """
    url = f"http://{host}:{port}/generate"
    timeout = aiohttp.ClientTimeout(total=_REQUEST_TIMEOUT_S)
    sem = asyncio.Semaphore(concurrency)
    # aiohttp's default TCPConnector caps at limit=100 total connections, which
    # silently serializes the "burst" once ``total`` exceeds that. Disable both
    # caps so all requests really do fire simultaneously.
    connector = aiohttp.TCPConnector(limit=0, limit_per_host=0)
    async with aiohttp.ClientSession(timeout=timeout, connector=connector) as session:
        tasks = [asyncio.create_task(_fire_one(session, url, payload, sem)) for _ in range(total)]
        t0 = time.time()
        try:
            done, pending = await asyncio.wait(tasks, timeout=deadline_s)
        except Exception as e:
            logger.error(f"[OOM_CHECK] wait() failed: {e!r}")
            done, pending = set(t for t in tasks if t.done()), set(t for t in tasks if not t.done())

        if pending:
            logger.info(
                f"[OOM_CHECK] deadline {deadline_s:.1f}s reached with {len(pending)} "
                f"in-flight request(s); cancelling."
            )
            for p in pending:
                p.cancel()
            # Let cancellations propagate. We swallow the resulting CancelledErrors;
            # the `async with` below will also close the session which disconnects
            # and prompts the server to abort the requests.
            await asyncio.gather(*pending, return_exceptions=True)

        outcomes: List[dict] = []
        cancelled_latency = round(time.time() - t0, 3)
        for t in tasks:
            if t in done and not t.cancelled():
                try:
                    outcomes.append(t.result())
                except asyncio.CancelledError:
                    outcomes.append(
                        {"class": "aborted", "status": None, "latency_s": cancelled_latency, "error": "cancelled"}
                    )
                except Exception as e:
                    outcomes.append(
                        {"class": "other", "status": None, "latency_s": cancelled_latency, "error": repr(e)}
                    )
            else:
                outcomes.append(
                    {"class": "aborted", "status": None, "latency_s": cancelled_latency, "error": "deadline_hit"}
                )
        return outcomes


def _build_multimodal_payload(image_size: int, max_new_tokens: int) -> dict:
    img_b64 = build_large_image_b64(image_size)
    return {
        "inputs": _MULTIMODAL_PROMPT,
        "parameters": {
            "do_sample": False,
            "ignore_eos": True,
            "max_new_tokens": max_new_tokens,
        },
        "multimodal_params": {
            "images": [{"type": "base64", "data": img_b64}],
        },
    }


def _build_text_payload(model_dir: str, trust_remote_code: bool, max_req_total_len: int, max_new_tokens: int) -> dict:
    n_input = int(0.9 * max_req_total_len)
    capped = max_req_total_len - n_input - _SAFETY_MARGIN_TOKENS
    effective_new = max(1, min(max_new_tokens, capped))
    input_str = build_input_string(model_dir, trust_remote_code, n_input)
    logger.info(f"[OOM_CHECK] text input built (tokens={n_input}, chars={len(input_str)}).")
    return {
        "inputs": input_str,
        "parameters": {
            "do_sample": False,
            "ignore_eos": True,
            "max_new_tokens": effective_new,
        },
    }


def run_oom_check(
    host: str,
    port: int,
    model_dir: str,
    trust_remote_code: bool,
    running_max_req_size: int,
    max_req_total_len: int,
    image_size: Optional[int] = None,
    total: Optional[int] = None,
    concurrency: Optional[int] = None,
    max_new_tokens: Optional[int] = None,
    deadline_s: Optional[float] = None,
) -> dict:
    if running_max_req_size <= 0:
        logger.warning(f"[OOM_CHECK] running_max_req_size={running_max_req_size}, skipping.")
        return {"skipped": True}

    # Resolve knobs: explicit kwarg > env var > default. Burst profile: total and
    # concurrency both default to 2 * running_max_req_size so the full wave fires
    # at t=0 and the server's ViT / LLM stages overlap under load. Keep all values
    # well above 0.
    default_wave = 2 * running_max_req_size
    if total is None:
        total = _env_int("LIGHTLLM_OOM_CHECK_TOTAL", default_wave)
    if concurrency is None:
        concurrency = _env_int("LIGHTLLM_OOM_CHECK_CONCURRENCY", default_wave)
    if max_new_tokens is None:
        max_new_tokens = _env_int("LIGHTLLM_OOM_CHECK_MAX_NEW_TOKENS", _DEFAULT_MAX_NEW_TOKENS)
    if image_size is None:
        image_size = _env_int("LIGHTLLM_OOM_CHECK_IMAGE_SIZE", _DEFAULT_IMAGE_SIZE)
    if deadline_s is None:
        deadline_s = _env_float("LIGHTLLM_OOM_CHECK_DEADLINE_S", _DEFAULT_DEADLINE_S)
    concurrency = max(1, min(concurrency, total))

    from lightllm.utils.config_utils import has_vision_module

    has_vision = has_vision_module(model_dir)
    is_qwen_vl = has_vision and _is_qwen_family_vl(model_dir)
    if has_vision and not is_qwen_vl:
        logger.warning(
            "[OOM_CHECK] model has a vision module but is not in the Qwen VL family; "
            "the multimodal prompt placeholder this probe builds won't match, "
            "falling back to text mode."
        )

    if not wait_health(host, port):
        logger.error(f"[OOM_CHECK] /health did not go green within {_HEALTH_POLL_TIMEOUT_S}s; aborting.")
        return {"skipped": True, "reason": "health_timeout"}

    if is_qwen_vl:
        logger.info(
            f"[OOM_CHECK] multimodal burst: image={image_size}x{image_size}, total={total}, "
            f"concurrency={concurrency}, max_new_tokens={max_new_tokens}, deadline={deadline_s:.1f}s"
        )
        payload = _build_multimodal_payload(image_size, max_new_tokens)
    else:
        logger.info(
            f"[OOM_CHECK] text burst: total={total}, concurrency={concurrency}, "
            f"max_new_tokens={max_new_tokens}, deadline={deadline_s:.1f}s"
        )
        payload = _build_text_payload(model_dir, trust_remote_code, max_req_total_len, max_new_tokens)

    t0 = time.time()
    outcomes = asyncio.run(_run_stress(host, port, payload, concurrency, total, deadline_s))
    duration = time.time() - t0

    summary = summarize(outcomes, duration)
    summary["mode"] = "multimodal" if is_qwen_vl else "text"
    logger.info(f"[OOM_CHECK_RESULT] {json.dumps(summary)}")
    return summary
