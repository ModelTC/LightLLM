"""OpenAI-compatible Vision provider adapter."""

from __future__ import annotations

import base64
import hashlib
from typing import Any, Awaitable, Callable, Optional

from fastapi import Request

from lightllm.utils.log_utils import init_logger

from ..visual_trace import VisualTraceRecorder
from .config import (
    VisualProxyTimeoutError,
    VisualProxyUpstreamError,
    VisualProxyUpstreamRejectedError,
)
from .constants import VISION_MAX_OUTPUT_TOKENS, VISION_TASK_MAX_CHARS
from .evidence import RegisteredImage
from .images import image_url_for_visual_provider
from .sanitizer import _IncrementalVisualTextCleaner, strip_backend_special_tokens
from .runtime import VisualProxyRuntime


logger = init_logger(__name__)
VisualResultCallback = Callable[[str], Awaitable[None]]


class VisualRemoteResult(str):
    """String-compatible Vision result with auditable provider metadata."""

    def __new__(
        cls,
        value: str,
        *,
        finish_reason: str = "",
        usage: Optional[dict[str, Any]] = None,
        image_digest_sha256: str = "",
    ):
        instance = str.__new__(cls, value)
        instance.finish_reason = finish_reason
        instance.usage = dict(usage or {})
        instance.image_digest_sha256 = image_digest_sha256
        return instance


def _image_digest(image_url: str) -> str:
    if not image_url.startswith("data:"):
        return ""
    payload = image_url.split(",", 1)[1]
    return hashlib.sha256(base64.b64decode(payload, validate=True)).hexdigest()


def _request_payload(model: str, image_url: str, task: str) -> dict[str, Any]:
    return {
        "model": model,
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": task},
                    {"type": "image_url", "image_url": {"url": image_url}},
                ],
            }
        ],
        "max_tokens": VISION_MAX_OUTPUT_TOKENS,
        "temperature": 0.0,
        "chat_template_kwargs": {"enable_thinking": False},
    }


def _remote_content(data: Any) -> str:
    if not isinstance(data, dict):
        raise VisualProxyUpstreamError("Visual remote returned a non-object response")
    choices = data.get("choices")
    if not isinstance(choices, list) or not choices or not isinstance(choices[0], dict):
        raise VisualProxyUpstreamError("Visual remote response has no choices[0]")
    message = choices[0].get("message")
    if not isinstance(message, dict):
        raise VisualProxyUpstreamError(
            "Visual remote response has no choices[0].message"
        )
    content = message.get("content")
    visible_text = ""
    if isinstance(content, str):
        visible_text = strip_backend_special_tokens(content)
    elif isinstance(content, list):
        text_parts = [
            str(part.get("text", ""))
            for part in content
            if isinstance(part, dict) and part.get("type") in {None, "text"}
        ]
        visible_text = strip_backend_special_tokens(
            "\n".join(part for part in text_parts if part)
        )
    if visible_text:
        return visible_text
    for reasoning_field in ("reasoning_content", "reasoning"):
        reasoning = message.get(reasoning_field)
        if isinstance(reasoning, str):
            reasoning_text = strip_backend_special_tokens(reasoning)
            if reasoning_text:
                return reasoning_text
    raise VisualProxyUpstreamError("Visual remote returned an empty assistant message")


def _visual_response_finish_reason(data: Any) -> str:
    if not isinstance(data, dict):
        return ""
    choices = data.get("choices")
    if not isinstance(choices, list) or not choices or not isinstance(choices[0], dict):
        return ""
    return str(choices[0].get("finish_reason") or "")


def _stream_error(payload: dict[str, Any]) -> Optional[str]:
    if "error" not in payload or payload.get("choices"):
        return None
    error = payload.get("error") or {}
    message = error.get("message") if isinstance(error, dict) else str(error)
    return str(message or "unknown error")


async def _stream_visual_result(
    runtime: VisualProxyRuntime,
    payload: dict[str, Any],
    raw_request: Optional[Request],
    trace_id: str,
    on_text: Optional[VisualResultCallback],
) -> tuple[VisualRemoteResult, dict[str, Any]]:
    visible_parts: list[str] = []
    reasoning_parts: list[str] = []
    cleaner = _IncrementalVisualTextCleaner()
    emitted_bytes = 0
    finish_reason = ""
    usage: dict[str, Any] = {}

    async def emit(text: str) -> None:
        nonlocal emitted_bytes
        if not text:
            return
        emitted_bytes += len(text.encode("utf-8"))
        if emitted_bytes > runtime.settings.max_remote_response_bytes:
            raise VisualProxyUpstreamError(
                "Visual upstream response exceeds the configured size limit"
            )
        if on_text is not None:
            await on_text(text)

    async for event in runtime.stream_json(payload, raw_request, trace_id):
        error = _stream_error(event)
        if error is not None:
            raise VisualProxyUpstreamError(f"Visual upstream stream failed: {error}")

        raw_usage = event.get("usage")
        if isinstance(raw_usage, dict):
            usage = raw_usage
        choices = event.get("choices") or []
        if not choices or not isinstance(choices[0], dict):
            continue

        choice = choices[0]
        if choice.get("finish_reason") is not None:
            finish_reason = str(choice["finish_reason"])
        delta = choice.get("delta") or {}
        if not isinstance(delta, dict):
            continue

        visible = delta.get("content")
        if isinstance(visible, str) and visible:
            visible_parts.append(visible)
            await emit(cleaner.feed(visible))
        reasoning = delta.get("reasoning_content")
        if not isinstance(reasoning, str):
            reasoning = delta.get("reasoning")
        if isinstance(reasoning, str) and reasoning:
            reasoning_parts.append(reasoning)

    visible_text = strip_backend_special_tokens("".join(visible_parts))
    reasoning_text = strip_backend_special_tokens("".join(reasoning_parts))
    if visible_text:
        await emit(cleaner.finish())
        text = visible_text
        source = "message.content"
    elif reasoning_text:
        # Visible content has priority. Reasoning stays buffered until EOF so
        # later content can never invalidate bytes already sent to the caller.
        await emit(reasoning_text)
        text = reasoning_text
        source = "message.reasoning"
    else:
        raise VisualProxyUpstreamError(
            "Visual remote returned an empty assistant message"
        )

    return (
        VisualRemoteResult(text, finish_reason=finish_reason, usage=usage),
        {"stream": True, "content": text, "content_source": source},
    )


def _records_circuit_failure(exc: BaseException) -> bool:
    return isinstance(
        exc, (VisualProxyUpstreamError, VisualProxyTimeoutError)
    ) and not isinstance(exc, VisualProxyUpstreamRejectedError)


def _trace(recorder: Optional[VisualTraceRecorder], event: str, **payload: Any) -> None:
    if recorder is not None and recorder.enabled:
        recorder.event(event, **payload)


async def call_visual_remote(
    runtime: VisualProxyRuntime,
    model: str,
    image: RegisteredImage,
    task: str,
    trace_id: str,
    raw_request: Optional[Request] = None,
    trace_recorder: Optional[VisualTraceRecorder] = None,
    result_callback: Optional[VisualResultCallback] = None,
) -> VisualRemoteResult:
    if len(task) > VISION_TASK_MAX_CHARS:
        raise ValueError(
            f"vision_reader task exceeds the {VISION_TASK_MAX_CHARS}-character limit"
        )

    settings = runtime.settings
    image_url = image.source
    if image_url.startswith(("http://", "https://")):
        image_url = await runtime.freeze_remote_image(image_url, raw_request, trace_id)
    image_url = image_url_for_visual_provider(image_url, settings)
    image_digest = _image_digest(image_url)
    payload = _request_payload(settings.remote_model or model, image_url, task)
    if result_callback is not None:
        payload["stream"] = True
    logger.info(
        "[visual-chat-proxy][visual_model_request] trace_id=%s url=%s image=%s origin=%s task_chars=%d",
        trace_id,
        settings.remote_url,
        image.tag,
        image.origin,
        len(task),
    )
    _trace(
        trace_recorder,
        "visual_model_request",
        visual_trace_id=trace_id,
        url=settings.remote_url,
        image_tag=image.tag,
        image_origin=image.origin,
        request=payload,
    )

    try:
        if result_callback is None:
            data = await runtime.post_json(payload, raw_request, trace_id)
            text = _remote_content(data)
            raw_usage = data.get("usage") if isinstance(data, dict) else None
            result = VisualRemoteResult(
                text,
                finish_reason=_visual_response_finish_reason(data),
                usage=raw_usage if isinstance(raw_usage, dict) else {},
            )
            trace_response: Any = data
        else:
            result, trace_response = await _stream_visual_result(
                runtime, payload, raw_request, trace_id, result_callback
            )
        if len(str(result).encode("utf-8")) > settings.max_remote_response_bytes:
            raise VisualProxyUpstreamError(
                "Visual upstream response exceeds the configured size limit"
            )
    except BaseException as exc:
        if _records_circuit_failure(exc):
            runtime.record_remote_failure(trace_id)
        _trace(
            trace_recorder,
            "visual_model_error",
            visual_trace_id=trace_id,
            error={"type": type(exc).__name__, "message": str(exc)},
        )
        raise

    _trace(
        trace_recorder,
        "visual_model_response",
        visual_trace_id=trace_id,
        response=trace_response,
    )
    runtime.record_remote_success()
    return VisualRemoteResult(
        str(result),
        finish_reason=result.finish_reason,
        usage=result.usage,
        image_digest_sha256=image_digest,
    )
