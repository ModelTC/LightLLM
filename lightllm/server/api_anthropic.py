"""Anthropic Messages API compatibility layer.

Translates incoming /v1/messages requests into LightLLM's internal chat
completions pipeline by delegating the hard parts (content-block parsing,
tool schema normalisation, stop-reason mapping) to LiteLLM's adapter.

The streaming path is added in a later task; this module currently
rejects stream=true with 501.
"""
from __future__ import annotations

import uuid
import ujson as json
from http import HTTPStatus
from typing import Any, Dict, Tuple

from fastapi import Request
from fastapi.responses import JSONResponse, Response

from lightllm.utils.log_utils import init_logger

from ._litellm_shim import get_anthropic_messages_adapter

logger = init_logger(__name__)


# ---------------------------------------------------------------------------
# Request translation
# ---------------------------------------------------------------------------


def _anthropic_to_chat_request(anthropic_body: Dict[str, Any]) -> Tuple[Dict[str, Any], Dict[str, str]]:
    """Translate an Anthropic Messages request body into a dict suitable
    for constructing a LightLLM ``ChatCompletionRequest``.

    Returns ``(chat_request_dict, tool_name_mapping)``. The mapping must
    be passed back to ``_chat_response_to_anthropic`` so that tool names
    truncated by LiteLLM's 64-character limit can be restored.
    """
    adapter = get_anthropic_messages_adapter()

    openai_request, tool_name_mapping = adapter.translate_anthropic_to_openai(anthropic_body)

    if hasattr(openai_request, "model_dump"):
        openai_dict = openai_request.model_dump(exclude_none=True)
    else:
        openai_dict = dict(openai_request)

    if "max_tokens" not in openai_dict and "max_completion_tokens" not in openai_dict:
        if "max_tokens" in anthropic_body:
            openai_dict["max_tokens"] = anthropic_body["max_tokens"]

    _UNKNOWN_FIELDS = {"extra_body", "metadata", "anthropic_version", "cache_control"}
    for key in list(openai_dict.keys()):
        if key in _UNKNOWN_FIELDS:
            openai_dict.pop(key, None)

    return openai_dict, tool_name_mapping


# ---------------------------------------------------------------------------
# Response translation
# ---------------------------------------------------------------------------


_FINISH_REASON_TO_STOP_REASON = {
    "stop": "end_turn",
    "length": "max_tokens",
    "tool_calls": "tool_use",
    None: "end_turn",
}


def _chat_response_to_anthropic(
    chat_response: Any,
    tool_name_mapping: Dict[str, str],
    requested_model: str,
) -> Dict[str, Any]:
    """Wrap a LightLLM ``ChatCompletionResponse`` into an Anthropic
    Messages response dict.

    LiteLLM's ``translate_openai_response_to_anthropic`` requires a
    ``litellm.ModelResponse`` object (discovered via Task 3's characterisation
    test). We construct one from the LightLLM response's dict form.
    """
    adapter = get_anthropic_messages_adapter()
    if hasattr(chat_response, "model_dump"):
        openai_dict = chat_response.model_dump(exclude_none=True)
    else:
        openai_dict = dict(chat_response)

    try:
        # Lazy import so this module stays importable when litellm is absent.
        from litellm import ModelResponse  # type: ignore

        model_response = ModelResponse(**openai_dict)
        anthropic_obj = adapter.translate_openai_response_to_anthropic(
            model_response, tool_name_mapping
        )
    except Exception as exc:
        logger.warning("LiteLLM response translation failed (%s); using fallback", exc)
        return _fallback_openai_to_anthropic(openai_dict, requested_model)

    if hasattr(anthropic_obj, "model_dump"):
        result = anthropic_obj.model_dump(exclude_none=True)
    else:
        result = dict(anthropic_obj)

    # Echo the client-provided model name.
    result["model"] = requested_model

    result.setdefault("id", f"msg_{uuid.uuid4().hex[:24]}")
    result.setdefault("type", "message")
    result.setdefault("role", "assistant")
    result.setdefault("stop_sequence", None)

    return result


def _fallback_openai_to_anthropic(openai_dict: Dict[str, Any], requested_model: str) -> Dict[str, Any]:
    """Minimal hand-built OpenAI->Anthropic translation for text-only responses.

    Used only when LiteLLM's adapter raises on the response path. Does
    not support tool_use; errors out loudly if tool calls are present
    since silently dropping them would corrupt the response.
    """
    choice = (openai_dict.get("choices") or [{}])[0]
    message = choice.get("message") or {}
    if message.get("tool_calls"):
        raise RuntimeError(
            "Fallback translator cannot handle tool_calls; LiteLLM adapter path is required."
        )
    text = message.get("content") or ""
    usage = openai_dict.get("usage") or {}
    finish_reason = choice.get("finish_reason")
    return {
        "id": f"msg_{uuid.uuid4().hex[:24]}",
        "type": "message",
        "role": "assistant",
        "model": requested_model,
        "content": [{"type": "text", "text": text}],
        "stop_reason": _FINISH_REASON_TO_STOP_REASON.get(finish_reason, "end_turn"),
        "stop_sequence": None,
        "usage": {
            "input_tokens": int(usage.get("prompt_tokens", 0)),
            "output_tokens": int(usage.get("completion_tokens", 0)),
            "cache_creation_input_tokens": 0,
            "cache_read_input_tokens": 0,
        },
    }


# ---------------------------------------------------------------------------
# Streaming bridge
# ---------------------------------------------------------------------------


def _sse_event(event_type: str, data_obj: Dict[str, Any]) -> bytes:
    """Encode an Anthropic-style SSE event."""
    return f"event: {event_type}\ndata: {json.dumps(data_obj)}\n\n".encode("utf-8")


async def _openai_sse_to_anthropic_events(
    openai_body_iterator,
    requested_model: str,
    message_id: str,
):
    """Async generator: consume OpenAI-format SSE bytes and yield
    Anthropic-format SSE event bytes.

    Only the text-only path is implemented here. Tool-use streaming
    requires additional state tracking and is handled in Task 7.
    """
    # State
    message_started = False
    text_block_open = False
    text_block_index = 0
    final_stop_reason = "end_turn"
    final_output_tokens = 0
    final_input_tokens = 0

    _OPENAI_TO_ANTHROPIC_STOP = {
        "stop": "end_turn",
        "length": "max_tokens",
        "tool_calls": "tool_use",
    }

    async for raw_line in openai_body_iterator:
        if not raw_line:
            continue
        # A single StreamingResponse chunk may contain multiple SSE lines.
        for line in raw_line.split(b"\n"):
            line = line.strip()
            if not line or not line.startswith(b"data: "):
                continue
            payload = line[len(b"data: "):]
            if payload == b"[DONE]":
                continue
            try:
                chunk = json.loads(payload.decode("utf-8"))
            except Exception:
                logger.debug("Skipping non-JSON SSE payload: %r", payload)
                continue

            # Usage-only chunk (emitted when stream_options.include_usage is set)
            usage = chunk.get("usage")
            if usage:
                final_input_tokens = int(usage.get("prompt_tokens", 0))
                final_output_tokens = int(usage.get("completion_tokens", final_output_tokens))

            choices = chunk.get("choices") or []
            if not choices:
                continue
            choice = choices[0]
            delta = choice.get("delta") or {}
            finish_reason = choice.get("finish_reason")

            # Emit message_start the first time we see any content
            if not message_started:
                message_started = True
                yield _sse_event(
                    "message_start",
                    {
                        "type": "message_start",
                        "message": {
                            "id": message_id,
                            "type": "message",
                            "role": "assistant",
                            "model": requested_model,
                            "content": [],
                            "stop_reason": None,
                            "stop_sequence": None,
                            "usage": {
                                "input_tokens": final_input_tokens,
                                "output_tokens": 0,
                                "cache_creation_input_tokens": 0,
                                "cache_read_input_tokens": 0,
                            },
                        },
                    },
                )

            content_piece = delta.get("content")
            if content_piece:
                if not text_block_open:
                    text_block_open = True
                    yield _sse_event(
                        "content_block_start",
                        {
                            "type": "content_block_start",
                            "index": text_block_index,
                            "content_block": {"type": "text", "text": ""},
                        },
                    )
                yield _sse_event(
                    "content_block_delta",
                    {
                        "type": "content_block_delta",
                        "index": text_block_index,
                        "delta": {"type": "text_delta", "text": content_piece},
                    },
                )
                final_output_tokens += 1

            if finish_reason:
                final_stop_reason = _OPENAI_TO_ANTHROPIC_STOP.get(finish_reason, "end_turn")

    # Close any open content block
    if text_block_open:
        yield _sse_event("content_block_stop", {"type": "content_block_stop", "index": text_block_index})

    # message_delta carries the final stop_reason and cumulative output_tokens
    if message_started:
        yield _sse_event(
            "message_delta",
            {
                "type": "message_delta",
                "delta": {"stop_reason": final_stop_reason, "stop_sequence": None},
                "usage": {"output_tokens": final_output_tokens},
            },
        )
        yield _sse_event("message_stop", {"type": "message_stop"})


# ---------------------------------------------------------------------------
# HTTP entry point
# ---------------------------------------------------------------------------


async def anthropic_messages_impl(raw_request: Request) -> Response:
    # Lazy imports to avoid pulling in heavy server deps at module import time.
    from .api_models import ChatCompletionRequest, ChatCompletionResponse
    from .api_openai import chat_completions_impl, create_error_response

    try:
        raw_body = await raw_request.json()
    except Exception as exc:
        return create_error_response(HTTPStatus.BAD_REQUEST, f"Invalid JSON body: {exc}")

    if not isinstance(raw_body, dict):
        return create_error_response(HTTPStatus.BAD_REQUEST, "Request body must be a JSON object")

    requested_model = raw_body.get("model", "default")
    is_stream = bool(raw_body.get("stream"))

    try:
        chat_dict, tool_name_mapping = _anthropic_to_chat_request(raw_body)
    except Exception as exc:
        logger.exception("Failed to translate Anthropic request")
        return create_error_response(HTTPStatus.BAD_REQUEST, f"Request translation failed: {exc}")

    # Force the downstream path to stream if the client asked for stream.
    chat_dict["stream"] = is_stream

    try:
        chat_request = ChatCompletionRequest(**chat_dict)
    except Exception as exc:
        logger.exception("Failed to build ChatCompletionRequest")
        return create_error_response(HTTPStatus.BAD_REQUEST, f"Invalid request after translation: {exc}")

    downstream = await chat_completions_impl(chat_request, raw_request)

    if is_stream:
        from fastapi.responses import StreamingResponse

        if not isinstance(downstream, StreamingResponse):
            return downstream  # error path

        message_id = f"msg_{uuid.uuid4().hex[:24]}"
        anthropic_stream = _openai_sse_to_anthropic_events(
            downstream.body_iterator, requested_model=requested_model, message_id=message_id
        )
        return StreamingResponse(anthropic_stream, media_type="text/event-stream")

    if not isinstance(downstream, ChatCompletionResponse):
        return downstream  # JSONResponse error

    anthropic_dict = _chat_response_to_anthropic(downstream, tool_name_mapping, requested_model)
    return JSONResponse(anthropic_dict)
