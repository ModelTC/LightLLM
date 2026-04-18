"""Anthropic Messages API compatibility layer.

Translates incoming /v1/messages requests into LightLLM's internal chat
completions pipeline by delegating the hard parts (content-block parsing,
tool schema normalisation, stop-reason mapping) to LiteLLM's adapter.

The streaming path intercepts the OpenAI-format SSE stream from
chat_completions_impl and re-emits it as the Anthropic event sequence
(message_start, content_block_*, message_delta, message_stop).
"""
from __future__ import annotations

import uuid
import ujson as json
from http import HTTPStatus
from typing import Any, Dict, Tuple

from fastapi import Request
from fastapi.responses import JSONResponse, Response

from lightllm.utils.log_utils import init_logger

logger = init_logger(__name__)

_cached_adapter: Any = None


def get_anthropic_messages_adapter() -> Any:
    """Return a cached instance of LiteLLM's Anthropic<->OpenAI adapter.

    The returned object exposes ``translate_anthropic_to_openai`` and
    ``translate_openai_response_to_anthropic`` methods.
    """
    global _cached_adapter
    if _cached_adapter is not None:
        return _cached_adapter

    try:
        from litellm.llms.anthropic.experimental_pass_through.adapters.transformation import (
            LiteLLMAnthropicMessagesAdapter,
        )
    except ImportError as exc:
        raise RuntimeError(
            "The Anthropic Messages API (/v1/messages) requires the 'litellm' package. "
            "Install it with: pip install 'litellm>=1.52.0,<1.85'. "
            f"Original error: {exc}"
        ) from exc

    _cached_adapter = LiteLLMAnthropicMessagesAdapter()
    return _cached_adapter


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

    # Forward LightLLM-specific fields nested under ``extra_body`` (OpenAI SDK
    # convention) so clients hitting /v1/messages can reach ChatCompletionRequest
    # options Anthropic's own schema does not expose — notably chat_template_kwargs
    # for models with optional thinking modes (Qwen3, DeepSeek). Fields already
    # produced by the Anthropic->OpenAI translation take precedence; unknown keys
    # are silently dropped by Pydantic (extra='ignore').
    extra_body = anthropic_body.get("extra_body")
    if isinstance(extra_body, dict):
        for k, v in extra_body.items():
            openai_dict.setdefault(k, v)

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
        anthropic_obj = adapter.translate_openai_response_to_anthropic(model_response, tool_name_mapping)
    except Exception as exc:
        logger.warning("LiteLLM response translation failed (%s); using fallback", exc)
        return _fallback_openai_to_anthropic(openai_dict, requested_model)

    if hasattr(anthropic_obj, "model_dump"):
        result = anthropic_obj.model_dump(exclude_none=True)
    else:
        result = dict(anthropic_obj)

    return _normalize_anthropic_response(result, requested_model)


def _normalize_anthropic_response(result: Dict[str, Any], requested_model: str) -> Dict[str, Any]:
    """Cosmetic clean-ups applied to every non-streaming Anthropic response:

    - echo the client-supplied model name (LiteLLM sometimes emits the
      upstream model id instead);
    - force the Anthropic ``msg_`` id prefix (LiteLLM passes LightLLM's
      raw numeric request id through, which confuses strict clients);
    - set default ``type`` / ``role`` / ``stop_sequence`` when missing;
    - drop empty text blocks (LiteLLM sometimes produces a leading
      ``{"type":"text","text":""}`` before a tool_use block);
    - strip the LiteLLM-specific ``provider_specific_fields`` leak from
      every content block.
    """
    result["model"] = requested_model

    if not str(result.get("id", "")).startswith("msg_"):
        result["id"] = f"msg_{uuid.uuid4().hex[:24]}"
    result.setdefault("type", "message")
    result.setdefault("role", "assistant")
    result.setdefault("stop_sequence", None)

    cleaned_content = []
    for block in result.get("content") or []:
        if not isinstance(block, dict):
            cleaned_content.append(block)
            continue
        if block.get("type") == "text" and not block.get("text"):
            continue
        block.pop("provider_specific_fields", None)
        cleaned_content.append(block)
    result["content"] = cleaned_content

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
        raise RuntimeError("Fallback translator cannot handle tool_calls; LiteLLM adapter path is required.")
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

    Handles both text deltas (emitted as text_delta content blocks) and
    tool-call deltas (emitted as tool_use content blocks whose arguments
    stream as input_json_delta events). Anthropic's protocol opens one
    content block at a time — when switching between a text block and a
    tool_use block (or between tool_use blocks) the current block is
    closed before the next is opened.
    """
    message_started = False
    next_content_index = 0

    # Currently open content block, if any.
    # current_open is either None or a tuple ("text"|"tool_use", anthropic_index).
    current_open = None

    text_block_index = None  # Anthropic index of the active text block.

    # Per-tool-call state keyed by OpenAI streaming tool_calls[i].index.
    # Each entry: {anthropic_index, id, name, started, buffered_args}
    tool_state: Dict[int, Dict[str, Any]] = {}

    final_stop_reason = "end_turn"
    final_output_tokens = 0
    final_input_tokens = 0

    _OPENAI_TO_ANTHROPIC_STOP = {
        "stop": "end_turn",
        "length": "max_tokens",
        "tool_calls": "tool_use",
    }

    async for raw_chunk in openai_body_iterator:
        if not raw_chunk:
            continue
        # chat_completions_impl yields str ("data: {...}\n\n"); some callers or
        # middlewares may hand us bytes. Normalise to str so the splitter below
        # does not have to branch on type.
        if isinstance(raw_chunk, (bytes, bytearray)):
            raw_chunk = raw_chunk.decode("utf-8", errors="replace")
        # A single StreamingResponse chunk may contain multiple SSE lines.
        for line in raw_chunk.split("\n"):
            line = line.strip()
            if not line or not line.startswith("data: "):
                continue
            payload = line[len("data: ") :]
            if payload == "[DONE]":
                continue
            try:
                chunk = json.loads(payload)
            except Exception:
                logger.debug("Skipping non-JSON SSE payload: %r", payload)
                continue

            # final_output_tokens is sourced exclusively from the trailing usage
            # chunk emitted by chat_completions_impl; we intentionally do not
            # estimate it per delta because that would diverge from the
            # tokenizer-accurate count on any upstream change.
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

            # Emit message_start the first time we see any content.
            # NOTE: The upstream usage chunk arrives AFTER all content chunks, so
            # final_input_tokens is still 0 here. message_start.message.usage.input_tokens
            # will always be 0 on this path — Anthropic clients that care about prompt
            # token counts should read message_delta.usage instead. Fixing this would
            # require buffering until the usage chunk arrives, trading streaming
            # latency for accurate prompt-token reporting at message_start time.
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

            # ---- Text delta ----
            content_piece = delta.get("content")
            if content_piece:
                if current_open is None or current_open[0] != "text":
                    if current_open is not None:
                        yield _sse_event(
                            "content_block_stop",
                            {"type": "content_block_stop", "index": current_open[1]},
                        )
                    text_block_index = next_content_index
                    next_content_index += 1
                    current_open = ("text", text_block_index)
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

            # ---- Tool-call deltas ----
            for tc in delta.get("tool_calls") or []:
                tc_idx = tc.get("index", 0)
                fn = tc.get("function") or {}
                state = tool_state.setdefault(
                    tc_idx,
                    {
                        "anthropic_index": None,
                        "id": None,
                        "name": None,
                        "started": False,
                        "buffered_args": "",
                    },
                )
                if tc.get("id"):
                    state["id"] = tc["id"]
                if fn.get("name"):
                    state["name"] = fn["name"]
                new_args = fn.get("arguments") or ""

                if not state["started"]:
                    # Buffer args until we know the tool name (required for
                    # content_block_start).
                    state["buffered_args"] += new_args
                    if not state["name"]:
                        continue
                    # Close whatever block is currently open (text or a
                    # previous tool_use) before opening this one.
                    if current_open is not None:
                        yield _sse_event(
                            "content_block_stop",
                            {"type": "content_block_stop", "index": current_open[1]},
                        )
                    state["anthropic_index"] = next_content_index
                    next_content_index += 1
                    current_open = ("tool_use", state["anthropic_index"])
                    state["started"] = True
                    yield _sse_event(
                        "content_block_start",
                        {
                            "type": "content_block_start",
                            "index": state["anthropic_index"],
                            "content_block": {
                                "type": "tool_use",
                                "id": state["id"] or f"toolu_{uuid.uuid4().hex[:24]}",
                                "name": state["name"],
                                "input": {},
                            },
                        },
                    )
                    if state["buffered_args"]:
                        yield _sse_event(
                            "content_block_delta",
                            {
                                "type": "content_block_delta",
                                "index": state["anthropic_index"],
                                "delta": {
                                    "type": "input_json_delta",
                                    "partial_json": state["buffered_args"],
                                },
                            },
                        )
                        state["buffered_args"] = ""
                else:
                    # Already started. If deltas for a different block are
                    # now arriving (unusual interleaving), close whatever's
                    # currently open and reopen... but in practice OpenAI
                    # streams tool_calls sequentially per index, so the
                    # current_open is this same block.
                    if new_args:
                        yield _sse_event(
                            "content_block_delta",
                            {
                                "type": "content_block_delta",
                                "index": state["anthropic_index"],
                                "delta": {
                                    "type": "input_json_delta",
                                    "partial_json": new_args,
                                },
                            },
                        )

            if finish_reason:
                final_stop_reason = _OPENAI_TO_ANTHROPIC_STOP.get(finish_reason, "end_turn")

    # Close any still-open content block.
    if current_open is not None:
        yield _sse_event(
            "content_block_stop",
            {"type": "content_block_stop", "index": current_open[1]},
        )

    # message_delta carries the final stop_reason and cumulative output_tokens.
    if message_started:
        yield _sse_event(
            "message_delta",
            {
                "type": "message_delta",
                "delta": {"stop_reason": final_stop_reason, "stop_sequence": None},
                "usage": {"input_tokens": final_input_tokens, "output_tokens": final_output_tokens},
            },
        )
        yield _sse_event("message_stop", {"type": "message_stop"})


# ---------------------------------------------------------------------------
# Error response helper
# ---------------------------------------------------------------------------


# HTTP status → Anthropic error type. Derived from
# https://docs.anthropic.com/en/api/errors ; values outside this map fall
# back to "api_error".
_STATUS_TO_ERROR_TYPE = {
    400: "invalid_request_error",
    401: "authentication_error",
    403: "permission_error",
    404: "not_found_error",
    413: "request_too_large",
    429: "rate_limit_error",
    500: "api_error",
    529: "overloaded_error",
}


def _anthropic_error_response(status: HTTPStatus, message: str) -> JSONResponse:
    """Return an Anthropic-shaped error envelope.

    Anthropic clients (including Claude Code) parse the {"type":"error",
    "error":{"type":..., "message":...}} shape; the OpenAI-style envelope
    from create_error_response hides the real message from them.
    """
    err_type = _STATUS_TO_ERROR_TYPE.get(int(status), "api_error")
    return JSONResponse(
        {"type": "error", "error": {"type": err_type, "message": message}},
        status_code=int(status),
    )


def _rewrap_openai_error_as_anthropic(resp: JSONResponse) -> JSONResponse:
    """Convert an OpenAI-format JSONResponse produced by create_error_response
    into Anthropic's error envelope. Best-effort: if we can't decode the body
    we leave the response alone so the caller still sees something."""
    try:
        body = json.loads(bytes(resp.body).decode("utf-8"))
        inner = (body or {}).get("error") or {}
        message = inner.get("message") or "request failed"
    except Exception:
        return resp
    return _anthropic_error_response(HTTPStatus(resp.status_code), message)


# ---------------------------------------------------------------------------
# HTTP entry point
# ---------------------------------------------------------------------------


async def anthropic_messages_impl(raw_request: Request) -> Response:
    # Lazy imports to avoid pulling in heavy server deps at module import time.
    from .api_models import ChatCompletionRequest, ChatCompletionResponse
    from .api_openai import chat_completions_impl

    try:
        raw_body = await raw_request.json()
    except Exception as exc:
        return _anthropic_error_response(HTTPStatus.BAD_REQUEST, f"Invalid JSON body: {exc}")

    if not isinstance(raw_body, dict):
        return _anthropic_error_response(HTTPStatus.BAD_REQUEST, "Request body must be a JSON object")

    requested_model = raw_body.get("model", "default")
    is_stream = bool(raw_body.get("stream"))

    try:
        chat_dict, tool_name_mapping = _anthropic_to_chat_request(raw_body)
    except Exception as exc:
        logger.exception("Failed to translate Anthropic request")
        return _anthropic_error_response(HTTPStatus.BAD_REQUEST, f"Request translation failed: {exc}")

    # Force the downstream path to stream if the client asked for stream.
    chat_dict["stream"] = is_stream

    try:
        chat_request = ChatCompletionRequest(**chat_dict)
    except Exception as exc:
        logger.exception("Failed to build ChatCompletionRequest")
        return _anthropic_error_response(HTTPStatus.BAD_REQUEST, f"Invalid request after translation: {exc}")

    downstream = await chat_completions_impl(chat_request, raw_request)

    if is_stream:
        from fastapi.responses import StreamingResponse

        if not isinstance(downstream, StreamingResponse):
            # chat_completions_impl returned an OpenAI-format error — rewrap it.
            if isinstance(downstream, JSONResponse):
                return _rewrap_openai_error_as_anthropic(downstream)
            return downstream

        message_id = f"msg_{uuid.uuid4().hex[:24]}"
        anthropic_stream = _openai_sse_to_anthropic_events(
            downstream.body_iterator, requested_model=requested_model, message_id=message_id
        )
        return StreamingResponse(anthropic_stream, media_type="text/event-stream")

    if not isinstance(downstream, ChatCompletionResponse):
        if isinstance(downstream, JSONResponse):
            return _rewrap_openai_error_as_anthropic(downstream)
        return downstream

    try:
        anthropic_dict = _chat_response_to_anthropic(downstream, tool_name_mapping, requested_model)
    except Exception as exc:
        logger.error("Failed to translate response to Anthropic format: %s", exc)
        return JSONResponse(_anthropic_error_response(500, str(exc)), status_code=500)
    return JSONResponse(anthropic_dict)
