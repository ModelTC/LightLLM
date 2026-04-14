"""Anthropic Messages API compatibility layer.

Translates incoming /v1/messages requests into LightLLM's internal chat
completions pipeline by delegating the hard parts (content-block parsing,
tool schema normalisation, stop-reason mapping) to LiteLLM's adapter.

The streaming path is added in a later task; this module currently
rejects stream=true with 501.
"""
from __future__ import annotations

import uuid
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
# HTTP entry point (non-streaming only in this task)
# ---------------------------------------------------------------------------


async def anthropic_messages_impl(raw_request: Request) -> Response:
    """Handle POST /v1/messages.

    Streaming support is added in a later task; this function currently
    rejects ``stream=true`` with a clear error.
    """
    # Lazy imports to avoid pulling in heavy server deps at module import time.
    from .api_models import ChatCompletionRequest, ChatCompletionResponse
    from .api_openai import chat_completions_impl, create_error_response

    try:
        raw_body = await raw_request.json()
    except Exception as exc:
        return create_error_response(HTTPStatus.BAD_REQUEST, f"Invalid JSON body: {exc}")

    if not isinstance(raw_body, dict):
        return create_error_response(HTTPStatus.BAD_REQUEST, "Request body must be a JSON object")

    if raw_body.get("stream"):
        return create_error_response(
            HTTPStatus.NOT_IMPLEMENTED,
            "Streaming is not yet implemented for /v1/messages",
        )

    requested_model = raw_body.get("model", "default")

    try:
        chat_dict, tool_name_mapping = _anthropic_to_chat_request(raw_body)
    except Exception as exc:
        logger.exception("Failed to translate Anthropic request")
        return create_error_response(HTTPStatus.BAD_REQUEST, f"Request translation failed: {exc}")

    try:
        chat_request = ChatCompletionRequest(**chat_dict)
    except Exception as exc:
        logger.exception("Failed to build ChatCompletionRequest")
        return create_error_response(HTTPStatus.BAD_REQUEST, f"Invalid request after translation: {exc}")

    chat_response_or_err = await chat_completions_impl(chat_request, raw_request)

    if not isinstance(chat_response_or_err, ChatCompletionResponse):
        # chat_completions_impl returned a JSONResponse (error). Pass through.
        return chat_response_or_err

    anthropic_dict = _chat_response_to_anthropic(
        chat_response_or_err, tool_name_mapping, requested_model
    )
    return JSONResponse(anthropic_dict)
