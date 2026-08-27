"""Assemble an internal OpenAI stream into one main-model turn."""

from __future__ import annotations

import time
import uuid
from typing import Any, Optional

from ..api_models import (
    ChatCompletionResponse,
    ChatCompletionResponseChoice,
    ChatMessage,
    PromptTokensDetails,
    UsageInfo,
)
from ..api_stream_obj import CustomStreamingResponse
from .config import VisualChatProxyError
from .constants import RAW_FUNCTION_CALL_LINE_PATTERN
from .sanitizer import StreamEventCallback, sanitize_reasoning
from .runtime import _iter_openai_sse_payloads


class _IncrementalReasoningSanitizer:
    """Stream safe reasoning while retaining possible split control markers."""

    _LINE_CONTROL_PREFIXES = (
        "function:",
        "recipient=functions.",
        "recipient = functions.",
        "to=functions.",
        "to = functions.",
        '"tool_calls":',
        "'tool_calls':",
    )

    def __init__(self) -> None:
        self._buffer = ""
        self._emitted = False

    def _sanitize_stable_prefix(self, value: str) -> str:
        if not value:
            return ""
        if not self._emitted:
            clean = sanitize_reasoning(value)
        else:
            sentinel = "__LIGHTLLM_STREAM_PREFIX__"
            clean = sanitize_reasoning(sentinel + value)
            if clean.startswith(sentinel):
                clean = clean[len(sentinel) :]
        if clean:
            self._emitted = True
        return clean

    def feed(self, value: Any) -> str:
        if not isinstance(value, str) or not value:
            return ""
        self._buffer += value
        marker_positions = [
            position
            for marker in ("<", "{")
            if (position := self._buffer.find(marker)) >= 0
        ]
        raw_line_marker = RAW_FUNCTION_CALL_LINE_PATTERN.search(self._buffer)
        if raw_line_marker is not None:
            marker_positions.append(raw_line_marker.start())
        if marker_positions:
            safe_end = min(marker_positions)
        else:
            line_start = self._buffer.rfind("\n") + 1
            line = self._buffer[line_start:]
            normalized_line = line.lstrip().lower()
            could_be_control_prefix = bool(normalized_line) and any(
                prefix.startswith(normalized_line)
                for prefix in self._LINE_CONTROL_PREFIXES
            )
            safe_end = line_start if could_be_control_prefix else len(self._buffer)
        # Outer whitespace is removed by sanitize_reasoning at the end of a
        # model turn. Retain it until a later non-whitespace token makes it
        # provably interior, so concatenated deltas match the non-stream body.
        while safe_end > 0 and self._buffer[safe_end - 1].isspace():
            safe_end -= 1
        if safe_end <= 0:
            return ""
        prefix = self._buffer[:safe_end]
        self._buffer = self._buffer[safe_end:]
        return self._sanitize_stable_prefix(prefix)

    def finish(self) -> str:
        raw = self._buffer
        self._buffer = ""
        return self._sanitize_stable_prefix(raw)


async def _consume_main_model_stream(
    response: CustomStreamingResponse,
    *,
    model: str,
    callback: StreamEventCallback,
    separate_from_prior_reasoning: bool,
    stream_content: bool,
    preserve_model_text: bool = False,
) -> ChatCompletionResponse:
    """Assemble one internal model turn, optionally preserving provider text exactly."""

    content_parts: list[str] = []
    reasoning_parts: list[str] = []
    tool_states: dict[int, dict[str, Any]] = {}
    usage = UsageInfo(prompt_tokens_details=PromptTokensDetails())
    finish_reason: Optional[str] = None
    response_id = f"chatcmpl-{uuid.uuid4().hex}"
    created = int(time.time())
    sanitizer = _IncrementalReasoningSanitizer()
    content_sanitizer = _IncrementalReasoningSanitizer()
    emitted_reasoning = False
    emitted_ready = False

    async def emit_reasoning(value: str) -> None:
        nonlocal emitted_reasoning
        if not value:
            return
        await callback(
            "reasoning",
            value,
            separate_from_prior_reasoning and not emitted_reasoning,
        )
        emitted_reasoning = True

    body_iterator = response.body_iterator
    try:
        async for payload in _iter_openai_sse_payloads(body_iterator):
            if "error" in payload and not payload.get("choices"):
                error = payload.get("error") or {}
                message = (
                    error.get("message") if isinstance(error, dict) else str(error)
                )
                raise VisualChatProxyError(
                    f"Main model stream failed: {message or 'unknown error'}"
                )
            if payload.get("id") is not None:
                response_id = str(payload["id"])
            if isinstance(payload.get("created"), int):
                created = payload["created"]
            raw_usage = payload.get("usage")
            if isinstance(raw_usage, dict):
                usage = UsageInfo.model_validate(raw_usage)
            choices = payload.get("choices") or []
            if not choices:
                continue
            if not emitted_ready:
                await callback("ready", "1", False)
                emitted_ready = True
            choice = choices[0]
            if not isinstance(choice, dict):
                continue
            delta = choice.get("delta") or {}
            if not isinstance(delta, dict):
                continue
            raw_reasoning = delta.get("reasoning")
            if not isinstance(raw_reasoning, str):
                raw_reasoning = delta.get("reasoning_content")
            if isinstance(raw_reasoning, str) and raw_reasoning:
                reasoning_parts.append(raw_reasoning)
                await emit_reasoning(
                    raw_reasoning
                    if preserve_model_text
                    else sanitizer.feed(raw_reasoning)
                )
            content = delta.get("content")
            if isinstance(content, str):
                content_parts.append(content)
                if stream_content:
                    clean_content = (
                        content
                        if preserve_model_text
                        else content_sanitizer.feed(content)
                    )
                    if clean_content:
                        await callback("content", clean_content, False)
            for raw_call in delta.get("tool_calls") or []:
                if not isinstance(raw_call, dict):
                    continue
                index = raw_call.get("index")
                if not isinstance(index, int):
                    index = len(tool_states)
                state = tool_states.setdefault(
                    index,
                    {"id": None, "type": "function", "name": None, "arguments": []},
                )
                if raw_call.get("id"):
                    state["id"] = str(raw_call["id"])
                if raw_call.get("type"):
                    state["type"] = raw_call["type"]
                function = raw_call.get("function") or {}
                if isinstance(function, dict):
                    if function.get("name"):
                        state["name"] = str(function["name"])
                    arguments = function.get("arguments")
                    if isinstance(arguments, str):
                        state["arguments"].append(arguments)
            if choice.get("finish_reason") is not None:
                finish_reason = choice["finish_reason"]
    finally:
        close = getattr(body_iterator, "aclose", None)
        if close is not None:
            await close()

    if not preserve_model_text:
        await emit_reasoning(sanitizer.finish())
    if stream_content and not preserve_model_text:
        clean_content = content_sanitizer.finish()
        if clean_content:
            await callback("content", clean_content, False)
    tool_calls = []
    for index in sorted(tool_states):
        state = tool_states[index]
        if not state["name"]:
            raise VisualChatProxyError(
                "Main model stream returned a tool call without a function name"
            )
        tool_calls.append(
            {
                "id": state["id"] or f"call_{uuid.uuid4().hex[:24]}",
                "index": index,
                "type": "function",
                "function": {
                    "name": state["name"],
                    "arguments": "".join(state["arguments"]),
                },
            }
        )
    if tool_calls and finish_reason == "stop" and not preserve_model_text:
        finish_reason = "tool_calls"
    message_data: dict[str, Any] = {
        "role": "assistant",
        "content": "".join(content_parts),
    }
    raw_reasoning = "".join(reasoning_parts)
    reasoning = (
        raw_reasoning if preserve_model_text else sanitize_reasoning(raw_reasoning)
    )
    if reasoning:
        message_data["reasoning"] = reasoning
    if tool_calls:
        message_data["tool_calls"] = tool_calls
    return ChatCompletionResponse(
        id=response_id,
        created=created,
        model=model,
        choices=[
            ChatCompletionResponseChoice(
                index=0,
                message=ChatMessage.model_validate(message_data),
                finish_reason=finish_reason,
            )
        ],
        usage=usage,
    )
