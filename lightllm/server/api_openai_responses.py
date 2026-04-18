"""OpenAI Responses API compatibility layer.

Translates incoming /v1/responses requests into LightLLM's internal chat
completions pipeline. The request shape (``input`` content parts,
``instructions``, function tools, ``max_output_tokens``) is reshaped into
the OpenAI Chat Completions shape, chat_completions_impl does the work,
and the response is re-packed into the Responses API envelope.

The streaming path intercepts the OpenAI-format SSE stream from
chat_completions_impl and re-emits it as Responses semantic events
(response.created, response.output_item.added, response.output_text.delta,
response.completed, ...).
"""
from __future__ import annotations

import time
import uuid
import ujson as json
from http import HTTPStatus
from typing import Any, Dict, List, Optional, Tuple

from fastapi import Request
from fastapi.responses import JSONResponse, Response

from lightllm.utils.log_utils import init_logger

logger = init_logger(__name__)


# ---------------------------------------------------------------------------
# Request translation: Responses API -> Chat Completions
# ---------------------------------------------------------------------------


def _input_content_to_chat_content(parts: List[Dict[str, Any]]) -> Any:
    """Convert a Responses-API content-part list into the OpenAI Chat
    Completions content representation.

    Chat Completions accepts either a bare string or a list of
    ``{type, text}`` / ``{type:image_url, image_url:{url}}`` parts.
    """
    if not isinstance(parts, list):
        return parts

    chat_parts: List[Dict[str, Any]] = []
    for part in parts:
        if not isinstance(part, dict):
            continue
        ptype = part.get("type")
        if ptype in ("input_text", "output_text", "text", "summary_text"):
            chat_parts.append({"type": "text", "text": part.get("text", "")})
        elif ptype in ("input_image", "image"):
            # Responses uses image_url (string OR {url, detail}); Chat uses
            # {type:image_url, image_url:{url, detail}}.
            image_url = part.get("image_url")
            if isinstance(image_url, str):
                image_url = {"url": image_url}
            elif not isinstance(image_url, dict):
                # Try file_id as fallback for uploaded image references.
                file_id = part.get("file_id")
                if file_id:
                    image_url = {"url": f"file://{file_id}"}
                else:
                    continue
            chat_parts.append({"type": "image_url", "image_url": image_url})
        else:
            # Unknown content type — drop it rather than fail, matching the
            # Anthropic adapter's forgiving posture toward unknown fields.
            continue

    # Collapse a single text part back to a bare string: the chat pipeline's
    # tokenizer templates tend to be happier with strings than one-element lists.
    if len(chat_parts) == 1 and chat_parts[0].get("type") == "text":
        return chat_parts[0].get("text", "")
    return chat_parts


def _responses_input_to_chat_messages(
    input_value: Any,
    instructions: Optional[str],
) -> List[Dict[str, Any]]:
    """Build the chat ``messages`` list from the Responses ``input`` field
    plus an optional top-level ``instructions`` string.
    """
    messages: List[Dict[str, Any]] = []

    if instructions:
        messages.append({"role": "system", "content": instructions})

    if input_value is None:
        return messages

    # Responses allows a plain string as shorthand for a single user turn.
    if isinstance(input_value, str):
        messages.append({"role": "user", "content": input_value})
        return messages

    if not isinstance(input_value, list):
        raise ValueError("'input' must be a string or a list of items")

    for item in input_value:
        if not isinstance(item, dict):
            continue

        # Top-level typed items (function_call, function_call_output) come in
        # alongside regular role-based messages in Responses input.
        item_type = item.get("type")

        if item_type == "function_call":
            # Assistant turn that issued a tool call. Chat Completions expects
            # this as an assistant message with a ``tool_calls`` array.
            call_id = item.get("call_id") or item.get("id") or f"call_{uuid.uuid4().hex[:24]}"
            messages.append(
                {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [
                        {
                            "id": call_id,
                            "type": "function",
                            "function": {
                                "name": item.get("name", ""),
                                "arguments": item.get("arguments", "") or "",
                            },
                        }
                    ],
                }
            )
            continue

        if item_type == "function_call_output":
            output = item.get("output")
            if not isinstance(output, str):
                output = json.dumps(output) if output is not None else ""
            messages.append(
                {
                    "role": "tool",
                    "tool_call_id": item.get("call_id") or "",
                    "content": output,
                }
            )
            continue

        if item_type == "reasoning":
            # We don't round-trip reasoning items through the chat pipeline;
            # dropping them matches our out-of-scope list.
            continue

        # Role-based message item (type="message" or no type but with role).
        role = item.get("role") or "user"
        if role == "developer":
            # Responses introduced "developer" as a stricter system role;
            # map to system for Chat Completions compatibility.
            role = "system"

        content = item.get("content")
        if isinstance(content, list):
            chat_content = _input_content_to_chat_content(content)
        else:
            chat_content = content if content is not None else ""

        messages.append({"role": role, "content": chat_content})

    return messages


def _responses_tools_to_chat_tools(tools: Any) -> Optional[List[Dict[str, Any]]]:
    """Filter the Responses ``tools`` list down to function tools and
    reshape each entry into the Chat Completions schema.

    Built-in tools (``web_search``, ``file_search``, ``code_interpreter``,
    ``computer_use_preview``) are dropped with a warning — they require
    side-channel capabilities LightLLM does not host.
    """
    if not isinstance(tools, list):
        return None

    chat_tools: List[Dict[str, Any]] = []
    for tool in tools:
        if not isinstance(tool, dict):
            continue
        if tool.get("type") != "function":
            logger.warning("Dropping unsupported Responses tool type: %s", tool.get("type"))
            continue
        # Responses has flat {type, name, description, parameters, strict}; Chat
        # nests under {type, function:{name, description, parameters}}.
        fn = {
            "name": tool.get("name", ""),
            "description": tool.get("description", ""),
            "parameters": tool.get("parameters") or {},
        }
        chat_tools.append({"type": "function", "function": fn})

    return chat_tools or None


def _responses_tool_choice_to_chat(tool_choice: Any) -> Any:
    """Translate ``tool_choice`` between the two envelopes.

    Shared string values (``auto``, ``none``, ``required``) pass through.
    ``{"type":"function","name":"X"}`` (Responses) → ``{"type":"function","function":{"name":"X"}}`` (Chat).
    """
    if tool_choice is None:
        return None
    if isinstance(tool_choice, str):
        return tool_choice
    if isinstance(tool_choice, dict):
        if tool_choice.get("type") == "function" and "name" in tool_choice:
            return {"type": "function", "function": {"name": tool_choice["name"]}}
        # Already in Chat shape, or a hosted-tool choice we can't honour.
        if "function" in tool_choice:
            return tool_choice
    return "auto"


def _responses_to_chat_request(body: Dict[str, Any]) -> Tuple[Dict[str, Any], str]:
    """Translate a Responses API request body into a dict suitable for
    constructing a LightLLM ``ChatCompletionRequest``.

    Returns ``(chat_dict, requested_model)``. ``requested_model`` is echoed
    into the response envelope since the Responses spec returns it verbatim.
    """
    requested_model = body.get("model") or "default"

    messages = _responses_input_to_chat_messages(body.get("input"), body.get("instructions"))

    chat_dict: Dict[str, Any] = {
        "model": requested_model,
        "messages": messages,
    }

    tools = _responses_tools_to_chat_tools(body.get("tools"))
    if tools is not None:
        chat_dict["tools"] = tools

    tool_choice = _responses_tool_choice_to_chat(body.get("tool_choice"))
    if tool_choice is not None:
        chat_dict["tool_choice"] = tool_choice

    if "max_output_tokens" in body and body["max_output_tokens"] is not None:
        chat_dict["max_tokens"] = body["max_output_tokens"]

    # Pass through fields that exist on both APIs with identical semantics.
    for key in ("temperature", "top_p", "stop", "user", "seed", "stream", "parallel_tool_calls"):
        if key in body and body[key] is not None:
            chat_dict[key] = body[key]

    return chat_dict, requested_model


# ---------------------------------------------------------------------------
# Response translation: Chat Completions -> Responses API
# ---------------------------------------------------------------------------

# Chat ``finish_reason`` → Responses ``status`` + optional incomplete reason.
_FINISH_TO_STATUS = {
    "stop": ("completed", None),
    "length": ("incomplete", "max_output_tokens"),
    "tool_calls": ("completed", None),
    "content_filter": ("incomplete", "content_filter"),
    None: ("completed", None),
}


def _new_response_id() -> str:
    return f"resp_{uuid.uuid4().hex[:24]}"


def _new_message_id() -> str:
    return f"msg_{uuid.uuid4().hex[:24]}"


def _new_fc_id() -> str:
    return f"fc_{uuid.uuid4().hex[:24]}"


def _usage_to_responses(usage: Dict[str, Any]) -> Dict[str, Any]:
    """Reshape Chat usage (prompt_tokens/completion_tokens/total_tokens)
    into Responses usage (input_tokens/output_tokens/total_tokens plus the
    nested ``*_tokens_details`` scaffolding some SDKs require)."""
    usage = usage or {}
    return {
        "input_tokens": int(usage.get("prompt_tokens", 0) or 0),
        "input_tokens_details": {"cached_tokens": 0},
        "output_tokens": int(usage.get("completion_tokens", 0) or 0),
        "output_tokens_details": {"reasoning_tokens": 0},
        "total_tokens": int(usage.get("total_tokens", 0) or 0),
    }


def _build_output_items(
    message: Dict[str, Any],
    finish_reason: Optional[str],
) -> List[Dict[str, Any]]:
    """Build the Responses ``output`` array from a Chat assistant message.

    Message items come first (if there's any text), then function_call items
    — downstream SDKs expect any narration to precede tool calls.
    """
    items: List[Dict[str, Any]] = []

    text = message.get("content")
    if isinstance(text, list):
        # Chat can emit list-content in rare multimodal cases; flatten to text.
        text = "".join(p.get("text", "") for p in text if isinstance(p, dict) and p.get("type") == "text")
    if text:
        items.append(
            {
                "type": "message",
                "id": _new_message_id(),
                "role": "assistant",
                "status": "completed",
                "content": [
                    {
                        "type": "output_text",
                        "text": text,
                        "annotations": [],
                    }
                ],
            }
        )

    for tc in message.get("tool_calls") or []:
        if not isinstance(tc, dict):
            continue
        fn = tc.get("function") or {}
        call_id = tc.get("id") or f"call_{uuid.uuid4().hex[:24]}"
        items.append(
            {
                "type": "function_call",
                "id": _new_fc_id(),
                "call_id": call_id,
                "name": fn.get("name", ""),
                "arguments": fn.get("arguments", "") or "",
                "status": "completed",
            }
        )

    # If the turn ended without any content AND without tool calls, emit an
    # empty message item so downstream SDKs that index ``output[0]`` don't crash.
    if not items:
        items.append(
            {
                "type": "message",
                "id": _new_message_id(),
                "role": "assistant",
                "status": "completed",
                "content": [],
            }
        )

    return items


def _chat_response_to_responses(
    chat_response: Any,
    requested_model: str,
    response_id: Optional[str] = None,
) -> Dict[str, Any]:
    """Pack a LightLLM ``ChatCompletionResponse`` into a Responses envelope."""
    if hasattr(chat_response, "model_dump"):
        chat_dict = chat_response.model_dump(exclude_none=True)
    else:
        chat_dict = dict(chat_response)

    choice = (chat_dict.get("choices") or [{}])[0]
    message = choice.get("message") or {}
    finish_reason = choice.get("finish_reason")

    status, incomplete_reason = _FINISH_TO_STATUS.get(finish_reason, ("completed", None))

    envelope: Dict[str, Any] = {
        "id": response_id or _new_response_id(),
        "object": "response",
        "created_at": int(chat_dict.get("created") or time.time()),
        "status": status,
        "error": None,
        "incomplete_details": {"reason": incomplete_reason} if incomplete_reason else None,
        "model": requested_model,
        "output": _build_output_items(message, finish_reason),
        "usage": _usage_to_responses(chat_dict.get("usage") or {}),
        "parallel_tool_calls": True,
        "previous_response_id": None,
        "reasoning": {"effort": None, "summary": None},
        "store": False,
        "temperature": None,
        "text": {"format": {"type": "text"}},
        "tool_choice": "auto",
        "tools": [],
        "top_p": None,
        "truncation": "disabled",
        "user": None,
        "metadata": {},
    }

    # Thread reasoning_content through if present (some models emit it via the
    # chat pipeline). We surface it as response.reasoning.summary[0].text so
    # clients that care can read it without a custom field.
    reasoning_content = message.get("reasoning_content")
    if reasoning_content:
        envelope["reasoning"] = {
            "effort": None,
            "summary": [{"type": "summary_text", "text": reasoning_content}],
        }

    return envelope


# ---------------------------------------------------------------------------
# Streaming bridge: OpenAI Chat SSE -> Responses semantic SSE
# ---------------------------------------------------------------------------


def _sse_event(event_type: str, data_obj: Dict[str, Any]) -> bytes:
    """Encode a Responses-API-style SSE event.

    The Responses stream sends both an ``event:`` line and a ``data:`` line
    (the data payload also carries a ``type`` field, but ``event:`` lets
    clients route without parsing JSON).
    """
    return f"event: {event_type}\ndata: {json.dumps(data_obj)}\n\n".encode("utf-8")


def _response_skeleton(
    response_id: str,
    requested_model: str,
    created_at: int,
    status: str,
) -> Dict[str, Any]:
    """Base Responses envelope used in response.created / response.completed."""
    return {
        "id": response_id,
        "object": "response",
        "created_at": created_at,
        "status": status,
        "error": None,
        "incomplete_details": None,
        "model": requested_model,
        "output": [],
        "usage": None,
        "parallel_tool_calls": True,
        "previous_response_id": None,
        "reasoning": {"effort": None, "summary": None},
        "store": False,
        "temperature": None,
        "text": {"format": {"type": "text"}},
        "tool_choice": "auto",
        "tools": [],
        "top_p": None,
        "truncation": "disabled",
        "user": None,
        "metadata": {},
    }


async def _openai_sse_to_responses_events(
    openai_body_iterator,
    requested_model: str,
    response_id: str,
):
    """Async generator: consume OpenAI chat SSE bytes and yield Responses
    semantic SSE events.

    Event layering:
      - Per-stream bookends: response.created, response.in_progress,
        response.completed.
      - Per output item (message or function_call): output_item.added +
        output_item.done, with text/argument delta events in between.
      - Per text content block inside a message: content_part.added +
        content_part.done, with output_text.delta events in between.
    """
    created_at = int(time.time())
    seq = 0

    def _next_seq() -> int:
        nonlocal seq
        cur = seq
        seq += 1
        return cur

    # --- response.created / response.in_progress ---
    created_resp = _response_skeleton(response_id, requested_model, created_at, "in_progress")
    yield _sse_event(
        "response.created",
        {"type": "response.created", "response": created_resp, "sequence_number": _next_seq()},
    )
    yield _sse_event(
        "response.in_progress",
        {"type": "response.in_progress", "response": created_resp, "sequence_number": _next_seq()},
    )

    # Per-stream state.
    output_index = 0
    final_status = "completed"
    incomplete_reason: Optional[str] = None
    final_usage = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
    finished_items: List[Dict[str, Any]] = []

    # Open message-item state (at most one open at a time).
    msg_open = False
    msg_id: Optional[str] = None
    msg_output_index: Optional[int] = None
    msg_text_buffer = ""
    msg_content_open = False  # whether content_part.added has been emitted

    # Open/finalised tool-call state, keyed by the OpenAI streaming index.
    # Each entry: {fc_id, call_id, name, arguments, output_index, started, closed}.
    tool_state: Dict[int, Dict[str, Any]] = {}
    current_tool_idx: Optional[int] = None  # OpenAI index of the currently-open tool call

    def _close_message_item():
        nonlocal msg_open, msg_content_open, msg_text_buffer, msg_id, msg_output_index
        if not msg_open:
            return
        events: List[bytes] = []
        # content_part.done + output_text.done
        if msg_content_open:
            events.append(
                _sse_event(
                    "response.output_text.done",
                    {
                        "type": "response.output_text.done",
                        "output_index": msg_output_index,
                        "item_id": msg_id,
                        "content_index": 0,
                        "text": msg_text_buffer,
                        "sequence_number": _next_seq(),
                    },
                )
            )
            events.append(
                _sse_event(
                    "response.content_part.done",
                    {
                        "type": "response.content_part.done",
                        "output_index": msg_output_index,
                        "item_id": msg_id,
                        "content_index": 0,
                        "part": {
                            "type": "output_text",
                            "text": msg_text_buffer,
                            "annotations": [],
                        },
                        "sequence_number": _next_seq(),
                    },
                )
            )
        final_item = {
            "type": "message",
            "id": msg_id,
            "role": "assistant",
            "status": "completed",
            "content": (
                [{"type": "output_text", "text": msg_text_buffer, "annotations": []}] if msg_content_open else []
            ),
        }
        events.append(
            _sse_event(
                "response.output_item.done",
                {
                    "type": "response.output_item.done",
                    "output_index": msg_output_index,
                    "item": final_item,
                    "sequence_number": _next_seq(),
                },
            )
        )
        finished_items.append(final_item)
        # Reset.
        msg_open = False
        msg_content_open = False
        msg_text_buffer = ""
        msg_id = None
        msg_output_index = None
        return events

    def _close_tool_item(idx: int):
        state = tool_state.get(idx)
        if not state or state.get("closed") or not state.get("started"):
            return []
        events: List[bytes] = []
        events.append(
            _sse_event(
                "response.function_call_arguments.done",
                {
                    "type": "response.function_call_arguments.done",
                    "output_index": state["output_index"],
                    "item_id": state["fc_id"],
                    "arguments": state["arguments"],
                    "sequence_number": _next_seq(),
                },
            )
        )
        final_item = {
            "type": "function_call",
            "id": state["fc_id"],
            "call_id": state["call_id"],
            "name": state["name"],
            "arguments": state["arguments"],
            "status": "completed",
        }
        events.append(
            _sse_event(
                "response.output_item.done",
                {
                    "type": "response.output_item.done",
                    "output_index": state["output_index"],
                    "item": final_item,
                    "sequence_number": _next_seq(),
                },
            )
        )
        state["closed"] = True
        finished_items.append(final_item)
        return events

    async for raw_chunk in openai_body_iterator:
        if not raw_chunk:
            continue
        if isinstance(raw_chunk, (bytes, bytearray)):
            raw_chunk = raw_chunk.decode("utf-8", errors="replace")
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

            usage = chunk.get("usage")
            if usage:
                # Chat's trailing usage chunk. Save for response.completed.
                for k in ("prompt_tokens", "completion_tokens", "total_tokens"):
                    if k in usage:
                        final_usage[k] = int(usage[k] or 0)

            choices = chunk.get("choices") or []
            if not choices:
                continue
            choice = choices[0]
            delta = choice.get("delta") or {}
            finish_reason = choice.get("finish_reason")

            # ---- Text delta ----
            content_piece = delta.get("content")
            if content_piece:
                # If a tool call is currently open, close it first — message
                # and function_call items cannot interleave.
                if current_tool_idx is not None:
                    for ev in _close_tool_item(current_tool_idx):
                        yield ev
                    current_tool_idx = None

                if not msg_open:
                    msg_open = True
                    msg_id = _new_message_id()
                    msg_output_index = output_index
                    output_index += 1
                    yield _sse_event(
                        "response.output_item.added",
                        {
                            "type": "response.output_item.added",
                            "output_index": msg_output_index,
                            "item": {
                                "type": "message",
                                "id": msg_id,
                                "role": "assistant",
                                "status": "in_progress",
                                "content": [],
                            },
                            "sequence_number": _next_seq(),
                        },
                    )

                if not msg_content_open:
                    msg_content_open = True
                    yield _sse_event(
                        "response.content_part.added",
                        {
                            "type": "response.content_part.added",
                            "output_index": msg_output_index,
                            "item_id": msg_id,
                            "content_index": 0,
                            "part": {"type": "output_text", "text": "", "annotations": []},
                            "sequence_number": _next_seq(),
                        },
                    )

                msg_text_buffer += content_piece
                yield _sse_event(
                    "response.output_text.delta",
                    {
                        "type": "response.output_text.delta",
                        "output_index": msg_output_index,
                        "item_id": msg_id,
                        "content_index": 0,
                        "delta": content_piece,
                        "sequence_number": _next_seq(),
                    },
                )

            # ---- Tool-call deltas ----
            for tc in delta.get("tool_calls") or []:
                tc_idx = tc.get("index", 0)
                fn = tc.get("function") or {}
                state = tool_state.setdefault(
                    tc_idx,
                    {
                        "fc_id": None,
                        "call_id": None,
                        "name": None,
                        "arguments": "",
                        "output_index": None,
                        "started": False,
                        "closed": False,
                    },
                )
                if tc.get("id") and not state["call_id"]:
                    state["call_id"] = tc["id"]
                if fn.get("name") and not state["name"]:
                    state["name"] = fn["name"]

                # Close the open message item before switching to a tool call
                # (or before switching between different tool calls).
                if not state["started"]:
                    if msg_open:
                        for ev in _close_message_item() or []:
                            yield ev
                    if current_tool_idx is not None and current_tool_idx != tc_idx:
                        for ev in _close_tool_item(current_tool_idx):
                            yield ev

                    # Can't emit output_item.added until we know the function name.
                    if not state["name"]:
                        # Buffer arguments until we have a name.
                        state["arguments"] += fn.get("arguments") or ""
                        continue

                    state["fc_id"] = _new_fc_id()
                    state["call_id"] = state["call_id"] or f"call_{uuid.uuid4().hex[:24]}"
                    state["output_index"] = output_index
                    output_index += 1
                    state["started"] = True
                    current_tool_idx = tc_idx
                    yield _sse_event(
                        "response.output_item.added",
                        {
                            "type": "response.output_item.added",
                            "output_index": state["output_index"],
                            "item": {
                                "type": "function_call",
                                "id": state["fc_id"],
                                "call_id": state["call_id"],
                                "name": state["name"],
                                "arguments": "",
                                "status": "in_progress",
                            },
                            "sequence_number": _next_seq(),
                        },
                    )
                    if state["arguments"]:
                        yield _sse_event(
                            "response.function_call_arguments.delta",
                            {
                                "type": "response.function_call_arguments.delta",
                                "output_index": state["output_index"],
                                "item_id": state["fc_id"],
                                "delta": state["arguments"],
                                "sequence_number": _next_seq(),
                            },
                        )

                new_args = fn.get("arguments") or ""
                if state["started"] and new_args:
                    state["arguments"] += new_args
                    current_tool_idx = tc_idx
                    yield _sse_event(
                        "response.function_call_arguments.delta",
                        {
                            "type": "response.function_call_arguments.delta",
                            "output_index": state["output_index"],
                            "item_id": state["fc_id"],
                            "delta": new_args,
                            "sequence_number": _next_seq(),
                        },
                    )

            if finish_reason:
                status, incomplete = _FINISH_TO_STATUS.get(finish_reason, ("completed", None))
                final_status = status
                incomplete_reason = incomplete

    # Close any still-open items.
    if msg_open:
        for ev in _close_message_item() or []:
            yield ev
    for idx in list(tool_state.keys()):
        for ev in _close_tool_item(idx):
            yield ev

    # Build the final response snapshot for response.completed.
    final_resp = _response_skeleton(response_id, requested_model, created_at, final_status)
    final_resp["output"] = finished_items
    final_resp["usage"] = _usage_to_responses(final_usage)
    if incomplete_reason:
        final_resp["incomplete_details"] = {"reason": incomplete_reason}

    yield _sse_event(
        "response.completed",
        {"type": "response.completed", "response": final_resp, "sequence_number": _next_seq()},
    )


# ---------------------------------------------------------------------------
# Error response helper
# ---------------------------------------------------------------------------

# HTTP status → Responses error ``type``. OpenAI's SDK surfaces this value
# unchanged, so unfamiliar types confuse error-handling code.
_STATUS_TO_ERROR_TYPE = {
    400: "invalid_request_error",
    401: "authentication_error",
    403: "permission_error",
    404: "not_found_error",
    413: "request_too_large",
    429: "rate_limit_exceeded",
    500: "server_error",
    529: "overloaded_error",
}


def _responses_error_response(status: HTTPStatus, message: str) -> JSONResponse:
    """Return a Responses-shaped error envelope."""
    err_type = _STATUS_TO_ERROR_TYPE.get(int(status), "server_error")
    return JSONResponse(
        {"error": {"type": err_type, "code": None, "message": message, "param": None}},
        status_code=int(status),
    )


def _rewrap_openai_error_as_responses(resp: JSONResponse) -> JSONResponse:
    """LightLLM's ``create_error_response`` emits ``{"message": ...}`` with no
    ``error`` envelope; wrap it in the Responses error shape."""
    try:
        body = json.loads(bytes(resp.body).decode("utf-8"))
        if isinstance(body, dict):
            # Either {"error": {"message": ...}} (OpenAI Chat) or {"message": ...} (LightLLM).
            inner = body.get("error") if isinstance(body.get("error"), dict) else body
            message = inner.get("message") or "request failed"
        else:
            message = "request failed"
    except Exception:
        return resp
    return _responses_error_response(HTTPStatus(resp.status_code), message)


# ---------------------------------------------------------------------------
# HTTP entry point
# ---------------------------------------------------------------------------


async def openai_responses_impl(raw_request: Request) -> Response:
    # Lazy imports keep this module importable without the full server stack.
    from .api_models import ChatCompletionRequest, ChatCompletionResponse
    from .api_openai import chat_completions_impl

    try:
        raw_body = await raw_request.json()
    except Exception as exc:
        return _responses_error_response(HTTPStatus.BAD_REQUEST, f"Invalid JSON body: {exc}")

    if not isinstance(raw_body, dict):
        return _responses_error_response(HTTPStatus.BAD_REQUEST, "Request body must be a JSON object")

    is_stream = bool(raw_body.get("stream"))

    try:
        chat_dict, requested_model = _responses_to_chat_request(raw_body)
    except Exception as exc:
        logger.exception("Failed to translate Responses request")
        return _responses_error_response(HTTPStatus.BAD_REQUEST, f"Request translation failed: {exc}")

    chat_dict["stream"] = is_stream
    if is_stream:
        # chat_completions_impl only emits the trailing usage chunk when
        # stream_options.include_usage is true; response.completed needs it
        # to report non-zero token counts.
        chat_dict["stream_options"] = {"include_usage": True}

    try:
        chat_request = ChatCompletionRequest(**chat_dict)
    except Exception as exc:
        logger.exception("Failed to build ChatCompletionRequest")
        return _responses_error_response(HTTPStatus.BAD_REQUEST, f"Invalid request after translation: {exc}")

    downstream = await chat_completions_impl(chat_request, raw_request)

    if is_stream:
        from fastapi.responses import StreamingResponse

        if not isinstance(downstream, StreamingResponse):
            if isinstance(downstream, JSONResponse):
                return _rewrap_openai_error_as_responses(downstream)
            return downstream

        response_id = _new_response_id()
        responses_stream = _openai_sse_to_responses_events(
            downstream.body_iterator,
            requested_model=requested_model,
            response_id=response_id,
        )
        return StreamingResponse(responses_stream, media_type="text/event-stream")

    if not isinstance(downstream, ChatCompletionResponse):
        if isinstance(downstream, JSONResponse):
            return _rewrap_openai_error_as_responses(downstream)
        return downstream

    try:
        responses_dict = _chat_response_to_responses(downstream, requested_model)
    except Exception as exc:
        logger.error("Failed to translate response to Responses format: %s", exc)
        return _responses_error_response(HTTPStatus.INTERNAL_SERVER_ERROR, str(exc))
    return JSONResponse(responses_dict)
