"""OpenAI Responses API compatibility layer.

Translates incoming /v1/responses requests into LightLLM's internal chat
completions pipeline (chat_completions_impl) and translates the result back
into Responses API shapes, mirroring the approach of api_anthropic.py.

Stateless only: ``store`` is ignored (echoed back as false) and
``previous_response_id`` is rejected, since LightLLM keeps no response store.
"""
from __future__ import annotations

import time
import uuid
import ujson as json
from http import HTTPStatus
from typing import Any, AsyncGenerator, Dict, List, Optional

from fastapi import Request
from fastapi.responses import JSONResponse, Response, StreamingResponse

from lightllm.server.visual_chat_proxy import (
    apply_visual_thinking_policy,
    VisualChatProxyError,
    VisualProxyCapacityError,
    VisualProxyTimeoutError,
    VisualProxyUpstreamError,
    should_use_visual_proxy,
    visual_chat_completions_impl,
)
from lightllm.utils.log_utils import init_logger

logger = init_logger(__name__)


# ---------------------------------------------------------------------------
# Request translation
# ---------------------------------------------------------------------------


def _content_parts_to_chat(parts: List[Any]) -> List[Dict[str, Any]]:
    chat_parts = []
    for part in parts:
        if isinstance(part, str):
            chat_parts.append({"type": "text", "text": part})
            continue
        ptype = part.get("type")
        if ptype in ("input_text", "output_text", "text"):
            chat_parts.append({"type": "text", "text": part.get("text", "")})
        elif ptype == "refusal":
            chat_parts.append({"type": "text", "text": part.get("refusal", "")})
        elif ptype == "input_image":
            url = part.get("image_url")
            if isinstance(url, dict):  # tolerate chat-style nesting
                url = url.get("url")
            if not url:
                raise ValueError("input_image requires an image_url")
            chat_parts.append({"type": "image_url", "image_url": {"url": url}})
        elif ptype == "input_audio":
            audio = part.get("input_audio") or {}
            url = audio.get("url") or part.get("audio_url")
            if not url:
                raise ValueError("input_audio requires a url (raw audio data is not supported)")
            chat_parts.append({"type": "audio_url", "audio_url": {"url": url}})
        else:
            raise ValueError(f"Unsupported input content type: {ptype}")
    return chat_parts


def _input_items_to_messages(items: List[Any]) -> List[Dict[str, Any]]:
    messages: List[Dict[str, Any]] = []
    for item in items:
        if not isinstance(item, dict):
            raise ValueError("input items must be objects")
        itype = item.get("type")
        if itype == "function_call":
            messages.append(
                {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [
                        {
                            "id": item.get("call_id") or item.get("id"),
                            "type": "function",
                            "function": {"name": item.get("name"), "arguments": item.get("arguments") or ""},
                        }
                    ],
                }
            )
        elif itype == "function_call_output":
            output = item.get("output")
            if not isinstance(output, str):
                output = json.dumps(output, ensure_ascii=False)
            messages.append({"role": "tool", "tool_call_id": item.get("call_id"), "content": output})
        elif itype == "reasoning":
            continue  # opaque reasoning items are not replayable; skip
        elif itype in (None, "message"):
            role = item.get("role", "user")
            if role == "developer":
                role = "system"
            content = item.get("content")
            if isinstance(content, list):
                content = _content_parts_to_chat(content)
            messages.append({"role": role, "content": content})
        else:
            raise ValueError(f"Unsupported input item type: {itype}")
    return messages


def _tools_to_chat(tools: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    chat_tools = []
    for tool in tools:
        if tool.get("type") != "function":
            # Clients (e.g. Codex) may send non-function tool types (web_search,
            # namespace, ...). The model can't call them anyway — skip, don't reject.
            logger.warning("Ignoring unsupported tool type: %s", tool.get("type"))
            continue
        if "function" in tool:  # already chat-completions shaped
            chat_tools.append(tool)
            continue
        chat_tools.append(
            {
                "type": "function",
                "function": {
                    "name": tool.get("name"),
                    "description": tool.get("description"),
                    "parameters": tool.get("parameters"),
                },
            }
        )
    return chat_tools


def _text_format_to_response_format(body: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    fmt = (body.get("text") or {}).get("format")
    if not fmt:
        return None
    ftype = fmt.get("type")
    if ftype == "text":
        return None
    if ftype == "json_object":
        return {"type": "json_object"}
    if ftype == "json_schema":
        return {
            "type": "json_schema",
            "json_schema": {
                "name": fmt.get("name", "response"),
                "description": fmt.get("description"),
                "schema": fmt.get("schema"),
                "strict": fmt.get("strict"),
            },
        }
    raise ValueError(f"Unsupported text.format type: {ftype}")


def _responses_to_chat_request(body: Dict[str, Any]) -> Dict[str, Any]:
    """Translate a Responses API request body into a dict suitable for
    constructing a LightLLM ``ChatCompletionRequest``."""
    messages: List[Dict[str, Any]] = []
    if body.get("instructions"):
        messages.append({"role": "system", "content": body["instructions"]})

    inp = body.get("input")
    if isinstance(inp, str):
        messages.append({"role": "user", "content": inp})
    elif isinstance(inp, list):
        messages.extend(_input_items_to_messages(inp))
    else:
        raise ValueError("'input' must be a string or an array of input items")

    # Chat templates (e.g. Qwen3.5) allow a single system message, and only at
    # the head. Clients (e.g. Codex) send instructions plus developer context
    # blocks interleaved with user turns — merge them all into one leading
    # system message, preserving relative order.
    system_texts = []
    for m in messages:
        if m["role"] == "system":
            content = m["content"]
            if isinstance(content, list):
                content = "\n".join(p.get("text", "") for p in content)
            if content:
                system_texts.append(content)
    messages = [m for m in messages if m["role"] != "system"]
    if system_texts:
        messages.insert(0, {"role": "system", "content": "\n\n".join(system_texts)})

    chat: Dict[str, Any] = {
        "model": body.get("model", "default"),
        "messages": messages,
        "stream": bool(body.get("stream")),
        "n": 1,
    }
    for src, dst in (
        ("temperature", "temperature"),
        ("top_p", "top_p"),
        ("max_output_tokens", "max_completion_tokens"),
        ("parallel_tool_calls", "parallel_tool_calls"),
        ("user", "user"),
    ):
        if body.get(src) is not None:
            chat[dst] = body[src]

    if body.get("tools"):
        chat["tools"] = _tools_to_chat(body["tools"])
    tool_choice = body.get("tool_choice")
    if tool_choice is not None:
        if isinstance(tool_choice, dict) and tool_choice.get("type") == "function":
            # Responses uses {"type": "function", "name": ...}; chat nests the name.
            chat["tool_choice"] = {"type": "function", "function": {"name": tool_choice.get("name")}}
        else:
            chat["tool_choice"] = tool_choice

    effort = (body.get("reasoning") or {}).get("effort")
    if effort in ("low", "medium", "high"):
        chat["reasoning_effort"] = effort

    response_format = _text_format_to_response_format(body)
    if response_format:
        chat["response_format"] = response_format

    # OpenAI SDK convention: LightLLM-specific knobs ride in extra_body.
    extra_body = body.get("extra_body")
    if isinstance(extra_body, dict):
        for k, v in extra_body.items():
            chat.setdefault(k, v)

    return chat


# ---------------------------------------------------------------------------
# Response translation
# ---------------------------------------------------------------------------


def _new_ids() -> Dict[str, str]:
    return {
        "response": f"resp_{uuid.uuid4().hex}",
        "message": f"msg_{uuid.uuid4().hex}",
        "reasoning": f"rs_{uuid.uuid4().hex}",
    }


def _response_envelope(body: Dict[str, Any], response_id: str, created_at: int) -> Dict[str, Any]:
    """Common Response-object skeleton echoing back request parameters."""
    return {
        "id": response_id,
        "object": "response",
        "created_at": created_at,
        "status": "in_progress",
        "background": False,
        "error": None,
        "incomplete_details": None,
        "instructions": body.get("instructions"),
        "max_output_tokens": body.get("max_output_tokens"),
        "model": body.get("model", "default"),
        "output": [],
        "parallel_tool_calls": body.get("parallel_tool_calls", True),
        "previous_response_id": None,
        "reasoning": body.get("reasoning"),
        "store": False,
        "temperature": body.get("temperature"),
        "text": body.get("text") or {"format": {"type": "text"}},
        "tool_choice": body.get("tool_choice", "auto"),
        "tools": body.get("tools") or [],
        "top_p": body.get("top_p"),
        "truncation": body.get("truncation", "disabled"),
        "usage": None,
        "user": body.get("user"),
        "metadata": body.get("metadata") or {},
    }


def _usage_to_responses(usage: Dict[str, Any]) -> Dict[str, Any]:
    input_tokens = int(usage.get("prompt_tokens", 0))
    output_tokens = int(usage.get("completion_tokens", 0))
    cached = int((usage.get("prompt_tokens_details") or {}).get("cached_tokens", 0) or 0)
    return {
        "input_tokens": input_tokens,
        "input_tokens_details": {"cached_tokens": cached},
        "output_tokens": output_tokens,
        "output_tokens_details": {"reasoning_tokens": 0},
        "total_tokens": input_tokens + output_tokens,
    }


def _chat_response_to_responses(chat_response: Any, body: Dict[str, Any]) -> Dict[str, Any]:
    if hasattr(chat_response, "model_dump"):
        openai_dict = chat_response.model_dump(exclude_none=True)
    else:
        openai_dict = dict(chat_response)

    ids = _new_ids()
    created_at = int(openai_dict.get("created") or time.time())
    result = _response_envelope(body, ids["response"], created_at)

    choice = (openai_dict.get("choices") or [{}])[0]
    message = choice.get("message") or {}
    finish_reason = choice.get("finish_reason")

    output: List[Dict[str, Any]] = []
    reasoning_text = message.get("reasoning") or message.get("reasoning_content")
    if reasoning_text:
        output.append(
            {
                "type": "reasoning",
                "id": ids["reasoning"],
                "summary": [],
                "content": [{"type": "reasoning_text", "text": reasoning_text}],
            }
        )
    text = message.get("content")
    if text:
        output.append(
            {
                "type": "message",
                "id": ids["message"],
                "status": "completed",
                "role": "assistant",
                "content": [{"type": "output_text", "text": text, "annotations": []}],
            }
        )
    for tc in message.get("tool_calls") or []:
        fn = tc.get("function") or {}
        output.append(
            {
                "type": "function_call",
                "id": f"fc_{uuid.uuid4().hex}",
                "call_id": tc.get("id"),
                "name": fn.get("name"),
                "arguments": fn.get("arguments") or "",
                "status": "completed",
            }
        )

    result["output"] = output
    result["usage"] = _usage_to_responses(openai_dict.get("usage") or {})
    if finish_reason == "length":
        result["status"] = "incomplete"
        result["incomplete_details"] = {"reason": "max_output_tokens"}
    else:
        result["status"] = "completed"
    return result


# ---------------------------------------------------------------------------
# Streaming bridge: OpenAI chat SSE -> Responses API SSE events
# ---------------------------------------------------------------------------


async def _openai_sse_to_responses_events(
    openai_body_iterator,
    body: Dict[str, Any],
) -> AsyncGenerator[bytes, None]:
    seq = 0

    def event(event_type: str, data: Dict[str, Any]) -> bytes:
        nonlocal seq
        seq += 1
        data = {"type": event_type, "sequence_number": seq, **data}
        return f"event: {event_type}\ndata: {json.dumps(data, ensure_ascii=False)}\n\n".encode("utf-8")

    ids = _new_ids()
    response = _response_envelope(body, ids["response"], int(time.time()))

    yield event("response.created", {"response": response})
    yield event("response.in_progress", {"response": response})

    output_index = -1
    # current open item: None | ("reasoning"|"message"|"function_call", item_dict)
    current: Optional[tuple] = None
    finish_reason = None
    usage: Dict[str, Any] = {}
    failed_error: Optional[Dict[str, Any]] = None

    def close_current():
        nonlocal current
        if current is None:
            return
        kind, item = current
        current = None
        if kind == "message":
            text = item["content"][0]["text"]
            yield event(
                "response.output_text.done",
                {"item_id": item["id"], "output_index": output_index, "content_index": 0, "text": text},
            )
            yield event(
                "response.content_part.done",
                {
                    "item_id": item["id"],
                    "output_index": output_index,
                    "content_index": 0,
                    "part": {"type": "output_text", "text": text, "annotations": []},
                },
            )
        elif kind == "reasoning":
            yield event(
                "response.reasoning_text.done",
                {
                    "item_id": item["id"],
                    "output_index": output_index,
                    "content_index": 0,
                    "text": item["content"][0]["text"],
                },
            )
        elif kind == "function_call":
            yield event(
                "response.function_call_arguments.done",
                {"item_id": item["id"], "output_index": output_index, "arguments": item["arguments"]},
            )
        item["status"] = "completed"
        response["output"].append(item)
        yield event("response.output_item.done", {"output_index": output_index, "item": item})

    def open_item(kind: str, item: Dict[str, Any]):
        nonlocal current, output_index
        yield from close_current()
        output_index += 1
        current = (kind, item)
        yield event("response.output_item.added", {"output_index": output_index, "item": item})
        if kind == "message":
            yield event(
                "response.content_part.added",
                {
                    "item_id": item["id"],
                    "output_index": output_index,
                    "content_index": 0,
                    "part": {"type": "output_text", "text": "", "annotations": []},
                },
            )

    async for raw_chunk in openai_body_iterator:
        if not raw_chunk:
            continue
        if isinstance(raw_chunk, (bytes, bytearray)):
            raw_chunk = raw_chunk.decode("utf-8", errors="replace")
        for line in raw_chunk.split("\n"):
            line = line.strip()
            if not line.startswith("data: "):
                continue
            payload = line[len("data: ") :]
            if payload == "[DONE]":
                continue
            try:
                chunk = json.loads(payload)
            except Exception:
                logger.debug("Skipping non-JSON SSE payload: %r", payload)
                continue

            if "error" in chunk and "choices" not in chunk:
                failed_error = chunk["error"]
                break

            if chunk.get("usage"):
                usage = chunk["usage"]
            choices = chunk.get("choices") or []
            if not choices:
                continue
            choice = choices[0]
            delta = choice.get("delta") or {}
            if choice.get("finish_reason"):
                finish_reason = choice["finish_reason"]

            reasoning_piece = delta.get("reasoning") or delta.get("reasoning_content")
            if reasoning_piece:
                if current is None or current[0] != "reasoning":
                    item = {
                        "type": "reasoning",
                        "id": f"rs_{uuid.uuid4().hex}",
                        "summary": [],
                        "content": [{"type": "reasoning_text", "text": ""}],
                        "status": "in_progress",
                    }
                    for e in open_item("reasoning", item):
                        yield e
                item = current[1]
                item["content"][0]["text"] += reasoning_piece
                yield event(
                    "response.reasoning_text.delta",
                    {
                        "item_id": item["id"],
                        "output_index": output_index,
                        "content_index": 0,
                        "delta": reasoning_piece,
                    },
                )

            content_piece = delta.get("content")
            if content_piece:
                if current is None or current[0] != "message":
                    item = {
                        "type": "message",
                        "id": f"msg_{uuid.uuid4().hex}",
                        "status": "in_progress",
                        "role": "assistant",
                        "content": [{"type": "output_text", "text": "", "annotations": []}],
                    }
                    for e in open_item("message", item):
                        yield e
                item = current[1]
                item["content"][0]["text"] += content_piece
                yield event(
                    "response.output_text.delta",
                    {
                        "item_id": item["id"],
                        "output_index": output_index,
                        "content_index": 0,
                        "delta": content_piece,
                    },
                )

            for tc in delta.get("tool_calls") or []:
                fn = tc.get("function") or {}
                # A new tool call starts when a name/id arrives; args stream after.
                if fn.get("name"):
                    item = {
                        "type": "function_call",
                        "id": f"fc_{uuid.uuid4().hex}",
                        "call_id": tc.get("id") or f"call_{uuid.uuid4().hex[:24]}",
                        "name": fn["name"],
                        "arguments": "",
                        "status": "in_progress",
                    }
                    for e in open_item("function_call", item):
                        yield e
                args = fn.get("arguments")
                if args and current is not None and current[0] == "function_call":
                    item = current[1]
                    item["arguments"] += args
                    yield event(
                        "response.function_call_arguments.delta",
                        {"item_id": item["id"], "output_index": output_index, "delta": args},
                    )
        if failed_error is not None:
            break

    for e in close_current():
        yield e

    if failed_error is not None:
        response["status"] = "failed"
        response["error"] = {"code": "server_error", "message": failed_error.get("message", "generation failed")}
        yield event("response.failed", {"response": response})
        return

    response["usage"] = _usage_to_responses(usage)
    if finish_reason == "length":
        response["status"] = "incomplete"
        response["incomplete_details"] = {"reason": "max_output_tokens"}
        yield event("response.incomplete", {"response": response})
    else:
        response["status"] = "completed"
        yield event("response.completed", {"response": response})


# ---------------------------------------------------------------------------
# HTTP entry point
# ---------------------------------------------------------------------------


async def _dispatch_chat_request(chat_request: Any, raw_request: Request, main_chat_handler: Any) -> Any:
    """Route translated Responses API requests through the visual proxy when needed."""
    # Imported lazily to avoid an api_http -> api_responses -> api_http import cycle.
    from .api_http import g_objs

    visual_remote_url = getattr(getattr(g_objs, "args", None), "visual_remote_url", None)
    if visual_remote_url and g_objs.visual_proxy_runtime is not None:
        chat_request = apply_visual_thinking_policy(chat_request, g_objs.visual_proxy_runtime.settings)
    if not should_use_visual_proxy(visual_remote_url, chat_request):
        return await main_chat_handler(chat_request, raw_request)

    runtime = g_objs.visual_proxy_runtime
    if runtime is None:
        raise VisualChatProxyError("Visual proxy runtime is not initialized")
    async with runtime.request_slot():
        return await visual_chat_completions_impl(
            request=chat_request,
            raw_request=raw_request,
            runtime=runtime,
            main_chat_handler=main_chat_handler,
        )


async def responses_impl(raw_request: Request) -> Response:
    # Lazy imports to avoid pulling heavy server deps at module import time.
    from .api_models import ChatCompletionRequest, ChatCompletionResponse
    from .api_openai import chat_completions_impl, create_error_response

    try:
        body = await raw_request.json()
    except Exception as exc:
        return create_error_response(HTTPStatus.BAD_REQUEST, f"Invalid JSON body: {exc}")
    if not isinstance(body, dict):
        return create_error_response(HTTPStatus.BAD_REQUEST, "Request body must be a JSON object")

    if body.get("previous_response_id"):
        return create_error_response(
            HTTPStatus.BAD_REQUEST,
            "previous_response_id is not supported (this server is stateless); "
            "resend the full conversation in 'input' instead",
            param="previous_response_id",
        )
    if body.get("background"):
        return create_error_response(
            HTTPStatus.BAD_REQUEST, "background responses are not supported", param="background"
        )

    try:
        chat_dict = _responses_to_chat_request(body)
        chat_request = ChatCompletionRequest(**chat_dict)
    except ValueError as exc:
        return create_error_response(HTTPStatus.BAD_REQUEST, str(exc))
    except Exception as exc:
        logger.exception("Failed to translate Responses API request")
        return create_error_response(HTTPStatus.BAD_REQUEST, f"Invalid request: {exc}")

    try:
        downstream = await _dispatch_chat_request(chat_request, raw_request, chat_completions_impl)
    except VisualProxyCapacityError as exc:
        logger.warning("%s", str(exc))
        return create_error_response(HTTPStatus.TOO_MANY_REQUESTS, str(exc), err_type="RateLimitError")
    except VisualProxyTimeoutError as exc:
        logger.warning("%s", str(exc))
        return create_error_response(HTTPStatus.GATEWAY_TIMEOUT, str(exc), err_type="UpstreamTimeoutError")
    except (VisualProxyUpstreamError, VisualChatProxyError) as exc:
        logger.warning("%s", str(exc))
        return create_error_response(HTTPStatus.BAD_GATEWAY, str(exc), err_type="UpstreamError")
    except ValueError as exc:
        return create_error_response(HTTPStatus.BAD_REQUEST, str(exc))

    if chat_request.stream:
        if not isinstance(downstream, StreamingResponse):
            return downstream  # error JSONResponse in OpenAI envelope; Responses uses the same shape
        return StreamingResponse(
            _openai_sse_to_responses_events(downstream.body_iterator, body),
            media_type="text/event-stream",
        )

    if not isinstance(downstream, ChatCompletionResponse):
        return downstream

    try:
        return JSONResponse(_chat_response_to_responses(downstream, body))
    except Exception as exc:
        logger.error("Failed to translate response to Responses API format: %s", exc, exc_info=True)
        return create_error_response(HTTPStatus.INTERNAL_SERVER_ERROR, str(exc))
