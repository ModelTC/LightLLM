from __future__ import annotations

from builtins import anext
import re
import time
import uuid
from json import JSONDecoder, JSONDecodeError
import ujson as json
from http import HTTPStatus
from typing import Any, AsyncGenerator, Dict, List, Optional

from fastapi import Request
from fastapi.responses import JSONResponse, Response

from lightllm.utils.envs_utils import get_env_start_args
from lightllm.utils.error_utils import SERVER_BUSY_MESSAGE, ServerBusyError
from lightllm.utils.log_utils import init_logger

from .api_errors import create_error_response, is_rate_limit_error
from .api_stream_obj import CustomStreamingResponse

logger = init_logger(__name__)


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
            if isinstance(url, dict):
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


def _reasoning_item_text(item: Dict[str, Any]) -> str:
    if item.get("encrypted_content"):
        raise ValueError("reasoning.encrypted_content is not supported; resend plaintext reasoning content")
    for field in ("content", "summary"):
        parts = item.get(field)
        if isinstance(parts, str):
            if parts:
                return parts
            continue
        if isinstance(parts, list):
            text = "".join(
                part if isinstance(part, str) else part.get("text", "")
                for part in parts
                if isinstance(part, (str, dict))
            )
            if text:
                return text
    return ""


def _input_items_to_messages(items: List[Any]) -> List[Dict[str, Any]]:
    messages: List[Dict[str, Any]] = []
    assistant_message: Optional[Dict[str, Any]] = None
    for item in items:
        if not isinstance(item, dict):
            raise ValueError("input items must be objects")
        itype = item.get("type")
        if itype in ("function_call", "custom_tool_call"):
            if assistant_message is None:
                assistant_message = {"role": "assistant", "content": ""}
                messages.append(assistant_message)
            name = item.get("name")
            if item.get("namespace"):
                name = f"{item['namespace']}__{name}"
            arguments = item.get("arguments") or ""
            if itype == "custom_tool_call":
                arguments = json.dumps({"input": item.get("input") or ""}, ensure_ascii=False)
            assistant_message.setdefault("tool_calls", []).append(
                {
                    "id": item.get("call_id") or item.get("id"),
                    "type": "function",
                    "function": {"name": name, "arguments": arguments},
                }
            )
        elif itype in ("function_call_output", "custom_tool_call_output"):
            assistant_message = None
            output = item.get("output")
            if isinstance(output, list):
                output = _content_parts_to_chat(output)
            elif not isinstance(output, str):
                output = json.dumps(output, ensure_ascii=False)
            messages.append({"role": "tool", "tool_call_id": item.get("call_id"), "content": output})
        elif itype == "reasoning":
            reasoning_text = _reasoning_item_text(item)
            if reasoning_text:
                if assistant_message is not None and assistant_message.get("phase") == "final_answer":
                    assistant_message = None
                if assistant_message is None:
                    assistant_message = {"role": "assistant", "content": ""}
                    messages.append(assistant_message)
                assistant_message["reasoning_content"] = assistant_message.get("reasoning_content", "") + reasoning_text
        elif itype in (None, "message"):
            role = item.get("role", "user")
            if role == "developer":
                role = "system"
            content = item.get("content")
            if isinstance(content, list):
                content = _content_parts_to_chat(content)
            if role == "assistant":
                phase = item.get("phase")
                if phase not in (None, "commentary", "final_answer"):
                    raise ValueError(f"Unsupported assistant phase: {phase}")
                if (
                    assistant_message is not None
                    and "phase" in assistant_message
                    and assistant_message["phase"] != phase
                ):
                    assistant_message = None
                if assistant_message is None:
                    assistant_message = {"role": "assistant", "content": ""}
                    messages.append(assistant_message)
                if phase is not None:
                    assistant_message["phase"] = phase
                previous_content = assistant_message["content"]
                if not previous_content:
                    assistant_message["content"] = content if content is not None else ""
                elif content:
                    if isinstance(previous_content, str) and isinstance(content, str):
                        assistant_message["content"] = previous_content + content
                    else:
                        if isinstance(previous_content, str):
                            previous_content = [{"type": "text", "text": previous_content}]
                        if isinstance(content, str):
                            content = [{"type": "text", "text": content}]
                        assistant_message["content"] = previous_content + content
            else:
                messages.append({"role": role, "content": content})
                assistant_message = None
        else:
            raise ValueError(f"Unsupported input item type: {itype}")
    return messages


def _tool_entries(tools, namespace=None):
    """Normalize Responses tool names for the shared function-tool path."""
    for tool in tools:
        if tool.get("type") == "namespace":
            if namespace is not None or not tool.get("name"):
                raise ValueError("Tool namespaces must be named and cannot be nested")
            yield from _tool_entries(tool.get("tools") or [], tool["name"])
            continue
        if tool.get("type") not in ("function", "custom"):
            raise ValueError(f"Unsupported tool type: {tool.get('type')}")
        definition = tool.get("function", tool)
        name = definition.get("name")
        if not name:
            raise ValueError("Tools require a name")
        yield f"{namespace}__{name}" if namespace else name, definition, namespace, tool["type"]


def _tools_to_chat(tools: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    chat_tools = []
    names = set()
    for name, definition, namespace, kind in _tool_entries(tools):
        if name in names:
            raise ValueError(f"Duplicate tool name: {name}")
        names.add(name)
        if definition.get("strict"):
            raise ValueError("Tool strict=true is not supported")
        description = definition.get("description")
        parameters = definition.get("parameters")
        if kind == "custom":
            fmt = definition.get("format") or {}
            if fmt.get("type") not in (None, "text", "grammar"):
                raise ValueError(f"Unsupported custom tool format: {fmt.get('type')}")
            if fmt.get("type") == "grammar":
                # The function shim describes the freeform grammar to the model;
                # it does not enforce that grammar during decoding.
                description = (
                    description or ""
                ) + f"\nInput format ({fmt.get('syntax', 'lark')}):\n{fmt.get('definition', '')}"
            parameters = {
                "type": "object",
                "properties": {"input": {"type": "string"}},
                "required": ["input"],
                "additionalProperties": False,
            }
        chat_tools.append(
            {
                "type": "function",
                "function": {
                    "name": name,
                    "description": description,
                    "parameters": parameters,
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
    extra_body = body.get("extra_body")
    if isinstance(extra_body, dict):
        body = {**extra_body, **body}
    truncation = body.get("truncation")
    if truncation not in (None, "disabled"):
        raise ValueError("Only truncation='disabled' is supported")

    effort = (body.get("reasoning") or {}).get("effort")
    if effort is not None and effort not in ("none", "minimal", "low", "medium", "high", "xhigh", "max"):
        raise ValueError("Unsupported reasoning.effort")
    if body.get("include"):
        raise ValueError("Responses include is not supported")
    summary = (body.get("reasoning") or {}).get("summary")
    if summary not in (None, "auto", "concise", "detailed"):
        raise ValueError("Unsupported reasoning.summary")

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
        ("seed", "seed"),
        ("top_k", "top_k"),
        ("repetition_penalty", "repetition_penalty"),
        ("presence_penalty", "presence_penalty"),
        ("frequency_penalty", "frequency_penalty"),
        ("stop", "stop"),
        ("ignore_eos", "ignore_eos"),
        ("do_sample", "do_sample"),
        ("include_stop_str_in_output", "include_stop_str_in_output"),
        ("chat_template_kwargs", "chat_template_kwargs"),
        ("separate_reasoning", "separate_reasoning"),
        ("stream_reasoning", "stream_reasoning"),
    ):
        if body.get(src) is not None:
            chat[dst] = body[src]

    if body.get("tools"):
        chat["tools"] = _tools_to_chat(body["tools"])
    tool_choice = body.get("tool_choice")
    if tool_choice is not None:
        if isinstance(tool_choice, dict) and tool_choice.get("type") in ("function", "custom"):
            name = tool_choice.get("name") or (tool_choice.get("function") or {}).get("name")
            if tool_choice.get("namespace"):
                name = f"{tool_choice['namespace']}__{name}"
            chat["tool_choice"] = {"type": "function", "function": {"name": name}}
        else:
            chat["tool_choice"] = tool_choice

    if effort is not None:
        chat["reasoning_effort"] = effort

    response_format = _text_format_to_response_format(body)
    if response_format:
        chat["response_format"] = response_format

    if isinstance(extra_body, dict):
        for key, value in extra_body.items():
            chat.setdefault(key, value)

    choice = chat.get("tool_choice", "auto")
    names = {tool["function"]["name"] for tool in chat.get("tools") or []}
    if choice == "required" and not names:
        raise ValueError("tool_choice=required needs at least one tool")
    if isinstance(choice, dict):
        if choice.get("type") != "function" or (choice.get("function") or {}).get("name") not in names:
            raise ValueError("tool_choice must name a declared function or custom tool")

    return chat


def _new_ids() -> Dict[str, str]:
    return {
        "response": f"resp_{uuid.uuid4().hex}",
        "message": f"msg_{uuid.uuid4().hex}",
        "reasoning": f"rs_{uuid.uuid4().hex}",
    }


def _response_envelope(body: Dict[str, Any], response_id: str, created_at: int) -> Dict[str, Any]:
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
    reasoning = int((usage.get("completion_tokens_details") or {}).get("reasoning_tokens", 0) or 0)
    return {
        "input_tokens": input_tokens,
        "input_tokens_details": {"cached_tokens": cached},
        "output_tokens": output_tokens,
        "output_tokens_details": {"reasoning_tokens": reasoning},
        "total_tokens": input_tokens + output_tokens,
    }


def _custom_tool_input(arguments: str, partial: bool = False) -> str:
    """Unwrap the function shim, including incomplete JSON string escapes."""
    if not partial:
        value = json.loads(arguments)
        if not isinstance(value, dict) or set(value) != {"input"} or not isinstance(value["input"], str):
            raise ValueError("Custom tool arguments must contain a single string input")
        return value["input"]
    match = re.match(r'\s*\{\s*"input"\s*:\s*(")', arguments)
    if match is None:
        return ""
    value = arguments[match.start(1) :]
    try:
        return JSONDecoder().raw_decode(value)[0]
    except JSONDecodeError:
        # A partial escape is at most a UTF-16 surrogate pair (12 bytes).
        for end in range(len(value), max(0, len(value) - 12), -1):
            try:
                text = json.loads(value[:end] + '"')
                if text and 0xD800 <= ord(text[-1]) <= 0xDFFF:
                    continue
                return text
            except ValueError:
                continue
        return ""


def _tool_call_item(name, arguments, call_id, tool_map, status):
    definition, namespace, kind = tool_map.get(name, ({"name": name}, None, "function"))
    item = {
        "type": "custom_tool_call" if kind == "custom" else "function_call",
        "id": f"{'ctc' if kind == 'custom' else 'fc'}_{uuid.uuid4().hex}",
        "call_id": call_id or f"call_{uuid.uuid4().hex[:24]}",
        "name": definition["name"],
        "status": status,
    }
    if namespace:
        item["namespace"] = namespace
    if kind == "custom":
        item["input"] = _custom_tool_input(arguments, partial=status != "completed")
    else:
        item["arguments"] = arguments
    return item


def _reasoning_output(text, body, item_id, status):
    return {
        "type": "reasoning",
        "id": item_id,
        "status": status,
        "summary": [{"type": "summary_text", "text": text}] if (body.get("reasoning") or {}).get("summary") else [],
        "content": [{"type": "reasoning_text", "text": text}],
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
    failed = finish_reason in ("error", "abort")
    output_status = "incomplete" if finish_reason == "length" or failed else "completed"
    tool_map = {
        name: (definition, namespace, kind)
        for name, definition, namespace, kind in _tool_entries(body.get("tools") or [])
    }

    output: List[Dict[str, Any]] = []
    reasoning_text = message.get("reasoning") or message.get("reasoning_content")
    if reasoning_text:
        output.append(_reasoning_output(reasoning_text, body, ids["reasoning"], output_status))
    text = message.get("content")
    if text:
        output.append(
            {
                "type": "message",
                "id": ids["message"],
                "status": output_status,
                "role": "assistant",
                "phase": "commentary" if message.get("tool_calls") else "final_answer",
                "content": [{"type": "output_text", "text": text, "annotations": []}],
            }
        )
    for tc in message.get("tool_calls") or []:
        fn = tc.get("function") or {}
        output.append(_tool_call_item(fn.get("name"), fn.get("arguments") or "", tc.get("id"), tool_map, output_status))

    result["output"] = output
    result["usage"] = _usage_to_responses(openai_dict.get("usage") or {})
    if failed:
        result["status"] = "failed"
        result["error"] = {"code": "server_error", "message": "Generation failed"}
    elif finish_reason == "length":
        result["status"] = "incomplete"
        result["incomplete_details"] = {"reason": "max_output_tokens"}
    else:
        result["status"] = "completed"
    return result


async def _openai_sse_to_responses_events(
    openai_body_iterator,
    body: Dict[str, Any],
    delay_response_start: bool = True,
) -> AsyncGenerator[bytes, None]:
    seq = 0

    def event(event_type: str, data: Dict[str, Any]) -> bytes:
        nonlocal seq
        data = {"type": event_type, "sequence_number": seq, **data}
        seq += 1
        return f"event: {event_type}\ndata: {json.dumps(data, ensure_ascii=False)}\n\n".encode("utf-8")

    ids = _new_ids()
    response = _response_envelope(body, ids["response"], int(time.time()))
    tool_map = {
        name: (definition, namespace, kind)
        for name, definition, namespace, kind in _tool_entries(body.get("tools") or [])
    }
    wants_summary = bool((body.get("reasoning") or {}).get("summary"))
    current = None
    tool_states = {}
    finish_reason = None
    usage = {}
    failed_error = None

    def close_item(state, phase="final_answer"):
        item, index = state["item"], state["index"]
        kind = item["type"]
        status = "incomplete" if finish_reason == "length" else "completed"
        common = {"item_id": item["id"], "output_index": index}
        if kind == "message":
            item["phase"] = phase
            text = item["content"][0]["text"]
            yield event("response.output_text.done", {**common, "content_index": 0, "text": text})
            yield event(
                "response.content_part.done",
                {
                    **common,
                    "content_index": 0,
                    "part": {"type": "output_text", "text": text, "annotations": []},
                },
            )
        elif kind == "reasoning":
            text = item["content"][0]["text"]
            item.update(_reasoning_output(text, body, item["id"], status))
            if wants_summary:
                yield event("response.reasoning_summary_text.done", {**common, "summary_index": 0, "text": text})
                yield event(
                    "response.reasoning_summary_part.done",
                    {
                        **common,
                        "summary_index": 0,
                        "part": {"type": "summary_text", "text": text},
                    },
                )
            else:
                yield event("response.reasoning_text.done", {**common, "content_index": 0, "text": text})
                yield event(
                    "response.content_part.done",
                    {
                        **common,
                        "content_index": 0,
                        "part": {"type": "reasoning_text", "text": text},
                    },
                )
        elif kind == "custom_tool_call":
            item["input"] = _custom_tool_input(state["arguments"], partial=status == "incomplete")
            yield event("response.custom_tool_call_input.done", {**common, "input": item["input"]})
        elif kind == "function_call":
            yield event(
                "response.function_call_arguments.done",
                {
                    **common,
                    "name": item["name"],
                    "arguments": item["arguments"],
                },
            )
        item["status"] = status
        yield event("response.output_item.done", {"output_index": index, "item": item})

    def open_item(item):
        index = len(response["output"])
        response["output"].append(item)
        state = {"item": item, "index": index}
        common = {"item_id": item["id"], "output_index": index}
        added = {**item, "content": []} if item["type"] in ("message", "reasoning") else item
        events = [event("response.output_item.added", {"output_index": index, "item": added})]
        if item["type"] == "message":
            events.append(
                event(
                    "response.content_part.added",
                    {
                        **common,
                        "content_index": 0,
                        "part": {"type": "output_text", "text": "", "annotations": []},
                    },
                )
            )
        elif item["type"] == "reasoning":
            if wants_summary:
                events.append(
                    event(
                        "response.reasoning_summary_part.added",
                        {
                            **common,
                            "summary_index": 0,
                            "part": {"type": "summary_text", "text": ""},
                        },
                    )
                )
            else:
                events.append(
                    event(
                        "response.content_part.added",
                        {
                            **common,
                            "content_index": 0,
                            "part": {"type": "reasoning_text", "text": ""},
                        },
                    )
                )
        return state, events

    iterator = openai_body_iterator.__aiter__()
    started = False
    try:
        # Run lazy admission before committing HTTP 200. The deployment opt-out
        # still allows immediate response.created for clients needing headers.
        first_chunk = await anext(iterator, None) if delay_response_start else None
        started = True
        yield event("response.created", {"response": response})
        yield event("response.in_progress", {"response": response})
        raw_chunk = first_chunk
        while True:
            if raw_chunk is None:
                raw_chunk = await anext(iterator, None)
                if raw_chunk is None:
                    break
            if isinstance(raw_chunk, (bytes, bytearray)):
                raw_chunk = raw_chunk.decode("utf-8")
            for line in raw_chunk.splitlines():
                if not line.startswith("data: "):
                    continue
                payload = line[len("data: ") :]
                if payload == "[DONE]":
                    continue
                chunk = json.loads(payload)
                if "error" in chunk and not chunk.get("choices"):
                    failed_error = chunk["error"]
                    break
                if chunk.get("usage"):
                    usage = chunk["usage"]
                choices = chunk.get("choices") or []
                if not choices:
                    continue
                choice = choices[0]
                delta = choice.get("delta") or {}
                finish_reason = choice.get("finish_reason") or finish_reason
                for kind, piece in (
                    ("reasoning", delta.get("reasoning") or delta.get("reasoning_content")),
                    ("message", delta.get("content")),
                ):
                    if not piece:
                        continue
                    if current is None or current["item"]["type"] != kind:
                        if current is not None:
                            for output_event in close_item(current, phase="commentary"):
                                yield output_event
                        if kind == "reasoning":
                            item = _reasoning_output("", body, f"rs_{uuid.uuid4().hex}", "in_progress")
                        else:
                            item = {
                                "type": "message",
                                "id": f"msg_{uuid.uuid4().hex}",
                                "status": "in_progress",
                                "role": "assistant",
                                "content": [{"type": "output_text", "text": "", "annotations": []}],
                            }
                        current, opening_events = open_item(item)
                        for output_event in opening_events:
                            yield output_event
                    item = current["item"]
                    item["content"][0]["text"] += piece
                    common = {"item_id": item["id"], "output_index": current["index"], "delta": piece}
                    if kind == "reasoning" and wants_summary:
                        yield event("response.reasoning_summary_text.delta", {**common, "summary_index": 0})
                    else:
                        event_type = (
                            "response.reasoning_text.delta" if kind == "reasoning" else "response.output_text.delta"
                        )
                        yield event(event_type, {**common, "content_index": 0})

                for tc in delta.get("tool_calls") or []:
                    fn = tc.get("function") or {}
                    index = tc.get("index", 0)
                    if index not in tool_states:
                        if not fn.get("name"):
                            raise ValueError("Tool argument delta has no preceding tool name")
                        if current is not None:
                            for output_event in close_item(current, phase="commentary"):
                                yield output_event
                            current = None
                        item = _tool_call_item(fn["name"], "", tc.get("id"), tool_map, "in_progress")
                        state, opening_events = open_item(item)
                        state.update(arguments="", emitted_input="")
                        tool_states[index] = state
                        for output_event in opening_events:
                            yield output_event
                    state = tool_states[index]
                    item = state["item"]
                    args = fn.get("arguments") or ""
                    state["arguments"] += args
                    common = {"item_id": item["id"], "output_index": state["index"]}
                    if item["type"] == "custom_tool_call":
                        text = _custom_tool_input(state["arguments"], partial=True)
                        piece = text[len(state["emitted_input"]) :]
                        state["emitted_input"] = text
                        item["input"] = text
                        if piece:
                            yield event("response.custom_tool_call_input.delta", {**common, "delta": piece})
                    else:
                        item["arguments"] = state["arguments"]
                        if args:
                            yield event("response.function_call_arguments.delta", {**common, "delta": args})
            raw_chunk = None
            if failed_error is not None:
                break
        if failed_error is None and finish_reason in (None, "error", "abort"):
            failed_error = {
                "message": "Generation failed" if finish_reason else "Generation stream ended without a finish reason"
            }
        if failed_error is None:
            # Validate custom payloads before emitting any terminal tool events.
            if finish_reason != "length":
                for state in tool_states.values():
                    if state["item"]["type"] == "custom_tool_call":
                        _custom_tool_input(state["arguments"])
            if current is not None:
                for output_event in close_item(current):
                    yield output_event
            for state in tool_states.values():
                for output_event in close_item(state):
                    yield output_event
    except Exception as exc:
        if not started:
            raise
        logger.exception("Responses stream failed")
        failed_error = (
            {"message": SERVER_BUSY_MESSAGE, "code": exc.status_code}
            if isinstance(exc, ServerBusyError)
            else {"message": str(exc)}
        )
    finally:
        await iterator.aclose()

    response["usage"] = _usage_to_responses(usage)
    if failed_error is not None:
        for item in response["output"]:
            if item["status"] == "in_progress":
                item["status"] = "incomplete"
        response["status"] = "failed"
        error_code = "rate_limit_exceeded" if is_rate_limit_error(failed_error) else "server_error"
        response["error"] = {"code": error_code, "message": failed_error.get("message", "generation failed")}
        yield event("response.failed", {"response": response})
    elif finish_reason == "length":
        response["status"] = "incomplete"
        response["incomplete_details"] = {"reason": "max_output_tokens"}
        yield event("response.incomplete", {"response": response})
    else:
        response["status"] = "completed"
        yield event("response.completed", {"response": response})


async def responses_impl(raw_request: Request) -> Response:
    from .api_models import ChatCompletionRequest, ChatCompletionResponse
    from .api_openai import chat_completions_impl

    try:
        body = await raw_request.json()
    except Exception as exc:
        return create_error_response(HTTPStatus.BAD_REQUEST, f"Invalid JSON body: {exc}")
    if not isinstance(body, dict):
        return create_error_response(HTTPStatus.BAD_REQUEST, "Request body must be a JSON object")
    if isinstance(body.get("extra_body"), dict):
        body = {**body["extra_body"], **body}

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

    downstream = await chat_completions_impl(chat_request, raw_request)

    if chat_request.stream:
        if not isinstance(downstream, CustomStreamingResponse):
            return downstream
        return CustomStreamingResponse(
            _openai_sse_to_responses_events(
                downstream.body_iterator,
                body,
                delay_response_start=not get_env_start_args().disable_delay_response_start,
            ),
            media_type="text/event-stream",
        )

    if not isinstance(downstream, ChatCompletionResponse):
        return downstream

    try:
        return JSONResponse(_chat_response_to_responses(downstream, body))
    except Exception as exc:
        logger.error("Failed to translate response to Responses API format: %s", exc, exc_info=True)
        return create_error_response(HTTPStatus.INTERNAL_SERVER_ERROR, str(exc))
