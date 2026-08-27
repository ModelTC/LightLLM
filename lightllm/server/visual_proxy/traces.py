"""Format and replay reversible public Vision traces."""

from __future__ import annotations

import copy
import json
import uuid
from html import escape as html_escape, unescape as html_unescape
from typing import Any, Optional

from .config import VisualChatProxyError
from .constants import (
    BUILTIN_TRACE_FORMATS,
    FUNCTION_PATTERN,
    IMAGE_ALIAS_PATTERN,
    IMAGE_TAG_PATTERN,
    MAX_REASONING_CONTEXT_BYTES,
    NATURAL_OBSERVATION_PATTERN,
    NATURAL_OBSERVATION_RESULT_MARKER,
    NATURAL_VISION_TRACE_PATTERN,
    PARAMETER_PATTERN,
    TOOL_CALL_PATTERN,
    VISION_READER_NAME,
    XML_BUILTIN_TRACE_PAIR_PATTERN,
)
from .sanitizer import _merge_reasoning_text, sanitize_reasoning


def _format_public_builtin_projection(
    image: str,
    task: str,
    result: str,
    offset: int,
    trace_format: str = "natural",
) -> str:
    if not task or not result:
        raise VisualChatProxyError("Cannot expose an empty builtin visual result")
    return _format_builtin_trace(trace_format, image, task, result, "", offset)


def _one_line_trace_text(value: str) -> str:
    return " ".join(str(value).split())


def _canonical_image_reference(value: str) -> str:
    match = IMAGE_TAG_PATTERN.fullmatch(str(value).strip())
    if match:
        return f"<image_{int(match.group(1))}/>"
    alias = IMAGE_ALIAS_PATTERN.fullmatch(str(value).strip())
    if alias:
        return f"<image_{int(alias.group(1))}/>"
    return _one_line_trace_text(value)


def _natural_trace_prefix(existing_reasoning: str, offset: int) -> str:
    if offset > 0:
        return "我接着"
    has_prior_observation = (
        "让内建读图能力完成这个任务：" in existing_reasoning
        or NATURAL_OBSERVATION_RESULT_MARKER in existing_reasoning
    )
    return "我接着" if has_prior_observation else "我先"


def _format_natural_builtin_trace(
    image: str,
    task: str,
    result: str,
    prefix: str,
) -> str:
    image = _canonical_image_reference(image)
    if not image or not task or not result:
        raise VisualChatProxyError(
            "Cannot format natural builtin trace with an empty image, task, or response"
        )
    encoded_task = json.dumps(str(task), ensure_ascii=False)
    encoded_result = json.dumps(str(result), ensure_ascii=False)
    return (
        f"{prefix}仔细看了图片 {image}，想确认 {encoded_task}"
        f"{NATURAL_OBSERVATION_RESULT_MARKER}{encoded_result}。"
    )


def _format_xml_builtin_trace(image: str, task: str, result: str) -> str:
    return (
        "<tool_call>\n"
        f"<function={VISION_READER_NAME}>\n"
        f"<parameter=image>\n{_canonical_image_reference(image)}\n</parameter>\n"
        f"<parameter=task>\n{html_escape(task.strip())}\n</parameter>\n"
        "</function>\n"
        "</tool_call>\n"
        f"<tool_response>\n{html_escape(result.strip())}\n</tool_response>"
    )


def _format_builtin_trace(
    trace_format: str,
    image: str,
    task: str,
    result: str,
    existing_reasoning: str,
    offset: int,
) -> str:
    if trace_format == "natural":
        return _format_natural_builtin_trace(
            image,
            task,
            result,
            _natural_trace_prefix(existing_reasoning, offset),
        )
    if trace_format == "xml":
        return _format_xml_builtin_trace(image, task, result)
    raise VisualChatProxyError(f"Unsupported builtin trace format: {trace_format}")


def _parse_natural_observation(line: str) -> Optional[dict[str, str]]:
    """Decode one observation sentence without relying on value delimiters."""

    match = NATURAL_OBSERVATION_PATTERN.match(line)
    if not match:
        return None
    decoder = json.JSONDecoder()
    try:
        task, task_end = decoder.raw_decode(line, match.end())
    except (json.JSONDecodeError, RecursionError):
        return None
    if not isinstance(task, str) or not line.startswith(
        NATURAL_OBSERVATION_RESULT_MARKER, task_end
    ):
        return None
    result_start = task_end + len(NATURAL_OBSERVATION_RESULT_MARKER)
    try:
        result, result_end = decoder.raw_decode(line, result_start)
    except (json.JSONDecodeError, RecursionError):
        return None
    if not isinstance(result, str) or line[result_end:].strip() != "。":
        return None
    return {
        "image": _canonical_image_reference(match.group(1)),
        "task": task,
        "tool_response": result,
    }


def _parse_natural_builtin_trace(reasoning: str) -> tuple[list[dict[str, str]], str]:
    lines = reasoning.splitlines()
    segments: list[dict[str, str]] = []
    current_reasoning: list[str] = []
    index = 0
    while index < len(lines):
        line = lines[index].strip()
        observation = _parse_natural_observation(line)
        if observation is not None:
            observation["reasoning"] = "\n".join(current_reasoning).strip()
            segments.append(observation)
            current_reasoning = []
            index += 1
            continue
        match = NATURAL_VISION_TRACE_PATTERN.fullmatch(line)
        if not match:
            current_reasoning.append(lines[index])
            index += 1
            continue
        if index + 1 >= len(lines):
            current_reasoning.append(lines[index])
            index += 1
            continue
        response_line = lines[index + 1].strip()
        if not response_line.startswith("读图结果："):
            current_reasoning.append(lines[index])
            index += 1
            continue
        response = response_line[len("读图结果：") :].strip()
        if not response:
            current_reasoning.extend(lines[index : index + 2])
            index += 2
            continue
        segments.append(
            {
                "reasoning": "\n".join(current_reasoning).strip(),
                "image": _canonical_image_reference(match.group(1)),
                "task": match.group(2).strip(),
                "tool_response": response,
            }
        )
        current_reasoning = []
        index += 2
    return segments, "\n".join(current_reasoning).strip()


def _parse_xml_tool_call(xml: str) -> tuple[str, dict[str, str]]:
    inner_match = TOOL_CALL_PATTERN.fullmatch(xml.strip())
    if not inner_match:
        raise ValueError("Malformed builtin tool_call block")
    function_match = FUNCTION_PATTERN.fullmatch(inner_match.group(1).strip())
    if not function_match:
        raise ValueError("Builtin tool_call has no valid <function=...> block")
    name = function_match.group(1).strip()
    arguments = {
        parameter_name.strip(): parameter_value.strip()
        for parameter_name, parameter_value in PARAMETER_PATTERN.findall(
            function_match.group(2)
        )
    }
    return name, arguments


def _parse_xml_builtin_trace(reasoning: str) -> tuple[list[dict[str, str]], str]:
    # Only the proxy-owned vision_reader pair is replayable. Other tools may
    # legitimately have raw serializations in historical reasoning; treating
    # every <tool_response> as a builtin vision trace caused unrelated agent
    # histories to fail with HTTP 400 before model inference.
    has_builtin_marker = f"<function={VISION_READER_NAME}>" in reasoning
    if not has_builtin_marker:
        return [], sanitize_reasoning(reasoning)
    segments: list[dict[str, str]] = []
    cursor = 0
    for match in XML_BUILTIN_TRACE_PAIR_PATTERN.finditer(reasoning):
        reasoning_before = sanitize_reasoning(reasoning[cursor : match.start()])
        name, arguments = _parse_xml_tool_call(match.group(1))
        if name != VISION_READER_NAME:
            raise ValueError(f"Unsupported builtin trace function: {name!r}")
        image = _canonical_image_reference(arguments.get("image", ""))
        task = html_unescape(arguments.get("task", "").strip())
        response = html_unescape(match.group(2).strip())
        if not image or not task or not response:
            raise ValueError(
                "Malformed XML builtin trace: empty image, task, or response"
            )
        segments.append(
            {
                "reasoning": reasoning_before,
                "image": image,
                "task": task,
                "tool_response": response,
            }
        )
        cursor = match.end()
    trailing = sanitize_reasoning(reasoning[cursor:])
    if not segments:
        raise ValueError("Malformed XML builtin vision_reader trace")
    return segments, trailing


def _trace_tool_call(image: str, task: str) -> dict[str, Any]:
    return {
        "id": f"call_builtin_{uuid.uuid4().hex[:24]}",
        "type": "function",
        "function": {
            "name": VISION_READER_NAME,
            "arguments": json.dumps(
                {"image": image, "task": task},
                ensure_ascii=False,
                separators=(",", ":"),
            ),
        },
    }


def expand_builtin_traces(
    messages: list[dict[str, Any]],
    trace_format: str,
) -> list[dict[str, Any]]:
    """Expand public builtin vision-reader traces for a model-native chat template."""

    if trace_format not in BUILTIN_TRACE_FORMATS:
        raise ValueError(f"Unsupported builtin trace format: {trace_format}")
    expanded: list[dict[str, Any]] = []
    reasoning_bytes = 0
    parser = (
        _parse_natural_builtin_trace
        if trace_format == "natural"
        else _parse_xml_builtin_trace
    )
    for message in messages:
        if message.get("role") != "assistant":
            expanded.append(copy.deepcopy(message))
            continue
        reasoning = _merge_reasoning_text(
            message.get("reasoning"), message.get("reasoning_content")
        )
        if not reasoning:
            expanded.append(copy.deepcopy(message))
            continue
        reasoning_bytes += len(reasoning.encode("utf-8"))
        if reasoning_bytes > MAX_REASONING_CONTEXT_BYTES:
            raise ValueError(
                "Visual reasoning history exceeds the configured size limit"
            )
        segments, final_reasoning = parser(reasoning)
        if not segments:
            clean_message = {
                key: copy.deepcopy(value)
                for key, value in message.items()
                if key not in {"reasoning", "reasoning_content"}
            }
            clean_reasoning = final_reasoning
            if clean_reasoning:
                clean_message["reasoning"] = clean_reasoning
            expanded.append(clean_message)
            continue
        for segment in segments:
            tool_call = _trace_tool_call(segment["image"], segment["task"])
            assistant_message: dict[str, Any] = {
                "role": "assistant",
                "content": "",
                "tool_calls": [tool_call],
            }
            clean_segment_reasoning = segment["reasoning"]
            if clean_segment_reasoning:
                assistant_message["reasoning"] = clean_segment_reasoning
            expanded.append(assistant_message)
            expanded.append(
                {
                    "role": "tool",
                    "name": VISION_READER_NAME,
                    "tool_call_id": tool_call["id"],
                    "content": segment["tool_response"],
                }
            )
        trailing_message = {
            key: copy.deepcopy(value)
            for key, value in message.items()
            if key not in {"reasoning", "reasoning_content"}
        }
        clean_final_reasoning = final_reasoning
        if clean_final_reasoning:
            trailing_message["reasoning"] = clean_final_reasoning
        if (
            trailing_message.get("content")
            or trailing_message.get("reasoning")
            or trailing_message.get("tool_calls")
        ):
            expanded.append(trailing_message)
    return expanded
