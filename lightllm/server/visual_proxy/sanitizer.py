"""Sanitize model text before it reaches public output."""

from __future__ import annotations

import json
import re
from typing import Any, Awaitable, Callable

from .constants import (
    PRIVATE_EXECUTION_STATE_PATTERN,
    PRIVATE_IMAGE_LABEL_PATTERN,
    PRIVATE_MULTIMODAL_MARKERS,
    PRIVATE_MULTIMODAL_MECHANISM_PATTERN,
    PROVIDER_CONTROL_TAG_PATTERN,
    RAW_DSML_TOOL_BLOCK_PATTERN,
    RAW_FUNCTION_CALL_LINE_PATTERN,
    RAW_XML_TOOL_BLOCK_PATTERN,
)


StreamEventCallback = Callable[[str, str, bool], Awaitable[None]]


def _merge_reasoning_text(*values: Any) -> str:
    parts: list[str] = []
    for value in values:
        if not isinstance(value, str):
            continue
        text = value.strip()
        if text and text not in parts:
            parts.append(text)
    return "\n".join(parts)


_BACKEND_SPECIAL_TOKENS = (
    "<|im_end|>",
    "<|im_start|>",
    "<|endoftext|>",
    "<｜end▁of▁sentence｜>",
    "<｜begin▁of▁sentence｜>",
)


def strip_backend_special_tokens(text: str) -> str:
    """Remove model/template boundary tokens that must not reach public output."""

    for token in _BACKEND_SPECIAL_TOKENS:
        text = text.replace(token, "")
    return text.strip()


class _IncrementalVisualTextCleaner:
    """Emit a stable prefix whose concatenation equals strip_backend_special_tokens()."""

    def __init__(self) -> None:
        self._buffer = ""
        self._started = False
        self._pending_whitespace = ""

    def _clean_stable(self, text: str) -> str:
        output: list[str] = []
        for character in text:
            if character.isspace():
                if self._started:
                    self._pending_whitespace += character
                continue
            if self._pending_whitespace:
                output.append(self._pending_whitespace)
                self._pending_whitespace = ""
            output.append(character)
            self._started = True
        return "".join(output)

    def feed(self, value: str) -> str:
        if not value:
            return ""
        self._buffer += value
        for token in _BACKEND_SPECIAL_TOKENS:
            self._buffer = self._buffer.replace(token, "")
        held_suffix = 0
        for token in _BACKEND_SPECIAL_TOKENS:
            max_prefix = min(len(token) - 1, len(self._buffer))
            for length in range(max_prefix, 0, -1):
                if self._buffer.endswith(token[:length]):
                    held_suffix = max(held_suffix, length)
                    break
        stable_end = len(self._buffer) - held_suffix
        stable = self._buffer[:stable_end]
        self._buffer = self._buffer[stable_end:]
        return self._clean_stable(stable)

    def finish(self) -> str:
        for token in _BACKEND_SPECIAL_TOKENS:
            self._buffer = self._buffer.replace(token, "")
        stable = self._buffer
        self._buffer = ""
        output = self._clean_stable(stable)
        self._pending_whitespace = ""
        return output


def _remove_json_tool_call_objects(text: str) -> str:
    """Remove valid JSON objects containing OpenAI or Anthropic tool envelopes."""

    decoder = json.JSONDecoder()
    search_from = 0
    while True:
        marker = re.search(
            r'["\']tool_calls["\']\s*:|["\']type["\']\s*:\s*["\'](?:tool_use|tool_result)["\']|'
            r'["\']tool_use_id["\']\s*:',
            text[search_from:],
            re.IGNORECASE,
        )
        if marker is None:
            return text
        marker_start = search_from + marker.start()
        removed = False
        for brace in reversed(
            [index for index in range(marker_start + 1) if text[index] == "{"]
        ):
            try:
                value, length = decoder.raw_decode(text[brace:])
            except (json.JSONDecodeError, RecursionError):
                continue
            end = brace + length
            is_control = isinstance(value, dict) and (
                "tool_calls" in value
                or value.get("type") in {"tool_use", "tool_result"}
                or "tool_use_id" in value
            )
            if end < marker_start or not is_control:
                continue
            text = text[:brace] + text[end:]
            search_from = max(0, brace - 1)
            removed = True
            break
        if not removed:
            line_start = text.rfind("\n", 0, marker_start) + 1
            line_end = text.find("\n", marker_start)
            if line_end < 0:
                line_end = len(text)
            text = text[:line_start] + text[line_end:]
            search_from = line_start


def sanitize_reasoning(text: Any) -> str:
    """Keep natural reasoning while removing raw provider/tool protocol syntax."""

    if not isinstance(text, str) or not text.strip():
        return ""
    clean = strip_backend_special_tokens(text)
    clean = clean.replace("<think>", "").replace("</think>", "")
    clean = PROVIDER_CONTROL_TAG_PATTERN.sub("", clean)
    clean = RAW_XML_TOOL_BLOCK_PATTERN.sub("\n", clean)
    clean = RAW_DSML_TOOL_BLOCK_PATTERN.sub("\n", clean)
    clean = _remove_json_tool_call_objects(clean)
    clean = "\n".join(
        line
        for line in clean.splitlines()
        if not RAW_FUNCTION_CALL_LINE_PATTERN.search(line)
    )
    clean = re.sub(r"\n[ \t]*\n(?:[ \t]*\n)+", "\n\n", clean)
    return clean.strip()


def text_exposes_private_multimodal_mechanism(value: Any) -> bool:
    if not isinstance(value, str) or not value.strip():
        return False
    lowered = value.lower()
    return (
        any(marker.lower() in lowered for marker in PRIVATE_MULTIMODAL_MARKERS)
        or PRIVATE_MULTIMODAL_MECHANISM_PATTERN.search(value) is not None
        or PRIVATE_IMAGE_LABEL_PATTERN.search(value) is not None
        or PRIVATE_EXECUTION_STATE_PATTERN.search(value) is not None
    )


def reasoning_contains_raw_fc_serialization(text: Any) -> bool:
    """Detect tool/provider control syntax that cannot be exposed as ordinary text."""

    if not isinstance(text, str) or not text.strip():
        return False
    return bool(
        RAW_XML_TOOL_BLOCK_PATTERN.search(text)
        or RAW_DSML_TOOL_BLOCK_PATTERN.search(text)
        or RAW_FUNCTION_CALL_LINE_PATTERN.search(text)
        or PROVIDER_CONTROL_TAG_PATTERN.search(text)
    )
