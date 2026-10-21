"""Shared boundaries and schema coercion for GLM's XML tool calls."""

import json
import math
from typing import Any, Iterator, Tuple


TOOL_START = "<tool_call>"
TOOL_END = "</tool_call>"
VALUE_START = "<arg_value>"
VALUE_END = "</arg_value>"
_TYPE_ALIASES = {
    **dict.fromkeys(("str", "text", "varchar", "char", "enum"), "string"),
    **dict.fromkeys(("int", "int32", "int64", "uint", "uint32", "uint64", "long", "short", "unsigned"), "integer"),
    **dict.fromkeys(("float", "float32", "float64", "double"), "number"),
    "bool": "boolean",
    "dict": "object",
    **dict.fromkeys(("arr", "list", "sequence"), "array"),
}


class GlmToolCallScanner:
    """Scan after a tool opener, treating argument values as literal text.

    ``scan`` returns the consumed length and whether the outer call closed.
    Callers retain the unconsumed suffix, which may be a partial delimiter,
    and pass it with the next chunk. Only ``</arg_value>`` closes a value;
    thinking and tool-call tags inside that value have no structural meaning.
    """

    def __init__(self):
        self.in_arg_value = False

    def scan(self, text: str) -> Tuple[int, bool]:
        offset = 0
        while offset < len(text):
            markers = (VALUE_END,) if self.in_arg_value else (VALUE_START, TOOL_END)
            positions = [(text.find(marker, offset), marker) for marker in markers]
            positions = [(position, marker) for position, marker in positions if position != -1]
            if not positions:
                max_hold = min(len(text) - offset, max(map(len, markers)) - 1)
                for length in range(max_hold, 0, -1):
                    if any(marker.startswith(text[-length:]) for marker in markers):
                        return len(text) - length, False
                return len(text), False

            position, marker = min(positions)
            offset = position + len(marker)
            if marker == TOOL_END:
                return offset, True
            self.in_arg_value = marker == VALUE_START

        return offset, False


def iter_glm_tool_call_spans(text: str) -> Iterator[Tuple[int, int]]:
    start = text.find(TOOL_START)
    while start != -1:
        body_start = start + len(TOOL_START)
        consumed, complete = GlmToolCallScanner().scan(text[body_start:])
        if not complete:
            return
        end = body_start + consumed
        yield start, end
        start = text.find(TOOL_START, end)


def _schema_types(schema: Any) -> set[str]:
    if not isinstance(schema, dict):
        return {"string"}
    types = set()
    declared = schema.get("type")
    if isinstance(declared, str):
        types.add(declared)
    elif isinstance(declared, list):
        types.update(item for item in declared if isinstance(item, str))
    for value in schema.get("enum", []) if isinstance(schema.get("enum"), list) else []:
        if value is None:
            types.add("null")
        elif isinstance(value, bool):
            types.add("boolean")
        elif isinstance(value, int):
            types.add("integer")
        elif isinstance(value, float):
            types.add("number")
        elif isinstance(value, str):
            types.add("string")
        elif isinstance(value, list):
            types.add("array")
        elif isinstance(value, dict):
            types.add("object")
    for field in ("anyOf", "oneOf", "allOf"):
        choices = schema.get(field)
        if isinstance(choices, list):
            for choice in choices:
                types.update(_schema_types(choice))
    return types or {"string"}


def _finite_json(value: Any) -> bool:
    if isinstance(value, float):
        return math.isfinite(value)
    if isinstance(value, dict):
        return all(_finite_json(item) for item in value.values())
    if isinstance(value, list):
        return all(_finite_json(item) for item in value)
    return True


def parse_glm_xml_value(value: str, schema: Any) -> Any:
    """Match vLLM's schema-based coercion without changing literal strings.

    Union types use null/integer/number/boolean/object/array/string priority.
    Unspecified fields remain strings. Non-finite numbers remain raw text so
    the resulting arguments are always valid JSON.
    """
    types = {_TYPE_ALIASES.get(item.strip().lower(), item.strip().lower()) for item in _schema_types(schema)}
    for candidate in ("null", "integer", "number", "boolean", "object", "array", "string"):
        if candidate not in types:
            continue
        try:
            if candidate == "null" and value.lower() == "null":
                return None
            if candidate == "integer":
                return int(value)
            if candidate == "number":
                number = float(value)
                if math.isfinite(number):
                    return number if number != int(number) else int(number)
            if candidate == "boolean":
                normalized = value.strip().lower()
                if normalized in ("true", "1"):
                    return True
                if normalized in ("false", "0"):
                    return False
            if candidate in ("object", "array"):
                parsed = json.loads(value)
                if _finite_json(parsed):
                    return parsed
            if candidate == "string":
                return value
        except (ValueError, TypeError):
            continue
    try:
        parsed = json.loads(value)
        return parsed if _finite_json(parsed) else value
    except ValueError:
        return value
