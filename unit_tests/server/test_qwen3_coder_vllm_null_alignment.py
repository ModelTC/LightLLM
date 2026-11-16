"""Qwen3-Coder null text compared with vLLM's parameter conversion.

vLLM keeps a null spelling only when the resolved schema is exactly string.
``null`` and ``None`` are both null spellings, case-insensitive, with surrounding
whitespace ignored. These cases come from ``convert_with_optional_schema`` and
``SchemaRoot::options`` in vLLM's parser.
"""

import json
from typing import Any, Optional

import pytest

from lightllm.server.api_models import Function, Tool
from lightllm.server.function_call_parser import FunctionCallParser

NULL_SPELLINGS = [
    "null",
    "NULL",
    "Null",
    "NuLl",
    " none ",
    "None",
    "NONE",
    "\tnull",
    " None ",
]

# Resolved type is exactly string, so the original characters stay.
VERBATIM_STRING_SCHEMAS = [
    {"type": "string"},
    {"type": "STRING"},
    {"type": " str "},
    {"type": "text"},
    {"type": "varchar"},
    {"type": "char"},
    {"type": "enum"},
    {"type": ["string"]},
    {"type": ["str", "text"]},
    {"type": ["string", 1]},
    {"enum": ["red", "green"]},
    {"enum": ["null", "auto"]},
    {"enum": ["red"]},
    {"type": "integer", "enum": ["a", "b"]},
    {"const": "ok"},
    {"const": "null"},
    {"anyOf": [{"type": "string"}]},
    {"oneOf": [{"type": "varchar"}]},
    {"allOf": [{"type": "string"}]},
    {"type": "string", "nullable": False},
]

# Anything other than a pure string schema coerces every null spelling to None.
COERCED_NULL_SCHEMAS = [
    {"type": "integer"},
    {"type": "int"},
    {"type": "int64"},
    {"type": "unsigned"},
    {"type": "unsigned_int"},
    {"type": "number"},
    {"type": "float"},
    {"type": "float64"},
    {"type": "double"},
    {"type": "boolean"},
    {"type": "bool"},
    {"type": "binary"},
    {"type": "object"},
    {"type": "dict"},
    {"type": "map"},
    {"type": "array"},
    {"type": "list"},
    {"type": "sequence"},
    {"type": "null"},
    {"type": "widget"},
    {"type": ["string", "null"]},
    {"type": ["integer", "null"]},
    {"type": ["integer", "string"]},
    {"type": ["object", "array"]},
    {"type": ["integer", "string"], "nullable": True},
    {"type": "string", "nullable": True},
    {"enum": ["red", "green"], "nullable": True},
    {"enum": [None, "auto"]},
    {"enum": ["red", None]},
    {"enum": [1, 2, 3]},
    {"enum": ["fast", 1]},
    {"enum": ["auto", True]},
    {"enum": []},
    {"const": None},
    {"const": 1},
    {"const": True},
    {"const": 1.5},
    {"anyOf": [{"type": "string"}, {"type": "null"}]},
    {"anyOf": [{"type": "integer"}, {"type": "string"}]},
    {"oneOf": [{"type": "integer"}, {"type": "string"}]},
    {"oneOf": []},
    {"allOf": [{"type": "string"}, {"type": "string"}]},
    {"allOf": []},
    {},
    {"properties": {"x": {"type": "integer"}}},
    {"items": {"type": "string"}},
    {"additionalProperties": True},
    {"type": [1, 2]},
    None,
]

NOT_NULL_TEXT = ["nil", "nullified", "none_", "nones", "", "hello", "0", "false"]


def _schema_label(schema: Any) -> str:
    if schema is None:
        return "missing"
    return json.dumps(schema, sort_keys=True, separators=(",", ":"), default=str)


def _parsed_value(schema: Any, value: str, chunk_size: Optional[int]) -> Any:
    properties = {} if schema is None else {"value": schema}
    tool = Tool(
        type="function",
        function=Function(name="echo", parameters={"type": "object", "properties": properties}),
    )
    parser = FunctionCallParser([tool], "qwen3_coder")
    text = f"<tool_call>\n<function=echo>\n<parameter=value>{value}</parameter>\n</function>\n</tool_call>"
    if chunk_size is None:
        _, calls = parser.parse_non_stream(text)
        arguments = calls[0].parameters
    else:
        argument_parts = []
        for offset in range(0, len(text), chunk_size):
            _, calls = parser.parse_stream_chunk(text[offset : offset + chunk_size])
            argument_parts.extend(call.parameters for call in calls)
        arguments = "".join(argument_parts)
    return json.loads(arguments)["value"]


@pytest.mark.parametrize("chunk_size", [None, 1], ids=["nonstream", "stream-1"])
@pytest.mark.parametrize("schema", VERBATIM_STRING_SCHEMAS, ids=_schema_label)
@pytest.mark.parametrize("value", NULL_SPELLINGS)
def test_pure_string_schema_keeps_null_spelling(schema, value, chunk_size):
    assert _parsed_value(schema=schema, value=value, chunk_size=chunk_size) == value


@pytest.mark.parametrize("chunk_size", [None, 1], ids=["nonstream", "stream-1"])
@pytest.mark.parametrize("schema", COERCED_NULL_SCHEMAS, ids=_schema_label)
@pytest.mark.parametrize("value", NULL_SPELLINGS)
def test_other_schemas_coerce_null_spelling(schema, value, chunk_size):
    assert _parsed_value(schema=schema, value=value, chunk_size=chunk_size) is None


@pytest.mark.parametrize("schema", VERBATIM_STRING_SCHEMAS[:8], ids=_schema_label)
@pytest.mark.parametrize("value", NOT_NULL_TEXT)
def test_pure_string_schema_keeps_other_text(schema, value):
    assert _parsed_value(schema=schema, value=value, chunk_size=None) == value


@pytest.mark.parametrize(
    ("schema", "value", "expected"),
    [
        ({"type": "integer"}, "nil", "nil"),
        ({"type": "integer"}, "nullified", "nullified"),
        ({"type": "integer"}, "42", 42),
        ({"type": "string", "nullable": True}, "hello", "hello"),
        ({"type": "string", "nullable": True}, "nil", "nil"),
        ({"enum": [None, "auto"]}, "auto", "auto"),
        ({"enum": ["red", "green"]}, "red", "red"),
        ({"const": None}, "hello", "hello"),
        ({}, "42", "42"),
        (None, "42", "42"),
    ],
)
def test_non_null_text_follows_the_same_schema_split(schema, value, expected):
    result = _parsed_value(schema=schema, value=value, chunk_size=None)
    assert type(result) is type(expected)
    assert result == expected


@pytest.mark.parametrize(
    ("property_schema", "parameters", "value", "expected"),
    [
        ({"$ref": "#/$defs/Name"}, {"$defs": {"Name": {"type": "string"}}}, "null", "null"),
        ({"$ref": "#/$defs/Name"}, {"$defs": {"Name": {"type": "string"}}}, "None", "None"),
        ({"$ref": "#/$defs/Count"}, {"$defs": {"Count": {"type": "integer"}}}, "None", None),
        (
            {"$ref": "#/$defs/Name", "nullable": True},
            {"$defs": {"Name": {"type": "string"}}},
            " null ",
            None,
        ),
        (
            {"$ref": "#/$defs/NullableName"},
            {"$defs": {"NullableName": {"type": "string", "nullable": True}}},
            "null",
            None,
        ),
        (
            {"anyOf": [{"type": "string", "nullable": True}]},
            {},
            "NONE",
            None,
        ),
        (
            {"allOf": [{"type": "string", "nullable": True}]},
            {},
            "null",
            None,
        ),
        (
            {"anyOf": [{"$ref": "#/$defs/Name"}, {"type": "null"}]},
            {"$defs": {"Name": {"type": "string"}}},
            "NONE",
            None,
        ),
        ({"$ref": "#/$defs/Missing"}, {"$defs": {}}, "null", None),
        (
            {"$ref": "#/$defs/Node"},
            {"$defs": {"Node": {"properties": {"child": {"$ref": "#/$defs/Node"}}}}},
            "null",
            None,
        ),
    ],
)
def test_ref_schemas_match_vllm_null_rules(property_schema, parameters, value, expected):
    parameters = {
        "type": "object",
        "properties": {"value": property_schema},
        **parameters,
    }
    tool = Tool(type="function", function=Function(name="echo", parameters=parameters))
    parser = FunctionCallParser([tool], "qwen3_coder")
    text = f"<tool_call>\n<function=echo>\n<parameter=value>{value}</parameter>\n</function>\n</tool_call>"
    _, calls = parser.parse_non_stream(text)
    result = json.loads(calls[0].parameters)["value"]
    assert type(result) is type(expected)
    assert result == expected
