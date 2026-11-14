"""Schema-aware parameter conversion through the public Qwen3-Coder parser."""

import json

import pytest

from lightllm.server.api_models import Function, Tool
from lightllm.server.function_call_parser import FunctionCallParser


@pytest.mark.parametrize("chunk_size", [None, 1, 7, 1024], ids=["nonstream", "stream-1", "stream-7", "stream-whole"])
@pytest.mark.parametrize(
    ("schema", "value", "expected"),
    [
        ({"type": "string"}, "null", "null"),
        ({"type": "string"}, "NULL", "NULL"),
        ({"type": "string"}, "NuLl", "NuLl"),
        ({"type": "string"}, "hello", "hello"),
        ({"type": "string"}, "true", "true"),
        ({"type": "string"}, "42", "42"),
        ({"type": "string"}, "None", "None"),
        ({"type": "integer"}, "42", 42),
        ({"type": "boolean"}, "true", True),
        ({"type": "object"}, '{"x":1}', {"x": 1}),
        ({"type": "array"}, "[1,2]", [1, 2]),
        ({"type": "integer"}, "null", None),
        ({"type": "boolean"}, "NULL", None),
        ({"type": "object"}, "null", None),
        ({"type": "array"}, "null", None),
        ({"type": ["string", "null"]}, "null", None),
        ({"type": ["object", "null"]}, "null", None),
        ({"type": ["integer", "null"]}, "null", None),
        ({}, "null", None),
        (None, "null", None),
    ],
)
def test_qwen3_coder_parameter_schema_values(schema, value, expected, chunk_size):
    properties = {} if schema is None else {"value": schema}
    tool = Tool(
        type="function",
        function=Function(name="echo", parameters={"type": "object", "properties": properties}),
    )
    parser = FunctionCallParser([tool], "qwen3_coder")
    text = f"<tool_call>\n<function=echo>\n<parameter=value>{value}</parameter>\n</function>\n</tool_call>"

    if chunk_size is None:
        normal_text, calls = parser.parse_non_stream(text)
        assert normal_text == ""
        assert len(calls) == 1
        assert calls[0].name == "echo"
        arguments = calls[0].parameters
    else:
        names = []
        argument_parts = []
        for offset in range(0, len(text), chunk_size):
            normal_text, calls = parser.parse_stream_chunk(text[offset : offset + chunk_size])
            assert normal_text == ""
            for call in calls:
                assert call.tool_index == 0
                if call.name is not None:
                    names.append(call.name)
                argument_parts.append(call.parameters)
        assert names == ["echo"]
        arguments = "".join(argument_parts)

    result = json.loads(arguments)["value"]
    assert type(result) is type(expected)
    assert result == expected
