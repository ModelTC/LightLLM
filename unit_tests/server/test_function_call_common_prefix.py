import json

import pytest

from lightllm.server.api_models import Tool
from lightllm.server.function_call_parser import FunctionCallParser, _find_common_prefix


def _reference_common_prefix(s1, s2):
    prefix = ""
    for a, b in zip(s1, s2):
        if a != b:
            break
        prefix += a
    return prefix


@pytest.mark.parametrize(
    "s1,s2",
    [
        ("", ""),
        ("", "abc"),
        ("abc", ""),
        ("abc", "abc"),
        ("abc", "abcd"),
        ("abcd", "abc"),
        ("abc", "abd"),
        ("xbc", "abc"),
        ('{"a": "x"}', '{"a": "xy"}'),
        ('{"a": "é😀"}', '{"a": "é😀b"}'),
        ("a" * 1000 + "b", "a" * 1000 + "c"),
    ],
)
def test_find_common_prefix_matches_reference(s1, s2):
    assert _find_common_prefix(s1, s2) == _reference_common_prefix(s1, s2)


def test_qwen25_streams_long_arguments_in_small_deltas():
    tools = [
        Tool.model_validate(
            {
                "type": "function",
                "function": {
                    "name": "write_file",
                    "parameters": {"type": "object", "properties": {"content": {"type": "string"}}},
                },
            }
        )
    ]
    arguments = {"content": "print(1)\n" * 500}
    payload = json.dumps(arguments)
    deltas = ["<tool_call>", '\n{"name": "write_file", "arguments": ']
    deltas += [payload[i : i + 4] for i in range(0, len(payload), 4)]
    deltas += ["}\n", "</tool_call>"]

    parser = FunctionCallParser(tools, "qwen25")
    streamed = ""
    names = []
    for delta in deltas:
        _, calls = parser.parse_stream_chunk(delta)
        for call in calls:
            if call.name:
                names.append(call.name)
            streamed += call.parameters or ""

    assert names == ["write_file"]
    assert json.loads(streamed) == arguments
