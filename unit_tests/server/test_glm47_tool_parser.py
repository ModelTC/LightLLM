import json

import pytest

from lightllm.server.api_models import Tool
from lightllm.server.function_call_parser import FunctionCallParser


TOOLS = [
    Tool.model_validate(
        {
            "type": "function",
            "function": {
                "name": "Read",
                "parameters": {"type": "object", "properties": {"file_path": {"type": "string"}}},
            },
        }
    ),
    Tool.model_validate(
        {
            "type": "function",
            "function": {
                "name": "Write",
                "parameters": {
                    "type": "object",
                    "properties": {"content": {"type": "string"}, "overwrite": {"type": "boolean"}},
                },
            },
        }
    ),
]


def collect_stream(text, chunk_size):
    parser = FunctionCallParser(TOOLS, "glm47")
    content = ""
    calls = {}
    for start in range(0, len(text), chunk_size):
        normal, deltas = parser.parse_stream_chunk(text[start : start + chunk_size])
        content += normal
        for delta in deltas:
            if delta.name is not None:
                assert delta.tool_index not in calls, "a tool call must have exactly one name event"
                calls[delta.tool_index] = {"name": delta.name, "arguments": ""}
            calls[delta.tool_index]["arguments"] += delta.parameters
    return content, calls


@pytest.mark.parametrize("separator", ["", "\n"])
@pytest.mark.parametrize("chunk_size", [1, 7, 39, 4096])
def test_repeated_tool_calls_have_separate_indices(separator, chunk_size):
    text = (
        f'<tool_call>Write{separator}<arg_key>content</arg_key><arg_value>{{"ready": true}}</arg_value>'
        "<arg_key>overwrite</arg_key><arg_value>false</arg_value></tool_call>"
        f"<tool_call>Read{separator}<arg_key>file_path</arg_key><arg_value>/tmp/a</arg_value></tool_call>"
        f"<tool_call>Read{separator}<arg_key>file_path</arg_key><arg_value>/tmp/b</arg_value></tool_call>"
    )
    _, calls = collect_stream(text, chunk_size)
    assert list(calls) == [0, 1, 2]
    assert [call["name"] for call in calls.values()] == ["Write", "Read", "Read"]
    assert [json.loads(call["arguments"]) for call in calls.values()] == [
        {"content": '{"ready": true}', "overwrite": False},
        {"file_path": "/tmp/a"},
        {"file_path": "/tmp/b"},
    ]
    _, nonstream = FunctionCallParser(TOOLS, "glm47").parse_non_stream(text)
    assert [call.tool_index for call in nonstream] == [0, 1, 2]
    assert [call.parameters for call in nonstream] == [call["arguments"] for call in calls.values()]


@pytest.mark.parametrize("value", ['{"count": 1}', "123", "false", "    return 1\n", '"quoted"'])
def test_string_arguments_preserve_type_and_whitespace(value):
    text = f"<tool_call>Write<arg_key>content</arg_key><arg_value>{value}</arg_value></tool_call>"
    _, calls = collect_stream(text, 1)
    assert json.loads(calls[0]["arguments"])["content"] == value


@pytest.mark.parametrize("chunk_size", [1, 7, 4096])
def test_text_around_tool_calls_survives_chunk_boundaries(chunk_size):
    text = (
        "Before <tool_call>Read<arg_key>file_path</arg_key><arg_value>/tmp/a</arg_value></tool_call>"
        " between <tool_call>Read<arg_key>file_path</arg_key><arg_value>/tmp/b</arg_value></tool_call> after"
    )
    content, calls = collect_stream(text, chunk_size)
    assert content == "Before  between  after"
    assert list(calls) == [0, 1]


def test_buffered_arguments_keep_the_stream_alive_without_repeating_the_name():
    parser = FunctionCallParser(TOOLS, "glm47")
    normal, calls = parser.parse_stream_chunk("<tool_call>Write<arg_key>content</arg_key><arg_value>")
    assert normal == ""
    assert len(calls) == 1
    assert calls[0].tool_index == 0
    assert calls[0].name == "Write"
    assert calls[0].parameters == ""

    value = '    line 1\n{"literal": true}\n' + "x" * 100_000
    for chunk in (value[:13], value[13:27], value[27:]):
        normal, calls = parser.parse_stream_chunk(chunk)
        assert normal == ""
        assert len(calls) == 1
        assert calls[0].tool_index == 0
        assert calls[0].name is None
        assert calls[0].parameters == ""

    normal, calls = parser.parse_stream_chunk("</arg_value></tool_call>")
    assert normal == ""
    assert len(calls) == 1
    assert calls[0].tool_index == 0
    assert calls[0].name is None
    assert json.loads(calls[0].parameters) == {"content": value}

    _, calls = parser.parse_stream_chunk("<tool_call>Write\n")
    assert len(calls) == 1
    assert calls[0].tool_index == 1
    assert calls[0].name == "Write"


def test_undefined_buffered_tool_does_not_consume_a_call_index():
    parser = FunctionCallParser(TOOLS, "glm47")
    _, calls = parser.parse_stream_chunk("<tool_call>unknown\n<arg_key>content</arg_key><arg_value>x")
    assert calls == []
    _, calls = parser.parse_stream_chunk("</arg_value></tool_call><tool_call>Read\n")
    assert len(calls) == 1
    assert calls[0].name == "Read"
    assert calls[0].tool_index == 0
