import json

import pytest

from lightllm.server.api_models import Tool
from lightllm.server.function_call_parser import FunctionCallParser

TOOLS = [
    Tool.model_validate(
        {
            "type": "function",
            "function": {
                "name": "write_file",
                "parameters": {"type": "object", "properties": {"content": {"type": "string"}}},
            },
        }
    ),
    Tool.model_validate(
        {
            "type": "function",
            "function": {
                "name": "get_weather",
                "parameters": {"type": "object", "properties": {"location": {"type": "string"}}},
            },
        }
    ),
]

DSML = "｜DSML｜"


def _dsml_invoke(name, param, value):
    return (
        f'<{DSML}invoke name="{name}">\n'
        f'<{DSML}parameter name="{param}" string="true">{value}</{DSML}parameter>\n'
        f"</{DSML}invoke>"
    )


def _stream(parser, deltas):
    names = []
    arguments = {}
    for delta in deltas:
        _, calls = parser.parse_stream_chunk(delta)
        for call in calls:
            if call.name:
                names.append(call.name)
            arguments[call.tool_index] = arguments.get(call.tool_index, "") + (call.parameters or "")
    return names, arguments


@pytest.mark.parametrize("tool_call_parser,block", [("deepseekv32", "function_calls"), ("deepseekv4", "tool_calls")])
def test_dsml_streams_long_invokes_in_small_deltas(tool_call_parser, block):
    content = "print(1)\n" * 500
    deltas = [f"<{DSML}{block}>", "\n", f"<{DSML}invoke", ' name="write_file">\n', f"<{DSML}parameter"]
    deltas += [' name="content" string="true">']
    deltas += [content[i : i + 4] for i in range(0, len(content), 4)]
    deltas += [f"</{DSML}parameter>", "\n", f"</{DSML}invoke>", "\n"]
    deltas += [f"<{DSML}invoke", ' name="get_weather">\n', f"<{DSML}parameter", ' name="location" string="true">']
    deltas += ["Hang", "zhou", f"</{DSML}parameter>", "\n", f"</{DSML}invoke>", "\n", f"</{DSML}{block}>"]

    parser = FunctionCallParser(TOOLS, tool_call_parser)
    names, arguments = _stream(parser, deltas)

    assert names == ["write_file", "get_weather"]
    assert json.loads(arguments[0]) == {"content": content}
    assert json.loads(arguments[1]) == {"location": "Hangzhou"}


def test_dsml_emits_buffered_invoke_on_next_delta():
    # Both invokes arrive in one delta; the second one is still in the buffer
    # and must be emitted by the next delta, which has no invoke end tag.
    text = f"<{DSML}function_calls>\n"
    text += _dsml_invoke("write_file", "content", "x = 1") + "\n"
    text += _dsml_invoke("get_weather", "location", "Paris") + "\n"

    parser = FunctionCallParser(TOOLS, "deepseekv32")
    names, arguments = _stream(parser, [text, f"</{DSML}function_calls>"])

    assert names == ["write_file", "get_weather"]
    assert json.loads(arguments[0]) == {"content": "x = 1"}
    assert json.loads(arguments[1]) == {"location": "Paris"}


def test_deepseekv31_streams_long_arguments_in_small_deltas():
    arguments = {"content": "print(1)\n" * 500}
    payload = json.dumps(arguments)
    deltas = ["<｜tool▁calls▁begin｜>", "<｜tool▁call▁begin｜>", "write_file", "<｜tool▁sep｜>"]
    deltas += [payload[i : i + 4] for i in range(0, len(payload), 4)]
    deltas += ["<｜tool▁call▁end｜>", "<｜tool▁calls▁end｜>"]

    parser = FunctionCallParser(TOOLS, "deepseekv31")
    names, streamed = _stream(parser, deltas)

    assert names == ["write_file"]
    assert json.loads(streamed[0]) == arguments
