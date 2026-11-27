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


@pytest.mark.parametrize("tool_call_parser,block", [("deepseekv32", "function_calls"), ("deepseekv4", "tool_calls")])
@pytest.mark.parametrize("chunk_size", [1, 5, 16, None])
@pytest.mark.parametrize(
    "tool_names",
    [
        ["undefined_tool"],
        ["undefined_tool", "get_weather"],
        ["write_file", "undefined_tool", "get_weather"],
        ["write_file", "get_weather", "undefined_tool"],
    ],
)
def test_dsml_stream_skips_undefined_tools(tool_call_parser, block, chunk_size, tool_names):
    invokes = [_dsml_invoke(name, "location" if name == "get_weather" else "content", "Paris") for name in tool_names]
    deltas = [f"<{DSML}{block}>"] + invokes + [f"</{DSML}{block}>"]
    text = "".join(deltas)
    if chunk_size is not None:
        deltas = [text[i : i + chunk_size] for i in range(0, len(text), chunk_size)]
    elif len(invokes) == 1:
        deltas = [text]

    _, expected = FunctionCallParser(TOOLS, tool_call_parser).parse_non_stream(text)
    parser = FunctionCallParser(TOOLS, tool_call_parser)
    names, arguments = _stream(parser, deltas)

    assert names == [call.name for call in expected]
    assert list(arguments) == list(range(len(expected)))
    assert [json.loads(value) for value in arguments.values()] == [json.loads(call.parameters) for call in expected]
    assert parser.detector.prev_tool_call_arr == [
        {"name": call.name, "arguments": json.loads(call.parameters)} for call in expected
    ]


@pytest.mark.parametrize("tool_call_parser,block", [("deepseekv32", "function_calls"), ("deepseekv4", "tool_calls")])
@pytest.mark.parametrize("split_valid_invoke", [False, True])
def test_dsml_skips_undefined_invokes_before_a_buffered_valid_call(tool_call_parser, block, split_valid_invoke):
    unknown = _dsml_invoke("undefined_tool", "content", "ignored")
    valid = _dsml_invoke("get_weather", "location", "Paris")
    prefix = f"<{DSML}{block}>" + unknown * 3
    suffix = f"</{DSML}{block}>"
    if split_valid_invoke:
        split = valid.index(f"<{DSML}parameter")
        deltas = [prefix + valid[:split], valid[split:] + suffix]
    else:
        deltas = [prefix + valid + suffix]

    parser = FunctionCallParser(TOOLS, tool_call_parser)
    names, arguments = _stream(parser, deltas)

    assert names == ["get_weather"]
    assert list(arguments) == [0]
    assert json.loads(arguments[0]) == {"location": "Paris"}
    assert parser.detector.prev_tool_call_arr == [{"name": "get_weather", "arguments": {"location": "Paris"}}]


def _v3_call(name, payload):
    return "<｜tool▁call▁begin｜>function<｜tool▁sep｜>" + name + "\n```json\n" + payload + "\n```<｜tool▁call▁end｜>"


def test_deepseekv3_streams_long_arguments_in_small_deltas():
    arguments = {"content": "print(1)\n" * 500}
    payload = json.dumps(arguments)
    call = _v3_call("write_file", payload)
    deltas = ["<｜tool▁calls▁begin｜>"] + [call[i : i + 4] for i in range(0, len(call), 4)]

    parser = FunctionCallParser(TOOLS, "deepseekv3")
    names, streamed = _stream(parser, deltas)

    assert names == ["write_file"]
    assert json.loads(streamed[0]) == arguments


def test_deepseekv3_holds_arguments_until_the_json_fence_closes():
    payload = json.dumps({"content": "x" * 200})
    head = "<｜tool▁calls▁begin｜><｜tool▁call▁begin｜>function<｜tool▁sep｜>write_file\n```json\n"
    parser = FunctionCallParser(TOOLS, "deepseekv3")
    names, streamed = _stream(parser, [head] + [payload[i : i + 4] for i in range(0, len(payload), 4)])
    assert names == []
    assert streamed == {}

    # The closing fence is split across deltas. The name is emitted on the delta that
    # completes it, and the arguments follow on the next delta, same as one regex search.
    names, streamed = _stream(parser, ["\n", "``", "`<｜tool▁call▁end｜>", "<｜tool▁calls▁end｜>"])
    assert names == ["write_file"]
    assert json.loads(streamed[0]) == {"content": "x" * 200}


def test_deepseekv3_keeps_a_fence_inside_the_argument():
    content = "line1\n```\nline2\n```python\ncode"
    payload = json.dumps({"content": content})
    text = "<｜tool▁calls▁begin｜>" + _v3_call("write_file", payload)
    deltas = [text[: len("<｜tool▁calls▁begin｜>")]] + [
        text[i : i + 4] for i in range(len("<｜tool▁calls▁begin｜>"), len(text), 4)
    ]

    parser = FunctionCallParser(TOOLS, "deepseekv3")
    names, streamed = _stream(parser, deltas)

    assert names == ["write_file"]
    assert json.loads(streamed[0]) == {"content": content}


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
