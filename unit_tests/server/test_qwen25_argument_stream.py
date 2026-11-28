import json

import partial_json_parser
import pytest

from lightllm.server.api_models import Tool
from lightllm.server.function_call_parser import FunctionCallParser
import lightllm.server.function_call_parser as parser_mod


TOOLS = [
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


def _stream(text, delta, parser_name="qwen25"):
    parser = FunctionCallParser(TOOLS, parser_name)
    names = []
    normal = []
    params = []
    for i in range(0, len(text), delta):
        chunk_normal, calls = parser.parse_stream_chunk(text[i : i + delta])
        if chunk_normal:
            normal.append(chunk_normal)
        for call in calls:
            if call.name:
                names.append(call.name)
            if call.parameters:
                params.append(call.parameters)
    return names, "".join(normal), params


def _qwen_call(arguments):
    return '<tool_call>\n{"name": "write_file", "arguments": ' + json.dumps(arguments) + "}\n</tool_call>"


def test_qwen25_plain_argument_chunks_match_character_deltas():
    text = _qwen_call({"content": "abcd"})
    names, normal, params = _stream(text, 1)
    assert names == ["write_file"]
    assert normal == ""
    assert params == ['{"content": "', "a", "b", "c", 'd"}']
    assert json.loads("".join(params)) == {"content": "abcd"}


@pytest.mark.parametrize("delta", [1, 4, 7])
def test_qwen25_argument_text_matches_json_dumps(delta):
    arguments = {"content": 'say "hi"\npath\\to é😀 ' + ("x" * 200)}
    names, _, params = _stream(_qwen_call(arguments), delta)
    assert names == ["write_file"]
    assert "".join(params) == json.dumps(arguments, ensure_ascii=False)


def test_qwen25_two_calls_and_unknown_name():
    first = json.dumps({"name": "write_file", "arguments": {"content": "x" * 40}})
    second = json.dumps({"name": "write_file", "arguments": {"content": "y" * 30}})
    text = f"<tool_call>\n{first}\n</tool_call><tool_call>\n{second}\n</tool_call>"
    names, normal, params = _stream(text, 4)
    assert names == ["write_file", "write_file"]
    assert normal == ""
    acc = "".join(params)
    split = acc.index("}{") + 1
    assert json.loads(acc[:split]) == {"content": "x" * 40}
    assert json.loads(acc[split:]) == {"content": "y" * 30}

    names, normal, params = _stream(
        '<tool_call>\n{"name": "nope", "arguments": {"content": "z"}}\n</tool_call>',
        3,
    )
    assert names == []
    assert params == []


def test_plain_argument_does_not_reparse_every_delta(monkeypatch):
    calls = {"n": 0}
    real = partial_json_parser.loads

    def counted(text, flags):
        calls["n"] += 1
        return real(text, flags)

    monkeypatch.setattr(parser_mod.partial_json_parser, "loads", counted)
    arguments = {"content": "x" * 2000}
    names, _, params = _stream(_qwen_call(arguments), 4)
    assert names == ["write_file"]
    assert "".join(params) == json.dumps(arguments)
    # The header is parsed; the plain body is not reparsed on every 4-character delta.
    assert calls["n"] < 40
