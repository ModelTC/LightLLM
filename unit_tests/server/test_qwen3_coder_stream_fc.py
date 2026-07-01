"""Unit tests for Qwen3-Coder XML streaming tool-call parsing.

These drive ``Qwen3CoderDetector.parse_streaming_increment`` directly (no server),
reassembling the per-tool argument string exactly the way ``api_openai.py`` does for
streamed responses, including the ``finish_reason == "stop"`` reconciliation. The key
invariant under test: the reassembled arguments are always valid JSON equal to what the
one-shot ``detect_and_parse`` would produce, for every chunk boundary.
"""

import json
import pytest

from lightllm.server.api_models import Function, Tool
from lightllm.server.function_call_parser import Qwen3CoderDetector

CHUNK_SIZES = [1, 2, 3, 5, 13, 10_000]


def _tool(name, properties):
    return Tool(
        type="function",
        function=Function(name=name, description="", parameters={"type": "object", "properties": properties}),
    )


def _stream_and_reassemble(text, tools, chunk):
    """Feed ``text`` to the detector in fixed-size chunks and rebuild the client-visible
    tool calls the way api_openai.py stream_results does (stop-rewrite on the last chunk)."""
    det = Qwen3CoderDetector()
    chunks = [text[i : i + chunk] for i in range(0, len(text), chunk)]
    per_tool = {}
    for ci, piece in enumerate(chunks):
        result = det.parse_streaming_increment(piece, tools)
        is_last = ci == len(chunks) - 1
        for call in result.calls:
            ti = call.tool_index
            per_tool.setdefault(ti, {"name": None, "args": ""})
            if call.name is not None:
                per_tool[ti]["name"] = call.name
            params = call.parameters
            if is_last and params:
                # Mirror api_openai.py:559-575 (REPLACE semantics).
                latest_delta_len = len(params)
                expected = json.dumps(det.prev_tool_call_arr[ti].get("arguments", {}), ensure_ascii=False)
                actual = det.streamed_args_for_tool[ti]
                if latest_delta_len > 0:
                    actual = actual[:-latest_delta_len]
                params = expected.replace(actual, "", 1)
            if params:
                per_tool[ti]["args"] += params
    return det, per_tool


def _assert_tool_calls(text, tools, expected, chunk_sizes=CHUNK_SIZES):
    """expected: {tool_index: (name, args_dict)}."""
    for chunk in chunk_sizes:
        det, per_tool = _stream_and_reassemble(text, tools, chunk)
        assert len(per_tool) == len(expected), f"chunk={chunk}: tool count {len(per_tool)} != {len(expected)}"
        for ti, (name, args) in expected.items():
            got = per_tool[ti]
            assert got["name"] == name, f"chunk={chunk}: tool {ti} name {got['name']!r} != {name!r}"
            parsed = json.loads(got["args"])  # must be valid JSON
            assert parsed == args, f"chunk={chunk}: tool {ti} args {parsed!r} != {args!r}"


@pytest.mark.parametrize("chunk", CHUNK_SIZES)
def test_single_string_param(chunk):
    text = (
        "<tool_call>\n<function=get_weather>\n<parameter=location>\n"
        "San Francisco\n</parameter>\n</function>\n</tool_call>"
    )
    _assert_tool_calls(
        text,
        [_tool("get_weather", {"location": {"type": "string"}})],
        {0: ("get_weather", {"location": "San Francisco"})},
        [chunk],
    )


def test_array_param_compact_spacing():
    # Regression: a non-string value whose raw text ("[1,2]") differs from json.dumps
    # ("[1, 2]") used to break the streamed-args prefix invariant -> duplicated/invalid JSON.
    text = "<tool_call>\n<function=calc>\n<parameter=nums>\n[1,2]\n</parameter>\n</function>\n</tool_call>"
    _assert_tool_calls(text, [_tool("calc", {"nums": {"type": "array"}})], {0: ("calc", {"nums": [1, 2]})})


def test_number_param_reformatted():
    # "1.0" is parsed to int 1 and json.dumps'd as "1"; the stream must agree.
    text = "<tool_call>\n<function=calc>\n<parameter=v>\n1.0\n</parameter>\n</function>\n</tool_call>"
    _assert_tool_calls(text, [_tool("calc", {"v": {"type": "number"}})], {0: ("calc", {"v": 1})})


def test_boolean_param():
    text = "<tool_call>\n<function=set>\n<parameter=flag>\ntrue\n</parameter>\n</function>\n</tool_call>"
    _assert_tool_calls(text, [_tool("set", {"flag": {"type": "boolean"}})], {0: ("set", {"flag": True})})


def test_object_param():
    text = '<tool_call>\n<function=f>\n<parameter=cfg>\n{"a":1,"b":[2,3]}\n</parameter>\n</function>\n</tool_call>'
    _assert_tool_calls(text, [_tool("f", {"cfg": {"type": "object"}})], {0: ("f", {"cfg": {"a": 1, "b": [2, 3]}})})


def test_two_params_mixed_types():
    text = (
        "<tool_call>\n<function=f>\n<parameter=city>\nNYC\n</parameter>\n"
        "<parameter=days>\n3\n</parameter>\n</function>\n</tool_call>"
    )
    _assert_tool_calls(
        text,
        [_tool("f", {"city": {"type": "string"}, "days": {"type": "integer"}})],
        {0: ("f", {"city": "NYC", "days": 3})},
    )


def test_multiline_string_value():
    text = (
        "<tool_call>\n<function=f>\n<parameter=code>\nline1\nline2\nline3\n" "</parameter>\n</function>\n</tool_call>"
    )
    _assert_tool_calls(text, [_tool("f", {"code": {"type": "string"}})], {0: ("f", {"code": "line1\nline2\nline3"})})


def test_string_with_json_special_chars():
    text = '<tool_call>\n<function=f>\n<parameter=s>\nsay "hi"\\path\n</parameter>\n</function>\n</tool_call>'
    _assert_tool_calls(text, [_tool("f", {"s": {"type": "string"}})], {0: ("f", {"s": 'say "hi"\\path'})})


def test_empty_string_value():
    text = "<tool_call>\n<function=f>\n<parameter=s>\n\n</parameter>\n</function>\n</tool_call>"
    _assert_tool_calls(text, [_tool("f", {"s": {"type": "string"}})], {0: ("f", {"s": ""})})


def test_no_param_function():
    text = "<tool_call>\n<function=ping>\n</function>\n</tool_call>"
    _assert_tool_calls(text, [_tool("ping", {})], {0: ("ping", {})})


def test_two_separate_tool_call_blocks():
    text = (
        "<tool_call>\n<function=a>\n<parameter=x>\nhi\n</parameter>\n</function>\n</tool_call>\n"
        "<tool_call>\n<function=b>\n<parameter=y>\nyo\n</parameter>\n</function>\n</tool_call>"
    )
    _assert_tool_calls(
        text,
        [_tool("a", {"x": {"type": "string"}}), _tool("b", {"y": {"type": "string"}})],
        {0: ("a", {"x": "hi"}), 1: ("b", {"y": "yo"})},
    )


def test_two_functions_in_one_block():
    # Regression: used to raise IndexError on the second function in a single block.
    text = (
        "<tool_call>\n<function=a>\n<parameter=x>\nhi\n</parameter>\n</function>\n"
        "<function=b>\n<parameter=y>\nyo\n</parameter>\n</function>\n</tool_call>"
    )
    _assert_tool_calls(
        text,
        [_tool("a", {"x": {"type": "string"}}), _tool("b", {"y": {"type": "string"}})],
        {0: ("a", {"x": "hi"}), 1: ("b", {"y": "yo"})},
    )


def test_undefined_then_valid_in_same_block():
    # Regression: an undefined first function used to discard the whole block, dropping
    # the valid call that followed it.
    text = (
        "<tool_call>\n<function=ghost>\n<parameter=x>\nhi\n</parameter>\n</function>\n"
        "<function=valid>\n<parameter=y>\nyo\n</parameter>\n</function>\n</tool_call>"
    )
    _assert_tool_calls(text, [_tool("valid", {"y": {"type": "string"}})], {0: ("valid", {"y": "yo"})})


def test_truncated_call_missing_function_close():
    # Regression: a typed value with no </function> before </tool_call> used to leave the
    # streamed args unterminated (missing closing brace).
    text = "<tool_call>\n<function=calc>\n<parameter=x>\n0.50\n</parameter>\n</tool_call>"
    _assert_tool_calls(text, [_tool("calc", {"x": {"type": "number"}})], {0: ("calc", {"x": 0.5})})


def test_streaming_matches_non_stream():
    # The reassembled streamed args must equal the one-shot detect_and_parse output.
    tools = [_tool("f", {"city": {"type": "string"}, "n": {"type": "integer"}, "tags": {"type": "array"}})]
    text = (
        "<tool_call>\n<function=f>\n<parameter=city>\nLondon\n</parameter>\n"
        '<parameter=n>\n7\n</parameter>\n<parameter=tags>\n["a","b"]\n</parameter>\n</function>\n</tool_call>'
    )
    oneshot = Qwen3CoderDetector().detect_and_parse(text, tools)
    expected_args = json.loads(oneshot.calls[0].parameters)
    _assert_tool_calls(text, tools, {0: ("f", expected_args)})


if __name__ == "__main__":
    import sys

    sys.exit(pytest.main([__file__, "-v"]))
