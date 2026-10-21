import json
import pytest

from lightllm.server.api_models import Function, Tool
from lightllm.server.function_call_parser import FunctionCallParser
from lightllm.utils.error_utils import GenerationError


@pytest.mark.parametrize("value,expected", [("hello", "hello"), ("12", 12), ("[1,2]", [1, 2])])
@pytest.mark.parametrize("stream", [False, True])
def test_dsv4_missing_string_attribute_preserves_parameter(value, expected, stream):
    text = _deepseek_v4_tool_block(value).replace(' string="true"', "")
    parser = _deepseek_v4_parser()
    if stream:
        calls = [call for chunk in text for call in parser.parse_stream_chunk(chunk)[1]]
        arguments = "".join(call.parameters for call in calls)
    else:
        arguments = parser.parse_non_stream(text)[1][0].parameters
    assert json.loads(arguments) == {"value": expected}


@pytest.mark.parametrize("stream", [False, True])
def test_dsv4_duplicate_parameters_fail_instead_of_rewriting_streamed_arguments(stream):
    text = _deepseek_v4_tool_block("old")
    repeated = '<｜DSML｜parameter name="value" string="true">new</｜DSML｜parameter>\n'
    text = text.replace("</｜DSML｜invoke>", repeated + "</｜DSML｜invoke>")
    parser = _deepseek_v4_parser()
    with pytest.raises(GenerationError, match="Duplicate"):
        if stream:
            for chunk in text:
                parser.parse_stream_chunk(chunk)
        else:
            parser.parse_non_stream(text)


def _deepseek_v4_parser() -> FunctionCallParser:
    tools = [
        Tool(
            function=Function(
                name="lookup",
                parameters={
                    "type": "object",
                    "properties": {"value": {"type": "string"}},
                },
            )
        )
    ]
    return FunctionCallParser(tools, "deepseekv4")


def _deepseek_v4_tool_block(value: str) -> str:
    return (
        "<｜DSML｜tool_calls>\n"
        '<｜DSML｜invoke name="lookup">\n'
        f'<｜DSML｜parameter name="value" string="true">{value}</｜DSML｜parameter>\n'
        "</｜DSML｜invoke>\n"
        "</｜DSML｜tool_calls>"
    )


def test_deepseek_v4_stream_preserves_less_than_split_across_chunks():
    parser = _deepseek_v4_parser()

    output = "".join(parser.parse_stream_chunk(chunk)[0] for chunk in ["9.11 ", "<", " 9.9"])

    assert output == "9.11 < 9.9"


def test_deepseek_v4_stream_still_parses_split_tool_call_tag():
    parser = _deepseek_v4_parser()
    chunks = [
        "before\n",
        "<｜DSML｜tool",
        '_calls>\n<｜DSML｜invoke name="lookup">\n'
        '<｜DSML｜parameter name="value" string="true">ok</｜DSML｜parameter>\n'
        "</｜DSML｜invoke>\n</｜DSML｜tool_calls>",
    ]

    normal_text = ""
    calls = []
    for chunk in chunks:
        parsed_text, parsed_calls = parser.parse_stream_chunk(chunk)
        normal_text += parsed_text
        calls.extend(parsed_calls)

    assert normal_text == "before\n"
    assert calls[0].name == "lookup"
    assert json.loads("".join(call.parameters for call in calls)) == {"value": "ok"}


def test_deepseek_v4_stream_emits_keepalive_deltas_while_parameter_is_incomplete():
    parser = _deepseek_v4_parser()

    _, calls = parser.parse_stream_chunk(
        "<｜DSML｜tool_calls>\n" '<｜DSML｜invoke name="lookup">\n' '<｜DSML｜parameter name="value" string="true">'
    )
    assert len(calls) == 1
    assert calls[0].name == "lookup"
    assert calls[0].parameters == ""

    streamed_calls = list(calls)
    for chunk in ("a", "b", "c" * 100_000):
        normal_text, calls = parser.parse_stream_chunk(chunk)
        assert normal_text == ""
        assert len(calls) == 1
        assert calls[0].tool_index == 0
        assert calls[0].name is None
        assert calls[0].parameters == ""
        streamed_calls.extend(calls)

    _, calls = parser.parse_stream_chunk("</｜DSML｜parameter>\n" "</｜DSML｜invoke>\n" "</｜DSML｜tool_calls>")
    streamed_calls.extend(calls)

    assert [call.name for call in streamed_calls].count("lookup") == 1
    assert json.loads("".join(call.parameters for call in streamed_calls)) == {"value": "ab" + "c" * 100_000}


def test_deepseek_v4_non_stream_merges_consecutive_tool_call_blocks():
    parser = _deepseek_v4_parser()
    completion = "\n\n".join(_deepseek_v4_tool_block(value) for value in ("a", "b", "c"))

    normal_text, calls = parser.parse_non_stream(completion)

    assert normal_text == ""
    assert [call.tool_index for call in calls] == [0, 1, 2]
    assert [json.loads(call.parameters) for call in calls] == [
        {"value": "a"},
        {"value": "b"},
        {"value": "c"},
    ]


def test_deepseek_v4_stream_merges_consecutive_tool_call_blocks_without_leaking_tags():
    parser = _deepseek_v4_parser()
    chunks = [_deepseek_v4_tool_block(value) for value in ("a", "b", "c")]

    normal_text = ""
    calls = []
    for chunk in chunks:
        parsed_text, parsed_calls = parser.parse_stream_chunk(chunk)
        normal_text += parsed_text
        calls.extend(parsed_calls)

    assert normal_text == ""
    assert [(call.tool_index, call.name) for call in calls if call.name] == [
        (0, "lookup"),
        (1, "lookup"),
        (2, "lookup"),
    ]
    assert [
        json.loads("".join(call.parameters for call in calls if call.tool_index == index)) for index in range(3)
    ] == [{"value": "a"}, {"value": "b"}, {"value": "c"}]


def test_deepseek_v4_stream_merges_character_split_tool_call_blocks_without_leaking_tags():
    parser = _deepseek_v4_parser()
    completion = "\n\n".join(_deepseek_v4_tool_block(value) for value in ("a", "b"))

    normal_text = ""
    calls = []
    for chunk in completion:
        parsed_text, parsed_calls = parser.parse_stream_chunk(chunk)
        normal_text += parsed_text
        calls.extend(parsed_calls)

    assert normal_text == ""
    assert [(call.tool_index, call.name) for call in calls if call.name] == [
        (0, "lookup"),
        (1, "lookup"),
    ]
    assert [
        json.loads("".join(call.parameters for call in calls if call.tool_index == index)) for index in range(2)
    ] == [{"value": "a"}, {"value": "b"}]


class _CountingRegex:
    def __init__(self, pattern):
        self.pattern = pattern
        self.match_count = 0
        self.findall_count = 0

    def match(self, *args, **kwargs):
        self.match_count += 1
        return self.pattern.match(*args, **kwargs)

    def findall(self, *args, **kwargs):
        self.findall_count += 1
        return self.pattern.findall(*args, **kwargs)


def test_deepseek_v4_character_stream_scans_only_when_end_tags_arrive():
    parser = _deepseek_v4_parser()
    detector = parser.detector
    detector.invoke_regex = _CountingRegex(detector.invoke_regex)
    detector.param_regex = _CountingRegex(detector.param_regex)
    detector.partial_invoke_regex = _CountingRegex(detector.partial_invoke_regex)
    value = "x" * 4096

    calls = []
    for chunk in _deepseek_v4_tool_block(value):
        _, parsed_calls = parser.parse_stream_chunk(chunk)
        calls.extend(parsed_calls)

    assert json.loads("".join(call.parameters for call in calls)) == {"value": value}
    assert detector.invoke_regex.match_count <= 2
    assert detector.param_regex.findall_count <= 3
    assert detector.partial_invoke_regex.match_count < 64
