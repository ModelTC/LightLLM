import json

from lightllm.server.api_models import Function, Tool
from lightllm.server.function_call_parser import KimiK3Detector


def _tools():
    return [
        Tool(
            type="function",
            function=Function(
                name="get_weather",
                description="Get weather",
                parameters={"type": "object"},
            ),
        )
    ]


def test_kimi_k3_xtml_argument_tool_call():
    text = (
        "<|close|>response<|sep|>"
        "<|open|>tools<|sep|>"
        '<|open|>call tool="get_weather" index="1"<|sep|>'
        '<|open|>argument key="city" type="string"<|sep|>上海'
        "<|close|>argument<|sep|>"
        '<|open|>argument key="days" type="number"<|sep|>3'
        "<|close|>argument<|sep|>"
        "<|close|>call<|sep|>"
        "<|close|>tools<|sep|>"
        "<|close|>message<|sep|>"
    )

    result = KimiK3Detector().detect_and_parse(text, _tools())

    assert result.normal_text == ""
    assert len(result.calls) == 1
    assert result.calls[0].tool_index == 0
    assert result.calls[0].name == "get_weather"
    assert json.loads(result.calls[0].parameters) == {"city": "上海", "days": 3}


def test_kimi_k3_xtml_json_tool_call_streaming():
    detector = KimiK3Detector()
    first = detector.parse_streaming_increment(
        "<|close|>response<|sep|><|open|>tools<|sep|>"
        '<|open|>call tool="get_weather" index="1"<|sep|>'
        '<|open|>json type="object"<|sep|>{"city":"Paris"}',
        _tools(),
    )
    second = detector.parse_streaming_increment(
        "<|close|>json<|sep|><|close|>call<|sep|>" "<|close|>tools<|sep|><|close|>message<|sep|>",
        _tools(),
    )

    assert first.normal_text == ""
    assert first.calls == []
    assert len(second.calls) == 1
    assert json.loads(second.calls[0].parameters) == {"city": "Paris"}


def test_kimi_k3_xtml_plain_response_is_cleaned_in_streaming_mode():
    detector = KimiK3Detector()
    result = detector.parse_streaming_increment(
        "hello<|close|>response<|sep|><|close|>message<|sep|>",
        _tools(),
    )

    assert result.normal_text == "hello"
    assert result.calls == []
