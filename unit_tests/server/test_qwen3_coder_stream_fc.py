"""Coverage for buffered Qwen3-Coder streaming keepalive deltas."""

import json
from types import SimpleNamespace

from lightllm.server.api_cli import make_argument_parser
from lightllm.server.api_models import Function, Tool
from lightllm.server.function_call_parser import FunctionCallParser, Qwen3CoderDetector


def _tool() -> Tool:
    return Tool(
        type="function",
        function=Function(
            name="write_file",
            description="",
            parameters={
                "type": "object",
                "properties": {"content": {"type": "string"}},
            },
        ),
    )


def test_qwen3_coder_selects_buffered_parser():
    assert FunctionCallParser.ToolCallParserEnum["qwen3_coder"] is Qwen3CoderDetector
    assert "qwen3_coder_legacy" not in FunctionCallParser.ToolCallParserEnum

    args = make_argument_parser().parse_args(["--tool_call_parser", "qwen3_coder"])
    assert args.tool_call_parser == "qwen3_coder"


def test_qwen3_coder_emits_one_tool_delta_for_every_decode_after_the_name():
    parser = FunctionCallParser([_tool()], "qwen3_coder")

    normal_text, calls = parser.parse_stream_chunk("<tool_call>\n<func")
    assert normal_text == ""
    assert calls == []

    normal_text, calls = parser.parse_stream_chunk("tion=write_file>\n<parameter=content>\n")
    assert normal_text == ""
    assert len(calls) == 1
    assert calls[0].tool_index == 0
    assert calls[0].name == "write_file"
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

    normal_text, calls = parser.parse_stream_chunk("\n</parameter>\n</function>\n</tool_call>")
    assert normal_text == ""
    assert len(calls) == 1
    assert calls[0].tool_index == 0
    assert calls[0].name is None
    assert json.loads(calls[0].parameters) == {"content": "ab" + "c" * 100_000}
    streamed_calls.extend(calls)

    assert [call.name for call in streamed_calls].count("write_file") == 1
    assert json.loads("".join(call.parameters for call in streamed_calls)) == {"content": "ab" + "c" * 100_000}


def test_qwen3_coder_accepts_an_undefined_tool_when_name_check_is_disabled(monkeypatch):
    import lightllm.server.function_call_parser as parser_module

    monkeypatch.setattr(parser_module, "ENABLE_TOOL_NAME_CHECK", False)
    parser = FunctionCallParser([_tool()], "qwen3_coder")

    _, calls = parser.parse_stream_chunk("<tool_call>\n<function=unknown>\n")
    assert len(calls) == 1
    assert calls[0].name == "unknown"
    assert calls[0].parameters == ""

    _, calls = parser.parse_stream_chunk("<parameter=value>x</parameter>\n</function>\n</tool_call>")
    assert len(calls) == 1
    assert calls[0].name is None
    assert json.loads(calls[0].parameters) == {"value": "x"}


def test_qwen3_coder_rejects_an_undefined_tool_when_name_check_is_enabled(monkeypatch):
    import lightllm.server.function_call_parser as parser_module

    monkeypatch.setattr(parser_module, "ENABLE_TOOL_NAME_CHECK", True)
    parser = FunctionCallParser([_tool()], "qwen3_coder")

    _, calls = parser.parse_stream_chunk("<tool_call>\n<function=unknown>\n")
    assert calls == []

    _, calls = parser.parse_stream_chunk("<parameter=value>x</parameter>\n</function>\n</tool_call>")
    assert calls == []


def test_api_tool_parser_keeps_its_two_item_return_contract():
    from lightllm.server.api_openai import _process_tools_stream

    request = SimpleNamespace(tools=[_tool()])
    parser_dict = {0: FunctionCallParser(request.tools, "qwen3_coder")}

    result = _process_tools_stream(
        0,
        "<tool_call>\n<function=write_file>\n<parameter=content>\n",
        parser_dict,
        request,
    )

    assert len(result) == 2
    normal_text, calls = result
    assert normal_text == ""
    assert len(calls) == 1
    assert calls[0].name == "write_file"


def test_empty_arguments_are_preserved_in_the_sse_payload():
    from lightllm.server.api_models import (
        ChatCompletionStreamResponse,
        ChatCompletionStreamResponseChoice,
        DeltaMessage,
        FunctionResponse,
        ToolCall,
    )
    from lightllm.server.api_openai import _serialize_sse_chunk

    chunk = ChatCompletionStreamResponse(
        id="chatcmpl-test",
        created=0,
        model="test",
        choices=[
            ChatCompletionStreamResponseChoice(
                index=0,
                delta=DeltaMessage(
                    tool_calls=[
                        ToolCall(
                            index=0,
                            function=FunctionResponse(arguments=""),
                        )
                    ]
                ),
                finish_reason=None,
            )
        ],
    )

    payload = json.loads(_serialize_sse_chunk(chunk, ("logprobs", "token_ids", "finish_reason")))
    function_delta = payload["choices"][0]["delta"]["tool_calls"][0]["function"]
    assert function_delta["arguments"] == ""
