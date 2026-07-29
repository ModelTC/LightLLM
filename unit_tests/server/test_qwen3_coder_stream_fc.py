"""Coverage for buffered Qwen3-Coder streaming and keepalive behavior."""

import json
from types import SimpleNamespace

from lightllm.server.api_models import Function, Tool
from lightllm.server.api_cli import make_argument_parser
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


def test_qwen3_coder_requests_empty_chunks_while_buffering_large_content():
    parser = FunctionCallParser([_tool()], "qwen3_coder")
    interval = parser.detector.KEEPALIVE_CHUNK_INTERVAL
    content_parts = []

    normal_text, calls = parser.parse_stream_chunk("<tool_call>\n<function=write_file>\n<parameter=content>\n")
    assert normal_text == ""
    assert calls == []
    assert parser.should_emit_empty_chunk is False

    for _ in range(interval - 2):
        normal_text, calls = parser.parse_stream_chunk("x")
        content_parts.append("x")
        assert normal_text == ""
        assert calls == []
        assert parser.should_emit_empty_chunk is False

    normal_text, calls = parser.parse_stream_chunk("y")
    content_parts.append("y")
    assert normal_text == ""
    assert calls == []
    assert parser.should_emit_empty_chunk is True

    normal_text, calls = parser.parse_stream_chunk("z" * 100_000)
    content_parts.append("z" * 100_000)
    assert normal_text == ""
    assert calls == []
    assert parser.should_emit_empty_chunk is False

    normal_text, calls = parser.parse_stream_chunk("\n</parameter>\n</function>\n</tool_call>")
    assert normal_text == ""
    assert len(calls) == 1
    assert calls[0].name == "write_file"
    assert json.loads(calls[0].parameters) == {"content": "".join(content_parts)}
    assert parser.should_emit_empty_chunk is False


def test_api_empty_chunk_signal_is_opt_in_for_buffered_qwen3_parser():
    from lightllm.server.api_openai import _process_tools_stream

    request = SimpleNamespace(tools=[_tool()])
    streaming_formats = {
        "llama3": [
            '<|python_tag|>{"name":"write_file","arguments":{"content":"abc',
            'def"}}',
        ],
        "qwen25": [
            '<tool_call>\n{"name":"write_file","arguments":{"content":"abc',
            'def"}}\n</tool_call>',
        ],
    }
    for parser_name, chunks in streaming_formats.items():
        parser_dict = {0: FunctionCallParser(request.tools, parser_name)}
        streamed_calls = []
        for chunk in chunks:
            _, calls, emit_empty_chunk = _process_tools_stream(0, chunk, parser_dict, request)
            streamed_calls.extend(calls)
            assert emit_empty_chunk is False
        assert streamed_calls[0].name == "write_file"
        assert json.loads("".join(call.parameters for call in streamed_calls)) == {"content": "abcdef"}

    parser_dict = {0: FunctionCallParser(request.tools, "qwen3_coder")}
    interval = parser_dict[0].detector.KEEPALIVE_CHUNK_INTERVAL
    for chunk_index in range(interval):
        delta = "<tool_call>\n<function=write_file>\n<parameter=content>\n" if chunk_index == 0 else "x"
        _, calls, emit_empty_chunk = _process_tools_stream(0, delta, parser_dict, request)
        assert calls == []
        assert emit_empty_chunk is (chunk_index == interval - 1)


def test_keepalive_chunk_uses_empty_delta_without_starting_a_text_block():
    from lightllm.server.api_models import (
        ChatCompletionStreamResponse,
        ChatCompletionStreamResponseChoice,
        DeltaMessage,
    )
    from lightllm.server.api_openai import _serialize_sse_chunk

    chunk = ChatCompletionStreamResponse(
        id="chatcmpl-test",
        created=0,
        model="test",
        choices=[
            ChatCompletionStreamResponseChoice(
                index=0,
                delta=DeltaMessage(),
                finish_reason=None,
            )
        ],
    )
    payload = json.loads(_serialize_sse_chunk(chunk, ("logprobs", "token_ids", "finish_reason")))
    assert payload["choices"][0]["delta"] == {}
    assert payload["choices"][0]["finish_reason"] is None


def test_large_complete_arguments_are_split_into_bounded_stream_deltas():
    from lightllm.server.api_openai import TOOL_ARGUMENT_STREAM_CHUNK_SIZE, _split_tool_argument_delta

    arguments = json.dumps(
        {"content": 'print("hello")\n' * 10_000},
        ensure_ascii=False,
    )
    deltas = _split_tool_argument_delta(arguments)

    assert "".join(deltas) == arguments
    assert deltas[0] == "{"
    assert deltas[-1] == "}"
    assert max(len(delta) for delta in deltas) <= TOOL_ARGUMENT_STREAM_CHUNK_SIZE
    assert len(deltas) > 3
