import asyncio
import json
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from lightllm.server import api_http, api_openai, build_prompt, reasoning_parser
from lightllm.server.api_models import ChatCompletionRequest, Tool
from lightllm.server.core.objs import FinishStatus
from lightllm.server.core.objs.start_args_type import StartArgs
from lightllm.server.function_call_parser import FunctionCallParser
from lightllm.server.httpserver.stop_seq_match import ReasoningStateMachine
from lightllm.server.reasoning_parser import ReasoningParser


TOOLS = [
    Tool.model_validate(
        {
            "type": "function",
            "function": {
                "name": "Write",
                "parameters": {"type": "object", "properties": {"content": {"type": "string"}}},
            },
        }
    )
]
VALUE = "line 1\n<think>literal</think>\nline 2"
TOOL_CALL = f"<tool_call>Write<arg_key>content</arg_key><arg_value>{VALUE}</arg_value></tool_call>"


@pytest.fixture(autouse=True)
def mock_token_ids(monkeypatch):
    monkeypatch.setattr(
        reasoning_parser, "get_token_id", {"<think>": 98, "</think>": 99, "<tool_call>": 77}.__getitem__
    )


@pytest.mark.parametrize(
    "prefix,force_reasoning",
    [
        ("<think>Need a file.", False),
        ("Need a file.", True),
        ("<think>Need a file.</think>", False),
        ("Need a file.</think>", True),
    ],
)
@pytest.mark.parametrize("chunk_size", [1, 7, 4096])
def test_glm_reasoning_and_tools_agree_across_stream_boundaries(prefix, force_reasoning, chunk_size):
    text = prefix + TOOL_CALL + TOOL_CALL
    parser = ReasoningParser("glm45", force_reasoning=force_reasoning)
    reasoning, content = parser.parse_non_stream(text)
    assert reasoning == "Need a file."
    assert content == TOOL_CALL + TOOL_CALL

    parser = ReasoningParser("glm45", force_reasoning=force_reasoning)
    tool_parser = FunctionCallParser(TOOLS, "glm47")
    reasoning_parts, content_parts, calls = [], [], {}
    for start in range(0, len(text), chunk_size):
        reasoning, content = parser.parse_stream_chunk(text[start : start + chunk_size])
        reasoning_parts.append(reasoning)
        content, deltas = tool_parser.parse_stream_chunk(content)
        content_parts.append(content)
        for delta in deltas:
            if delta.name is not None:
                assert delta.tool_index not in calls
                calls[delta.tool_index] = {"name": delta.name, "arguments": ""}
            calls[delta.tool_index]["arguments"] += delta.parameters

    assert parser.flush() == ("", "")
    assert "".join(reasoning_parts) == "Need a file."
    assert "".join(content_parts) == ""
    assert list(calls) == [0, 1]
    assert all(call["name"] == "Write" for call in calls.values())
    assert all(json.loads(call["arguments"]) == {"content": VALUE} for call in calls.values())


@pytest.mark.parametrize("suffix", ["<tool_ca", "</thi", "<tool_calling>"])
def test_truncated_or_similar_markers_remain_reasoning(suffix):
    parser = ReasoningParser("glm45", force_reasoning=True)
    reasoning, content = parser.parse_stream_chunk("Keep thinking." + suffix)
    tail_reasoning, tail_content = parser.flush()
    assert reasoning + tail_reasoning == "Keep thinking." + suffix
    assert content + tail_content == ""


def test_non_thinking_glm_preserves_tool_payload():
    parser = ReasoningParser("glm45", force_reasoning=False)
    assert parser.parse_non_stream(TOOL_CALL) == ("", TOOL_CALL)
    assert parser.parse_stream_chunk(TOOL_CALL) == ("", TOOL_CALL)


def test_other_reasoning_parsers_keep_their_existing_boundary():
    parser = ReasoningParser("qwen3", force_reasoning=True)
    reasoning, content = parser.parse_non_stream("Still thinking." + TOOL_CALL.replace(VALUE, "literal"))
    assert reasoning.startswith("Still thinking.<tool_call>")
    assert content == ""


def test_glm_tool_parsing_keeps_existing_reasoning_token_accounting():
    parser = ReasoningParser("glm45", force_reasoning=True)
    for token_id in [1, 2, 77, 3, 99, 4]:
        parser.update_reasoning_token_count(token_id)
    assert parser.reasoning_tokens == 4


def test_glm_tools_inside_reasoning_do_not_enable_string_stops():
    tokenizer = SimpleNamespace(encode=lambda text, **kwargs: {"<think>": [98], "</think>": [99]}[text])
    state = ReasoningStateMachine("glm45", tokenizer, 1, [])
    assert not any(state.handle(token_id) for token_id in [1, 77, 2, 3])
    assert state.in_reasoning
    assert state.handle(99) is False
    assert not state.in_reasoning
    assert state.handle(4) is True


@pytest.mark.parametrize("stream", [False, True])
@pytest.mark.parametrize("reasoning_end", ["missing", "before_tool", "after_tool"])
def test_chat_completion_exposes_glm_tool_calls(monkeypatch, stream, reasoning_end):
    args = StartArgs(reasoning_parser="glm45", tool_call_parser="glm47")
    monkeypatch.setattr(api_openai, "get_env_start_args", lambda: args)
    monkeypatch.setattr(build_prompt, "tokenizer_supports_force_thinking", lambda: True)
    monkeypatch.setattr(api_openai, "build_prompt", AsyncMock(return_value="<think>"))
    segments = [("Need a file.", 1)]
    if reasoning_end == "before_tool":
        segments.append(("</think>", 99))
    segments.extend([("<tool_call>", 77), (TOOL_CALL[len("<tool_call>") :], 2)])
    if reasoning_end == "after_tool":
        segments.append(("</think>", 99))

    async def generate(*args, **kwargs):
        for index, (text, token_id) in enumerate(segments):
            finish = FinishStatus(FinishStatus.FINISHED_STOP) if index == len(segments) - 1 else FinishStatus()
            yield 0, text, {"id": token_id, "prompt_tokens": 1}, finish

    manager = SimpleNamespace(tokenizer=SimpleNamespace(), generate=generate, output_grammar_compiler=None)
    monkeypatch.setattr(api_http, "g_objs", SimpleNamespace(args=args, httpserver_manager=manager))
    request = ChatCompletionRequest(model="glm", messages=[], tools=TOOLS, stream=stream)

    async def run():
        response = await api_openai.chat_completions_impl(request, None)
        if not stream:
            choice = response.choices[0]
            return (
                choice.message.reasoning,
                choice.message.content,
                choice.message.tool_calls[0].function.arguments,
                choice.finish_reason,
            )

        reasoning, content, arguments, finish_reason = "", "", "", None
        async for chunk in response.body_iterator:
            chunk = chunk.decode() if isinstance(chunk, bytes) else chunk
            if chunk.strip() == "data: [DONE]":
                continue
            payload = json.loads(chunk.removeprefix("data: "))
            for choice in payload["choices"]:
                delta = choice["delta"]
                reasoning += delta.get("reasoning") or ""
                content += delta.get("content") or ""
                for call in delta.get("tool_calls") or []:
                    assert call["index"] == 0
                    assert call["function"].get("name") in (None, "Write")
                    arguments += call["function"].get("arguments") or ""
                finish_reason = choice.get("finish_reason") or finish_reason
        return reasoning, content, arguments, finish_reason

    reasoning, content, arguments, finish_reason = asyncio.run(run())
    assert reasoning == "Need a file."
    assert content == ""
    assert json.loads(arguments) == {"content": VALUE}
    assert finish_reason == "tool_calls"
