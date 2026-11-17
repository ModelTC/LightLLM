import asyncio
import json
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
from pydantic import ValidationError

from lightllm.models.deepseek_v41.tokenizer import DeepSeekV41Tokenizer
from lightllm.server.api_models import ChatCompletionRequest, Tool
from lightllm.server.function_call_parser import FunctionCallParser


@pytest.mark.parametrize("effort", [1, 50, 75, 100, "low", "high", "max"])
def test_reasoning_effort_reaches_checkpoint_encoder(tmp_path, effort):
    encoder = tmp_path / "encoding"
    encoder.mkdir()
    # Verify the wrapper loads the checkpoint's encoder and forwards numeric efforts unchanged.
    (encoder / "encoding.py").write_text(
        "def encode_messages(messages, **kwargs):\n    return kwargs['reasoning_effort']\n"
    )
    request = ChatCompletionRequest(
        model="v41", messages=[{"role": "user", "content": "hello"}], reasoning_effort=effort
    )
    tokenizer = DeepSeekV41Tokenizer(
        SimpleNamespace(), str(tmp_path), {"model_type": "deepseek_v41", "text_config": {}}
    )
    assert (
        tokenizer.apply_chat_template(request.model_dump()["messages"], reasoning_effort=request.reasoning_effort)
        == effort
    )


@pytest.mark.parametrize("effort,expected", [("none", None), ("minimal", None), ("medium", "low"), ("xhigh", "high")])
def test_api_effort_aliases_reach_checkpoint(tmp_path, effort, expected):
    encoder = tmp_path / "encoding"
    encoder.mkdir()
    (encoder / "encoding.py").write_text(
        "def encode_messages(messages, **kwargs):\n    return kwargs['reasoning_effort']\n"
    )
    request = ChatCompletionRequest(
        model="v41", messages=[{"role": "user", "content": "hello"}], reasoning_effort=effort
    )
    tokenizer = DeepSeekV41Tokenizer(
        SimpleNamespace(), str(tmp_path), {"model_type": "deepseek_v41", "text_config": {}}
    )
    assert (
        tokenizer.apply_chat_template(
            request.model_dump()["messages"], reasoning_effort=request.reasoning_effort, **request.chat_template_kwargs
        )
        == expected
    )


@pytest.mark.parametrize("effort", [0, 101, True, 1.5, "75"])
def test_numeric_reasoning_effort_bounds(effort):
    with pytest.raises(ValidationError):
        ChatCompletionRequest(model="v41", messages=[], reasoning_effort=effort)


@pytest.mark.parametrize("chunk_size", [1, 7, 4096])
def test_space_prefixed_dsml_round_trip(chunk_size):
    tools = [Tool.model_validate({"type": "function", "function": {"name": "lookup", "parameters": {}}})]
    text = (
        '<｜DSML｜ calls><｜DSML｜ invoke name="lookup">'
        '<｜DSML｜ parameter name="text" string="true"> 中文\n</｜DSML｜ parameter>'
        '<｜DSML｜ parameter name="count" string="false">2</｜DSML｜ parameter>'
        "</｜DSML｜ invoke></｜DSML｜ calls>"
    )
    _, calls = FunctionCallParser(tools, "deepseekv41").parse_non_stream(text)
    assert len(calls) == 1
    assert calls[0].name == "lookup"
    assert json.loads(calls[0].parameters) == {"text": " 中文\n", "count": 2}
    parser = FunctionCallParser(tools, "deepseekv41")
    names, arguments = [], ""
    for start in range(0, len(text), chunk_size):
        content, deltas = parser.parse_stream_chunk(text[start : start + chunk_size])
        assert not content
        for delta in deltas:
            assert delta.name is not None or delta.parameters
            if delta.name is not None:
                names.append(delta.name)
            arguments += delta.parameters
    assert names == ["lookup"]
    assert json.loads(arguments) == {"text": " 中文\n", "count": 2}


@pytest.mark.parametrize("chunk_size", [1, 7, 4096])
def test_parallel_calls_and_surrounding_text(chunk_size):
    tools = [Tool.model_validate({"type": "function", "function": {"name": "lookup", "parameters": {}}})]
    invoke = (
        '<｜DSML｜ invoke name="lookup"><｜DSML｜ parameter name="id" string="false">{}</｜DSML｜ parameter></｜DSML｜ invoke>'
    )
    text = "before <｜DSML｜ calls>" + invoke.format(1) + invoke.format(2) + "</｜DSML｜ calls> after"
    parser = FunctionCallParser(tools, "deepseekv41")
    content, calls = "", {}
    for start in range(0, len(text), chunk_size):
        normal, deltas = parser.parse_stream_chunk(text[start : start + chunk_size])
        content += normal
        for delta in deltas:
            if delta.name is not None:
                assert delta.tool_index not in calls
                calls[delta.tool_index] = {"name": delta.name, "arguments": ""}
            calls[delta.tool_index]["arguments"] += delta.parameters
    assert content == "before  after"
    assert [call["name"] for call in calls.values()] == ["lookup", "lookup"]
    assert [json.loads(call["arguments"]) for call in calls.values()] == [{"id": 1}, {"id": 2}]


def test_non_stream_chat_returns_v41_tool_calls(monkeypatch):
    from lightllm.server import api_http, api_openai
    from lightllm.server.core.objs import FinishStatus
    from lightllm.server.core.objs.start_args_type import StartArgs

    text = (
        '<｜DSML｜ calls><｜DSML｜ invoke name="lookup">'
        '<｜DSML｜ parameter name="city" string="true">北京</｜DSML｜ parameter>'
        "</｜DSML｜ invoke></｜DSML｜ calls>"
    )

    async def generate(*args, **kwargs):
        yield 0, text, {"id": 1, "prompt_tokens": 1}, FinishStatus(FinishStatus.FINISHED_STOP)

    args = StartArgs(reasoning_parser=None, tool_call_parser="deepseekv41")
    manager = SimpleNamespace(tokenizer=None, output_grammar_compiler=None, generate=generate)
    monkeypatch.setattr(api_openai, "get_env_start_args", lambda: args)
    monkeypatch.setattr(api_openai, "build_prompt", AsyncMock(return_value="prompt"))
    monkeypatch.setattr(api_http, "g_objs", SimpleNamespace(args=args, httpserver_manager=manager))
    request = ChatCompletionRequest(
        model="v41",
        messages=[],
        tools=[Tool.model_validate({"type": "function", "function": {"name": "lookup", "parameters": {}}})],
        tool_choice="auto",
    )
    response = asyncio.run(api_openai.chat_completions_impl(request, None))
    choice = response.choices[0]
    assert choice.finish_reason == "tool_calls"
    assert choice.message.content == ""
    assert len(choice.message.tool_calls) == 1
    call = choice.message.tool_calls[0]
    assert call.function.name == "lookup"
    assert json.loads(call.function.arguments) == {"city": "北京"}


def test_anthropic_stream_sends_write_content_before_dsml_closes(monkeypatch):
    from lightllm.server import api_http, api_openai
    from lightllm.server.api_anthropic import _openai_sse_to_anthropic_events
    from lightllm.server.core.objs import FinishStatus
    from lightllm.server.core.objs.start_args_type import StartArgs

    lines = [f'# line {i:03d}\nprint("北京", r"C:\\tmp")\n<code>\n' for i in range(1024)]
    deltas, events = [], []

    async def generate(*args, **kwargs):
        start = (
            '<｜DSML｜ calls><｜DSML｜ invoke name="Write">'
            '<｜DSML｜ parameter name="file_path" string="true">street.py</｜DSML｜ parameter>'
            '<｜DSML｜ parameter name="content" string="true">'
        )
        yield 0, start, {"prompt_tokens": 1}, FinishStatus(FinishStatus.NO_FINISH)
        for i, line in enumerate(lines):
            yield 0, line, {"prompt_tokens": 1}, FinishStatus(FinishStatus.NO_FINISH)
            assert deltas and f"# line {i:03d}" in deltas[-1], f"Write content stalled at line {i}"
        yield 0, "</｜DSML｜ para", {"prompt_tokens": 1}, FinishStatus(FinishStatus.NO_FINISH)
        yield 0, "meter></｜DSML｜ invoke></｜DSML｜ calls>", {"prompt_tokens": 1}, FinishStatus(FinishStatus.FINISHED_STOP)

    args = StartArgs(reasoning_parser=None, tool_call_parser="deepseekv41")
    manager = SimpleNamespace(tokenizer=None, output_grammar_compiler=None, generate=generate)
    monkeypatch.setattr(api_openai, "get_env_start_args", lambda: args)
    monkeypatch.setattr(api_openai, "build_prompt", AsyncMock(return_value="prompt"))
    monkeypatch.setattr(api_http, "g_objs", SimpleNamespace(args=args, httpserver_manager=manager))
    request = ChatCompletionRequest(
        model="v41",
        messages=[],
        tools=[Tool.model_validate({"type": "function", "function": {"name": "Write", "parameters": {}}})],
        tool_choice="auto",
        stream=True,
    )

    async def run():
        response = await api_openai.chat_completions_impl(request, None)
        async for raw in _openai_sse_to_anthropic_events(response.body_iterator, "v41", "msg_test"):
            event = json.loads(raw.decode().splitlines()[1].removeprefix("data: "))
            events.append(event)
            if event.get("delta", {}).get("type") == "input_json_delta":
                deltas.append(event["delta"]["partial_json"])

    asyncio.run(run())
    assert json.loads("".join(deltas)) == {"file_path": "street.py", "content": "".join(lines)}
    assert [event["content_block"]["name"] for event in events if event["type"] == "content_block_start"] == ["Write"]
    assert sum(event["type"] == "content_block_stop" for event in events) == 1
    assert [event["delta"]["stop_reason"] for event in events if event["type"] == "message_delta"] == ["tool_use"]
