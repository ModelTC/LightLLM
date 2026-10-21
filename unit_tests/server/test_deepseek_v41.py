import json
from types import SimpleNamespace

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
    tokenizer = DeepSeekV41Tokenizer(SimpleNamespace(), str(tmp_path), {"model_type": "deepseek_v41"})
    assert (
        tokenizer.apply_chat_template(request.model_dump()["messages"], reasoning_effort=request.reasoning_effort)
        == effort
    )


@pytest.mark.parametrize("effort", ["none", "minimal", "medium", "xhigh"])
def test_other_api_efforts_use_checkpoint_default(tmp_path, effort):
    encoder = tmp_path / "encoding"
    encoder.mkdir()
    (encoder / "encoding.py").write_text(
        "def encode_messages(messages, **kwargs):\n    return kwargs['reasoning_effort']\n"
    )
    request = ChatCompletionRequest(
        model="v41", messages=[{"role": "user", "content": "hello"}], reasoning_effort=effort
    )
    tokenizer = DeepSeekV41Tokenizer(SimpleNamespace(), str(tmp_path), {"model_type": "deepseek_v41"})
    assert (
        tokenizer.apply_chat_template(
            request.model_dump()["messages"], reasoning_effort=request.reasoning_effort, **request.chat_template_kwargs
        )
        is None
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
