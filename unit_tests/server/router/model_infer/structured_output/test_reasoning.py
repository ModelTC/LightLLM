import asyncio
import ctypes
import json
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
import torch

from lightllm.server import api_http, api_openai, build_prompt
from lightllm.server.api_models import ChatCompletionRequest
from lightllm.server.core.objs import FinishStatus
from lightllm.server.core.objs.sampling_params import SamplingParams
from lightllm.server.core.objs.start_args_type import StartArgs
from lightllm.server.router.model_infer.infer_batch import InferSamplingParams
from lightllm.server.router.model_infer.structured_output.state import ConstraintState


@pytest.mark.parametrize(
    "parser,template_kwargs,effort,marker",
    [
        ("qwen3", None, None, "</think>"),
        ("qwen3", {"enable_thinking": True}, None, "</think>"),
        ("qwen3", {"enable_thinking": False}, None, None),
        ("qwen3", {"thinking": False}, None, None),
        ("qwen3", None, "none", None),
        ("deepseek-v3", None, None, None),
        ("deepseek-v3", {"thinking": True}, None, "</think>"),
        ("qwen3-thinking", None, None, "</think>"),
        ("kimi", None, None, "◁/think▷"),
        ("gpt-oss", None, None, "<|channel|>final<|message|>"),
        ("gemma4", None, None, "<channel|>answer"),
        (None, None, None, None),
    ],
)
def test_chat_grammar_activation_matches_template_thinking(monkeypatch, parser, template_kwargs, effort, marker):
    monkeypatch.setattr(api_openai, "get_env_start_args", lambda: StartArgs(reasoning_parser=parser))
    monkeypatch.setattr(build_prompt, "tokenizer_supports_force_thinking", lambda: True)
    encoded = []

    def encode(text, add_special_tokens):
        assert not add_special_tokens
        encoded.append(text)
        return [7, 8]

    request = ChatCompletionRequest(
        model="test",
        messages=[],
        chat_template_kwargs=template_kwargs,
        reasoning_effort=effort,
    )
    result = api_openai._get_guided_reasoning_end(request, SimpleNamespace(encode=encode))
    assert encoded == ([marker] if marker else [])
    assert result == ([7, 8] if marker else [])


@pytest.mark.parametrize("response_type", ["json_schema", "json_object", "text"])
@pytest.mark.parametrize("thinking", [False, True])
@pytest.mark.parametrize("budget", [None, -1, 0, 2])
@pytest.mark.parametrize("parser", ["qwen3", "gpt-oss"])
@pytest.mark.parametrize("stream", [False, True])
def test_chat_passes_reasoning_marker_through_shared_sampling_params(
    monkeypatch, compiler, runtime_start_args, response_type, thinking, budget, parser, stream
):
    args = StartArgs(reasoning_parser=parser)
    monkeypatch.setattr(api_openai, "get_env_start_args", lambda: args)
    monkeypatch.setattr("lightllm.server.core.objs.sampling_params.get_env_start_args", lambda: args)
    monkeypatch.setattr(build_prompt, "tokenizer_supports_force_thinking", lambda: True)

    async def prompt(*args):
        return "prompt"

    monkeypatch.setattr(api_openai, "build_prompt", prompt)
    captured = []

    class RequestCaptured(Exception):
        pass

    async def generate(prompt, params, *args, **kwargs):
        # Exercise the real API-to-shared-memory and grammar-state boundary.
        copy = SamplingParams.from_buffer_copy(bytes(params))
        payload = await copy.get_compiled_grammar(compiler)
        shared_req = SimpleNamespace(
            sample_params=copy,
            get_compiled_grammar=lambda: payload,
            input_len=1,
            shm_prompt_ids=SimpleNamespace(arr=[ord("P")]),
        )
        infer_params = InferSamplingParams(shared_req, vocab_size=257)
        captured.append(compiler.output_constraint_state_factory.create_state(shared_req, infer_params))
        raise RequestCaptured
        yield

    tokenizer = SimpleNamespace(encode=lambda text, **kwargs: list(text.encode()))
    manager = SimpleNamespace(tokenizer=tokenizer, generate=generate, output_grammar_compiler=compiler)
    monkeypatch.setattr(
        api_http,
        "g_objs",
        SimpleNamespace(httpserver_manager=manager, metric_client=SimpleNamespace(counter_inc=lambda *_: None)),
    )
    response_format = {"type": response_type}
    if response_type == "json_schema":
        response_format["json_schema"] = {"name": "result", "schema": {"type": "object"}}
    request = ChatCompletionRequest(
        model="test",
        messages=[],
        response_format=response_format,
        chat_template_kwargs={"enable_thinking": thinking},
        thinking_token_budget=budget,
        stream=stream,
    )
    if parser == "gpt-oss" and budget is not None and budget >= 0:
        response = asyncio.run(api_openai.chat_completions_impl(request, None))
        assert response.status_code == 400
        assert "thinking_token_budget is not supported by the gpt-oss" in response.body.decode()
        assert captured == []
        return

    async def run():
        response = await api_openai.chat_completions_impl(request, None)
        if stream:
            async for _ in response.body_iterator:
                pass

    with pytest.raises(RequestCaptured):
        asyncio.run(run())
    thinking = thinking or parser == "gpt-oss"
    budget_active = thinking and budget is not None and budget >= 0
    marker = b"</think>" if parser == "qwen3" else b"<|channel|>final<|message|>"
    expected = tuple(marker) if thinking and (response_type != "text" or budget_active) else ()
    state = captured[0]
    if response_type == "text" and not budget_active:
        assert state is None
    else:
        assert state.error is None
        assert state.reasoning_end_token_ids == expected
        assert state.in_reasoning == bool(expected)
        if budget_active:
            assert state.thinking_token_budget == budget
        else:
            assert type(state) is ConstraintState
            assert state.thinking_token_budget == -1


def test_reasoning_marker_survives_serialization_and_parameter_reuse():
    params = SamplingParams()
    params.init(None, guided_reasoning_end=[11, 12])
    copy = SamplingParams.from_buffer_copy(ctypes.string_at(ctypes.addressof(params), ctypes.sizeof(params)))
    round_trip = SamplingParams()
    round_trip.init(None, **copy.to_origin_dict())
    assert round_trip.constraint_out_params.guided_reasoning_end.to_list() == [11, 12]
    round_trip.init(None)
    assert round_trip.constraint_out_params.guided_reasoning_end.to_list() == []


@pytest.mark.parametrize(
    "enabled,parser,message", [(False, "qwen3", "--disable_thinking_budget"), (True, None, "reasoning parser")]
)
def test_budget_unavailable_service_returns_bad_request(monkeypatch, enabled, parser, message):
    monkeypatch.setattr(
        api_openai,
        "get_env_start_args",
        lambda: StartArgs(reasoning_parser=parser, disable_thinking_budget=not enabled),
    )
    monkeypatch.setattr(api_openai, "build_prompt", AsyncMock(return_value="prompt"))
    monkeypatch.setattr(
        api_http,
        "g_objs",
        SimpleNamespace(
            httpserver_manager=SimpleNamespace(), metric_client=SimpleNamespace(counter_inc=lambda *_: None)
        ),
    )
    request = ChatCompletionRequest(messages=[], thinking_token_budget=1)
    response = asyncio.run(api_openai.chat_completions_impl(request, None))
    assert response.status_code == 400
    assert message in response.body.decode()


@pytest.mark.parametrize("budget", [True, 1.5, "2", -2, 2 ** 31])
def test_chat_rejects_invalid_thinking_budget(budget):
    with pytest.raises(ValueError, match="thinking_token_budget"):
        ChatCompletionRequest(messages=[], thinking_token_budget=budget)


@pytest.mark.parametrize(
    "parser,marker",
    [
        ("deepseek-r1", "</think>"),
        ("qwen3-thinking", "</think>"),
        ("gpt-oss", "<|channel|>final<|message|>"),
        ("minimax", "</think>"),
        ("qwen3", None),
    ],
)
def test_fixed_reasoning_does_not_require_a_template_thinking_switch(monkeypatch, parser, marker):
    monkeypatch.setattr(api_openai, "get_env_start_args", lambda: StartArgs(reasoning_parser=parser))
    monkeypatch.setattr(build_prompt, "tokenizer_supports_force_thinking", lambda: False)
    tokenizer = SimpleNamespace(encode=lambda text, **kwargs: list(text.encode()))
    request = ChatCompletionRequest(model="test", messages=[])
    assert api_openai._is_force_thinking_mode(request) == (marker is not None)
    assert api_openai._get_guided_reasoning_end(request, tokenizer) == (list(marker.encode()) if marker else [])


@pytest.mark.parametrize("stream", [False, True])
@pytest.mark.parametrize("response_type", ["json_object", "json_schema"])
@pytest.mark.parametrize("parser,budget", [("deepseek-r1", None), ("deepseek-r1", 2), ("qwen3", 0), ("qwen3", 2)])
def test_deepseek_r1_json_answer_stays_in_content(
    monkeypatch, compiler, runtime_start_args, stream, response_type, parser, budget
):
    args = StartArgs(reasoning_parser=parser, disable_thinking_budget=budget is None)
    monkeypatch.setattr(api_openai, "get_env_start_args", lambda: args)
    monkeypatch.setattr("lightllm.server.core.objs.sampling_params.get_env_start_args", lambda: args)
    monkeypatch.setattr(build_prompt, "tokenizer_supports_force_thinking", lambda: parser == "qwen3")
    monkeypatch.setattr(api_openai, "build_prompt", AsyncMock(return_value="<think>"))
    monkeypatch.setattr("lightllm.server.reasoning_parser.get_token_id", lambda token: ord(">"))
    tokenizer = SimpleNamespace(encode=lambda text, **kwargs: list(text.encode()))
    reasoning = "Let me think." if budget is None else "xy"[:budget]
    prefix = reasoning + "</think>"
    answer = '{"ok":true}'

    async def generate(prompt, params, *args, **kwargs):
        payload = await params.get_compiled_grammar(compiler)
        state = ConstraintState.from_grammar(
            compiler.output_constraint_state_factory.get_grammar(payload), params.constraint_out_params
        )
        bitmask = torch.empty((1, 9), dtype=torch.int32)
        for index, token_id in enumerate([*tokenizer.encode(prefix + answer), 256]):
            # The real matcher must allow thinking and its delimiter before constraining JSON.
            masked = state.fill_masks(bitmask, [])
            assert masked == (index >= (len(prefix) if budget is None else len(reasoning)))
            if masked:
                assert (int(bitmask[0, token_id // 32]) >> (token_id % 32)) & 1
            state.commit(token_id)
            assert state.error is None
            finish = FinishStatus(FinishStatus.FINISHED_STOP) if token_id == 256 else FinishStatus()
            yield 0, "" if token_id == 256 else chr(token_id), {"id": token_id, "prompt_tokens": 1}, finish

    manager = SimpleNamespace(tokenizer=tokenizer, generate=generate, output_grammar_compiler=compiler)
    monkeypatch.setattr(api_http, "g_objs", SimpleNamespace(httpserver_manager=manager))
    response_format = {"type": response_type}
    if response_type == "json_schema":
        response_format["json_schema"] = {"name": "result", "schema": {"const": {"ok": True}}}
    request = ChatCompletionRequest(
        model="test", messages=[], response_format=response_format, stream=stream, thinking_token_budget=budget
    )

    async def run():
        response = await api_openai.chat_completions_impl(request, None)
        if not stream:
            message = response.choices[0].message
            return message.reasoning, message.content
        deltas = []
        async for chunk in response.body_iterator:
            chunk = chunk.decode() if isinstance(chunk, bytes) else chunk
            if chunk.strip() != "data: [DONE]":
                for choice in json.loads(chunk.removeprefix("data: "))["choices"]:
                    deltas.append(choice["delta"])
        return "".join(delta.get("reasoning", "") for delta in deltas), "".join(
            delta.get("content", "") for delta in deltas
        )

    actual_reasoning, content = asyncio.run(run())
    assert (actual_reasoning or "") == reasoning
    assert content == answer
