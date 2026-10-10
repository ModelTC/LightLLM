import asyncio
import ctypes
from types import SimpleNamespace

import pytest

from lightllm.server import api_http, api_openai, build_prompt
from lightllm.server.api_models import ChatCompletionRequest
from lightllm.server.core.objs.sampling_params import SamplingParams
from lightllm.server.core.objs.start_args_type import StartArgs
from lightllm.server.router.model_infer.infer_batch import InferSamplingParams


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
def test_chat_passes_reasoning_marker_through_shared_sampling_params(monkeypatch, compiler, response_type, thinking):
    monkeypatch.setattr(api_openai, "get_env_start_args", lambda: StartArgs(reasoning_parser="qwen3"))
    monkeypatch.setattr(build_prompt, "tokenizer_supports_force_thinking", lambda: True)

    async def prompt(*args):
        return "prompt"

    monkeypatch.setattr(api_openai, "build_prompt", prompt)
    captured = []

    class RequestCaptured(Exception):
        pass

    def generate(prompt, params, *args, **kwargs):
        # Exercise the real API-to-shared-memory and inference parameter boundary.
        copy = SamplingParams.from_buffer_copy(bytes(params))
        captured.append(InferSamplingParams(SimpleNamespace(sample_params=copy), vocab_size=257))
        raise RequestCaptured

    tokenizer = SimpleNamespace(encode=lambda *args, **kwargs: [7, 8])
    manager = SimpleNamespace(tokenizer=tokenizer, generate=generate, output_grammar_compiler=compiler)
    monkeypatch.setattr(api_http, "g_objs", SimpleNamespace(httpserver_manager=manager))
    response_format = {"type": response_type}
    if response_type == "json_schema":
        response_format["json_schema"] = {"name": "result", "schema": {"type": "object"}}
    request = ChatCompletionRequest(
        model="test",
        messages=[],
        response_format=response_format,
        chat_template_kwargs={"enable_thinking": thinking},
    )
    with pytest.raises(RequestCaptured):
        asyncio.run(api_openai.chat_completions_impl(request, None))
    expected = (7, 8) if thinking and response_type != "text" else ()
    assert captured[0].guided_reasoning_end == expected


def test_reasoning_marker_survives_serialization_and_parameter_reuse():
    params = SamplingParams()
    params.init(None, guided_reasoning_end=[11, 12])
    copy = SamplingParams.from_buffer_copy(ctypes.string_at(ctypes.addressof(params), ctypes.sizeof(params)))
    round_trip = SamplingParams()
    round_trip.init(None, **copy.to_origin_dict())
    assert round_trip.guided_reasoning_end.to_list() == [11, 12]
    round_trip.init(None)
    assert round_trip.guided_reasoning_end.to_list() == []


def test_grammar_uses_existing_force_thinking_decision(monkeypatch):
    monkeypatch.setattr(api_openai, "get_env_start_args", lambda: StartArgs(reasoning_parser="deepseek-r1"))
    monkeypatch.setattr(build_prompt, "tokenizer_supports_force_thinking", lambda: False)
    tokenizer = SimpleNamespace(encode=lambda *args, **kwargs: [99])
    request = ChatCompletionRequest(model="test", messages=[])
    assert not api_openai._is_force_thinking_mode(request)
    assert api_openai._get_guided_reasoning_end(request, tokenizer) == []
