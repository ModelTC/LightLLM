import asyncio
import json
from types import SimpleNamespace

import pytest
from pydantic import ValidationError

import lightllm.server.api_openai as api_openai
import lightllm.server.build_prompt as build_prompt_module
from lightllm.server.api_models import ChatCompletionRequest


def _request(**kwargs):
    return ChatCompletionRequest(messages=[{"role": "user", "content": "hello"}], **kwargs)


@pytest.mark.parametrize("effort", ["none", "minimal", "low", "medium", "high", "xhigh", "max"])
def test_reasoning_effort_is_accepted_and_controls_thinking(effort):
    request = _request(reasoning_effort=effort)

    assert request.reasoning_effort == ("high" if effort == "none" else effort)
    assert request.chat_template_kwargs["enable_thinking"] is True
    assert request.chat_template_kwargs["thinking"] is True


@pytest.mark.parametrize("kwargs", [{}, {"reasoning_effort": None}])
def test_omitted_or_null_reasoning_effort_keeps_default(kwargs):
    request = _request(**kwargs)

    assert request.reasoning_effort is None
    assert request.chat_template_kwargs is None


@pytest.mark.parametrize("stream", [False, True])
def test_json_none_reasoning_effort_is_normalized_before_serialization(stream):
    request = ChatCompletionRequest.model_validate_json(
        json.dumps({"messages": [{"role": "user", "content": "hello"}], "reasoning_effort": "none", "stream": stream})
    )

    assert request.reasoning_effort == "high"
    assert request.model_dump()["reasoning_effort"] == "high"
    assert request.chat_template_kwargs == {"enable_thinking": True, "thinking": True}


def test_unsupported_reasoning_effort_is_rejected():
    with pytest.raises(ValidationError):
        _request(reasoning_effort="ultra")


@pytest.mark.parametrize("template_key", ["thinking", "enable_thinking"])
@pytest.mark.parametrize(("effort", "explicit_value"), [("none", True), ("none", False), ("high", False)])
def test_explicit_thinking_setting_overrides_reasoning_effort(template_key, effort, explicit_value):
    request = _request(reasoning_effort=effort, chat_template_kwargs={template_key: explicit_value})

    assert request.chat_template_kwargs["enable_thinking"] is explicit_value
    assert request.chat_template_kwargs["thinking"] is explicit_value


@pytest.mark.parametrize("effort", ["none", "high"])
@pytest.mark.parametrize("parser", ["glm45", "qwen3"])
def test_reasoning_effort_is_forwarded_to_chat_template(monkeypatch, effort, parser):
    class RecordingTokenizer:
        def apply_chat_template(self, **kwargs):
            self.kwargs = kwargs
            return "rendered prompt"

    tokenizer = RecordingTokenizer()
    monkeypatch.setattr(build_prompt_module, "tokenizer", tokenizer)
    monkeypatch.setattr(build_prompt_module, "get_model_type_v1", lambda: None)
    monkeypatch.setattr(build_prompt_module, "tokenizer_supports_force_thinking", lambda: True)
    monkeypatch.setattr(api_openai, "get_env_start_args", lambda: SimpleNamespace(reasoning_parser=parser))

    prompt = asyncio.run(build_prompt_module.build_prompt(_request(reasoning_effort=effort), tools=None))

    assert prompt == "rendered prompt"
    assert tokenizer.kwargs["reasoning_effort"] == "high"
    assert tokenizer.kwargs["enable_thinking"] is True
    assert tokenizer.kwargs["thinking"] is True
