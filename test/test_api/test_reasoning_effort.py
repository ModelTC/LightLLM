import asyncio
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

    expected = effort != "none"
    assert request.chat_template_kwargs["enable_thinking"] is expected
    assert request.chat_template_kwargs["thinking"] is expected


def test_unsupported_reasoning_effort_is_rejected():
    with pytest.raises(ValidationError):
        _request(reasoning_effort="ultra")


@pytest.mark.parametrize("template_key", ["thinking", "enable_thinking"])
@pytest.mark.parametrize(("effort", "explicit_value"), [("none", True), ("high", False)])
def test_explicit_thinking_setting_overrides_reasoning_effort(template_key, effort, explicit_value):
    request = _request(reasoning_effort=effort, chat_template_kwargs={template_key: explicit_value})

    assert request.chat_template_kwargs["enable_thinking"] is explicit_value
    assert request.chat_template_kwargs["thinking"] is explicit_value


def test_reasoning_effort_is_forwarded_to_chat_template(monkeypatch):
    class RecordingTokenizer:
        def apply_chat_template(self, **kwargs):
            self.kwargs = kwargs
            return "rendered prompt"

    tokenizer = RecordingTokenizer()
    monkeypatch.setattr(build_prompt_module, "tokenizer", tokenizer)
    monkeypatch.setattr(build_prompt_module, "get_model_type_v1", lambda: None)
    monkeypatch.setattr(build_prompt_module, "tokenizer_supports_force_thinking", lambda: True)
    monkeypatch.setattr(api_openai, "get_env_start_args", lambda: SimpleNamespace(reasoning_parser="qwen3"))

    prompt = asyncio.run(build_prompt_module.build_prompt(_request(reasoning_effort="none"), tools=None))

    assert prompt == "rendered prompt"
    assert tokenizer.kwargs["reasoning_effort"] == "none"
    assert tokenizer.kwargs["enable_thinking"] is False
    assert tokenizer.kwargs["thinking"] is False
