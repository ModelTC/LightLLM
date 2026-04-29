"""Unit test for Anthropic -> OpenAI request translation with extra_body.

Verifies that ``extra_body.chat_template_kwargs`` (and other backend-specific
fields nested under ``extra_body`` per OpenAI SDK convention) survive the
/v1/messages request translation, so clients can opt out of model-default
thinking modes on engines that expose the toggle through
ChatCompletionRequest.chat_template_kwargs.

No server required — calls the pure translation helper directly.
"""

import pytest

pytest.importorskip("litellm")

from lightllm.server.api_anthropic import _anthropic_to_chat_request


def _base_body():
    return {
        "model": "test-model",
        "max_tokens": 32,
        "messages": [{"role": "user", "content": "hi"}],
    }


def test_extra_body_chat_template_kwargs_forwarded():
    body = _base_body()
    body["extra_body"] = {"chat_template_kwargs": {"enable_thinking": False}}

    chat_dict, _ = _anthropic_to_chat_request(body)

    assert chat_dict.get("chat_template_kwargs") == {"enable_thinking": False}
    assert "extra_body" not in chat_dict


def test_extra_body_multiple_fields_forwarded():
    body = _base_body()
    body["extra_body"] = {
        "chat_template_kwargs": {"enable_thinking": False},
        "do_sample": False,
        "top_k": 5,
    }

    chat_dict, _ = _anthropic_to_chat_request(body)

    assert chat_dict.get("chat_template_kwargs") == {"enable_thinking": False}
    assert chat_dict.get("do_sample") is False
    assert chat_dict.get("top_k") == 5


def test_top_level_openai_field_beats_extra_body_duplicate():
    # If a field ends up in openai_dict via the Anthropic->OpenAI translation
    # AND the same key appears in extra_body, the translation path wins.
    body = _base_body()
    body["temperature"] = 0.1  # translated by litellm -> openai_dict["temperature"] = 0.1
    body["extra_body"] = {"temperature": 0.9}

    chat_dict, _ = _anthropic_to_chat_request(body)

    assert chat_dict.get("temperature") == 0.1


def test_missing_extra_body_is_noop():
    body = _base_body()
    chat_dict, _ = _anthropic_to_chat_request(body)
    assert "extra_body" not in chat_dict
    assert "chat_template_kwargs" not in chat_dict


def test_non_dict_extra_body_is_ignored():
    body = _base_body()
    body["extra_body"] = "not-a-dict"
    chat_dict, _ = _anthropic_to_chat_request(body)
    assert "extra_body" not in chat_dict
