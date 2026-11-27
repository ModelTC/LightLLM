import copy
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from lightllm.models.deepseek_v4.tokenizer import DeepSeekV4Tokenizer


IMAGE_TOKEN = "<｜deepseek_image｜>"


class _Tokenizer:
    def convert_tokens_to_ids(self, token):
        assert token == IMAGE_TOKEN
        return 99

    def encode(self, prompt, **kwargs):
        ids = []
        for index, text in enumerate(prompt.split(IMAGE_TOKEN)):
            if index:
                ids.append(99)
            ids.extend([7] * len(text))
        return ids


@pytest.fixture
def tokenizer(tmp_path):
    # An empty model directory selects the bundled encoder used by default.
    return DeepSeekV4Tokenizer(_Tokenizer(), str(tmp_path), {"vision_n_layers": 1})


@pytest.mark.parametrize("tokenize", [False, True])
def test_mixed_content_preserves_image_order_and_expands_placeholders(tokenizer, tokenize):
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "before"},
                {"type": "image_url", "image_url": {"url": "first"}},
                {"type": "text", "text": "between"},
                {"type": "image_url", "image_url": {"url": "second"}},
                {"type": "text", "text": "after"},
            ],
        }
    ]
    original = copy.deepcopy(messages)
    prompt = tokenizer.apply_chat_template(messages=messages, tokenize=tokenize)
    if not tokenize:
        assert f"before{IMAGE_TOKEN}between{IMAGE_TOKEN}after" in prompt
    images = [SimpleNamespace(token_id=1000, token_num=13), SimpleNamespace(token_id=2000, token_num=17)]
    ids = tokenizer.encode(prompt, SimpleNamespace(images=images))
    for image in images:
        expected = list(range(image.token_id + image.start_idx % 4, image.token_id + image.token_num))
        assert ids[image.start_idx : image.start_idx + len(expected)] == expected
    assert images[0].start_idx < images[1].start_idx
    assert 99 not in ids
    assert messages == original


@pytest.mark.parametrize(
    "kwargs,thinking",
    [
        ({"reasoning_effort": "none"}, False),
        ({"reasoning_effort": "none", "thinking": False}, False),
        ({"reasoning_effort": "none", "enable_thinking": False}, False),
        ({"reasoning_effort": "none", "thinking": True}, True),
        ({"reasoning_effort": "high"}, True),
        ({"reasoning_effort": "max"}, True),
        ({}, True),
    ],
)
def test_reasoning_effort_uses_encoder_modes(tokenizer, kwargs, thinking):
    prompt = tokenizer.apply_chat_template(messages=[{"role": "user", "content": "hello"}], **kwargs)
    assert prompt.endswith("<think>" if thinking else "</think>")


def test_unknown_reasoning_effort_is_not_silently_replaced(tokenizer):
    with pytest.raises(AssertionError, match="Invalid reasoning effort: low"):
        tokenizer.apply_chat_template(messages=[{"role": "user", "content": "hello"}], reasoning_effort="low")


def test_chat_template_forwards_tools_and_encoder_options(tokenizer, monkeypatch):
    render = Mock(return_value="rendered prompt")
    monkeypatch.setattr(tokenizer, "_get_encoding_module", lambda: SimpleNamespace(encode_messages=render))
    messages = [
        {"role": "system", "content": "Use tools."},
        {"role": "user", "content": [{"type": "text", "text": "Look up 上海."}]},
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [{"type": "function", "function": {"name": "lookup", "arguments": {"city": "上海"}}}],
        },
    ]
    original = copy.deepcopy(messages)
    tool = {"name": "lookup", "parameters": {"type": "object"}}

    prompt = tokenizer.apply_chat_template(
        conversation=messages,
        tools=[tool],
        thinking=False,
        reasoning_effort="high",
        add_default_bos_token=False,
        drop_thinking=False,
    )
    assert prompt == "rendered prompt"

    expected = copy.deepcopy(original)
    expected[0]["tools"] = [{"type": "function", "function": tool}]
    expected[1]["content"] = "Look up 上海."
    expected[2]["tool_calls"][0]["function"]["arguments"] = '{"city": "上海"}'
    render.assert_called_once_with(
        expected, thinking_mode="chat", reasoning_effort="high", add_default_bos_token=False, drop_thinking=False
    )
    assert messages == original


def test_v4_vision_encoder_receives_structured_images(tokenizer, monkeypatch):
    render = Mock(return_value=f"before{IMAGE_TOKEN}after")
    encoder = SimpleNamespace(encode_messages=render, process_image_messages=lambda messages: messages)
    monkeypatch.setattr(tokenizer, "_get_encoding_module", lambda: encoder)
    messages = [
        {"role": "user", "content": [{"type": "text", "text": "one"}, {"type": "text", "text": "two"}]},
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "before"},
                {"type": "image_url", "image_url": {"url": "image"}},
                {"type": "text", "text": "after"},
            ],
        },
    ]
    original = copy.deepcopy(messages)

    assert tokenizer.apply_chat_template(messages=messages) == f"before{IMAGE_TOKEN}after"
    assert render.call_args.args[0] == [{"role": "user", "content": "onetwo"}, original[1]]
    assert messages == original
