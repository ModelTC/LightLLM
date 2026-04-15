"""Unit tests for the Anthropic API translation layer.

These tests import the translation helpers directly and do not require
a running LightLLM server. They do require 'litellm' to be installed —
tests are skipped if it is not available.
"""
import asyncio

import pytest

litellm = pytest.importorskip("litellm")


def test_shim_imports_adapter():
    from lightllm.server._litellm_shim import get_anthropic_messages_adapter

    adapter = get_anthropic_messages_adapter()
    assert hasattr(adapter, "translate_anthropic_to_openai")
    assert hasattr(adapter, "translate_openai_response_to_anthropic")


def test_shim_raises_clear_error_when_litellm_missing(monkeypatch):
    import sys

    from lightllm.server import _litellm_shim

    monkeypatch.setitem(sys.modules, "litellm", None)
    _litellm_shim._cached_adapter = None
    _litellm_shim._import_checked = False  # reset module-level cache

    with pytest.raises(RuntimeError, match="--enable_anthropic_api requires"):
        _litellm_shim.get_anthropic_messages_adapter()


def test_adapter_round_trip_minimal_text():
    """Lock down LiteLLM adapter I/O shapes for a minimal text request.

    If this test breaks after a LiteLLM upgrade, the adapter's contract
    has shifted and _litellm_shim.py may need updating.
    """
    from lightllm.server._litellm_shim import get_anthropic_messages_adapter
    from litellm import ModelResponse

    adapter = get_anthropic_messages_adapter()

    anthropic_request = {
        "model": "claude-opus-4-6",
        "max_tokens": 128,
        "system": "You are a terse assistant.",
        "messages": [
            {"role": "user", "content": "Say hi."},
        ],
    }

    # Direction 1: Anthropic request -> OpenAI request
    openai_request, tool_name_mapping = adapter.translate_anthropic_to_openai(anthropic_request)

    # Should be a dict-like / pydantic model with messages field
    openai_dict = (
        openai_request.model_dump(exclude_none=True)
        if hasattr(openai_request, "model_dump")
        else dict(openai_request)
    )
    assert "messages" in openai_dict
    messages = openai_dict["messages"]

    # System prompt should be injected as a system-role message
    assert any(m.get("role") == "system" for m in messages), messages
    # User content should be preserved
    assert any(m.get("role") == "user" for m in messages), messages
    assert isinstance(tool_name_mapping, dict)

    # Direction 2: OpenAI response -> Anthropic response
    fake_openai_response_dict = {
        "id": "chatcmpl-test",
        "object": "chat.completion",
        "created": 0,
        "model": "local-model",
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": "Hi."},
                "finish_reason": "stop",
            }
        ],
        "usage": {"prompt_tokens": 5, "completion_tokens": 2, "total_tokens": 7},
    }
    # Wrap dict in ModelResponse for adapter consumption
    fake_openai_response = ModelResponse(**fake_openai_response_dict)
    anthropic_response = adapter.translate_openai_response_to_anthropic(
        fake_openai_response, tool_name_mapping
    )

    resp_dict = (
        anthropic_response.model_dump(exclude_none=True)
        if hasattr(anthropic_response, "model_dump")
        else dict(anthropic_response)
    )
    assert resp_dict.get("type") == "message"
    assert resp_dict.get("role") == "assistant"
    content = resp_dict.get("content")
    assert isinstance(content, list) and len(content) >= 1
    assert content[0].get("type") == "text"
    assert "Hi" in content[0].get("text", "")
    # Stop reasons: Anthropic uses end_turn/tool_use/max_tokens/stop_sequence
    assert resp_dict.get("stop_reason") in {"end_turn", "stop_sequence", None}


def test_anthropic_to_chat_request_dict_minimal_text():
    """_anthropic_to_chat_request should return a dict suitable for
    constructing a LightLLM ChatCompletionRequest."""
    from lightllm.server.api_anthropic import _anthropic_to_chat_request

    anthropic_body = {
        "model": "claude-opus-4-6",
        "max_tokens": 64,
        "system": "Be terse.",
        "messages": [{"role": "user", "content": "hello"}],
        "temperature": 0.4,
    }
    chat_request_dict, tool_name_mapping = _anthropic_to_chat_request(anthropic_body)

    assert "messages" in chat_request_dict
    assert any(m.get("role") == "system" for m in chat_request_dict["messages"])
    assert any(m.get("role") == "user" for m in chat_request_dict["messages"])
    # max_tokens must be propagated
    assert chat_request_dict.get("max_tokens") == 64 or chat_request_dict.get("max_completion_tokens") == 64
    assert isinstance(tool_name_mapping, dict)


def test_chat_response_to_anthropic_minimal_text():
    """_chat_response_to_anthropic should wrap a ChatCompletionResponse
    into an Anthropic message dict."""
    from lightllm.server.api_anthropic import _chat_response_to_anthropic
    from lightllm.server.api_models import (
        ChatCompletionResponse,
        ChatCompletionResponseChoice,
        ChatMessage,
        UsageInfo,
    )

    chat_resp = ChatCompletionResponse(
        id="chatcmpl-xyz",
        model="local-model",
        choices=[
            ChatCompletionResponseChoice(
                index=0,
                message=ChatMessage(role="assistant", content="Hello."),
                finish_reason="stop",
            )
        ],
        usage=UsageInfo(prompt_tokens=3, completion_tokens=2, total_tokens=5),
    )
    anthropic_dict = _chat_response_to_anthropic(chat_resp, tool_name_mapping={}, requested_model="claude-opus-4-6")

    assert anthropic_dict["type"] == "message"
    assert anthropic_dict["role"] == "assistant"
    assert anthropic_dict["model"] == "claude-opus-4-6"
    content = anthropic_dict["content"]
    assert isinstance(content, list) and len(content) >= 1
    assert content[0]["type"] == "text"
    assert "Hello" in content[0]["text"]
    assert anthropic_dict["stop_reason"] in {"end_turn", "stop_sequence"}
    assert anthropic_dict["usage"]["input_tokens"] == 3
    assert anthropic_dict["usage"]["output_tokens"] == 2


def _run(coro):
    return asyncio.run(coro)


def test_stream_bridge_emits_anthropic_event_sequence_text_only():
    """Feed a canned OpenAI SSE stream through the bridge and assert we
    get the expected Anthropic event sequence."""
    from lightllm.server.api_anthropic import _openai_sse_to_anthropic_events

    # Simulate three OpenAI chunks: 'Hel', 'lo', finish_reason=stop.
    openai_chunks = [
        b'data: {"id":"x","object":"chat.completion.chunk","created":0,"model":"m","choices":[{"index":0,"delta":{"role":"assistant","content":"Hel"},"finish_reason":null}]}\n\n',
        b'data: {"id":"x","object":"chat.completion.chunk","created":0,"model":"m","choices":[{"index":0,"delta":{"content":"lo"},"finish_reason":null}]}\n\n',
        b'data: {"id":"x","object":"chat.completion.chunk","created":0,"model":"m","choices":[{"index":0,"delta":{},"finish_reason":"stop"}]}\n\n',
        b'data: {"id":"x","object":"chat.completion.chunk","created":0,"model":"m","choices":[],"usage":{"prompt_tokens":5,"completion_tokens":2,"total_tokens":7}}\n\n',
        b"data: [DONE]\n\n",
    ]

    async def fake_body_iterator():
        for c in openai_chunks:
            yield c

    async def collect():
        out = []
        async for event_bytes in _openai_sse_to_anthropic_events(
            fake_body_iterator(), requested_model="claude-opus-4-6", message_id="msg_test"
        ):
            out.append(event_bytes.decode("utf-8"))
        return out

    events = _run(collect())
    joined = "".join(events)

    # Required event types appear in order
    must_appear_in_order = [
        "event: message_start",
        "event: content_block_start",
        'content_block_delta',
        "event: content_block_stop",
        "event: message_delta",
        "event: message_stop",
    ]
    last_idx = -1
    for needle in must_appear_in_order:
        idx = joined.find(needle, last_idx + 1)
        assert idx > last_idx, f"missing or out-of-order event: {needle}\nfull:\n{joined}"
        last_idx = idx

    # Text deltas preserve the original content
    assert "Hel" in joined and "lo" in joined
    # end_turn stop reason is surfaced
    assert "end_turn" in joined
    # Final usage output_tokens is included in message_delta
    assert '"output_tokens"' in joined


def test_anthropic_to_chat_request_with_tools():
    from lightllm.server.api_anthropic import _anthropic_to_chat_request

    anthropic_body = {
        "model": "claude-opus-4-6",
        "max_tokens": 256,
        "messages": [{"role": "user", "content": "What's the weather in SF?"}],
        "tools": [
            {
                "name": "get_weather",
                "description": "Return the current weather for a city",
                "input_schema": {
                    "type": "object",
                    "properties": {"city": {"type": "string"}},
                    "required": ["city"],
                },
            }
        ],
    }
    chat_dict, tool_name_mapping = _anthropic_to_chat_request(anthropic_body)

    assert "tools" in chat_dict
    assert isinstance(chat_dict["tools"], list) and len(chat_dict["tools"]) == 1
    tool_entry = chat_dict["tools"][0]
    # OpenAI tool format: {"type": "function", "function": {"name", "description", "parameters"}}
    assert tool_entry.get("type") == "function"
    fn = tool_entry.get("function") or {}
    assert fn.get("name") == "get_weather"
    # input_schema should have been renamed to parameters
    assert "parameters" in fn
    assert fn["parameters"]["properties"]["city"]["type"] == "string"


def test_chat_response_to_anthropic_with_tool_call():
    from lightllm.server.api_anthropic import _chat_response_to_anthropic
    from lightllm.server.api_models import (
        ChatCompletionResponse,
        ChatCompletionResponseChoice,
        ChatMessage,
        FunctionResponse,
        ToolCall,
        UsageInfo,
    )

    chat_resp = ChatCompletionResponse(
        id="chatcmpl-tool",
        model="local-model",
        choices=[
            ChatCompletionResponseChoice(
                index=0,
                message=ChatMessage(
                    role="assistant",
                    content="",
                    tool_calls=[
                        ToolCall(
                            id="call_abc123",
                            index=0,
                            type="function",
                            function=FunctionResponse(
                                name="get_weather",
                                arguments='{"city":"San Francisco"}',
                            ),
                        )
                    ],
                ),
                finish_reason="tool_calls",
            )
        ],
        usage=UsageInfo(prompt_tokens=20, completion_tokens=12, total_tokens=32),
    )
    anthropic_dict = _chat_response_to_anthropic(
        chat_resp, tool_name_mapping={}, requested_model="claude-opus-4-6"
    )

    assert anthropic_dict["stop_reason"] == "tool_use"
    content = anthropic_dict["content"]
    tool_blocks = [b for b in content if b.get("type") == "tool_use"]
    assert len(tool_blocks) == 1
    tool_block = tool_blocks[0]
    assert tool_block["name"] == "get_weather"
    assert isinstance(tool_block["input"], dict)
    assert tool_block["input"].get("city") == "San Francisco"
    assert tool_block.get("id", "").startswith("toolu_") or tool_block.get("id") == "call_abc123"


def test_anthropic_to_chat_request_with_image_content_block():
    """Vision smoke test: an Anthropic image content block must survive
    translation without raising or being silently dropped. We do not
    assert the exact OpenAI shape here because LiteLLM's adapter controls
    that contract and may normalise it in different ways across releases."""
    from lightllm.server.api_anthropic import _anthropic_to_chat_request

    anthropic_body = {
        "model": "claude-opus-4-6",
        "max_tokens": 64,
        "messages": [
            {
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "source": {
                            "type": "base64",
                            "media_type": "image/png",
                            "data": "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR4nGNgAAIAAAUAAen63NgAAAAASUVORK5CYII=",
                        },
                    },
                    {"type": "text", "text": "What do you see?"},
                ],
            }
        ],
    }
    chat_dict, _ = _anthropic_to_chat_request(anthropic_body)

    assert "messages" in chat_dict
    # The user message must still be present after translation — the exact
    # shape of its content is left to the adapter, but it must exist.
    user_messages = [m for m in chat_dict["messages"] if m.get("role") == "user"]
    assert user_messages, f"user message was dropped during translation: {chat_dict['messages']}"
