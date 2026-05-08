"""Unit test for Anthropic -> OpenAI request translation with extra_body.

Verifies that ``extra_body.chat_template_kwargs`` (and other backend-specific
fields nested under ``extra_body`` per OpenAI SDK convention) survive the
/v1/messages request translation, so clients can opt out of model-default
thinking modes on engines that expose the toggle through
ChatCompletionRequest.chat_template_kwargs.

No server required — calls the pure translation helper directly.
"""

import asyncio
import pytest
import ujson as json

pytest.importorskip("litellm")

from lightllm.server.api_anthropic import _anthropic_to_chat_request, _openai_sse_to_anthropic_events


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


# Helpers for streaming test
def _chunk(delta, finish_reason=None, usage=None):
    obj = {"choices": [{"delta": delta, "finish_reason": finish_reason}]}
    if usage is not None:
        obj["usage"] = usage
    return f"data: {json.dumps(obj)}\n\n"


def test_interleaved_tool_calls_do_not_emit_against_closed_block():
    """Deltas for tool-call idx=1 arriving after idx=0 started must not
    stream into the (now-closed) idx=0 block."""

    async def chunks():
        yield _chunk(
            {
                "tool_calls": [
                    {"index": 0, "id": "call_a", "function": {"name": "fn_a", "arguments": '{"x":1'}},
                ]
            }
        )
        yield _chunk(
            {
                "tool_calls": [
                    {"index": 1, "id": "call_b", "function": {"name": "fn_b", "arguments": '{"y":2'}},
                ]
            }
        )
        yield _chunk(
            {
                "tool_calls": [
                    {"index": 0, "function": {"arguments": "}"}},
                ]
            }
        )
        yield _chunk({}, finish_reason="tool_calls", usage={"prompt_tokens": 3, "completion_tokens": 4})

    async def run():
        out = []
        async for ev in _openai_sse_to_anthropic_events(chunks(), "m", "msg_x"):
            out.append(ev.decode("utf-8"))
        return out

    events = asyncio.run(run())
    index_of_delta = []
    currently_open = None
    for raw in events:
        lines = raw.strip().split("\n")
        etype = lines[0].split(": ", 1)[1]
        data = json.loads(lines[1].split(": ", 1)[1])
        if etype == "content_block_start":
            currently_open = data["index"]
        elif etype == "content_block_stop":
            currently_open = None
        elif etype == "content_block_delta":
            assert (
                currently_open == data["index"]
            ), f"delta for index {data['index']} but open block is {currently_open}"
            index_of_delta.append(data["index"])
    assert index_of_delta, "no deltas observed"


def test_chat_response_translation_failure_returns_valid_json():
    """If response translation raises, the error path must return a clean
    Anthropic-shaped JSONResponse — not a JSONResponse wrapped in another
    JSONResponse."""
    from fastapi.responses import JSONResponse

    from lightllm.server import api_anthropic

    # Exercise the helper directly; the bug in anthropic_messages_impl was
    # wrapping this return value in another JSONResponse.
    resp = api_anthropic._anthropic_error_response(api_anthropic.HTTPStatus.INTERNAL_SERVER_ERROR, "synthetic")
    assert isinstance(resp, JSONResponse)
    body = bytes(resp.body).decode("utf-8")
    assert '"type":"error"' in body
    assert '"message":"synthetic"' in body
    assert resp.status_code == 500


def test_unknown_fields_emit_debug_log(caplog):
    """Silently-dropped Anthropic fields should at least emit a debug log so
    users can trace 'my metadata isn't propagating' without adding prints."""
    import logging

    from lightllm.server.api_anthropic import _anthropic_to_chat_request

    body = {
        "model": "m",
        "messages": [{"role": "user", "content": "hi"}],
        "max_tokens": 8,
        "metadata": {"user_id": "abc"},
        "anthropic_version": "2023-06-01",
    }
    # Set logger to DEBUG so caplog can capture it
    logger = logging.getLogger("lightllm.server.api_anthropic")
    logger.setLevel(logging.DEBUG)

    # Manually add caplog's handler to the logger to intercept logs
    # (works even with propagate=False)
    caplog_handler = logging.Handler()
    caplog_handler.emit = lambda record: caplog.records.append(record)
    logger.addHandler(caplog_handler)

    try:
        try:
            _anthropic_to_chat_request(body)
        except RuntimeError:
            import pytest

            pytest.skip("litellm not available; cannot exercise drop path")
        joined = "\n".join(rec.getMessage() for rec in caplog.records)
        assert "metadata" in joined or "anthropic_version" in joined
    finally:
        logger.removeHandler(caplog_handler)
