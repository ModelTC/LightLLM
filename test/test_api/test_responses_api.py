#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Smoke tests for LightLLM's OpenAI /v1/responses compatibility endpoint.

Assumes a LightLLM server is running locally (default: http://localhost:18888)
with any loaded model. Run:

    python test/test_api/test_responses_api.py
    python test/test_api/test_responses_api.py --base-url http://localhost:18888

Tests hit the endpoint with raw requests (no OpenAI SDK dependency) so the
script can run against a fresh checkout without extra pip installs.
"""

import argparse
import asyncio
import json
import sys
import time
from typing import Any, Dict, Iterator, List

import requests
import ujson


BASE_URL = "http://localhost:18888"
MODEL_NAME = "test-model"  # echoed back by /v1/responses, model identity is ignored


def _post(url: str, body: Dict[str, Any], stream: bool = False) -> requests.Response:
    resp = requests.post(
        url,
        headers={"Content-Type": "application/json"},
        json=body,
        stream=stream,
        timeout=120,
    )
    if not resp.ok:
        raise AssertionError(f"POST {url} failed: HTTP {resp.status_code} — {resp.text}")
    return resp


def _parse_sse(resp: requests.Response) -> Iterator[Dict[str, Any]]:
    """Yield decoded ``data:`` payloads from an SSE stream, preserving order.

    Each yielded dict comes from a ``data:`` line's JSON payload. The ``event:``
    header line is dropped because the payload also carries a ``type`` field.
    """
    for raw_line in resp.iter_lines(decode_unicode=True):
        if not raw_line:
            continue
        line = raw_line.strip()
        if line.startswith("data: "):
            payload = line[len("data: ") :]
            if payload == "[DONE]":
                return
            try:
                yield json.loads(payload)
            except json.JSONDecodeError as exc:
                raise AssertionError(f"Malformed SSE JSON: {payload!r} ({exc})")


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def _extract_output_text(output: List[Dict[str, Any]]) -> str:
    """Concatenate output_text pieces from Responses ``output`` items."""
    text = ""
    for item in output:
        if item.get("type") != "message":
            continue
        for part in item.get("content") or []:
            if part.get("type") == "output_text":
                text += part.get("text", "")
    return text


# Reasoning-capable models can burn the entire output budget on hidden
# reasoning before producing any output_text. These tests use a generous
# budget so the model has headroom, and fall back to a warning instead of
# failing when the server reports status=incomplete.
TEXT_TOKEN_BUDGET = 256


def test_simple_string_input(base_url: str) -> None:
    print("\n[1] simple string input, non-streaming")
    body = {
        "model": MODEL_NAME,
        "input": "Say hello in one short sentence.",
        "max_output_tokens": TEXT_TOKEN_BUDGET,
    }
    data = _post(f"{base_url}/v1/responses", body).json()

    assert data["object"] == "response", data
    assert data["model"] == MODEL_NAME, data
    assert data["status"] in ("completed", "incomplete"), data
    assert isinstance(data["output"], list) and data["output"], "empty output array"
    first = data["output"][0]
    assert first["type"] == "message", first
    assert data["usage"]["input_tokens"] > 0, data["usage"]
    assert data["usage"]["output_tokens"] > 0, data["usage"]

    text = _extract_output_text(data["output"])
    if data["status"] == "incomplete":
        assert data["incomplete_details"] and data["incomplete_details"].get("reason"), data
        print(f"  note — status=incomplete ({data['incomplete_details']['reason']}); text={text!r}")
    else:
        assert text, f"completed response has no output_text: {data}"
        print(f"  ok — text: {text!r}  usage: {data['usage']}")


def test_array_input_with_instructions(base_url: str) -> None:
    print("\n[2] array input + instructions, non-streaming")
    body = {
        "model": MODEL_NAME,
        "instructions": "Answer with just the digit.",
        "input": [
            {
                "role": "user",
                "content": [
                    {"type": "input_text", "text": "What is two plus two?"},
                ],
            }
        ],
        "max_output_tokens": TEXT_TOKEN_BUDGET,
    }
    data = _post(f"{base_url}/v1/responses", body).json()
    assert data["output"][0]["type"] == "message", data
    text = _extract_output_text(data["output"])
    if data["status"] == "incomplete":
        print(f"  note — status=incomplete; text={text!r}")
    else:
        print(f"  ok — text: {text!r}")


def test_streaming_text(base_url: str) -> None:
    print("\n[3] streaming text")
    body = {
        "model": MODEL_NAME,
        "input": "Count from one to three.",
        "max_output_tokens": TEXT_TOKEN_BUDGET,
        "stream": True,
    }
    resp = _post(f"{base_url}/v1/responses", body, stream=True)
    event_types: List[str] = []
    collected_text = ""
    completed_payload: Dict[str, Any] = {}
    for event in _parse_sse(resp):
        event_types.append(event["type"])
        if event["type"] == "response.output_text.delta":
            collected_text += event["delta"]
        elif event["type"] == "response.completed":
            completed_payload = event["response"]

    print(f"  event sequence: {event_types}")
    assert event_types[0] == "response.created", event_types
    assert "response.in_progress" in event_types, event_types
    assert event_types[-1] == "response.completed", event_types

    if completed_payload.get("status") == "incomplete":
        print(f"  note — status=incomplete ({completed_payload['incomplete_details']}); text={collected_text!r}")
        return

    assert completed_payload.get("usage", {}).get("output_tokens", 0) > 0, completed_payload

    # For a truly completed text response the full event choreography should appear.
    for et in (
        "response.output_item.added",
        "response.content_part.added",
        "response.output_text.delta",
        "response.output_text.done",
        "response.content_part.done",
        "response.output_item.done",
    ):
        assert et in event_types, (et, event_types)
    assert collected_text, "no text deltas accumulated on completed stream"
    print(f"  ok — final text: {collected_text!r}  usage: {completed_payload['usage']}")


def test_function_tool_nonstreaming(base_url: str) -> None:
    print("\n[4] function tool, non-streaming")
    tools = [
        {
            "type": "function",
            "name": "get_current_weather",
            "description": "Get the current weather for a given city.",
            "parameters": {
                "type": "object",
                "properties": {
                    "location": {"type": "string", "description": "City name"},
                },
                "required": ["location"],
            },
        }
    ]
    body = {
        "model": MODEL_NAME,
        "input": "What's the weather in San Francisco right now? Use the tool.",
        "tools": tools,
        "tool_choice": "auto",
        "max_output_tokens": 128,
    }
    data = _post(f"{base_url}/v1/responses", body).json()
    # Model may or may not emit a tool call depending on model capability;
    # just assert the envelope is well-formed.
    assert data["object"] == "response", data
    types = [item["type"] for item in data["output"]]
    print(f"  output types: {types}")
    for item in data["output"]:
        if item["type"] == "function_call":
            assert item["name"] == "get_current_weather", item
            assert item["call_id"].startswith("call_") or item["call_id"], item
            # arguments should be a JSON string (possibly empty if model chose not to populate)
            try:
                json.loads(item["arguments"]) if item["arguments"] else None
            except json.JSONDecodeError:
                print(f"  warning — function_call arguments is not valid JSON: {item['arguments']!r}")
            print(f"  ok — function_call: name={item['name']} args={item['arguments']!r}")
            return
    print("  note — model did not emit a function_call; envelope still valid")


def test_function_tool_streaming(base_url: str) -> None:
    print("\n[5] function tool, streaming")
    tools = [
        {
            "type": "function",
            "name": "lookup_city",
            "description": "Look up a city by name.",
            "parameters": {
                "type": "object",
                "properties": {"name": {"type": "string"}},
                "required": ["name"],
            },
        }
    ]
    body = {
        "model": MODEL_NAME,
        "input": "Please call lookup_city for Tokyo.",
        "tools": tools,
        "tool_choice": "required",
        "max_output_tokens": 128,
        "stream": True,
    }
    resp = _post(f"{base_url}/v1/responses", body, stream=True)
    event_types: List[str] = []
    fc_args = ""
    for event in _parse_sse(resp):
        event_types.append(event["type"])
        if event["type"] == "response.function_call_arguments.delta":
            fc_args += event["delta"]
    print(f"  event sequence (first 12): {event_types[:12]}")
    print(f"  event sequence (total): {len(event_types)} events")
    assert event_types[0] == "response.created", event_types
    assert event_types[-1] == "response.completed", event_types
    if "response.function_call_arguments.delta" in event_types:
        print(f"  ok — streamed function_call args: {fc_args!r}")
    else:
        print("  note — model did not emit a function_call (tool_choice=required was best-effort)")


def test_function_call_round_trip(base_url: str) -> None:
    print("\n[6] function_call_output round-trip")
    body = {
        "model": MODEL_NAME,
        "input": [
            {"role": "user", "content": "What's the weather in Paris?"},
            {
                "type": "function_call",
                "call_id": "call_abc",
                "name": "get_weather",
                "arguments": json.dumps({"location": "Paris"}),
            },
            {
                "type": "function_call_output",
                "call_id": "call_abc",
                "output": "sunny, 22C",
            },
        ],
        "max_output_tokens": 64,
    }
    data = _post(f"{base_url}/v1/responses", body).json()
    assert data["object"] == "response", data
    print(f"  ok — assistant responded after tool output; status={data['status']}")


def test_input_image_file_id_is_dropped():
    """file_id must not become file:// — path-traversal mitigation."""
    from lightllm.server.api_openai_responses import _input_content_to_chat_content

    parts = [{"type": "input_image", "file_id": "/etc/passwd"}]
    result = _input_content_to_chat_content(parts)
    # With file_id dropped, no image part is emitted, so the result collapses
    # to either an empty list or an empty string via the single-text-part shortcut.
    assert result in ([], ""), f"file_id leaked through: {result!r}"

    parts2 = [{"type": "input_image", "image_url": "https://example.com/cat.png"}]
    result2 = _input_content_to_chat_content(parts2)
    assert result2 == [{"type": "image_url", "image_url": {"url": "https://example.com/cat.png"}}]


def _rc(delta, finish_reason=None, usage=None):
    obj = {"choices": [{"delta": delta, "finish_reason": finish_reason}]}
    if usage is not None:
        obj["usage"] = usage
    return f"data: {ujson.dumps(obj)}\n\n"


def test_responses_interleaved_tool_calls_do_not_emit_against_closed_item():
    from lightllm.server.api_openai_responses import _openai_sse_to_responses_events

    async def chunks():
        yield _rc(
            {
                "tool_calls": [
                    {"index": 0, "id": "call_a", "function": {"name": "fn_a", "arguments": '{"x":1'}},
                ]
            }
        )
        yield _rc(
            {
                "tool_calls": [
                    {"index": 1, "id": "call_b", "function": {"name": "fn_b", "arguments": '{"y":2'}},
                ]
            }
        )
        yield _rc({"tool_calls": [{"index": 0, "function": {"arguments": "}"}}]})
        yield _rc({}, finish_reason="tool_calls", usage={"prompt_tokens": 3, "completion_tokens": 5, "total_tokens": 8})

    async def run():
        out = []
        async for ev in _openai_sse_to_responses_events(chunks(), "m", "resp_x"):
            out.append(ev.decode("utf-8"))
        return out

    events = asyncio.run(run())

    open_item_id = None
    delta_count = 0
    for raw in events:
        lines = raw.strip().split("\n")
        etype = lines[0].split(": ", 1)[1]
        data = ujson.loads(lines[1].split(": ", 1)[1])
        if etype == "response.output_item.added" and data["item"]["type"] == "function_call":
            open_item_id = data["item"]["id"]
        elif etype == "response.output_item.done" and data["item"]["type"] == "function_call":
            if open_item_id == data["item"]["id"]:
                open_item_id = None
        elif etype == "response.function_call_arguments.delta":
            delta_count += 1
            assert data["item_id"] == open_item_id, f"delta item_id={data['item_id']} but open item_id={open_item_id}"
    assert delta_count >= 2


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", default=BASE_URL, help=f"LightLLM server URL (default {BASE_URL})")
    args = parser.parse_args()

    tests = [
        test_simple_string_input,
        test_array_input_with_instructions,
        test_streaming_text,
        test_function_tool_nonstreaming,
        test_function_tool_streaming,
        test_function_call_round_trip,
    ]

    failures: List[str] = []
    for fn in tests:
        start = time.time()
        try:
            fn(args.base_url)
            print(f"  [{fn.__name__}] PASS ({time.time() - start:.2f}s)")
        except AssertionError as exc:
            failures.append(f"{fn.__name__}: {exc}")
            print(f"  [{fn.__name__}] FAIL — {exc}")
        except Exception as exc:
            failures.append(f"{fn.__name__}: {exc!r}")
            print(f"  [{fn.__name__}] ERROR — {exc!r}")

    print("\n" + "=" * 60)
    if failures:
        print(f"{len(failures)} FAILURE(S):")
        for f in failures:
            print(f"  - {f}")
        return 1
    print(f"All {len(tests)} tests passed.")
    return 0


def test_hosted_tool_choice_rejected():
    from lightllm.server.api_openai_responses import _responses_tool_choice_to_chat, UnsupportedToolChoiceError

    # Valid forms still pass through.
    assert _responses_tool_choice_to_chat(None) is None
    assert _responses_tool_choice_to_chat("auto") == "auto"
    assert _responses_tool_choice_to_chat({"type": "function", "name": "X"}) == {
        "type": "function",
        "function": {"name": "X"},
    }

    # Hosted / unknown dict shapes raise.
    import pytest

    with pytest.raises(UnsupportedToolChoiceError):
        _responses_tool_choice_to_chat({"type": "web_search"})
    with pytest.raises(UnsupportedToolChoiceError):
        _responses_tool_choice_to_chat({"type": "file_search"})


def test_response_format_forwarded_from_top_level():
    from lightllm.server.api_openai_responses import _responses_to_chat_request

    body = {
        "model": "m",
        "input": "hi",
        "response_format": {"type": "json_object"},
    }
    chat_dict, _ = _responses_to_chat_request(body)
    assert chat_dict["response_format"] == {"type": "json_object"}


def test_text_format_forwarded_as_response_format():
    from lightllm.server.api_openai_responses import _responses_to_chat_request

    schema = {"type": "object", "properties": {"answer": {"type": "string"}}}
    body = {
        "model": "m",
        "input": "hi",
        "text": {"format": {"type": "json_schema", "name": "Ans", "schema": schema, "strict": True}},
    }
    chat_dict, _ = _responses_to_chat_request(body)
    assert chat_dict["response_format"] == {
        "type": "json_schema",
        "json_schema": {"name": "Ans", "schema": schema, "strict": True},
    }


def test_top_level_response_format_wins_over_text_format():
    from lightllm.server.api_openai_responses import _responses_to_chat_request

    body = {
        "model": "m",
        "input": "hi",
        "response_format": {"type": "json_object"},
        "text": {"format": {"type": "json_schema", "name": "X", "schema": {}}},
    }
    chat_dict, _ = _responses_to_chat_request(body)
    assert chat_dict["response_format"] == {"type": "json_object"}


def test_responses_lifecycle_endpoints_are_registered():
    """Stub GET / DELETE / POST .../cancel for /v1/responses/{id} must
    exist and return Responses-shaped error envelopes."""
    from lightllm.server.api_openai_responses import _stateless_lifecycle_error

    resp = _stateless_lifecycle_error("retrieve")
    assert resp.status_code == 404
    body = bytes(resp.body).decode("utf-8")
    assert '"error"' in body
    assert "stateless" in body.lower()

    cancel = _stateless_lifecycle_error("cancel")
    assert cancel.status_code == 400


def test_responses_lifecycle_routes_exist_on_app():
    from lightllm.server.api_http import app

    paths = {(r.path, tuple(sorted(r.methods))) for r in app.routes if hasattr(r, "methods") and r.methods}
    assert ("/v1/responses/{response_id}", ("GET",)) in paths
    assert ("/v1/responses/{response_id}", ("DELETE",)) in paths
    assert ("/v1/responses/{response_id}/cancel", ("POST",)) in paths


if __name__ == "__main__":
    sys.exit(main())
