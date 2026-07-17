import asyncio

import pytest
import ujson as json

from lightllm.server.api_responses import (
    _chat_response_to_responses,
    _openai_sse_to_responses_events,
    _responses_to_chat_request,
)


async def _chunks(*payloads):
    for payload in payloads:
        yield f"data: {json.dumps(payload)}\n\n"


def _collect_events(*payloads):
    async def collect():
        return [event async for event in _openai_sse_to_responses_events(_chunks(*payloads), {"input": "hi"})]

    raw_events = asyncio.run(collect())
    return [json.loads(event.decode().split("data: ", 1)[1]) for event in raw_events]


def test_function_call_arguments_done_includes_name():
    events = _collect_events(
        {
            "choices": [
                {
                    "delta": {
                        "tool_calls": [
                            {
                                "id": "call_1",
                                "function": {"name": "get_weather", "arguments": '{"city":"Paris"}'},
                            }
                        ]
                    }
                }
            ]
        }
    )

    done = next(event for event in events if event["type"] == "response.function_call_arguments.done")
    assert done["name"] == "get_weather"


def test_content_array_function_output_is_preserved():
    request = _responses_to_chat_request(
        {
            "input": [
                {
                    "type": "function_call_output",
                    "call_id": "call_1",
                    "output": [
                        {"type": "input_text", "text": "sunny"},
                        {"type": "input_image", "image_url": "https://example.com/weather.png"},
                    ],
                }
            ]
        }
    )

    assert request["messages"] == [
        {
            "role": "tool",
            "tool_call_id": "call_1",
            "content": [
                {"type": "text", "text": "sunny"},
                {"type": "image_url", "image_url": {"url": "https://example.com/weather.png"}},
            ],
        }
    ]


def test_non_streamed_truncated_function_call_is_incomplete():
    response = _chat_response_to_responses(
        {
            "choices": [
                {
                    "finish_reason": "length",
                    "message": {
                        "tool_calls": [
                            {
                                "id": "call_1",
                                "function": {"name": "get_weather", "arguments": '{"city":'},
                            }
                        ]
                    },
                }
            ]
        },
        {"input": "hi"},
    )

    assert response["status"] == "incomplete"
    assert response["output"][0]["status"] == "incomplete"


def test_streamed_truncated_function_call_is_incomplete():
    events = _collect_events(
        {
            "choices": [
                {
                    "delta": {
                        "tool_calls": [
                            {
                                "id": "call_1",
                                "function": {"name": "get_weather", "arguments": '{"city":'},
                            }
                        ]
                    }
                }
            ]
        },
        {"choices": [{"delta": {}, "finish_reason": "length"}]},
    )

    item_done = next(event for event in events if event["type"] == "response.output_item.done")
    response_done = events[-1]
    assert item_done["item"]["status"] == "incomplete"
    assert response_done["type"] == "response.incomplete"
    assert response_done["response"]["output"][0]["status"] == "incomplete"


def test_streamed_message_adds_text_content_once():
    events = _collect_events({"choices": [{"delta": {"content": "hello"}, "finish_reason": "stop"}]})

    item_added = next(event for event in events if event["type"] == "response.output_item.added")
    part_added = next(event for event in events if event["type"] == "response.content_part.added")
    assert item_added["item"]["content"] == []
    assert part_added["part"] == {"type": "output_text", "text": "", "annotations": []}


def test_route_maps_downstream_value_error_to_bad_request(monkeypatch):
    from types import SimpleNamespace

    from lightllm.server import api_http, api_responses

    async def raise_value_error(raw_request):
        raise ValueError("Unrecognized image input.")

    monkeypatch.setattr(api_http, "get_env_start_args", lambda: SimpleNamespace(run_mode="normal"))
    monkeypatch.setattr(api_http.g_objs, "metric_client", SimpleNamespace(counter_inc=lambda *args: None))
    monkeypatch.setattr(api_responses, "responses_impl", raise_value_error)

    response = asyncio.run(api_http.openai_responses(None))
    assert response.status_code == 400


@pytest.mark.parametrize(
    "partial_payload",
    [
        {"choices": [{"delta": {"content": "partial"}}]},
        {
            "choices": [
                {
                    "delta": {
                        "tool_calls": [
                            {
                                "id": "call_1",
                                "function": {"name": "get_weather", "arguments": '{"city":'},
                            }
                        ]
                    }
                }
            ]
        },
    ],
)
def test_stream_failure_does_not_complete_partial_item(partial_payload):
    events = _collect_events(partial_payload, {"error": {"message": "generation failed"}})
    event_types = [event["type"] for event in events]

    assert "response.output_text.done" not in event_types
    assert "response.function_call_arguments.done" not in event_types
    assert "response.output_item.done" not in event_types
    assert event_types[-1] == "response.failed"


@pytest.mark.parametrize("effort", ["none", "minimal", "xhigh"])
def test_unsupported_reasoning_effort_is_rejected(effort):
    with pytest.raises(ValueError, match="reasoning.effort"):
        _responses_to_chat_request({"input": "hi", "reasoning": {"effort": effort}})


def test_supported_reasoning_effort_is_forwarded():
    request = _responses_to_chat_request({"input": "hi", "reasoning": {"effort": "high"}})
    assert request["reasoning_effort"] == "high"


def test_automatic_truncation_is_rejected():
    with pytest.raises(ValueError, match="truncation"):
        _responses_to_chat_request({"input": "hi", "truncation": "auto"})
