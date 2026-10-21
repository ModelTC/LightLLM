import asyncio

import pytest
import ujson as json

from lightllm.server.api_responses import (
    _chat_response_to_responses,
    _openai_sse_to_responses_events,
    _responses_to_chat_request,
)
from lightllm.server.api_models import ChatCompletionRequest


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


def test_replayed_function_call_preserves_content_after_validation():
    request = _responses_to_chat_request(
        {
            "input": [
                {
                    "type": "function_call",
                    "call_id": "call_1",
                    "name": "get_weather",
                    "arguments": '{"city":"Paris"}',
                }
            ]
        }
    )

    chat_request = ChatCompletionRequest(**request)
    message = chat_request.messages[0].model_dump(exclude_none=True)
    assert message["content"] == ""


def test_parallel_function_calls_share_one_assistant_message():
    request = _responses_to_chat_request(
        {
            "input": [
                {
                    "type": "function_call",
                    "call_id": "call_paris",
                    "name": "get_weather",
                    "arguments": '{"city":"Paris"}',
                },
                {
                    "type": "function_call",
                    "call_id": "call_tokyo",
                    "name": "get_weather",
                    "arguments": '{"city":"Tokyo"}',
                },
                {"type": "function_call_output", "call_id": "call_paris", "output": "sunny"},
                {"type": "function_call_output", "call_id": "call_tokyo", "output": "rainy"},
            ]
        }
    )

    assert request["messages"] == [
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [
                {
                    "id": "call_paris",
                    "type": "function",
                    "function": {"name": "get_weather", "arguments": '{"city":"Paris"}'},
                },
                {
                    "id": "call_tokyo",
                    "type": "function",
                    "function": {"name": "get_weather", "arguments": '{"city":"Tokyo"}'},
                },
            ],
        },
        {"role": "tool", "tool_call_id": "call_paris", "content": "sunny"},
        {"role": "tool", "tool_call_id": "call_tokyo", "content": "rainy"},
    ]


@pytest.mark.parametrize(
    ("field", "part_type"),
    [("content", "reasoning_text"), ("summary", "summary_text")],
)
def test_replayed_reasoning_is_attached_to_assistant_tool_call(field, part_type):
    request = _responses_to_chat_request(
        {
            "input": [
                {
                    "type": "reasoning",
                    field: [{"type": part_type, "text": "I should check the weather."}],
                },
                {
                    "type": "function_call",
                    "call_id": "call_1",
                    "name": "get_weather",
                    "arguments": '{"city":"Paris"}',
                },
                {"type": "function_call_output", "call_id": "call_1", "output": "sunny"},
            ]
        }
    )

    assert request["messages"][0]["reasoning_content"] == "I should check the weather."
    assert request["messages"][0]["tool_calls"][0]["id"] == "call_1"
    assert len(request["messages"]) == 2


def test_replayed_reasoning_is_attached_to_assistant_message():
    request = _responses_to_chat_request(
        {
            "input": [
                {
                    "type": "reasoning",
                    "content": [{"type": "reasoning_text", "text": "I found the answer."}],
                },
                {
                    "type": "message",
                    "role": "assistant",
                    "content": [{"type": "output_text", "text": "It is sunny."}],
                },
                {
                    "type": "message",
                    "role": "user",
                    "content": [{"type": "input_text", "text": "Thanks"}],
                },
            ]
        }
    )

    assert request["messages"] == [
        {
            "role": "assistant",
            "content": [{"type": "text", "text": "It is sunny."}],
            "reasoning_content": "I found the answer.",
        },
        {"role": "user", "content": [{"type": "text", "text": "Thanks"}]},
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


@pytest.mark.parametrize("effort", ["low", "medium", "high", "none", "minimal", "xhigh"])
def test_reasoning_effort_is_rejected(effort):
    with pytest.raises(ValueError, match="reasoning.effort is not supported"):
        _responses_to_chat_request({"input": "hi", "reasoning": {"effort": effort}})


def test_automatic_truncation_is_rejected():
    with pytest.raises(ValueError, match="truncation"):
        _responses_to_chat_request({"input": "hi", "truncation": "auto"})
