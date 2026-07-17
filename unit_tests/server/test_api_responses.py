import asyncio

import pytest
import ujson as json

from lightllm.server.api_responses import _openai_sse_to_responses_events, _responses_to_chat_request


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
