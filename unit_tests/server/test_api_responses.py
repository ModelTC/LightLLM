import asyncio
from builtins import anext
import json as std_json

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


def _collect_events(*payloads, body=None, finish=True):
    if finish and not any(
        p.get("error") or any(c.get("finish_reason") for c in p.get("choices", [])) for p in payloads
    ):
        payloads += ({"choices": [{"delta": {}, "finish_reason": "stop"}]},)

    async def collect():
        return [event async for event in _openai_sse_to_responses_events(_chunks(*payloads), body or {"input": "hi"})]

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


@pytest.mark.parametrize("with_reasoning", [False, True])
def test_replayed_commentary_and_function_calls_preserve_assistant_turns(with_reasoning):
    items = [{"role": "user", "content": "Check the files."}]
    for i in range(3):
        message = {
            "content": f"Checking file {i}.",
            "tool_calls": [
                {
                    "id": f"call_{i}",
                    "function": {"name": "read_file", "arguments": json.dumps({"path": f"file_{i}"})},
                }
            ],
        }
        if with_reasoning:
            message["reasoning"] = f"I need to inspect file {i}."
        response = _chat_response_to_responses(
            {"choices": [{"finish_reason": "tool_calls", "message": message}]},
            {"input": "Check the files."},
        )
        items.extend(response["output"])
        items.append({"type": "function_call_output", "call_id": f"call_{i}", "output": "ok"})

    messages = _responses_to_chat_request({"input": items})["messages"]

    assert [message["role"] for message in messages] == ["user"] + ["assistant", "tool"] * 3
    for i, message in enumerate(messages[1::2]):
        assert message["content"] == [{"type": "text", "text": f"Checking file {i}."}]
        assert message["tool_calls"][0]["id"] == f"call_{i}"
        if with_reasoning:
            assert message["reasoning_content"] == f"I need to inspect file {i}."


@pytest.mark.parametrize(
    ("first", "second", "expected"),
    [
        ("First ", "second.", "First second."),
        (
            [{"type": "output_text", "text": "First "}],
            [{"type": "output_text", "text": "second."}],
            [{"type": "text", "text": "First "}, {"type": "text", "text": "second."}],
        ),
        (
            "First ",
            [{"type": "output_text", "text": "second."}],
            [{"type": "text", "text": "First "}, {"type": "text", "text": "second."}],
        ),
        (
            [{"type": "output_text", "text": "First "}],
            "second.",
            [{"type": "text", "text": "First "}, {"type": "text", "text": "second."}],
        ),
        ("First", None, "First"),
        ("First", "", "First"),
    ],
)
def test_replayed_assistant_text_parts_are_preserved(first, second, expected):
    request = _responses_to_chat_request(
        {
            "input": [
                {"type": "message", "role": "assistant", "content": first},
                {"type": "message", "role": "assistant", "content": second},
                {"type": "function_call", "call_id": "call_1", "name": "read_file", "arguments": "{}"},
            ]
        }
    )

    assert len(request["messages"]) == 1
    assert request["messages"][0]["content"] == expected
    assert request["messages"][0]["tool_calls"][0]["id"] == "call_1"


@pytest.mark.parametrize(
    "boundary",
    [
        {"role": "user", "content": "Next task."},
        {"role": "system", "content": "New instruction."},
        {"role": "developer", "content": "New instruction."},
        {"type": "function_call_output", "call_id": "call_before", "output": "ok"},
    ],
)
def test_replayed_assistant_turns_stop_at_message_boundaries(boundary):
    request = _responses_to_chat_request(
        {
            "input": [
                {"role": "assistant", "content": "Before."},
                {"type": "function_call", "call_id": "call_before", "name": "read_file", "arguments": "{}"},
                boundary,
                {"type": "reasoning", "content": [{"type": "reasoning_text", "text": "Next step."}]},
                {"role": "assistant", "content": "After."},
                {"type": "function_call", "call_id": "call_after", "name": "read_file", "arguments": "{}"},
            ]
        }
    )

    messages = request["messages"]
    # System/developer instructions are moved to the start by the request adapter.
    assistants = [message for message in messages if message["role"] == "assistant"]
    assert len(messages) == 3
    assert len(assistants) == 2
    assert assistants[0]["content"] == "Before."
    assert "reasoning_content" not in assistants[0]
    assert assistants[0]["tool_calls"][0]["id"] == "call_before"
    assert assistants[1]["content"] == "After."
    assert assistants[1]["reasoning_content"] == "Next step."
    assert assistants[1]["tool_calls"][0]["id"] == "call_after"


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


@pytest.mark.parametrize("effort", ["invalid", True, 0])
def test_invalid_reasoning_effort_is_rejected(effort):
    with pytest.raises(ValueError, match="reasoning.effort"):
        _responses_to_chat_request({"input": "hi", "reasoning": {"effort": effort}})


def test_automatic_truncation_is_rejected():
    with pytest.raises(ValueError, match="truncation"):
        _responses_to_chat_request({"input": "hi", "truncation": "auto"})


def test_responses_passes_supported_chat_parameters():
    params = {
        "seed": 1234,
        "top_k": 1,
        "repetition_penalty": 1.1,
        "presence_penalty": 0.2,
        "frequency_penalty": 0.3,
        "stop": ["END"],
        "ignore_eos": True,
        "chat_template_kwargs": {"enable_thinking": False},
    }
    request = ChatCompletionRequest(**_responses_to_chat_request({"input": "hi", **params}))
    for key, value in params.items():
        actual = getattr(request, key)
        if isinstance(value, dict):
            assert {k: actual[k] for k in value} == value
        else:
            assert actual == value


@pytest.mark.parametrize("effort", ["none", "minimal", "low", "medium", "high", "xhigh", "max"])
def test_responses_reasoning_effort_matches_chat(effort):
    request = ChatCompletionRequest(**_responses_to_chat_request({"input": "hi", "reasoning": {"effort": effort}}))
    assert request.reasoning_effort == effort
    assert request.chat_template_kwargs["enable_thinking"] == (effort != "none")


def test_distinct_assistant_phases_survive_chat_validation():
    request = ChatCompletionRequest(
        **_responses_to_chat_request(
            {
                "input": [
                    {"role": "assistant", "phase": "commentary", "content": "working"},
                    {"role": "assistant", "phase": "final_answer", "content": "answer"},
                ]
            }
        )
    )
    assert [m.content for m in request.messages] == ["working", "answer"]
    assert [m.phase for m in request.messages] == ["commentary", "final_answer"]


def test_reasoning_commentary_and_call_remain_one_message_with_phase():
    request = _responses_to_chat_request(
        {
            "input": [
                {"type": "reasoning", "content": [{"type": "reasoning_text", "text": "plan"}]},
                {"role": "assistant", "phase": "commentary", "content": "working"},
                {"type": "function_call", "call_id": "call_1", "name": "read", "arguments": "{}"},
            ]
        }
    )
    assert len(request["messages"]) == 1
    assert request["messages"][0]["phase"] == "commentary"
    assert request["messages"][0]["reasoning_content"] == "plan"


@pytest.mark.parametrize(
    "tool",
    [
        {"type": "web_search"},
        {"type": "function", "name": "read", "strict": True, "parameters": {"type": "object"}},
    ],
)
def test_unsupported_tools_fail_explicitly(tool):
    with pytest.raises(ValueError):
        _responses_to_chat_request({"input": "hi", "tools": [tool]})


@pytest.mark.parametrize(
    "choice,tools",
    [
        ("required", []),
        ({"type": "function", "name": "missing"}, [{"type": "function", "name": "read", "parameters": {}}]),
    ],
)
def test_invalid_tool_choice_is_rejected(choice, tools):
    with pytest.raises(ValueError, match="tool_choice"):
        _responses_to_chat_request({"input": "hi", "tool_choice": choice, "tools": tools})


def test_namespace_tool_roundtrip():
    tools = [{"type": "namespace", "name": "fs", "tools": [{"type": "function", "name": "read", "parameters": {}}]}]
    body = {"input": "hi", "tools": tools, "tool_choice": {"type": "function", "name": "read", "namespace": "fs"}}
    request = _responses_to_chat_request(body)
    assert request["tools"][0]["function"]["name"] == "fs__read"
    assert request["tool_choice"]["function"]["name"] == "fs__read"
    result = _chat_response_to_responses(
        {
            "choices": [
                {
                    "finish_reason": "tool_calls",
                    "message": {"tool_calls": [{"id": "call_1", "function": {"name": "fs__read", "arguments": "{}"}}]},
                }
            ]
        },
        body,
    )
    call = result["output"][0]
    assert call["name"] == "read" and call["namespace"] == "fs"
    replay = _responses_to_chat_request({"input": [call], "tools": tools})
    assert replay["messages"][0]["tool_calls"][0]["function"]["name"] == "fs__read"


def test_custom_tool_roundtrip():
    payload = '*** Begin Patch\n中文 "quoted" \\ path\n*** End Patch'
    body = {"input": "edit", "tools": [{"type": "custom", "name": "apply_patch", "format": {"type": "text"}}]}
    request = _responses_to_chat_request(body)
    assert request["tools"][0]["function"]["parameters"]["properties"]["input"]["type"] == "string"
    result = _chat_response_to_responses(
        {
            "choices": [
                {
                    "finish_reason": "tool_calls",
                    "message": {
                        "tool_calls": [
                            {
                                "id": "call_1",
                                "function": {"name": "apply_patch", "arguments": json.dumps({"input": payload})},
                            }
                        ]
                    },
                }
            ]
        },
        body,
    )
    call = result["output"][0]
    assert call["type"] == "custom_tool_call" and call["input"] == payload
    replay = _responses_to_chat_request(
        {
            "input": [call, {"type": "custom_tool_call_output", "call_id": "call_1", "output": "ok"}],
            "tools": body["tools"],
        }
    )
    assert json.loads(replay["messages"][0]["tool_calls"][0]["function"]["arguments"]) == {"input": payload}
    assert replay["messages"][1]["role"] == "tool"


@pytest.mark.parametrize("stream", [False, True])
def test_requested_reasoning_summary_is_returned(stream):
    body = {"input": "hi", "reasoning": {"summary": "auto"}}
    if stream:
        events = _collect_events({"choices": [{"delta": {"reasoning": "plan"}}]}, body=body)
        assert any(e["type"] == "response.reasoning_summary_part.added" for e in events)
        assert any(e["type"] == "response.reasoning_summary_text.delta" for e in events)
        result = events[-1]["response"]
    else:
        result = _chat_response_to_responses(
            {"choices": [{"message": {"reasoning": "plan"}, "finish_reason": "stop"}]}, body
        )
    assert result["output"][0]["summary"] == [{"type": "summary_text", "text": "plan"}]


def test_opaque_reasoning_is_not_silently_dropped():
    with pytest.raises(ValueError, match="encrypted_content"):
        _responses_to_chat_request({"input": [{"type": "reasoning", "summary": [], "encrypted_content": "opaque"}]})


def test_unsupported_encrypted_reasoning_include_is_rejected():
    with pytest.raises(ValueError, match="include"):
        _responses_to_chat_request({"input": "hi", "include": ["reasoning.encrypted_content"]})


@pytest.mark.parametrize("stream", [False, True])
def test_generation_error_never_reports_completed(stream):
    if stream:
        events = _collect_events(
            {"choices": [{"delta": {"content": "partial"}}]}, {"choices": [{"delta": {}, "finish_reason": "error"}]}
        )
        assert events[-1]["type"] == "response.failed"
        assert not any(e["type"] == "response.output_item.done" for e in events)
        result = events[-1]["response"]
    else:
        result = _chat_response_to_responses(
            {"choices": [{"message": {"content": "partial"}, "finish_reason": "error"}]}, {"input": "hi"}
        )
    assert result["status"] == "failed"
    assert result["error"] is not None
    assert all(i.get("status") != "completed" for i in result["output"])


def test_unfinished_upstream_stream_is_failed():
    events = _collect_events({"choices": [{"delta": {"content": "partial"}}]}, finish=False)
    assert events[-1]["type"] == "response.failed"


def test_interleaved_tool_arguments_are_routed_by_index():
    def call(index, **fn):
        return {"choices": [{"delta": {"tool_calls": [{"index": index, "id": f"call_{index}", "function": fn}]}}]}

    events = _collect_events(
        call(0, name="first", arguments="{"),
        call(1, name="second", arguments="{"),
        call(0, arguments='"a":1}'),
        call(1, arguments='"b":2}'),
    )
    output = events[-1]["response"]["output"]
    assert [(i["name"], i["arguments"]) for i in output] == [("first", '{"a":1}'), ("second", '{"b":2}')]
    for done in [e for e in events if e["type"] == "response.function_call_arguments.done"]:
        assert json.loads(done["arguments"])


def test_custom_tool_stream_unwraps_fragmented_unicode_input():
    payload = '中文 😀\n"quoted" \\ path'
    arguments = std_json.dumps({"input": payload}, ensure_ascii=True)
    chunks = [
        {
            "choices": [
                {
                    "delta": {
                        "tool_calls": [
                            {"index": 0, "id": "call_1", "function": {"name": "apply_patch", "arguments": ""}}
                        ]
                    }
                }
            ]
        }
    ]
    chunks += [
        {"choices": [{"delta": {"tool_calls": [{"index": 0, "function": {"arguments": ch}}]}}]} for ch in arguments
    ]
    events = _collect_events(*chunks, body={"input": "hi", "tools": [{"type": "custom", "name": "apply_patch"}]})
    assert "".join(e["delta"] for e in events if e["type"] == "response.custom_tool_call_input.delta") == payload
    assert events[-1]["response"]["output"][0]["input"] == payload


@pytest.mark.parametrize("with_call", [False, True])
def test_output_message_phase_is_set_when_item_finishes(with_call):
    payloads = [{"choices": [{"delta": {"content": "working"}}]}]
    if with_call:
        payloads.append(
            {
                "choices": [
                    {
                        "delta": {
                            "tool_calls": [
                                {"index": 0, "id": "call_1", "function": {"name": "read", "arguments": "{}"}}
                            ]
                        }
                    }
                ]
            }
        )
    events = _collect_events(*payloads)
    message = next(i for i in events[-1]["response"]["output"] if i["type"] == "message")
    assert message["phase"] == ("commentary" if with_call else "final_answer")


def test_early_stream_error_does_not_send_http_200(monkeypatch):
    from types import SimpleNamespace
    from lightllm.server import api_stream_obj

    monkeypatch.setattr(
        api_stream_obj,
        "get_env_start_args",
        lambda: SimpleNamespace(disable_delay_response_start=False, run_mode="normal"),
    )
    sent = []

    async def fail():
        raise ValueError("bad request")
        yield

    async def send(message):
        sent.append(message)

    async def run():
        response = api_stream_obj.CustomStreamingResponse(_openai_sse_to_responses_events(fail(), {"input": "hi"}))
        with pytest.raises(ValueError, match="bad request"):
            await response.stream_response(send)

    asyncio.run(run())
    assert sent == []


def test_late_stream_exception_is_failed_and_upstream_closed():
    closed = []

    async def fail():
        try:
            yield 'data: {"choices":[{"delta":{"content":"partial"}}]}\n\n'
            raise RuntimeError("worker failed")
        finally:
            closed.append(True)

    async def run():
        return [
            json.loads(e.decode().split("data: ", 1)[1])
            async for e in _openai_sse_to_responses_events(fail(), {"input": "hi"})
        ]

    events = asyncio.run(run())
    assert events[-1]["type"] == "response.failed"
    assert closed == [True]


def test_reasoning_after_final_answer_belongs_to_next_phase():
    request = _responses_to_chat_request(
        {
            "input": [
                {"role": "assistant", "phase": "final_answer", "content": "previous"},
                {"type": "reasoning", "content": [{"type": "reasoning_text", "text": "new plan"}]},
                {"role": "assistant", "phase": "commentary", "content": "working"},
            ]
        }
    )
    first, second = request["messages"]
    assert "reasoning_content" not in first
    assert second["reasoning_content"] == "new plan"
    assert second["phase"] == "commentary"


def test_legacy_extra_body_and_top_level_parameters_have_same_meaning():
    request = _responses_to_chat_request({"input": "hi", "top_k": 1, "extra_body": {"top_k": 2, "seed": 1234}})
    assert request["top_k"] == 1 and request["seed"] == 1234
    with pytest.raises(ValueError, match="strict"):
        _responses_to_chat_request(
            {"input": "hi", "extra_body": {"tools": [{"type": "function", "name": "read", "strict": True}]}}
        )


def test_colliding_namespace_and_flat_tool_names_are_rejected():
    with pytest.raises(ValueError, match="Duplicate"):
        _responses_to_chat_request(
            {
                "input": "hi",
                "tools": [
                    {"type": "function", "name": "fs__read"},
                    {"type": "namespace", "name": "fs", "tools": [{"type": "function", "name": "read"}]},
                ],
            }
        )


def test_custom_grammar_is_included_in_tool_description():
    request = _responses_to_chat_request(
        {
            "input": "hi",
            "tools": [
                {
                    "type": "custom",
                    "name": "patch",
                    "format": {"type": "grammar", "syntax": "lark", "definition": 'start: "patch"'},
                }
            ],
        }
    )
    assert 'start: "patch"' in request["tools"][0]["function"]["description"]


def test_raw_reasoning_stream_has_content_part_lifecycle():
    events = _collect_events({"choices": [{"delta": {"reasoning": "plan"}}]})
    parts = [e for e in events if e["type"] in ("response.content_part.added", "response.content_part.done")]
    assert [p["part"]["type"] for p in parts] == ["reasoning_text", "reasoning_text"]
    assert parts[0]["item_id"] == parts[1]["item_id"]


def test_streamed_namespace_tool_preserves_output_order_and_ids():
    body = {
        "input": "hi",
        "tools": [{"type": "namespace", "name": "fs", "tools": [{"type": "function", "name": "read"}]}],
    }
    events = _collect_events(
        {"choices": [{"delta": {"content": "checking"}}]},
        {
            "choices": [
                {
                    "delta": {
                        "tool_calls": [
                            {"index": 0, "id": "call_1", "function": {"name": "fs__read", "arguments": "{}"}}
                        ]
                    }
                }
            ]
        },
        body=body,
    )
    message, call = events[-1]["response"]["output"]
    assert message["phase"] == "commentary"
    assert call["namespace"] == "fs" and call["name"] == "read" and call["call_id"] == "call_1"
    assert [e["output_index"] for e in events if e["type"] == "response.output_item.added"] == [0, 1]
    assert [e["sequence_number"] for e in events] == list(range(len(events)))


@pytest.mark.parametrize("truncated", [False, True])
def test_invalid_custom_payload_is_not_completed(truncated):
    events = _collect_events(
        {
            "choices": [
                {
                    "delta": {
                        "tool_calls": [
                            {
                                "index": 0,
                                "id": "call_1",
                                "function": {"name": "patch", "arguments": '{"input":"partial'},
                            }
                        ]
                    }
                }
            ]
        },
        {"choices": [{"delta": {}, "finish_reason": "length" if truncated else "tool_calls"}]},
        body={"input": "hi", "tools": [{"type": "custom", "name": "patch"}]},
    )
    assert events[-1]["type"] == ("response.incomplete" if truncated else "response.failed")
    assert events[-1]["response"]["output"][0]["status"] == "incomplete"
    if truncated:
        assert events[-1]["response"]["output"][0]["input"] == "partial"
    else:
        assert not any(e["type"] == "response.custom_tool_call_input.done" for e in events)


def test_stream_opt_out_reports_early_busy_error_in_stream():
    from lightllm.utils.error_utils import ServerBusyError

    async def busy():
        raise ServerBusyError()
        yield

    async def run():
        return [
            json.loads(e.decode().split("data: ", 1)[1])
            async for e in _openai_sse_to_responses_events(busy(), {"input": "hi"}, delay_response_start=False)
        ]

    events = asyncio.run(run())
    assert events[0]["type"] == "response.created"
    assert events[-1]["type"] == "response.failed"
    error = events[-1]["response"]["error"]
    assert error["code"] == "rate_limit_exceeded"
    ResponseError = pytest.importorskip("openai.types.responses.response_error").ResponseError

    ResponseError.model_validate(error)


def test_closing_responses_stream_closes_prefetched_upstream():
    closed = []

    async def upstream():
        try:
            yield 'data: {"choices":[{"delta":{"content":"partial"}}]}\n\n'
            yield 'data: {"choices":[{"delta":{},"finish_reason":"stop"}]}\n\n'
        finally:
            closed.append(True)

    async def run():
        stream = _openai_sse_to_responses_events(upstream(), {"input": "hi"})
        await anext(stream)
        await stream.aclose()

    asyncio.run(run())
    assert closed == [True]


@pytest.mark.parametrize("legacy_extra_body", [False, True])
def test_route_roundtrips_custom_tool_through_chat_models(monkeypatch, legacy_extra_body):
    from types import SimpleNamespace
    from lightllm.server import api_openai, api_responses
    from lightllm.server.api_models import ChatCompletionResponse

    payload = "patch text"
    body = {
        "input": "edit",
        "top_k": 1,
        "chat_template_kwargs": {"enable_thinking": False},
        "tools": [{"type": "custom", "name": "patch"}],
    }
    if legacy_extra_body:
        body = {"input": body.pop("input"), "extra_body": body}

    async def request_json():
        return body

    async def generate(request, raw_request, *, preserve_abort=False):
        assert preserve_abort
        assert request.top_k == 1
        assert request.chat_template_kwargs["thinking"] is False
        assert request.tools[0].function.parameters["properties"]["input"]["type"] == "string"
        return ChatCompletionResponse(
            id="chat_1",
            created=1,
            model="model",
            choices=[
                {
                    "index": 0,
                    "finish_reason": "tool_calls",
                    "message": {
                        "role": "assistant",
                        "content": "",
                        "tool_calls": [
                            {"id": "call_1", "function": {"name": "patch", "arguments": json.dumps({"input": payload})}}
                        ],
                    },
                }
            ],
            usage={"prompt_tokens": 5, "completion_tokens": 6, "total_tokens": 11},
        )

    monkeypatch.setattr(api_openai, "chat_completions_impl", generate)
    result = asyncio.run(api_responses.responses_impl(SimpleNamespace(json=request_json)))
    output = json.loads(result.body)["output"]
    assert output[0]["type"] == "custom_tool_call" and output[0]["input"] == payload


@pytest.mark.parametrize("top_tools", [[], None])
def test_top_level_empty_tools_override_legacy_extra_body(top_tools):
    request = ChatCompletionRequest(
        **_responses_to_chat_request(
            {
                "input": "hi",
                "tools": top_tools,
                "extra_body": {"tools": [{"type": "function", "function": {"name": "hidden", "strict": True}}]},
            }
        )
    )
    assert not request.tools


def test_top_level_null_sampling_parameter_overrides_legacy_extra_body():
    request = _responses_to_chat_request({"input": "hi", "temperature": None, "extra_body": {"temperature": 0.2}})
    assert request["temperature"] is None


@pytest.mark.parametrize(
    "arguments",
    [
        '{"input":"first","input":"second"}',
        '{"input":"\\ud83d"}',
        '{"input":"a\\ude00b"}',
    ],
)
def test_malformed_custom_input_fails_without_completing_tool(arguments):
    events = _collect_events(
        {
            "choices": [
                {
                    "delta": {
                        "tool_calls": [
                            {"index": 0, "id": "call_1", "function": {"name": "patch", "arguments": arguments}}
                        ]
                    }
                }
            ]
        },
        body={"input": "hi", "tools": [{"type": "custom", "name": "patch"}]},
    )
    assert events[-1]["type"] == "response.failed"
    assert not any(e["type"] == "response.custom_tool_call_input.done" for e in events)
    assert not any(e["type"] == "response.output_item.done" for e in events)


@pytest.mark.parametrize(
    "arguments",
    [
        '{"\\u0069nput":"payload"}',
        ' { "in\\u0070ut" : "中文\\n😀" } ',
    ],
)
def test_escaped_custom_input_keys_preserve_streamed_payload(arguments):
    payloads = [
        {
            "choices": [
                {
                    "delta": {
                        "tool_calls": [{"index": 0, "id": "call_1", "function": {"name": "patch", "arguments": ""}}]
                    }
                }
            ]
        }
    ]
    payloads += [
        {"choices": [{"delta": {"tool_calls": [{"index": 0, "function": {"arguments": char}}]}}]} for char in arguments
    ]
    events = _collect_events(*payloads, body={"input": "hi", "tools": [{"type": "custom", "name": "patch"}]})
    streamed = "".join(e["delta"] for e in events if e["type"] == "response.custom_tool_call_input.delta")
    assert events[-1]["type"] == "response.completed"
    assert streamed == json.loads(arguments)["input"] == events[-1]["response"]["output"][0]["input"]


@pytest.mark.parametrize("stream", [False, True])
@pytest.mark.parametrize(
    "choice,parallel,names",
    [
        ("required", True, []),
        ({"type": "function", "name": "read"}, True, ["write"]),
        ({"type": "function", "name": "read"}, True, []),
        ("auto", False, ["read", "write"]),
    ],
)
def test_output_tool_constraints_are_enforced_before_completion(stream, choice, parallel, names):
    body = {
        "input": "hi",
        "tool_choice": choice,
        "parallel_tool_calls": parallel,
        "tools": [{"type": "function", "name": name, "parameters": {}} for name in ["read", "write"]],
    }
    calls = [
        {"index": i, "id": f"call_{i}", "function": {"name": name, "arguments": "{}"}} for i, name in enumerate(names)
    ]
    if stream:
        events = _collect_events(
            {
                "choices": [
                    {
                        "delta": {"content": "text", "tool_calls": calls},
                        "finish_reason": "tool_calls" if calls else "stop",
                    }
                ]
            },
            body=body,
        )
        assert events[-1]["type"] == "response.failed"
        assert not any(
            e["type"] in ("response.function_call_arguments.done", "response.custom_tool_call_input.done")
            for e in events
        )
        response = events[-1]["response"]
    else:
        response = _chat_response_to_responses(
            {
                "choices": [
                    {
                        "message": {"content": "text", "tool_calls": calls},
                        "finish_reason": "tool_calls" if calls else "stop",
                    }
                ]
            },
            body,
        )
    assert response["status"] == "failed"
    assert all(i["status"] != "completed" for i in response["output"] if i["type"] == "function_call")


def test_truncated_required_tool_request_remains_incomplete():
    events = _collect_events(
        {"choices": [{"delta": {"reasoning": "plan"}, "finish_reason": "length"}]},
        body={"input": "hi", "tool_choice": "required"},
    )
    assert events[-1]["type"] == "response.incomplete"


@pytest.fixture
def native_chat_pipeline(monkeypatch):
    from types import SimpleNamespace
    from lightllm.server import api_http, api_openai, api_responses, api_stream_obj
    from lightllm.server.core.objs import FinishStatus

    class Sampling:
        def init(self, **kwargs):
            pass

        async def verify_cpu_async(self, compiler):
            pass

    args = SimpleNamespace(
        reasoning_parser=None, tool_call_parser="deepseekv4", disable_delay_response_start=False, run_mode="normal"
    )
    monkeypatch.setattr(api_openai, "SamplingParams", Sampling)

    async def prompt(request, tools):
        return "Prompt"

    monkeypatch.setattr(api_openai, "build_prompt", prompt)
    for module in [api_openai, api_responses, api_stream_obj]:
        monkeypatch.setattr(module, "get_env_start_args", lambda: args)
    monkeypatch.setattr(api_http.g_objs, "args", args, raising=False)
    monkeypatch.setattr(api_http.g_objs, "metric_client", SimpleNamespace(counter_inc=lambda *a: None))
    closed = []
    generators = []

    def manager_generate(finish=None, output="text"):
        async def generate(*a, **kw):
            try:
                yield 80, output if finish != "abort" else "", {"prompt_tokens": 4}, FinishStatus(
                    {
                        None: FinishStatus.NO_FINISH,
                        "abort": FinishStatus.FINISHED_ABORTED,
                        "stop": FinishStatus.FINISHED_STOP,
                        "error": FinishStatus.FINISHED_ERROR,
                    }[finish]
                )
                if finish is None:
                    await asyncio.Event().wait()
            finally:
                await asyncio.sleep(0)
                closed.append(True)

        def start(*a, **kw):
            generator = generate(*a, **kw)
            generators.append(generator)
            return generator

        monkeypatch.setattr(
            api_http.g_objs,
            "httpserver_manager",
            SimpleNamespace(tokenizer=None, output_grammar_compiler=None, generate=start),
            raising=False,
        )

    return SimpleNamespace(set_finish=manager_generate, closed=closed, generators=generators, args=args)


@pytest.mark.parametrize("stream", [False, True])
@pytest.mark.parametrize("duplicate", [False, True])
@pytest.mark.parametrize("implicit", [False, True])
@pytest.mark.parametrize("value", ["hello", "<think>literal</think>"])
def test_native_dsv4_implicit_tool_exit_and_parameter_errors(
    native_chat_pipeline, monkeypatch, stream, duplicate, implicit, value
):
    from types import SimpleNamespace
    from lightllm.server.api_responses import responses_impl
    from lightllm.utils.error_utils import GenerationError

    native_chat_pipeline.args.reasoning_parser = "deepseek-v4"
    monkeypatch.setattr("lightllm.server.reasoning_parser.get_token_id", lambda token: 99)
    parameter = f'<｜DSML｜parameter name="value" string="true">{value}</｜DSML｜parameter>\n'
    output = (
        ("plan\n\n" if implicit else "plan</think>\n\n")
        + '<｜DSML｜tool_calls>\n<｜DSML｜invoke name="lookup">\n'
        + parameter * (2 if duplicate else 1)
        + "</｜DSML｜invoke>\n</｜DSML｜tool_calls>"
    )
    native_chat_pipeline.set_finish("stop", output)

    async def request_json():
        return {"input": "hi", "stream": stream, "tools": [{"type": "function", "name": "lookup", "parameters": {}}]}

    async def run():
        if duplicate and not stream:
            with pytest.raises(GenerationError, match="Duplicate"):
                await responses_impl(SimpleNamespace(json=request_json))
            return
        response = await responses_impl(SimpleNamespace(json=request_json))
        if stream:
            events = [json.loads(chunk.decode().split("data: ", 1)[1]) async for chunk in response.body_iterator]
            result = events[-1]["response"]
        else:
            result = json.loads(response.body)
        if duplicate:
            assert result["status"] == "failed"
            assert not any(event["type"] == "response.function_call_arguments.done" for event in events)
        else:
            assert result["status"] == "completed"
            calls = [item for item in result["output"] if item["type"] == "function_call"]
            assert len(calls) == 1 and calls[0]["name"] == "lookup"
            assert json.loads(calls[0]["arguments"]) == {"value": value}
            thought = next(item for item in result["output"] if item["type"] == "reasoning")
            assert thought["content"][0]["text"].rstrip() == "plan"

    asyncio.run(run())
    assert native_chat_pipeline.closed == [True]


@pytest.mark.parametrize("stream", [False, True])
@pytest.mark.parametrize("finish", ["abort", "error"])
def test_native_chat_preserves_failure_for_responses(native_chat_pipeline, stream, finish):
    from types import SimpleNamespace
    from lightllm.server.api_responses import responses_impl
    from lightllm.utils.error_utils import GenerationError

    native_chat_pipeline.set_finish(finish)

    async def request_json():
        return {"input": "hi", "stream": stream}

    async def run():
        response = await responses_impl(SimpleNamespace(json=request_json))
        if stream:
            if finish == "error":
                with pytest.raises(GenerationError, match="before producing output"):
                    _ = [chunk async for chunk in response.body_iterator]
            else:
                events = [json.loads(e.decode().split("data: ", 1)[1]) async for e in response.body_iterator]
                assert events[-1]["type"] == "response.failed"
        else:
            assert json.loads(response.body)["status"] == "failed"

    asyncio.run(run())
    assert native_chat_pipeline.closed == [True]


def test_plain_chat_abort_compatibility_is_preserved(native_chat_pipeline):
    from types import SimpleNamespace
    from lightllm.server.api_openai import chat_completions_impl

    native_chat_pipeline.set_finish("abort")
    response = asyncio.run(
        chat_completions_impl(ChatCompletionRequest(messages=[], max_completion_tokens=32), SimpleNamespace())
    )
    assert response.choices[0].finish_reason == "stop"


@pytest.mark.parametrize("early_headers", [False, True])
@pytest.mark.parametrize("cancel_during_send", [False, True])
def test_native_responses_closes_backend_before_http_stream_exits(
    native_chat_pipeline, early_headers, cancel_during_send
):
    import anyio
    from types import SimpleNamespace
    from lightllm.server.api_responses import responses_impl

    native_chat_pipeline.set_finish(None)
    native_chat_pipeline.args.disable_delay_response_start = early_headers

    async def request_json():
        return {"input": "hi", "stream": True}

    async def run():
        response = await responses_impl(SimpleNamespace(json=request_json))
        with anyio.CancelScope() as scope:

            async def send(message):
                if message["type"] == "http.response.body":
                    if cancel_during_send:
                        scope.cancel()
                        await anyio.sleep(0)
                    raise RuntimeError("send failed")

            if cancel_during_send:
                await response.stream_response(send)
            else:
                with pytest.raises(RuntimeError, match="send failed"):
                    await response.stream_response(send)
        assert native_chat_pipeline.closed == ([True] if not early_headers else [])
        # With immediate headers, no backend has been started yet.
        if not early_headers:
            assert all(g.ag_frame is None for g in native_chat_pipeline.generators)

    asyncio.run(run())


def test_native_responses_closes_started_backend_with_immediate_headers(native_chat_pipeline):
    import anyio
    from types import SimpleNamespace
    from lightllm.server.api_responses import responses_impl

    native_chat_pipeline.set_finish(None)
    native_chat_pipeline.args.disable_delay_response_start = True

    async def request_json():
        return {"input": "hi", "stream": True}

    async def run():
        response = await responses_impl(SimpleNamespace(json=request_json))
        with anyio.CancelScope() as scope:

            async def send(message):
                if message["type"] == "http.response.body" and b"response.output_text.delta" in message["body"]:
                    scope.cancel()
                    await anyio.sleep(0)

            await response.stream_response(send)
        assert native_chat_pipeline.closed == [True]

    asyncio.run(run())
