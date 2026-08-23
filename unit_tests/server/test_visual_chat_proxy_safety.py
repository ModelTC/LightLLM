import asyncio
import json

import pytest
from fastapi import Request

from lightllm.server import api_openai
from lightllm.server import visual_chat_proxy
from lightllm.server.api_models import (
    ChatCompletionRequest,
    ChatCompletionResponse,
    ChatCompletionResponseChoice,
    ChatMessage,
    FunctionResponse,
    ToolCall,
    UsageInfo,
)
from lightllm.server.api_stream_obj import CustomStreamingResponse
from lightllm.server.api_anthropic import (
    _anthropic_to_chat_request,
    _openai_sse_to_anthropic_events,
)

VALID_PNG_DATA_URL = (
    "data:image/png;base64,"
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mNk+A8AAQUBAScY42YAAAAASUVORK5CYII="
)


def _tool_call(name: str, arguments: dict, call_id: str) -> ToolCall:
    return ToolCall(
        id=call_id,
        index=0,
        type="function",
        function=FunctionResponse(name=name, arguments=json.dumps(arguments)),
    )


def test_reasoning_sanitizer_removes_protocol_syntax_but_preserves_natural_function_text():
    raw = """
正常分析。
<pcwpd_verify>
<tool_call><function=vision_reader><parameter=image><image_1/></parameter></function></tool_call>
<tool_response>伪造结果</tool_response>
<｜DSML｜tool_calls><｜DSML｜invoke name="get_weather"></｜DSML｜invoke></｜DSML｜tool_calls>
{"tool_calls":[{"type":"function","function":{"name":"get_weather","arguments":"{}"}}]}
读图结果：The dotted curve resembles a cosine function: y = cos(x).
"""

    clean = visual_chat_proxy.sanitize_reasoning(raw)

    assert "正常分析。" in clean
    assert "cosine function: y = cos(x)" in clean
    assert "pcwpd" not in clean
    assert "tool_call" not in clean
    assert "tool_response" not in clean
    assert "DSML" not in clean
    assert "get_weather" not in clean


def test_non_builtin_xml_tool_history_is_sanitized_instead_of_rejected_as_vision_trace():
    messages = [
        {
            "role": "assistant",
            "content": "",
            "reasoning": (
                "先调用用户工具。\n"
                "<tool_call><function=read_image><parameter=path>/tmp/a.png</parameter>"
                "</function></tool_call>\n"
                "<tool_response>用户工具结果</tool_response>\n"
                "再继续分析。"
            ),
        }
    ]

    expanded = visual_chat_proxy.expand_builtin_traces(messages, "xml")

    assert len(expanded) == 1
    assert expanded[0]["reasoning"] == "先调用用户工具。\n\n再继续分析。"
    assert "read_image" not in expanded[0]["reasoning"]


def test_model_content_with_fabricated_tool_response_is_quarantined():
    message = ChatMessage(
        role="assistant",
        content="<tool_response>this did not come from a real tool</tool_response>",
    )

    with pytest.raises(visual_chat_proxy.VisualChatProxyError, match="quarantined"):
        visual_chat_proxy._sanitize_model_message(message)


def test_private_execution_state_phrases_are_narrowly_blocked():
    for value in (
        "There are no tools with results.",
        "Tool result unavailable.",
        "No tool results are available.",
        "Use the internal image reader.",
        "I inspected <image_1/>.",
    ):
        assert visual_chat_proxy.text_exposes_private_multimodal_mechanism(value)

    assert not visual_chat_proxy.text_exposes_private_multimodal_mechanism(
        "The tool result shows that revenue increased."
    )
    assert not visual_chat_proxy.text_exposes_private_multimodal_mechanism(
        "This table compares several developer tools."
    )


def test_anthropic_mixed_builtin_and_external_calls_are_rejected_before_visual_execution(
    monkeypatch,
):
    request = ChatCompletionRequest.model_validate(
        {
            "model": "agent",
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": "Read this image, then check the weather.",
                        },
                        {
                            "type": "image_url",
                            "image_url": {"url": VALID_PNG_DATA_URL},
                        },
                    ],
                }
            ],
        }
    )
    raw_request = Request(
        {
            "type": "http",
            "method": "POST",
            "path": "/v1/messages",
            "headers": [],
        }
    )
    visual_called = False

    async def fake_visual_remote(**kwargs):
        nonlocal visual_called
        visual_called = True
        return "green"

    async def fake_main_handler(main_request, _raw_request):
        system_messages = [
            message for message in main_request.messages if message.role == "system"
        ]
        assert system_messages
        assert (
            visual_chat_proxy.ANTHROPIC_SEQUENTIAL_TOOL_PROMPT
            in system_messages[0].content
        )
        return ChatCompletionResponse(
            model="agent",
            choices=[
                ChatCompletionResponseChoice(
                    index=0,
                    finish_reason="tool_calls",
                    message=ChatMessage(
                        role="assistant",
                        content="",
                        tool_calls=[
                            _tool_call(
                                "vision_reader",
                                {"image": "<image_1/>", "task": "read the image"},
                                "builtin_1",
                            ),
                            _tool_call(
                                "get_weather", {"city": "Shanghai"}, "external_1"
                            ),
                        ],
                    ),
                )
            ],
            usage=UsageInfo(prompt_tokens=1, completion_tokens=1, total_tokens=2),
        )

    monkeypatch.setattr(visual_chat_proxy, "call_visual_remote", fake_visual_remote)
    runtime = visual_chat_proxy.VisualProxyRuntime(
        visual_chat_proxy.VisualProxySettings(
            remote_url="http://127.0.0.1:18180/generate"
        )
    )

    async def run_test():
        try:
            with pytest.raises(
                visual_chat_proxy.VisualChatProxyError,
                match="mixed builtin and external",
            ):
                await visual_chat_proxy.visual_chat_completions_impl(
                    request=request,
                    raw_request=raw_request,
                    runtime=runtime,
                    main_chat_handler=fake_main_handler,
                )
        finally:
            await runtime.close()

    asyncio.run(run_test())

    assert not visual_called


def test_anthropic_exposes_private_reader_then_caller_tools_in_separate_phases(
    monkeypatch,
):
    request = ChatCompletionRequest.model_validate(
        {
            "model": "agent",
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "Read the image, then check weather."},
                        {"type": "image_url", "image_url": {"url": VALID_PNG_DATA_URL}},
                    ],
                }
            ],
            "tools": [
                {
                    "type": "function",
                    "function": {
                        "name": "get_weather",
                        "description": "Get weather.",
                        "parameters": {
                            "type": "object",
                            "properties": {"city": {"type": "string"}},
                        },
                    },
                }
            ],
        }
    )
    raw_request = Request(
        {"type": "http", "method": "POST", "path": "/v1/messages", "headers": []}
    )
    calls = 0

    async def fake_visual_remote(**_kwargs):
        return "The image shows Shanghai."

    async def fake_main_handler(main_request, _raw_request):
        nonlocal calls
        calls += 1
        tool_names = [tool.function.name for tool in main_request.tools or []]
        if calls == 1:
            assert tool_names == ["vision_reader"]
            message = ChatMessage(
                role="assistant",
                content="",
                tool_calls=[
                    _tool_call(
                        "vision_reader",
                        {"image": "<image_1/>", "task": "identify the city"},
                        "builtin_1",
                    )
                ],
            )
        else:
            assert tool_names == ["get_weather"]
            message = ChatMessage(
                role="assistant",
                content="",
                tool_calls=[
                    _tool_call("get_weather", {"city": "Shanghai"}, "external_1")
                ],
            )
        return ChatCompletionResponse(
            model="agent",
            choices=[
                ChatCompletionResponseChoice(
                    index=0,
                    finish_reason="tool_calls",
                    message=message,
                )
            ],
            usage=UsageInfo(prompt_tokens=1, completion_tokens=1, total_tokens=2),
        )

    monkeypatch.setattr(visual_chat_proxy, "call_visual_remote", fake_visual_remote)
    runtime = visual_chat_proxy.VisualProxyRuntime(
        visual_chat_proxy.VisualProxySettings(
            remote_url="http://127.0.0.1:18180/generate"
        )
    )

    async def run_test():
        try:
            return await visual_chat_proxy.visual_chat_completions_impl(
                request=request,
                raw_request=raw_request,
                runtime=runtime,
                main_chat_handler=fake_main_handler,
            )
        finally:
            await runtime.close()

    response = asyncio.run(run_test())

    assert calls == 2
    assert response.choices[0].finish_reason == "tool_calls"
    assert response.choices[0].message.tool_calls[0].function.name == "get_weather"


def test_visual_trace_format_defaults_to_natural():
    settings = visual_chat_proxy.VisualProxySettings(
        remote_url="http://127.0.0.1:18180/generate"
    )

    assert settings.builtin_trace_format == "natural"


def test_visual_request_matches_nova_formal_profile():
    captured = {}

    class FakeRuntime:
        settings = visual_chat_proxy.VisualProxySettings(
            remote_url="http://127.0.0.1:18180/v1/chat/completions",
            remote_model="vision-model",
        )

        def ensure_remote_available(self):
            return None

        def record_remote_success(self):
            return None

        def record_remote_failure(self, _trace_id):
            return None

        async def post_json(self, payload, raw_request, trace_id):
            captured.update(payload)
            return {
                "choices": [
                    {"message": {"content": "The title is LightLLM."}}
                ]
            }

    image = visual_chat_proxy.RegisteredImage(
        tag="<image_1/>",
        source=VALID_PNG_DATA_URL,
        origin="user",
    )

    result = asyncio.run(
        visual_chat_proxy.call_visual_remote(
            runtime=FakeRuntime(),
            model="agent-model",
            image=image,
            task="read the title",
            trace_id="nova-profile",
        )
    )

    assert result == "The title is LightLLM."
    assert captured == {
        "model": "vision-model",
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "read the title"},
                    {
                        "type": "image_url",
                        "image_url": {"url": VALID_PNG_DATA_URL},
                    },
                ],
            }
        ],
        "max_tokens": 8192,
        "temperature": 0.0,
        "chat_template_kwargs": {"enable_thinking": False},
    }


def test_visual_response_falls_back_to_clean_reasoning_like_nova():
    response = {
        "choices": [
            {
                "message": {
                    "content": None,
                    "reasoning_content": "<|im_start|>The chart peaks in June.<|im_end|>",
                }
            }
        ]
    }

    assert visual_chat_proxy._remote_content(response) == "The chart peaks in June."


def test_visual_stream_assembles_same_result_and_requests_streaming():
    captured = {}
    streamed = []

    class FakeRuntime:
        settings = visual_chat_proxy.VisualProxySettings(
            remote_url="http://127.0.0.1:18180/v1/chat/completions",
            remote_model="vision-model",
        )

        def ensure_remote_available(self):
            return None

        def record_remote_success(self):
            return None

        def record_remote_failure(self, _trace_id):
            return None

        async def stream_json(self, payload, raw_request, trace_id):
            captured.update(payload)
            yield {"choices": [{"delta": {"content": "  The title "}}]}
            assert "".join(streamed) == "The title"
            yield {"choices": [{"delta": {"content": "is LightLLM.<|im_"}}]}
            yield {"choices": [{"delta": {"content": "end|>  "}}]}

    async def collect(value):
        streamed.append(value)

    image = visual_chat_proxy.RegisteredImage(
        tag="<image_1/>",
        source=VALID_PNG_DATA_URL,
        origin="user",
    )
    result = asyncio.run(
        visual_chat_proxy.call_visual_remote(
            runtime=FakeRuntime(),
            model="agent-model",
            image=image,
            task="read the title",
            trace_id="visual-stream",
            result_callback=collect,
        )
    )

    assert result == "The title is LightLLM."
    assert "".join(streamed) == result
    assert captured["stream"] is True
    assert captured["messages"][0]["content"][0] == {
        "type": "text",
        "text": "read the title",
    }


def test_visual_stream_buffers_reasoning_fallback_until_content_priority_is_known():
    streamed = []

    class FakeRuntime:
        settings = visual_chat_proxy.VisualProxySettings(
            remote_url="http://127.0.0.1:18180/v1/chat/completions"
        )

        def ensure_remote_available(self):
            return None

        def record_remote_success(self):
            return None

        def record_remote_failure(self, _trace_id):
            return None

        async def stream_json(self, payload, raw_request, trace_id):
            yield {"choices": [{"delta": {"reasoning_content": "fallback "}}]}
            assert streamed == []
            yield {"choices": [{"delta": {"reasoning_content": "answer"}}]}

    async def collect(value):
        streamed.append(value)

    result = asyncio.run(
        visual_chat_proxy.call_visual_remote(
            runtime=FakeRuntime(),
            model="agent-model",
            image=visual_chat_proxy.RegisteredImage(
                tag="<image_1/>",
                source=VALID_PNG_DATA_URL,
                origin="user",
            ),
            task="inspect",
            trace_id="visual-reasoning-stream",
            result_callback=collect,
        )
    )

    assert result == "fallback answer"
    assert "".join(streamed) == result


def test_builtin_tool_definition_matches_nova_profile():
    function = visual_chat_proxy.BUILTIN_VISION_READER_TOOL["function"]

    assert function["description"] == (
        "BUILTIN tag-based vision reader. Call it before answering anything that depends on an image, "
        "screenshot, chart, table, UI, OCR, object count, color, position, layout, or PDF appearance. "
        "The image argument must be an exact <image_n/> tag already visible in the conversation, never a "
        "path, URL, file URI, or base64 value."
    )
    assert function["parameters"]["properties"]["image"]["description"] == (
        "An exact conversation image tag such as <image_1/>."
    )
    assert function["parameters"]["properties"]["task"]["description"] == (
        "The visual inspection task or question."
    )


def test_explicit_legacy_xml_trace_remains_replayable_at_public_boundary():
    trace = visual_chat_proxy._format_xml_builtin_trace(
        "<image_1/>", "read the title", "The title is LightLLM."
    )

    message = visual_chat_proxy._message_with_reasoning(
        ChatMessage(role="assistant", content="done"), [trace]
    )

    assert "<function=vision_reader>" in message.reasoning
    assert "<tool_response>" in message.reasoning


@pytest.mark.parametrize("effort", ["none", "off", "disabled"])
def test_openai_disabled_effort_aliases_turn_thinking_off(effort):
    request = ChatCompletionRequest(
        model="agent",
        messages=[{"role": "user", "content": "hi"}],
        reasoning_effort=effort,
    )
    settings = visual_chat_proxy.VisualProxySettings(
        remote_url="http://127.0.0.1:18180/generate"
    )

    resolved = visual_chat_proxy.apply_visual_thinking_policy(request, settings)

    assert resolved.chat_template_kwargs["enable_thinking"] is False
    assert resolved.reasoning_effort == "none"


@pytest.mark.parametrize("effort", ["minimal", "low", "medium", "high", "max", "xhigh"])
def test_openai_enabled_effort_levels_turn_thinking_on(effort):
    request = ChatCompletionRequest(
        model="agent",
        messages=[{"role": "user", "content": "hi"}],
        reasoning_effort=effort,
    )
    settings = visual_chat_proxy.VisualProxySettings(
        remote_url="http://127.0.0.1:18180/generate"
    )

    resolved = visual_chat_proxy.apply_visual_thinking_policy(request, settings)

    assert resolved.chat_template_kwargs["enable_thinking"] is True
    assert resolved.reasoning_effort == effort


def test_visual_proxy_buffers_private_turns_and_projects_only_verified_results(monkeypatch):
    first_turn_gate = asyncio.Event()
    second_turn_gate = asyncio.Event()
    main_requests = []

    def sse(payload):
        return f"data: {json.dumps(payload)}\n\n"

    async def first_turn():
        yield sse(
            {
                "choices": [
                    {
                        "delta": {"role": "assistant", "content": ""},
                        "finish_reason": None,
                    }
                ]
            }
        )
        yield sse(
            {
                "choices": [
                    {
                        "delta": {"reasoning": "I am inspecting the image now. " * 4},
                        "finish_reason": None,
                    }
                ]
            }
        )
        await first_turn_gate.wait()
        yield sse(
            {
                "choices": [
                    {
                        "delta": {
                            "tool_calls": [
                                {
                                    "index": 0,
                                    "id": "builtin_1",
                                    "type": "function",
                                    "function": {
                                        "name": "vision_reader",
                                        "arguments": '{"image":"<image_1/>",',
                                    },
                                }
                            ]
                        },
                        "finish_reason": None,
                    }
                ]
            }
        )
        yield sse(
            {
                "choices": [
                    {
                        "delta": {
                            "tool_calls": [
                                {
                                    "index": 0,
                                    "function": {
                                        "arguments": '"task":"read the title"}'
                                    },
                                }
                            ]
                        },
                        "finish_reason": "tool_calls",
                    }
                ]
            }
        )
        yield sse(
            {
                "choices": [],
                "usage": {
                    "prompt_tokens": 3,
                    "completion_tokens": 2,
                    "total_tokens": 5,
                },
            }
        )
        yield "data: [DONE]\n\n"

    async def second_turn():
        yield sse(
            {
                "choices": [
                    {
                        "delta": {
                            "reasoning": "The visual evidence is sufficient. " * 3
                        },
                        "finish_reason": None,
                    }
                ]
            }
        )
        yield sse(
            {
                "choices": [
                    {
                        "delta": {"content": "The title is LightLLM."},
                        "finish_reason": "stop",
                    }
                ]
            }
        )
        await second_turn_gate.wait()
        yield sse(
            {
                "choices": [],
                "usage": {
                    "prompt_tokens": 5,
                    "completion_tokens": 3,
                    "total_tokens": 8,
                },
            }
        )
        yield "data: [DONE]\n\n"

    async def fake_main(request, _raw_request):
        main_requests.append(request)
        assert request.stream_options.include_usage is True
        iterator = first_turn() if len(main_requests) == 1 else second_turn()
        return CustomStreamingResponse(iterator, media_type="text/event-stream")

    async def fake_visual_remote(**kwargs):
        callback = kwargs.get("result_callback")
        if callback is not None:
            await callback("The title ")
            await callback("is LightLLM.")
        return "The title is LightLLM."

    monkeypatch.setattr(visual_chat_proxy, "call_visual_remote", fake_visual_remote)
    request = ChatCompletionRequest.model_validate(
        {
            "model": "agent",
            "stream": True,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "Read the title."},
                        {
                            "type": "image_url",
                            "image_url": {"url": VALID_PNG_DATA_URL},
                        },
                    ],
                }
            ],
        }
    )
    raw_request = Request(
        {
            "type": "http",
            "method": "POST",
            "path": "/v1/chat/completions",
            "headers": [],
        }
    )
    runtime = visual_chat_proxy.VisualProxyRuntime(
        visual_chat_proxy.VisualProxySettings(
            remote_url="http://127.0.0.1:18180/generate"
        ),
        client=object(),
    )

    async def run_test():
        response = await visual_chat_proxy.visual_chat_completions_impl(
            request=request,
            raw_request=raw_request,
            runtime=runtime,
            main_chat_handler=fake_main,
        )
        assert main_requests == []  # The streaming agent loop is lazy.
        assert (
            runtime._request_semaphore._value
            == runtime.settings.max_inflight_requests - 1
        )
        iterator = response.body_iterator.__aiter__()
        chunks = [await iterator.__anext__()]
        assert chunks == [": nova-stream-open\n\n"]
        chunks.append(await asyncio.wait_for(iterator.__anext__(), timeout=1))
        assert not any('"reasoning"' in str(chunk) for chunk in chunks)
        assert not first_turn_gate.is_set()
        assert len(main_requests) == 1
        assert main_requests[0].stream is True
        first_turn_gate.set()
        for _ in range(100):
            if len(main_requests) == 2:
                break
            await asyncio.sleep(0.01)
        assert len(main_requests) == 2
        assert not second_turn_gate.is_set()
        second_turn_gate.set()
        async for chunk in iterator:
            chunks.append(chunk)
        return "".join(
            chunk.decode() if isinstance(chunk, bytes) else chunk for chunk in chunks
        )

    body = asyncio.run(run_test())

    assert "I am inspecting the image now." in body
    assert "The visual evidence is sufficient." in body
    assert "我先查看了图片 <image_1/>，The title is LightLLM." in body
    assert "read the title" not in body
    assert '"content": "The title is LightLLM."' in body
    assert '"name": "vision_reader"' not in body
    assert body.endswith("data: [DONE]\n\n")
    assert len(main_requests) == 2
    assert all(item.stream is True for item in main_requests)
    tool_results = [
        message.content
        for message in main_requests[1].messages
        if message.role == "tool" and message.name == "vision_reader"
    ]
    assert tool_results == ["The title is LightLLM."]
    assert runtime._request_semaphore._value == runtime.settings.max_inflight_requests

    payloads = [
        json.loads(line[len("data: ") :])
        for line in body.splitlines()
        if line.startswith("data: {")
    ]
    streamed_reasoning = "".join(
        choice.get("delta", {}).get("reasoning", "")
        for payload in payloads
        for choice in payload.get("choices", [])
    )
    streamed_content = "".join(
        choice.get("delta", {}).get("content", "")
        for payload in payloads
        for choice in payload.get("choices", [])
    )
    expected_reasoning = "\n".join(
        [
            visual_chat_proxy.sanitize_reasoning("I am inspecting the image now. " * 4),
            visual_chat_proxy._format_public_builtin_projection(
                "<image_1/>", "The title is LightLLM.", 0
            ),
            visual_chat_proxy.sanitize_reasoning(
                "The visual evidence is sufficient. " * 3
            ),
        ]
    )
    assert streamed_reasoning == expected_reasoning
    assert streamed_content == "The title is LightLLM."


def test_empty_output_retry_does_not_replay_rejected_assistant_turn(monkeypatch):
    main_requests = []

    async def fake_main(request, _raw_request):
        main_requests.append(request)
        if len(main_requests) == 1:
            return ChatCompletionResponse(
                model="agent",
                choices=[
                    ChatCompletionResponseChoice(
                        index=0,
                        finish_reason="stop",
                        message=ChatMessage(
                            role="assistant",
                            content="",
                            reasoning="discarded retry reasoning",
                        ),
                    )
                ],
                usage=UsageInfo(prompt_tokens=1, completion_tokens=1, total_tokens=2),
            )
        if len(main_requests) == 2:
            return ChatCompletionResponse(
                model="agent",
                choices=[
                    ChatCompletionResponseChoice(
                        index=0,
                        finish_reason="tool_calls",
                        message=ChatMessage(
                            role="assistant",
                            content="",
                            tool_calls=[
                                _tool_call(
                                    "vision_reader",
                                    {"image": "<image_1/>", "task": "inspect the image"},
                                    "builtin_1",
                                )
                            ],
                        ),
                    )
                ],
                usage=UsageInfo(prompt_tokens=1, completion_tokens=1, total_tokens=2),
            )
        return ChatCompletionResponse(
            model="agent",
            choices=[
                ChatCompletionResponseChoice(
                    index=0,
                    finish_reason="stop",
                    message=ChatMessage(
                        role="assistant",
                        content="done",
                        reasoning="accepted reasoning",
                    ),
                )
            ],
            usage=UsageInfo(prompt_tokens=1, completion_tokens=1, total_tokens=2),
        )

    async def fake_visual_remote(**_kwargs):
        return "No relevant visual content."

    monkeypatch.setattr(visual_chat_proxy, "call_visual_remote", fake_visual_remote)

    request = ChatCompletionRequest.model_validate(
        {
            "model": "agent",
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "Do not inspect this image."},
                        {
                            "type": "image_url",
                            "image_url": {"url": VALID_PNG_DATA_URL},
                        },
                    ],
                }
            ],
        }
    )
    raw_request = Request(
        {"type": "http", "method": "POST", "path": "/v1/chat/completions", "headers": []}
    )
    runtime = visual_chat_proxy.VisualProxyRuntime(
        visual_chat_proxy.VisualProxySettings(remote_url="http://127.0.0.1:18180/generate"),
        client=object(),
    )

    response = asyncio.run(
        visual_chat_proxy.visual_chat_completions_impl(
            request=request,
            raw_request=raw_request,
            runtime=runtime,
            main_chat_handler=fake_main,
        )
    )

    second_history = [message.model_dump(exclude_none=True) for message in main_requests[1].messages]
    assert not any("discarded retry reasoning" in json.dumps(message) for message in second_history)
    assert second_history[-1] == visual_chat_proxy._build_empty_output_retry_feedback()
    assert "discarded retry reasoning" not in (response.choices[0].message.reasoning or "")
    assert "accepted reasoning" in (response.choices[0].message.reasoning or "")


def test_visual_guardrail_does_not_replay_rejected_assistant_turn(monkeypatch):
    main_requests = []

    async def fake_main(request, _raw_request):
        main_requests.append(request)
        if len(main_requests) == 1:
            return ChatCompletionResponse(
                model="agent",
                choices=[
                    ChatCompletionResponseChoice(
                        index=0,
                        finish_reason="stop",
                        message=ChatMessage(
                            role="assistant",
                            content="The image is red.",
                            reasoning="discarded visual guess",
                        ),
                    )
                ],
                usage=UsageInfo(prompt_tokens=1, completion_tokens=1, total_tokens=2),
            )
        if len(main_requests) == 2:
            return ChatCompletionResponse(
                model="agent",
                choices=[
                    ChatCompletionResponseChoice(
                        index=0,
                        finish_reason="tool_calls",
                        message=ChatMessage(
                            role="assistant",
                            content="",
                            tool_calls=[
                                _tool_call(
                                    "vision_reader",
                                    {"image": "<image_1/>", "task": "identify the color"},
                                    "builtin_1",
                                )
                            ],
                        ),
                    )
                ],
                usage=UsageInfo(prompt_tokens=1, completion_tokens=1, total_tokens=2),
            )
        return ChatCompletionResponse(
            model="agent",
            choices=[
                ChatCompletionResponseChoice(
                    index=0,
                    finish_reason="stop",
                    message=ChatMessage(role="assistant", content="The image is blue."),
                )
            ],
            usage=UsageInfo(prompt_tokens=1, completion_tokens=1, total_tokens=2),
        )

    async def fake_visual_remote(**_kwargs):
        return "The image is blue."

    monkeypatch.setattr(visual_chat_proxy, "call_visual_remote", fake_visual_remote)
    request = ChatCompletionRequest.model_validate(
        {
            "model": "agent",
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "What color is this image?"},
                        {
                            "type": "image_url",
                            "image_url": {"url": VALID_PNG_DATA_URL},
                        },
                    ],
                }
            ],
        }
    )
    raw_request = Request(
        {"type": "http", "method": "POST", "path": "/v1/chat/completions", "headers": []}
    )
    runtime = visual_chat_proxy.VisualProxyRuntime(
        visual_chat_proxy.VisualProxySettings(remote_url="http://127.0.0.1:18180/generate"),
        client=object(),
    )

    response = asyncio.run(
        visual_chat_proxy.visual_chat_completions_impl(
            request=request,
            raw_request=raw_request,
            runtime=runtime,
            main_chat_handler=fake_main,
        )
    )

    second_history = [message.model_dump(exclude_none=True) for message in main_requests[1].messages]
    assert not any("discarded visual guess" in json.dumps(message) for message in second_history)
    assert not any("The image is red." in json.dumps(message) for message in second_history[:-1])
    assert second_history[-1] == visual_chat_proxy._build_output_guardrail_feedback(
        ["<image_1/>"], "The image is red."
    )
    assert "discarded visual guess" not in (response.choices[0].message.reasoning or "")
    assert response.choices[0].message.content == "The image is blue."


def test_anthropic_stream_maps_proxy_reasoning_to_thinking_deltas():
    async def openai_stream():
        yield 'data: {"choices":[{"delta":{"reasoning":"读图结果：标题是 LightLLM。"},"finish_reason":null}]}\n\n'
        yield 'data: {"choices":[{"delta":{"content":"最终答案"},"finish_reason":"stop"}]}\n\n'
        yield 'data: {"choices":[],"usage":{"prompt_tokens":3,"completion_tokens":2,"total_tokens":5}}\n\n'
        yield "data: [DONE]\n\n"

    async def collect():
        return [
            event.decode()
            async for event in _openai_sse_to_anthropic_events(
                openai_stream(), requested_model="agent", message_id="msg_test"
            )
        ]

    body = "".join(asyncio.run(collect()))

    assert '"type":"thinking"' in body
    assert '"type":"thinking_delta"' in body
    assert json.dumps("读图结果：标题是 LightLLM。")[1:-1] in body
    assert '"type":"text_delta"' in body
    assert json.dumps("最终答案")[1:-1] in body


def test_closing_visual_stream_cancels_generation_and_releases_request_slot():
    cancellation_count = 0
    next_request_id = 10_000
    active_request_ids = set()
    aborted_request_ids = []

    async def endless_stream(request_id):
        nonlocal cancellation_count
        active_request_ids.add(request_id)
        try:
            yield 'data: {"choices":[{"delta":{"reasoning":"working"},"finish_reason":null}]}\n\n'
            await asyncio.Event().wait()
        finally:
            cancellation_count += 1
            active_request_ids.discard(request_id)
            aborted_request_ids.append(request_id)

    async def fake_main(_request, _raw_request):
        nonlocal next_request_id
        request_id = next_request_id
        next_request_id += 1
        return CustomStreamingResponse(endless_stream(request_id), media_type="text/event-stream")

    request = ChatCompletionRequest.model_validate(
        {
            "model": "agent",
            "stream": True,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "Inspect this."},
                        {
                            "type": "image_url",
                            "image_url": {"url": VALID_PNG_DATA_URL},
                        },
                    ],
                }
            ],
        }
    )
    raw_request = Request(
        {
            "type": "http",
            "method": "POST",
            "path": "/v1/chat/completions",
            "headers": [],
        }
    )
    runtime = visual_chat_proxy.VisualProxyRuntime(
        visual_chat_proxy.VisualProxySettings(
            remote_url="http://127.0.0.1:18180/generate"
        ),
        client=object(),
    )

    async def run_test():
        for _ in range(25):
            response = await visual_chat_proxy.visual_chat_completions_impl(
                request=request,
                raw_request=raw_request,
                runtime=runtime,
                main_chat_handler=fake_main,
            )
            iterator = response.body_iterator.__aiter__()
            await iterator.__anext__()
            await iterator.aclose()
            await asyncio.sleep(0)
            assert active_request_ids == set()
            assert runtime._request_semaphore._value == runtime.settings.max_inflight_requests

    asyncio.run(run_test())

    assert cancellation_count == 25
    assert aborted_request_ids == list(range(10_000, 10_025))
    assert active_request_ids == set()
    assert runtime._request_semaphore._value == runtime.settings.max_inflight_requests


def test_proxy_cancellation_closes_nested_dsv4_alloc_request_iterator():
    active_request_ids = set()
    aborted_request_ids = []

    async def run_test():
        started = asyncio.Event()

        async def allocated_dsv4_stream():
            request_id = 20_001
            active_request_ids.add(request_id)
            started.set()
            try:
                yield 'data: {"choices":[{"delta":{"reasoning":"working"},"finish_reason":null}]}\n\n'
                await asyncio.Event().wait()
            finally:
                active_request_ids.discard(request_id)
                aborted_request_ids.append(request_id)

        response = CustomStreamingResponse(
            api_openai._safe_stream_wrapper(allocated_dsv4_stream()),
            media_type="text/event-stream",
        )

        async def callback(_kind, _value, _separate):
            return None

        consume_task = asyncio.create_task(
            visual_chat_proxy._consume_main_model_stream(
                response,
                model="agent",
                callback=callback,
                separate_from_prior_reasoning=False,
                stream_content=False,
            )
        )
        await started.wait()
        assert active_request_ids == {20_001}
        consume_task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await consume_task
        await asyncio.sleep(0)

    asyncio.run(run_test())

    assert active_request_ids == set()
    assert aborted_request_ids == [20_001]


def test_invalid_remote_200_opens_circuit_before_another_dsv4_allocation():
    class InvalidRemoteClient:
        async def post(self, _url, json):
            return visual_chat_proxy.httpx.Response(200, json={"choices": []})

    settings = visual_chat_proxy.VisualProxySettings(
        remote_url="http://127.0.0.1:18180/generate",
        remote_max_retries=0,
        circuit_failure_threshold=2,
        circuit_recovery_seconds=60.0,
    )
    runtime = visual_chat_proxy.VisualProxyRuntime(settings, client=InvalidRemoteClient())
    request = ChatCompletionRequest.model_validate(
        {
            "model": "agent",
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "Inspect."},
                        {"type": "image_url", "image_url": {"url": VALID_PNG_DATA_URL}},
                    ],
                }
            ],
        }
    )
    raw_request = Request({"type": "http", "method": "POST", "path": "/v1/chat/completions", "headers": []})
    dsv4_allocations = 0
    active_request_ids = set()

    async def allocate_then_request_vision(_request, _raw_request):
        nonlocal dsv4_allocations
        dsv4_allocations += 1
        request_id = 30_000 + dsv4_allocations
        active_request_ids.add(request_id)
        try:
            return ChatCompletionResponse(
                model="agent",
                choices=[
                    ChatCompletionResponseChoice(
                        index=0,
                        finish_reason="tool_calls",
                        message=ChatMessage(
                            role="assistant",
                            content="",
                            tool_calls=[
                                _tool_call(
                                    "vision_reader",
                                    {"image": "<image_1/>", "task": "inspect"},
                                    f"vision-{request_id}",
                                )
                            ],
                        ),
                    )
                ],
                usage=UsageInfo(prompt_tokens=1, completion_tokens=1, total_tokens=2),
            )
        finally:
            active_request_ids.discard(request_id)

    async def run_test():
        for call_index in range(2):
            with pytest.raises(
                visual_chat_proxy.VisualProxyUpstreamError,
                match=r"choices\[0\]",
            ):
                await visual_chat_proxy.visual_chat_completions_impl(
                    request=request,
                    raw_request=raw_request,
                    runtime=runtime,
                    main_chat_handler=allocate_then_request_vision,
                )
            assert dsv4_allocations == call_index + 1
            assert active_request_ids == set()
            assert runtime._request_semaphore._value == settings.max_inflight_requests
            assert runtime._semaphore._value == settings.remote_max_concurrency

        with pytest.raises(
            visual_chat_proxy.VisualProxyUpstreamError,
            match="circuit breaker is open",
        ):
            await visual_chat_proxy.visual_chat_completions_impl(
                request=request,
                raw_request=raw_request,
                runtime=runtime,
                main_chat_handler=allocate_then_request_vision,
            )
        assert dsv4_allocations == 2
        assert active_request_ids == set()
        assert runtime._request_semaphore._value == settings.max_inflight_requests
        assert runtime._semaphore._value == settings.remote_max_concurrency
        await runtime.close()

    asyncio.run(run_test())


def test_visual_proxy_never_intercepts_text_only_dsv4_requests():
    request = ChatCompletionRequest.model_validate(
        {
            "model": "agent",
            "messages": [{"role": "user", "content": "Keep this on the native DSV4 path."}],
        }
    )

    assert not visual_chat_proxy.should_use_visual_proxy(
        "http://127.0.0.1:18180/generate",
        request,
    )


def test_conflicting_thinking_controls_fail_closed():
    request = ChatCompletionRequest.model_validate(
        {
            "model": "agent",
            "messages": [{"role": "user", "content": "inspect"}],
            "reasoning_effort": "high",
            "chat_template_kwargs": {"enable_thinking": False},
        }
    )
    with pytest.raises(ValueError, match="Conflicting thinking controls"):
        visual_chat_proxy.apply_visual_thinking_policy(
            request,
            visual_chat_proxy.VisualProxySettings(
                remote_url="http://127.0.0.1:18180/generate"
            ),
        )


@pytest.mark.parametrize("seed", [True, False, 1.5, "1"])
def test_seed_rejects_non_integer_values_before_coercion(seed):
    with pytest.raises(ValueError, match="seed must be an integer"):
        ChatCompletionRequest.model_validate(
            {
                "model": "agent",
                "messages": [{"role": "user", "content": "inspect"}],
                "seed": seed,
            }
        )


@pytest.mark.parametrize("parallel_tool_calls", [0, 1, "false", "true"])
def test_parallel_tool_calls_requires_a_real_boolean(parallel_tool_calls):
    with pytest.raises(ValueError, match="parallel_tool_calls must be a boolean"):
        ChatCompletionRequest.model_validate(
            {
                "model": "agent",
                "messages": [{"role": "user", "content": "inspect"}],
                "parallel_tool_calls": parallel_tool_calls,
            }
        )


def test_request_thinking_policy_defaults_on_and_force_off_hides_public_reasoning():
    request = ChatCompletionRequest.model_validate(
        {
            "model": "agent",
            "messages": [{"role": "user", "content": "inspect"}],
        }
    )
    request_settings = visual_chat_proxy.VisualProxySettings(
        remote_url="http://127.0.0.1:18180/generate",
        thinking_policy="request",
    )
    normalized = visual_chat_proxy.apply_visual_thinking_policy(request, request_settings)
    assert normalized.chat_template_kwargs["enable_thinking"] is True
    assert visual_chat_proxy.resolve_public_reasoning_enabled(None, request_settings) is True

    force_off_settings = visual_chat_proxy.VisualProxySettings(
        remote_url="http://127.0.0.1:18180/generate",
        thinking_policy="force_off",
    )
    assert visual_chat_proxy.resolve_public_reasoning_enabled(True, force_off_settings) is False


def test_anthropic_maps_seed_parallel_control_and_rejects_conflicting_thinking():
    converted, _ = _anthropic_to_chat_request(
        {
            "model": "agent",
            "max_tokens": 64,
            "seed": 42,
            "messages": [{"role": "user", "content": "inspect"}],
            "tools": [
                {
                    "name": "lookup",
                    "description": "Look something up.",
                    "input_schema": {"type": "object", "properties": {}},
                }
            ],
            "tool_choice": {"type": "auto", "disable_parallel_tool_use": True},
        }
    )
    assert converted["seed"] == 42
    assert converted["parallel_tool_calls"] is False

    with pytest.raises(ValueError, match="disable_parallel_tool_use must be a boolean"):
        _anthropic_to_chat_request(
            {
                "model": "agent",
                "max_tokens": 64,
                "messages": [{"role": "user", "content": "inspect"}],
                "tool_choice": {"type": "auto", "disable_parallel_tool_use": "true"},
            }
        )

    with pytest.raises(ValueError, match="must agree"):
        _anthropic_to_chat_request(
            {
                "model": "agent",
                "max_tokens": 64,
                "messages": [{"role": "user", "content": "inspect"}],
                "thinking": {"type": "disabled"},
                "extra_body": {"chat_template_kwargs": {"enable_thinking": True}},
            }
        )


def test_parallel_tool_violation_on_last_step_keeps_specific_failure(monkeypatch):
    monkeypatch.setattr(visual_chat_proxy, "MAX_AGENT_STEPS", 1)
    request = ChatCompletionRequest.model_validate(
        {
            "model": "agent",
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "Inspect and call one tool."},
                        {"type": "image_url", "image_url": {"url": VALID_PNG_DATA_URL}},
                    ],
                }
            ],
            "parallel_tool_calls": False,
            "tools": [
                {
                    "type": "function",
                    "function": {
                        "name": name,
                        "description": name,
                        "parameters": {"type": "object", "properties": {}},
                    },
                }
                for name in ("alpha", "beta")
            ],
        }
    )

    async def fake_main(_request, _raw_request):
        return ChatCompletionResponse(
            model="agent",
            choices=[
                ChatCompletionResponseChoice(
                    index=0,
                    finish_reason="tool_calls",
                    message=ChatMessage(
                        role="assistant",
                        content="",
                        tool_calls=[
                            _tool_call("alpha", {}, "external_1"),
                            _tool_call("beta", {}, "external_2"),
                        ],
                    ),
                )
            ],
            usage=UsageInfo(prompt_tokens=1, completion_tokens=1, total_tokens=2),
        )

    runtime = visual_chat_proxy.VisualProxyRuntime(
        visual_chat_proxy.VisualProxySettings(
            remote_url="http://127.0.0.1:18180/generate",
            empty_output_retries=1,
        )
    )
    raw_request = Request(
        {"type": "http", "method": "POST", "path": "/v1/chat/completions", "headers": []}
    )

    async def run_test():
        try:
            with pytest.raises(
                visual_chat_proxy.VisualChatProxyError,
                match="parallel_tool_calls_violation",
            ):
                await visual_chat_proxy.visual_chat_completions_impl(
                    request=request,
                    raw_request=raw_request,
                    runtime=runtime,
                    main_chat_handler=fake_main,
                )
        finally:
            await runtime.close()

    asyncio.run(run_test())


def test_explicit_thinking_off_hides_public_visual_reasoning(monkeypatch):
    calls = 0

    async def fake_main(request, _raw_request):
        nonlocal calls
        calls += 1
        assert request.chat_template_kwargs["enable_thinking"] is False
        if calls == 1:
            message = ChatMessage(
                role="assistant",
                content="",
                reasoning="private planning",
                tool_calls=[
                    _tool_call(
                        "vision_reader",
                        {"image": "<image_1/>", "task": "read title"},
                        "builtin_1",
                    )
                ],
            )
            finish_reason = "tool_calls"
        else:
            message = ChatMessage(
                role="assistant",
                content="The title is LightLLM.",
                reasoning="private final reasoning",
            )
            finish_reason = "stop"
        return ChatCompletionResponse(
            model="agent",
            choices=[
                ChatCompletionResponseChoice(
                    index=0,
                    finish_reason=finish_reason,
                    message=message,
                )
            ],
            usage=UsageInfo(prompt_tokens=1, completion_tokens=1, total_tokens=2),
        )

    async def fake_visual_remote(**_kwargs):
        return "The title is LightLLM."

    monkeypatch.setattr(visual_chat_proxy, "call_visual_remote", fake_visual_remote)
    request = ChatCompletionRequest.model_validate(
        {
            "model": "agent",
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "Read the title."},
                        {"type": "image_url", "image_url": {"url": VALID_PNG_DATA_URL}},
                    ],
                }
            ],
            "chat_template_kwargs": {"enable_thinking": False},
        }
    )
    runtime = visual_chat_proxy.VisualProxyRuntime(
        visual_chat_proxy.VisualProxySettings(
            remote_url="http://127.0.0.1:18180/generate"
        ),
        client=object(),
    )
    raw_request = Request(
        {"type": "http", "method": "POST", "path": "/v1/chat/completions", "headers": []}
    )

    async def run_test():
        try:
            return await visual_chat_proxy.visual_chat_completions_impl(
                request=request,
                raw_request=raw_request,
                runtime=runtime,
                main_chat_handler=fake_main,
            )
        finally:
            await runtime.close()

    response = asyncio.run(run_test())
    message = response.choices[0].message
    assert message.content == "The title is LightLLM."
    assert message.reasoning is None
    assert message.reasoning_content is None


def test_all_current_turn_images_require_independent_evidence(monkeypatch):
    main_requests = []
    visual_results = iter(["first result", "second result"])

    async def fake_main(request, _raw_request):
        main_requests.append(request)
        step = len(main_requests)
        if step == 1:
            message = ChatMessage(
                role="assistant",
                content="",
                tool_calls=[
                    _tool_call("vision_reader", {"image": "<image_1/>", "task": "first"}, "v1")
                ],
            )
            finish_reason = "tool_calls"
        elif step == 2:
            message = ChatMessage(role="assistant", content="Both images are handled.", reasoning="discard me")
            finish_reason = "stop"
        elif step == 3:
            assert "<image_2/>" in str(request.messages[-1].content)
            assert "<image_1/>" not in str(request.messages[-1].content).split("Available image tags:", 1)[-1]
            message = ChatMessage(
                role="assistant",
                content="",
                tool_calls=[
                    _tool_call("vision_reader", {"image": "<image_2/>", "task": "second"}, "v2")
                ],
            )
            finish_reason = "tool_calls"
        else:
            message = ChatMessage(role="assistant", content="Both images are handled.")
            finish_reason = "stop"
        return ChatCompletionResponse(
            model="agent",
            choices=[ChatCompletionResponseChoice(index=0, finish_reason=finish_reason, message=message)],
            usage=UsageInfo(prompt_tokens=1, completion_tokens=1, total_tokens=2),
        )

    async def fake_visual_remote(**_kwargs):
        return next(visual_results)

    monkeypatch.setattr(visual_chat_proxy, "call_visual_remote", fake_visual_remote)
    request = ChatCompletionRequest.model_validate(
        {
            "model": "agent",
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "Compare both images."},
                        {"type": "image_url", "image_url": {"url": VALID_PNG_DATA_URL}},
                        {"type": "image_url", "image_url": {"url": VALID_PNG_DATA_URL}},
                    ],
                }
            ],
        }
    )
    runtime = visual_chat_proxy.VisualProxyRuntime(
        visual_chat_proxy.VisualProxySettings(remote_url="http://127.0.0.1:18180/generate"),
        client=object(),
    )
    raw_request = Request(
        {"type": "http", "method": "POST", "path": "/v1/chat/completions", "headers": []}
    )

    async def run_test():
        try:
            return await visual_chat_proxy.visual_chat_completions_impl(
                request=request,
                raw_request=raw_request,
                runtime=runtime,
                main_chat_handler=fake_main,
            )
        finally:
            await runtime.close()

    response = asyncio.run(run_test())
    reasoning = response.choices[0].message.reasoning or ""
    assert "我先查看了图片 <image_1/>，first result" in reasoning
    assert "我接着查看了图片 <image_2/>，second result" in reasoning
    assert "discard me" not in reasoning
    assert "vision_reader" not in reasoning
    assert "first" not in reasoning.replace("first result", "")


def test_latest_failed_reread_does_not_fall_back_in_public_projection(monkeypatch):
    main_calls = 0
    visual_calls = 0

    async def fake_main(_request, _raw_request):
        nonlocal main_calls
        main_calls += 1
        if main_calls <= 3:
            image = "<image_1/>" if main_calls in {1, 3} else "<image_2/>"
            message = ChatMessage(
                role="assistant",
                content="",
                tool_calls=[
                    _tool_call(
                        "vision_reader",
                        {"image": image, "task": f"task-{main_calls}"},
                        f"v{main_calls}",
                    )
                ],
            )
            finish_reason = "tool_calls"
        else:
            message = ChatMessage(role="assistant", content="Final answer.")
            finish_reason = "stop"
        return ChatCompletionResponse(
            model="agent",
            choices=[ChatCompletionResponseChoice(index=0, finish_reason=finish_reason, message=message)],
            usage=UsageInfo(prompt_tokens=1, completion_tokens=1, total_tokens=2),
        )

    async def fake_visual_remote(**_kwargs):
        nonlocal visual_calls
        visual_calls += 1
        if visual_calls == 1:
            return "old image-one result"
        if visual_calls == 2:
            return "image-two result"
        return visual_chat_proxy.VisualRemoteResult(
            "truncated image-one result",
            finish_reason="length",
        )

    monkeypatch.setattr(visual_chat_proxy, "call_visual_remote", fake_visual_remote)
    request = ChatCompletionRequest.model_validate(
        {
            "model": "agent",
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "Compare both images."},
                        {"type": "image_url", "image_url": {"url": VALID_PNG_DATA_URL}},
                        {"type": "image_url", "image_url": {"url": VALID_PNG_DATA_URL}},
                    ],
                }
            ],
        }
    )
    runtime = visual_chat_proxy.VisualProxyRuntime(
        visual_chat_proxy.VisualProxySettings(remote_url="http://127.0.0.1:18180/generate"),
        client=object(),
    )
    raw_request = Request(
        {"type": "http", "method": "POST", "path": "/v1/chat/completions", "headers": []}
    )

    async def run_test():
        try:
            return await visual_chat_proxy.visual_chat_completions_impl(
                request=request,
                raw_request=raw_request,
                runtime=runtime,
                main_chat_handler=fake_main,
            )
        finally:
            await runtime.close()

    response = asyncio.run(run_test())
    reasoning = response.choices[0].message.reasoning or ""
    assert "image-two result" in reasoning
    assert "old image-one result" not in reasoning
    assert "truncated image-one result" not in reasoning


def test_remote_image_target_rejects_mixed_private_dns_and_pins_public_ip(monkeypatch):
    settings = visual_chat_proxy.VisualProxySettings(
        remote_url="http://127.0.0.1:18180/generate",
        allow_remote_image_urls=True,
        remote_image_hosts=("images.example.com",),
    )
    monkeypatch.setattr(
        visual_chat_proxy.socket,
        "getaddrinfo",
        lambda *_args, **_kwargs: [
            (2, 1, 6, "", ("8.8.8.8", 443)),
            (2, 1, 6, "", ("127.0.0.1", 443)),
        ],
    )
    with pytest.raises(ValueError, match="private or non-global"):
        visual_chat_proxy._pinned_remote_image_request(
            "https://images.example.com/a.png",
            settings,
        )


def test_remote_image_download_falls_back_across_validated_addresses(monkeypatch):
    settings = visual_chat_proxy.VisualProxySettings(
        remote_url="http://127.0.0.1:18180/generate",
        allow_remote_image_urls=True,
        remote_image_hosts=("images.example.com",),
    )
    monkeypatch.setattr(
        visual_chat_proxy.socket,
        "getaddrinfo",
        lambda *_args, **_kwargs: [
            (2, 1, 6, "", ("8.8.8.8", 443)),
            (2, 1, 6, "", ("8.8.4.4", 443)),
        ],
    )
    attempts = []
    png_bytes = visual_chat_proxy.base64.b64decode(VALID_PNG_DATA_URL.split(",", 1)[1])

    class FakeImageClient:
        def build_request(self, method, url, **kwargs):
            return visual_chat_proxy.httpx.Request(method, url, **kwargs)

        async def send(self, request, stream):
            assert stream is True
            attempts.append(str(request.url))
            if request.url.host == "8.8.4.4":
                raise visual_chat_proxy.httpx.ConnectError("unreachable", request=request)
            return visual_chat_proxy.httpx.Response(
                200,
                headers={"Content-Type": "image/png"},
                content=png_bytes,
                request=request,
            )

        async def aclose(self):
            return None

    runtime = visual_chat_proxy.VisualProxyRuntime(settings, client=object())

    async def run_test():
        await runtime.image_client.aclose()
        runtime.image_client = FakeImageClient()
        try:
            return await runtime.freeze_remote_image(
                "https://images.example.com/a.png",
                request=None,
                trace_id="fallback",
            )
        finally:
            await runtime.close()

    frozen = asyncio.run(run_test())
    assert frozen == VALID_PNG_DATA_URL
    assert attempts == [
        "https://8.8.4.4/a.png",
        "https://8.8.8.8/a.png",
    ]

    monkeypatch.setattr(
        visual_chat_proxy.socket,
        "getaddrinfo",
        lambda *_args, **_kwargs: [
            (2, 1, 6, "", ("8.8.8.8", 443)),
            (2, 1, 6, "", ("8.8.4.4", 443)),
        ],
    )
    pinned_urls, host_header, sni = visual_chat_proxy._pinned_remote_image_request(
        "https://images.example.com/a.png?x=1",
        settings,
    )
    assert pinned_urls == (
        "https://8.8.4.4:443/a.png?x=1",
        "https://8.8.8.8:443/a.png?x=1",
    )
    assert host_header == "images.example.com"
    assert sni == "images.example.com"

    monkeypatch.setattr(
        visual_chat_proxy.socket,
        "getaddrinfo",
        lambda *_args, **_kwargs: [(2, 1, 6, "", ("224.0.0.1", 443))],
    )
    with pytest.raises(ValueError, match="private or non-global"):
        visual_chat_proxy._pinned_remote_image_request(
            "https://images.example.com/a.png",
            settings,
        )
