import asyncio
import json
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pytest
import httpx
from fastapi import Request
from fastapi.responses import StreamingResponse
from jinja2 import Environment
from lightllm.utils.error_utils import ClientDisconnected

from lightllm.server.api_models import (
    ChatCompletionRequest,
    ChatCompletionResponse,
    ChatCompletionResponseChoice,
    ChatMessage,
    Function,
    FunctionResponse,
    Tool,
    ToolCall,
    UsageInfo,
)
from lightllm.server import visual_chat_proxy
from lightllm.server.api_cli import make_argument_parser
from lightllm.server.visual_chat_proxy import (
    apply_visual_thinking_policy,
    ImageRegistry,
    RegisteredImage,
    VisualChatProxyError,
    VisualProxyCapacityError,
    VisualProxySettings,
    VisualProxyRuntime,
    VisualProxyTimeoutError,
    VisualProxyUpstreamError,
    call_visual_remote,
    replace_images_with_tags,
    should_use_visual_proxy,
    visual_chat_completions_impl,
)


def _runtime(client=None, **overrides):
    settings = VisualProxySettings(
        remote_url="https://vision.test/v1/chat/completions",
        builtin_trace_format="natural",
    )
    settings = replace(settings, **overrides)
    settings.validate()
    return VisualProxyRuntime(settings, client=client)


def _raw_request():
    return Request(
        {
            "type": "http",
            "method": "POST",
            "path": "/v1/chat/completions",
            "headers": [],
        }
    )


def _response(message, finish_reason="stop", prompt_tokens=3, completion_tokens=2):
    return ChatCompletionResponse(
        model="agent",
        choices=[
            ChatCompletionResponseChoice(
                index=0, message=message, finish_reason=finish_reason
            )
        ],
        usage=UsageInfo(
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=prompt_tokens + completion_tokens,
        ),
    )


def _multimodal_request(**overrides):
    payload = {
        "model": "agent",
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "What color is the square?"},
                    {
                        "type": "image_url",
                        "image_url": {"url": "data:image/png;base64,AAAA"},
                    },
                ],
            }
        ],
        "max_tokens": 64,
    }
    payload.update(overrides)
    return ChatCompletionRequest.model_validate(payload)


def _call(name, arguments, call_id):
    return ToolCall(
        id=call_id,
        index=0,
        type="function",
        function=FunctionResponse(name=name, arguments=json.dumps(arguments)),
    )


def test_replace_images_with_tags_keeps_pixels_out_of_agent_messages():
    registry = ImageRegistry()
    messages = replace_images_with_tags(
        [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "compare"},
                    {
                        "type": "image_url",
                        "image_url": {"url": "https://example.test/a.png"},
                    },
                    {
                        "type": "image_url",
                        "image_url": {"url": "data:image/png;base64,BBBB"},
                    },
                ],
            }
        ],
        registry,
    )

    assert messages[0]["content"] == [
        {"type": "text", "text": "compare"},
        {"type": "text", "text": "<image_1/>"},
        {"type": "text", "text": "<image_2/>"},
    ]
    assert registry.resolve("<image_1/>").source == "https://example.test/a.png"
    assert registry.resolve("image <image_2/>").source == "data:image/png;base64,BBBB"
    assert registry.resolve("image_1").tag == "<image_1/>"
    assert registry.resolve("Picture 2").tag == "<image_2/>"


@pytest.mark.parametrize("trace_format", ["natural", "xml"])
def test_builtin_trace_formats_round_trip_into_native_template_messages(trace_format):
    if trace_format == "natural":
        reasoning = (
            "先判断任务。\n"
            "我先查看了图片 <image_1/>，让内建读图能力完成这个任务：识别主要颜色。\n"
            "读图结果：主要颜色是绿色。\n"
            "现在调用天气工具。"
        )
    else:
        reasoning = (
            "先判断任务。\n"
            "<tool_call>\n"
            "<function=vision_reader>\n"
            "<parameter=image>\n<image_1/>\n</parameter>\n"
            "<parameter=task>\n识别主要颜色\n</parameter>\n"
            "</function>\n"
            "</tool_call>\n"
            "<tool_response>\n主要颜色是绿色。\n</tool_response>\n"
            "现在调用天气工具。"
        )
    messages = [
        {
            "role": "assistant",
            "content": "",
            "reasoning": reasoning,
            "tool_calls": [
                {
                    "id": "weather_1",
                    "type": "function",
                    "function": {
                        "name": "get_weather",
                        "arguments": '{"city":"上海"}',
                    },
                }
            ],
        }
    ]

    expanded = visual_chat_proxy.expand_builtin_traces(messages, trace_format)

    assert [message["role"] for message in expanded] == [
        "assistant",
        "tool",
        "assistant",
    ]
    assert expanded[0]["reasoning"] == "先判断任务。"
    assert expanded[0]["tool_calls"][0]["function"]["name"] == "vision_reader"
    arguments = json.loads(expanded[0]["tool_calls"][0]["function"]["arguments"])
    assert arguments == {"image": "<image_1/>", "task": "识别主要颜色"}
    assert expanded[1]["name"] == "vision_reader"
    assert expanded[1]["content"] == visual_chat_proxy._format_visual_tool_result(
        "主要颜色是绿色。"
    )
    assert expanded[2]["reasoning"] == "现在调用天气工具。"
    assert expanded[2]["tool_calls"][0]["function"]["name"] == "get_weather"


def test_malformed_builtin_trace_is_rejected_before_model_inference():
    messages = [
        {
            "role": "assistant",
            "content": "",
            "reasoning": ("我先查看了图片 <image_1/>，让内建读图能力完成这个任务：识别颜色。"),
        }
    ]
    with pytest.raises(ValueError, match="missing reading result"):
        visual_chat_proxy.expand_builtin_traces(messages, "natural")


def test_proxy_activation_is_strictly_opt_in_and_multimodal():
    multimodal = _multimodal_request()
    text_only = ChatCompletionRequest(
        model="agent", messages=[{"role": "user", "content": "hello"}]
    )

    assert not should_use_visual_proxy(None, multimodal)
    assert not should_use_visual_proxy("", multimodal)
    assert not should_use_visual_proxy("http://vision.test", text_only)
    assert should_use_visual_proxy("http://vision.test", multimodal)


def test_visual_remote_receives_openai_multimodal_payload(monkeypatch):
    recorded = {}

    class FakeResponse:
        status_code = 200
        text = ""

        def json(self):
            return {
                "choices": [
                    {"message": {"role": "assistant", "content": "remote result"}}
                ]
            }

    class FakeClient:
        async def post(self, url, json):
            recorded["url"] = url
            recorded["payload"] = json
            return FakeResponse()

    runtime = _runtime(client=FakeClient(), remote_model="vision-model")
    result = asyncio.run(
        call_visual_remote(
            runtime=runtime,
            model="agent",
            image=RegisteredImage(
                tag="<image_1/>", source="data:image/png;base64,AAAA", origin="user"
            ),
            task="Read the image",
            trace_id="trace-test",
        )
    )

    assert result == "remote result"
    assert recorded["url"] == runtime.settings.remote_url
    payload = recorded["payload"]
    assert payload["model"] == "vision-model"
    assert payload["stream"] is False
    assert payload["messages"][1]["content"] == [
        {"type": "image_url", "image_url": {"url": "data:image/png;base64,AAAA"}},
        {"type": "text", "text": "Read the image"},
    ]


@pytest.mark.parametrize(
    ("policy", "template_kwargs", "effort", "expected_enabled", "expected_effort"),
    [
        ("request", None, None, False, None),
        ("request", {"enable_thinking": True}, None, True, "high"),
        ("request", {"enable_thinking": True}, "low", True, "low"),
        ("force_on", {"enable_thinking": False}, None, True, "high"),
        ("force_off", {"enable_thinking": True}, "low", False, None),
    ],
)
def test_visual_thinking_policy(
    policy, template_kwargs, effort, expected_enabled, expected_effort
):
    request = _multimodal_request(
        chat_template_kwargs=template_kwargs,
        reasoning_effort=effort,
    )
    settings = replace(_runtime().settings, thinking_policy=policy)

    resolved = apply_visual_thinking_policy(request, settings)

    assert resolved.chat_template_kwargs["enable_thinking"] is expected_enabled
    assert resolved.chat_template_kwargs["thinking"] is expected_enabled
    assert resolved.reasoning_effort == expected_effort
    force_on_prompts = [
        message
        for message in resolved.messages
        if message.role == "system"
        and message.content == visual_chat_proxy.NOVA_FORCE_ON_SYSTEM_PROMPT
    ]
    assert len(force_on_prompts) == (1 if policy == "force_on" else 0)


def test_force_on_system_prompt_is_idempotent_and_preserves_user_system():
    request = _multimodal_request(
        messages=[
            {"role": "system", "content": "User-provided system prompt."},
            {"role": "user", "content": "What is shown?"},
        ]
    )
    settings = replace(_runtime().settings, thinking_policy="force_on")

    resolved = apply_visual_thinking_policy(request, settings)
    resolved = apply_visual_thinking_policy(resolved, settings)

    system_contents = [message.content for message in resolved.messages if message.role == "system"]
    assert system_contents == [
        visual_chat_proxy.NOVA_FORCE_ON_SYSTEM_PROMPT,
        "User-provided system prompt.",
    ]


def test_nova_accuracy_mode_uses_exact_generate_prompt_and_parameters():
    recorded = {}

    class FakeResponse:
        status_code = 200
        text = ""

        def json(self):
            return {
                "generated_text": "<think>visual reasoning</think>\nremote result<|im_end|>"
            }

    class FakeClient:
        async def post(self, url, json):
            recorded["url"] = url
            recorded["payload"] = json
            return FakeResponse()

    runtime = _runtime(
        client=FakeClient(),
        remote_url="https://vision.test/generate",
        remote_model="vision-model",
        nova_accuracy_compat=True,
    )
    result = asyncio.run(
        call_visual_remote(
            runtime=runtime,
            model="agent",
            image=RegisteredImage(
                tag="<image_1/>",
                source="data:image/png;base64,AAAA",
                origin="user",
            ),
            task="Read the image",
            trace_id="trace-nova",
        )
    )

    assert result == "remote result"
    assert recorded["url"] == "https://vision.test/generate"
    assert recorded["payload"] == {
        "inputs": (
            "<|im_start|>system\n"
            "You are the builtin vision_reader. Inspect the attached image and answer only the requested visual "
            "task. Ground every statement in the image; do not mention tool calls.\n"
            "<|im_end|>\n"
            "<|im_start|>user\n"
            "<|vision_start|><|image_pad|><|vision_end|>\n"
            "Read the image\n"
            "<|im_end|>\n"
            "<|im_start|>assistant\n"
        ),
        "parameters": {"max_new_tokens": 512, "temperature": 0.0},
        "multimodal_params": {"images": [{"type": "base64", "data": "AAAA"}]},
        "model": "vision-model",
    }


def test_nova_accuracy_mode_keeps_model_facing_visual_result_unmodified(
    monkeypatch,
):
    main_requests = []

    async def fake_main(request, raw_request):
        payload = request.model_dump(exclude_none=True)
        main_requests.append(payload)
        assert payload["chat_template_kwargs"] == {
            "enable_builtin_vision_reader": True,
            "render_vision_placeholders": False,
            "enable_thinking": False,
            "thinking": False,
        }
        if len(main_requests) == 1:
            return _response(
                ChatMessage(
                    role="assistant",
                    tool_calls=[
                        _call(
                            "vision_reader",
                            {"image": "<image_1/>", "task": "Read it"},
                            "reader_1",
                        )
                    ],
                ),
                finish_reason="tool_calls",
            )
        assert payload["messages"][-1]["role"] == "assistant"
        assert "raw visual result" in payload["messages"][-1]["reasoning"]
        assert "<tool_response>" in payload["messages"][-1]["reasoning"]
        return _response(ChatMessage(role="assistant", content="done"))

    async def fake_remote(**kwargs):
        return "raw visual result"

    monkeypatch.setattr(visual_chat_proxy, "call_visual_remote", fake_remote)
    response = asyncio.run(
        visual_chat_completions_impl(
            request=_multimodal_request(),
            raw_request=_raw_request(),
            runtime=_runtime(
                client=object(),
                remote_url="https://vision.test/generate",
                nova_accuracy_compat=True,
                builtin_trace_format="xml",
            ),
            main_chat_handler=fake_main,
        )
    )

    assert response.choices[0].message.content == "done"
    assert "raw visual result" in (response.choices[0].message.reasoning or "")
    assert "UNTRUSTED_VISUAL_OBSERVATION" not in json.dumps(
        main_requests, ensure_ascii=False
    )


def test_multiple_builtin_calls_use_distinct_idempotency_keys(monkeypatch):
    remote_trace_ids = []
    main_calls = 0

    async def fake_main(request, raw_request):
        nonlocal main_calls
        main_calls += 1
        if main_calls == 1:
            return _response(
                ChatMessage(
                    role="assistant",
                    tool_calls=[
                        _call(
                            "vision_reader",
                            {"image": "<image_1/>", "task": "first"},
                            "reader_1",
                        ),
                        _call(
                            "vision_reader",
                            {"image": "<image_1/>", "task": "second"},
                            "reader_2",
                        ),
                    ],
                ),
                finish_reason="tool_calls",
            )
        return _response(ChatMessage(role="assistant", content="done"))

    async def fake_remote(**kwargs):
        remote_trace_ids.append(kwargs["trace_id"])
        return "observation"

    monkeypatch.setattr(visual_chat_proxy, "call_visual_remote", fake_remote)
    asyncio.run(
        visual_chat_completions_impl(
            request=_multimodal_request(),
            raw_request=_raw_request(),
            runtime=_runtime(client=object()),
            main_chat_handler=fake_main,
        )
    )

    assert len(remote_trace_ids) == 2
    assert len(set(remote_trace_ids)) == 2
    assert remote_trace_ids[0].endswith("-s1-c0")
    assert remote_trace_ids[1].endswith("-s1-c1")


def test_builtin_reader_runs_remote_then_returns_openai_completion(monkeypatch):
    main_requests = []
    remote_calls = []

    async def fake_main(request, raw_request):
        main_requests.append(request.model_dump(exclude_none=True))
        if len(main_requests) == 1:
            return _response(
                ChatMessage(
                    role="assistant",
                    content="",
                    reasoning="I need to inspect the tagged image.",
                    tool_calls=[
                        _call(
                            "vision_reader",
                            {
                                "image": "<image_1/>",
                                "task": "Identify the square's color.",
                            },
                            "call_visual_1",
                        )
                    ],
                ),
                finish_reason="tool_calls",
            )
        assert main_requests[-1]["messages"][-1] == {
            "role": "tool",
            "content": visual_chat_proxy._format_visual_tool_result(
                "The square is green."
            ),
            "tool_call_id": "call_visual_1",
            "name": "vision_reader",
        }
        return _response(
            ChatMessage(
                role="assistant",
                content="The square is green.",
                reasoning="The reader result gives me enough evidence to answer.",
            )
        )

    async def fake_remote(**kwargs):
        remote_calls.append(kwargs)
        return "The square is green."

    monkeypatch.setattr(visual_chat_proxy, "call_visual_remote", fake_remote)
    runtime = _runtime(client=object())
    response = asyncio.run(
        visual_chat_completions_impl(
            request=_multimodal_request(),
            raw_request=_raw_request(),
            runtime=runtime,
            main_chat_handler=fake_main,
        )
    )

    assert isinstance(response, ChatCompletionResponse)
    assert response.choices[0].message.content == "The square is green."
    final_reasoning = response.choices[0].message.reasoning or ""
    assert "I need to inspect the tagged image." in final_reasoning
    assert "The reader result gives me enough evidence to answer." in final_reasoning
    assert response.choices[0].message.reasoning_content is None
    assert response.choices[0].finish_reason == "stop"
    assert response.usage.total_tokens == 10
    assert len(remote_calls) == 1
    assert remote_calls[0]["image"].source == "data:image/png;base64,AAAA"
    assert remote_calls[0]["task"] == "Identify the square's color."

    first_payload = main_requests[0]
    serialized = json.dumps(first_payload)
    assert "<image_1/>" in serialized
    assert "data:image/png;base64,AAAA" not in serialized
    assert first_payload["tools"][-1]["function"]["name"] == "vision_reader"

    assert (
        "我先查看了图片 <image_1/>，让内建读图能力完成这个任务：" "Identify the square's color."
    ) in final_reasoning
    assert "读图结果：The square is green." in final_reasoning
    assert "lightllm_vision_reader_trace" not in final_reasoning


def test_invalid_reader_arguments_are_returned_then_model_can_retry(monkeypatch):
    main_requests = []
    remote_calls = []

    async def fake_main(request, raw_request):
        main_requests.append(request)
        if len(main_requests) == 1:
            return _response(
                ChatMessage(
                    role="assistant",
                    content="",
                    tool_calls=[
                        _call(
                            "vision_reader", {"image": "image_1"}, "call_missing_task"
                        )
                    ],
                ),
                finish_reason="tool_calls",
            )
        if len(main_requests) == 2:
            assert "requires both arguments" in main_requests[-1].messages[-1].content
            return _response(
                ChatMessage(
                    role="assistant",
                    content="",
                    tool_calls=[
                        _call(
                            "vision_reader",
                            {
                                "image": "<image_1/>",
                                "task": "Identify the square color.",
                            },
                            "call_retry",
                        )
                    ],
                ),
                finish_reason="tool_calls",
            )
        return _response(ChatMessage(role="assistant", content="The square is green."))

    async def fake_remote(**kwargs):
        remote_calls.append(kwargs)
        return "The square is green."

    monkeypatch.setattr(visual_chat_proxy, "call_visual_remote", fake_remote)
    runtime = _runtime(client=object())
    response = asyncio.run(
        visual_chat_completions_impl(
            request=_multimodal_request(),
            raw_request=_raw_request(),
            runtime=runtime,
            main_chat_handler=fake_main,
        )
    )

    assert response.choices[0].message.content == "The square is green."
    assert len(remote_calls) == 1
    assert remote_calls[0]["image"].tag == "<image_1/>"
    assert remote_calls[0]["task"] == "Identify the square color."


def test_external_tools_stay_public_when_mixed_with_builtin(monkeypatch):
    main_requests = []

    async def fake_main(request, raw_request):
        main_requests.append(request.model_dump(exclude_none=True))
        return _response(
            ChatMessage(
                role="assistant",
                content="",
                tool_calls=[
                    _call(
                        "vision_reader",
                        {"image": "<image_1/>", "task": "Read the city."},
                        "builtin_1",
                    ),
                    _call("get_weather", {"city": "Shanghai"}, "external_1"),
                ],
            ),
            finish_reason="tool_calls",
        )

    async def fake_remote(**kwargs):
        return "The image says Shanghai."

    monkeypatch.setattr(visual_chat_proxy, "call_visual_remote", fake_remote)
    runtime = _runtime(client=object())
    weather = Tool(
        type="function",
        function=Function(
            name="get_weather",
            description="Get weather",
            parameters={"type": "object", "properties": {"city": {"type": "string"}}},
        ),
    )
    response = asyncio.run(
        visual_chat_completions_impl(
            request=_multimodal_request(tools=[weather.model_dump()]),
            raw_request=_raw_request(),
            runtime=runtime,
            main_chat_handler=fake_main,
        )
    )

    message = response.choices[0].message
    assert response.choices[0].finish_reason == "tool_calls"
    assert [call.function.name for call in message.tool_calls] == ["get_weather"]
    assert message.tool_calls[0].id == "external_1"
    assert len(main_requests) == 1
    trace_visual_result = "The image says Shanghai."
    internal_visual_result = visual_chat_proxy._format_visual_tool_result(
        trace_visual_result
    )
    assert trace_visual_result in (message.reasoning or "")
    assert "lightllm_vision_reader_trace" not in (message.reasoning or "")

    replay_messages = _multimodal_request().model_dump(
        by_alias=True, exclude_none=True
    )["messages"]
    replay_messages.extend(
        [
            message.model_dump(by_alias=True, exclude_none=True),
            {
                "role": "tool",
                "tool_call_id": "external_1",
                "name": "get_weather",
                "content": "Sunny, 24 C",
            },
        ]
    )

    async def replay_main(request, raw_request):
        replay_payload = request.model_dump(by_alias=True, exclude_none=True)
        replay_messages = replay_payload["messages"]
        assert [item["role"] for item in replay_messages[-4:]] == [
            "assistant",
            "tool",
            "assistant",
            "tool",
        ]
        assert replay_messages[-3]["name"] == "vision_reader"
        assert internal_visual_result in replay_messages[-3]["content"]
        assert replay_messages[-2]["tool_calls"][0]["function"]["name"] == "get_weather"
        assert replay_payload["chat_template_kwargs"]["enable_thinking"] is False
        assert replay_payload["chat_template_kwargs"]["thinking"] is False
        return _response(ChatMessage(role="assistant", content="Shanghai is sunny."))

    replay_response = asyncio.run(
        visual_chat_completions_impl(
            request=_multimodal_request(
                messages=replay_messages,
                tools=[weather.model_dump()],
                chat_template_kwargs={"enable_thinking": False},
            ),
            raw_request=_raw_request(),
            runtime=runtime,
            main_chat_handler=replay_main,
        )
    )
    assert replay_response.choices[0].message.content == "Shanghai is sunny."


def test_startup_settings_load_upstream_auth_and_headers(monkeypatch):
    args = SimpleNamespace(visual_remote_url="https://vision.test/v1")
    monkeypatch.setenv("LIGHTLLM_VISUAL_REMOTE_API_KEY", "upstream-token")
    monkeypatch.setenv("LIGHTLLM_VISUAL_REMOTE_HEADERS", '{"X-Tenant":"tenant-a"}')
    monkeypatch.setenv("THINKING_POLICY", "force_on")
    monkeypatch.setenv("EMPTY_OUTPUT_RETRIES", "3")
    settings = VisualProxySettings.from_args(args)
    assert settings.remote_url == "https://vision.test/v1/chat/completions"
    assert settings.builtin_trace_format == "xml"
    assert settings.thinking_policy == "force_on"
    assert settings.empty_output_retries == 3
    assert not settings.allow_local_files
    assert not settings.allow_remote_image_urls
    runtime = VisualProxyRuntime(settings)
    assert runtime.client.headers["authorization"] == "Bearer upstream-token"
    assert runtime.client.headers["x-tenant"] == "tenant-a"
    asyncio.run(runtime.close())

    insecure_args = SimpleNamespace(visual_remote_url="http://vision.test/v1")
    with pytest.raises(ValueError, match="must use HTTPS"):
        VisualProxySettings.from_args(insecure_args)

    loopback_args = SimpleNamespace(visual_remote_url="http://127.0.0.1:18180/v1")
    assert VisualProxySettings.from_args(loopback_args).remote_url.startswith(
        "http://127.0.0.1:18180/"
    )


def test_nova_accuracy_startup_selects_bundled_template_and_generate_protocol():
    args = SimpleNamespace(
        visual_remote_url="http://127.0.0.1:18180/v1",
        visual_nova_accuracy_compat=True,
        chat_template=None,
        tool_call_parser=None,
        reasoning_parser=None,
    )

    visual_chat_proxy.validate_visual_proxy_startup(args)
    settings = VisualProxySettings.from_args(args)

    assert settings.nova_accuracy_compat is True
    assert settings.remote_url == "http://127.0.0.1:18180/generate"
    assert Path(args.chat_template).resolve() == (
        visual_chat_proxy.NOVA_ACCURACY_TEMPLATE_PATH.resolve()
    )
    assert args.tool_call_parser == "qwen3_coder"
    assert args.reasoning_parser == "qwen3"
    template = Path(args.chat_template).read_text(encoding="utf-8")
    assert "enable_builtin_vision_reader" in template
    assert "SensenNova-v6.7-Flash" in template


def test_nova_template_deduplicates_external_tool_xml_from_reasoning():
    template = visual_chat_proxy.NOVA_ACCURACY_TEMPLATE_PATH.read_text(encoding="utf-8")
    env = Environment()
    env.globals["raise_exception"] = lambda message: (_ for _ in ()).throw(
        RuntimeError(message)
    )
    rendered = env.from_string(template).render(
        messages=[
            {"role": "user", "content": "查天气"},
            {
                "role": "assistant",
                "content": "",
                "reasoning_content": (
                    "先调用天气工具。\n<tool_call>\n<function=weather>\n"
                    "</function>\n</tool_call>"
                ),
                "tool_calls": [
                    {"function": {"name": "weather", "arguments": {"city": "上海"}}}
                ],
            },
            {"role": "tool", "content": "上海今天多云。"},
        ],
        tools=[
            {
                "type": "function",
                "function": {
                    "name": "weather",
                    "description": "Get weather.",
                    "parameters": {
                        "type": "object",
                        "properties": {"city": {"type": "string"}},
                    },
                },
            }
        ],
        enable_builtin_vision_reader=True,
        add_generation_prompt=True,
    )

    assert "先调用天气工具。" in rendered
    assert rendered.count("<function=weather>") == 1
    assert "<tool_response>\n上海今天多云。\n</tool_response>" in rendered


def test_nova_accuracy_startup_rejects_incompatible_parsers():
    args = SimpleNamespace(
        visual_remote_url="http://127.0.0.1:18180",
        visual_nova_accuracy_compat=True,
        chat_template=None,
        tool_call_parser="llama3",
        reasoning_parser="qwen3",
    )
    with pytest.raises(ValueError, match="qwen3_coder"):
        visual_chat_proxy.validate_visual_proxy_startup(args)

    args.tool_call_parser = "qwen3_coder"
    args.chat_template = "/tmp/model-native-template.jinja"
    with pytest.raises(ValueError, match="conflicting --chat_template"):
        visual_chat_proxy.validate_visual_proxy_startup(args)


def test_builtin_trace_format_cli_default_and_aliases():
    parser = make_argument_parser()
    assert parser.parse_args([]).visual_builtin_trace_format == "xml"
    assert parser.parse_args([]).visual_nova_accuracy_compat is False
    assert parser.parse_args([]).visual_thinking_policy is None
    assert parser.parse_args([]).visual_empty_output_retries is None
    assert (
        parser.parse_args(
            ["--visual_thinking_policy", "force_off"]
        ).visual_thinking_policy
        == "force_off"
    )
    assert (
        parser.parse_args(["--visual_nova_accuracy_compat"]).visual_nova_accuracy_compat
        is True
    )
    assert (
        parser.parse_args(
            ["--builtin-trace-format", "natural"]
        ).visual_builtin_trace_format
        == "natural"
    )
    assert (
        parser.parse_args(
            ["--visual_builtin_trace_format", "natural"]
        ).visual_builtin_trace_format
        == "natural"
    )


def test_image_sources_are_bounded_and_local_files_are_rooted(tmp_path):
    settings = _runtime(client=object()).settings
    assert visual_chat_proxy._remote_image_url(
        "data:image/png;base64,AAAA", settings
    ).startswith("data:image/png;base64,")
    with pytest.raises(ValueError, match="invalid base64"):
        visual_chat_proxy._remote_image_url("data:image/png;base64,%%%", settings)
    with pytest.raises(ValueError, match="supported raster"):
        visual_chat_proxy._remote_image_url("data:image/svg+xml;base64,AAAA", settings)
    with pytest.raises(ValueError, match="Remote image URLs are disabled"):
        visual_chat_proxy._remote_image_url("https://example.test/image.png", settings)

    root = tmp_path / "allowed"
    root.mkdir()
    image = root / "image.png"
    image.write_bytes(b"png")
    outside = tmp_path / "outside.png"
    outside.write_bytes(b"png")
    local_settings = replace(
        settings,
        allow_local_files=True,
        local_file_roots=(root.resolve(),),
        max_image_bytes=8,
    )
    assert visual_chat_proxy._remote_image_url(
        image.as_uri(), local_settings
    ).startswith("data:image/png;base64,")
    with pytest.raises(ValueError, match="outside"):
        visual_chat_proxy._remote_image_url(outside.as_uri(), local_settings)
    with pytest.raises(ValueError, match="byte limit"):
        visual_chat_proxy._remote_image_url(
            image.as_uri(), replace(local_settings, max_image_bytes=2)
        )


def test_remote_images_require_an_exact_https_host_allowlist():
    settings = _runtime(
        client=object(),
        allow_remote_image_urls=True,
        remote_image_hosts=("images.example.com",),
    ).settings
    source = "https://images.example.com/a.png"
    assert visual_chat_proxy._remote_image_url(source, settings) == source
    with pytest.raises(ValueError, match="allowlist"):
        visual_chat_proxy._remote_image_url("https://evil.example.com/a.png", settings)
    with pytest.raises(ValueError, match="Plain HTTP"):
        visual_chat_proxy._remote_image_url("http://images.example.com/a.png", settings)


def test_image_registry_enforces_request_limit():
    registry = ImageRegistry(max_images=1)
    registry.add("data:image/png;base64,AAAA", "user")
    with pytest.raises(ValueError, match="at most 1 images"):
        registry.add("data:image/png;base64,BBBB", "user")

    registry = ImageRegistry(max_images=2, max_total_image_bytes=4)
    registry.add("data:image/png;base64,AAAA", "user", byte_size=3)
    with pytest.raises(ValueError, match="request limit"):
        registry.add("data:image/png;base64,BBBB", "user", byte_size=3)


def test_visual_upstream_retries_transient_failures():
    class FakeResponse:
        def __init__(self, status_code, payload=None):
            self.status_code = status_code
            self.headers = {}
            self._payload = payload

        def json(self):
            return self._payload

    class FakeClient:
        def __init__(self):
            self.responses = [
                FakeResponse(503),
                FakeResponse(200, {"choices": [{"message": {"content": "ok"}}]}),
            ]
            self.calls = 0

        async def post(self, url, json):
            self.calls += 1
            return self.responses.pop(0)

    client = FakeClient()
    runtime = _runtime(client=client, remote_max_retries=1, remote_timeout=5.0)
    data = asyncio.run(runtime.post_json({"model": "vision"}, None, "retry-test"))
    assert data["choices"][0]["message"]["content"] == "ok"
    assert client.calls == 2


def test_visual_upstream_timeout_is_bounded():
    class SlowClient:
        async def post(self, url, json):
            await asyncio.sleep(1)
            raise AssertionError("unreachable")

    runtime = _runtime(client=SlowClient(), remote_max_retries=0, remote_timeout=0.01)
    with pytest.raises(VisualProxyTimeoutError):
        asyncio.run(runtime.post_json({"model": "vision"}, None, "timeout-test"))


def test_visual_upstream_is_cancelled_when_client_disconnects():
    class DisconnectRequest:
        async def is_disconnected(self):
            return True

    class CancellableClient:
        def __init__(self):
            self.cancelled = False

        async def post(self, url, json):
            try:
                await asyncio.sleep(10)
            finally:
                self.cancelled = True

    async def run_test():
        client = CancellableClient()
        runtime = _runtime(client=client, remote_max_retries=0)
        with pytest.raises(ClientDisconnected):
            await runtime.post_json(
                {"model": "vision"}, DisconnectRequest(), "disconnect-test"
            )
        assert client.cancelled

    asyncio.run(run_test())


def test_visual_upstream_concurrency_queue_is_bounded():
    class SuccessResponse:
        status_code = 200
        headers = {}

        def json(self):
            return {"choices": [{"message": {"content": "ok"}}]}

    class BlockingClient:
        def __init__(self):
            self.started = asyncio.Event()
            self.release = asyncio.Event()

        async def post(self, url, json):
            self.started.set()
            await self.release.wait()
            return SuccessResponse()

    async def run_test():
        client = BlockingClient()
        runtime = _runtime(
            client=client,
            remote_max_concurrency=1,
            remote_queue_timeout=0.01,
        )
        first = asyncio.create_task(
            runtime.post_json({"model": "vision"}, None, "first")
        )
        await client.started.wait()
        with pytest.raises(VisualProxyCapacityError, match="saturated"):
            await runtime.post_json({"model": "vision"}, None, "second")
        client.release.set()
        await first

    asyncio.run(run_test())


def test_visual_proxy_request_queue_is_bounded():
    async def run_test():
        runtime = _runtime(
            client=object(),
            max_inflight_requests=1,
            remote_queue_timeout=0.01,
        )
        entered = asyncio.Event()
        release = asyncio.Event()

        async def hold_slot():
            async with runtime.request_slot():
                entered.set()
                await release.wait()

        first = asyncio.create_task(hold_slot())
        await entered.wait()
        with pytest.raises(VisualProxyCapacityError, match="request limit"):
            async with runtime.request_slot():
                raise AssertionError("unreachable")
        release.set()
        await first

    asyncio.run(run_test())


def test_visual_upstream_response_body_is_bounded():
    async def handler(request):
        assert request.headers["idempotency-key"] == "lightllm-body-limit-test"
        return httpx.Response(200, content=b"x" * 32)

    async def run_test():
        client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
        runtime = _runtime(client=client, max_upstream_body_bytes=16)
        try:
            with pytest.raises(VisualProxyUpstreamError, match="body exceeds"):
                await runtime.post_json({"model": "vision"}, None, "body-limit-test")
        finally:
            await client.aclose()

    asyncio.run(run_test())


def test_visual_upstream_circuit_breaker_opens_after_threshold():
    class FailedResponse:
        status_code = 503
        headers = {}

    class FailedClient:
        def __init__(self):
            self.calls = 0

        async def post(self, url, json):
            self.calls += 1
            return FailedResponse()

    async def run_test():
        client = FailedClient()
        runtime = _runtime(
            client=client,
            remote_max_retries=0,
            circuit_failure_threshold=2,
            circuit_recovery_seconds=60.0,
        )
        for _ in range(2):
            with pytest.raises(VisualProxyUpstreamError):
                await runtime.post_json({"model": "vision"}, None, "circuit-test")
        with pytest.raises(VisualProxyUpstreamError, match="circuit breaker"):
            await runtime.post_json({"model": "vision"}, None, "circuit-test")
        assert client.calls == 2

    asyncio.run(run_test())


def test_non_retryable_upstream_4xx_does_not_open_circuit():
    class RejectedResponse:
        status_code = 400
        headers = {}

    class RejectedClient:
        def __init__(self):
            self.calls = 0

        async def post(self, url, json):
            self.calls += 1
            return RejectedResponse()

    async def run_test():
        client = RejectedClient()
        runtime = _runtime(
            client=client,
            remote_max_retries=0,
            circuit_failure_threshold=1,
        )
        for _ in range(2):
            with pytest.raises(VisualProxyUpstreamError, match="HTTP 400"):
                await runtime.post_json({"model": "vision"}, None, "rejected-test")
        assert client.calls == 2

    asyncio.run(run_test())


def test_user_vision_reader_disables_builtin_and_stays_external():
    vision_reader = Tool(
        type="function",
        function=Function(
            name="vision_reader",
            description="collision",
            parameters={"type": "object", "properties": {}},
        ),
    )

    requests = []

    async def fake_main(request, raw_request):
        requests.append(request.model_dump(exclude_none=True))
        return _response(
            ChatMessage(
                role="assistant",
                content="",
                tool_calls=[
                    _call(
                        "vision_reader", {"path": "/tmp/image.png"}, "external_reader"
                    )
                ],
            ),
            finish_reason="tool_calls",
        )

    response = asyncio.run(
        visual_chat_completions_impl(
            request=_multimodal_request(tools=[vision_reader.model_dump()]),
            raw_request=_raw_request(),
            runtime=_runtime(client=object()),
            main_chat_handler=fake_main,
        )
    )

    assert len(requests) == 1
    assert [tool["function"]["name"] for tool in requests[0]["tools"]] == [
        "vision_reader"
    ]
    assert response.choices[0].finish_reason == "tool_calls"
    assert response.choices[0].message.tool_calls[0].function.name == "vision_reader"


def test_complete_agent_loop_timeout_is_bounded():
    async def slow_main(request, raw_request):
        await asyncio.sleep(1)
        return _response(ChatMessage(role="assistant", content="late"))

    with pytest.raises(VisualProxyTimeoutError, match="agent loop"):
        asyncio.run(
            visual_chat_completions_impl(
                request=_multimodal_request(),
                raw_request=_raw_request(),
                runtime=_runtime(client=object(), agent_timeout=0.01),
                main_chat_handler=slow_main,
            )
        )


def test_visual_proxy_limits_number_of_choices():
    async def should_not_run(request, raw_request):
        raise AssertionError("main model must not run when n exceeds the proxy limit")

    with pytest.raises(ValueError, match="at most 2 choices"):
        asyncio.run(
            visual_chat_completions_impl(
                request=_multimodal_request(n=3),
                raw_request=_raw_request(),
                runtime=_runtime(client=object(), max_choices=2),
                main_chat_handler=should_not_run,
            )
        )


def test_output_guardrail_forces_builtin_evidence_before_visual_answer(monkeypatch):
    main_requests = []

    async def fake_main(request, raw_request):
        main_requests.append(request.model_dump(by_alias=True, exclude_none=True))
        if len(main_requests) == 1:
            return _response(
                ChatMessage(
                    role="assistant",
                    content="The square is blue.",
                    reasoning="I guessed from the question.",
                )
            )
        if len(main_requests) == 2:
            assert (
                "Output guardrail blocked"
                in main_requests[-1]["messages"][-1]["content"]
            )
            return _response(
                ChatMessage(
                    role="assistant",
                    content="",
                    reasoning="I must inspect it first.",
                    tool_calls=[
                        _call(
                            "vision_reader",
                            {
                                "image": "<image_1/>",
                                "task": "Identify the square color.",
                            },
                            "reader_after_guardrail",
                        )
                    ],
                ),
                finish_reason="tool_calls",
            )
        return _response(
            ChatMessage(
                role="assistant",
                content="The square is green.",
                reasoning="The visual evidence supports green.",
            )
        )

    async def fake_remote(**kwargs):
        return "The square is green."

    monkeypatch.setattr(visual_chat_proxy, "call_visual_remote", fake_remote)
    response = asyncio.run(
        visual_chat_completions_impl(
            request=_multimodal_request(),
            raw_request=_raw_request(),
            runtime=_runtime(client=object()),
            main_chat_handler=fake_main,
        )
    )

    assert len(main_requests) == 3
    assert response.choices[0].message.content == "The square is green."
    reasoning = response.choices[0].message.reasoning or ""
    assert "I guessed from the question." in reasoning
    assert "I must inspect it first." in reasoning
    assert "读图结果：" in reasoning
    assert "The visual evidence supports green." in reasoning


def test_empty_output_is_retried_before_returning_final_answer():
    main_requests = []

    async def fake_main(request, raw_request):
        main_requests.append(request.model_dump(by_alias=True, exclude_none=True))
        if len(main_requests) == 1:
            return _response(
                ChatMessage(role="assistant", content="", reasoning="reasoning only")
            )
        assert (
            "previous generation ended with no user-visible answer"
            in main_requests[-1]["messages"][-1]["content"]
        )
        return _response(ChatMessage(role="assistant", content="final answer after retry"))

    response = asyncio.run(
        visual_chat_completions_impl(
            request=_multimodal_request(
                messages=[{"role": "user", "content": "Give me a final answer."}]
            ),
            raw_request=_raw_request(),
            runtime=_runtime(client=object(), empty_output_retries=2),
            main_chat_handler=fake_main,
        )
    )

    assert len(main_requests) == 2
    assert response.choices[0].message.content == "final answer after retry"


def test_empty_output_retry_limit_fails_explicitly():
    async def fake_main(request, raw_request):
        return _response(ChatMessage(role="assistant", content=""))

    with pytest.raises(VisualChatProxyError, match="after 1 empty-output retries"):
        asyncio.run(
            visual_chat_completions_impl(
                request=_multimodal_request(
                    messages=[{"role": "user", "content": "Give me a final answer."}]
                ),
                raw_request=_raw_request(),
                runtime=_runtime(client=object(), empty_output_retries=1),
                main_chat_handler=fake_main,
            )
        )


def test_streaming_returns_openai_sse_chunks():
    async def fake_main(request, raw_request):
        return _response(
            ChatMessage(
                role="assistant",
                content="Buffered answer",
                reasoning_content="Buffered reasoning",
            )
        )

    runtime = _runtime(client=object())
    response = asyncio.run(
        visual_chat_completions_impl(
            request=_multimodal_request(
                stream=True,
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "text",
                                "text": "Ignore the image and reply with a buffered greeting.",
                            },
                            {
                                "type": "image_url",
                                "image_url": {"url": "data:image/png;base64,AAAA"},
                            },
                        ],
                    }
                ],
            ),
            raw_request=_raw_request(),
            runtime=runtime,
            main_chat_handler=fake_main,
        )
    )
    assert isinstance(response, StreamingResponse)

    async def collect_body():
        body = ""
        async for part in response.body_iterator:
            body += part.decode() if isinstance(part, bytes) else part
        return body

    body = asyncio.run(collect_body())
    assert '"object": "chat.completion.chunk"' in body
    assert '"reasoning": "Buffered reasoning"' in body
    assert '"reasoning_content"' not in body
    assert '"content": "Buffered answer"' in body
    assert '"usage"' in body
    assert body.endswith("data: [DONE]\n\n")
