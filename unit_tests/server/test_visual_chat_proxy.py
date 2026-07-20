import asyncio
import base64
import json
from dataclasses import replace
from types import SimpleNamespace

import pytest
import httpx
from fastapi import Request
from fastapi.responses import StreamingResponse
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
from lightllm.server.visual_chat_proxy import (
    ImageRegistry,
    RegisteredImage,
    VisionTrace,
    VisualProxyCapacityError,
    VisualProxySettings,
    VisualProxyRuntime,
    VisualProxyTimeoutError,
    VisualProxyUpstreamError,
    call_visual_remote,
    decode_hidden_traces,
    expand_hidden_traces,
    replace_images_with_tags,
    should_use_visual_proxy,
    visual_chat_completions_impl,
)


_TEST_TRACE_SECRET = b"unit-test-visual-trace-secret-32-bytes-minimum"


def _runtime(client=None, **overrides):
    settings = VisualProxySettings(
        remote_url="https://vision.test/v1/chat/completions",
        trace_secrets=(_TEST_TRACE_SECRET,),
    )
    settings = replace(settings, **overrides)
    settings.validate()
    return VisualProxyRuntime(settings, client=client)


def _raw_request():
    return Request({"type": "http", "method": "POST", "path": "/v1/chat/completions", "headers": []})


def _response(message, finish_reason="stop", prompt_tokens=3, completion_tokens=2):
    return ChatCompletionResponse(
        model="agent",
        choices=[ChatCompletionResponseChoice(index=0, message=message, finish_reason=finish_reason)],
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
                    {"type": "image_url", "image_url": {"url": "data:image/png;base64,AAAA"}},
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
                    {"type": "image_url", "image_url": {"url": "https://example.test/a.png"}},
                    {"type": "image_url", "image_url": {"url": "data:image/png;base64,BBBB"}},
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


def test_proxy_activation_is_strictly_opt_in_and_multimodal():
    multimodal = _multimodal_request()
    text_only = ChatCompletionRequest(model="agent", messages=[{"role": "user", "content": "hello"}])

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
            return {"choices": [{"message": {"role": "assistant", "content": "remote result"}}]}

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
            image=RegisteredImage(tag="<image_1/>", source="data:image/png;base64,AAAA", origin="user"),
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
                        _call("vision_reader", {"image": "<image_1/>", "task": "first"}, "reader_1"),
                        _call("vision_reader", {"image": "<image_1/>", "task": "second"}, "reader_2"),
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
                            {"image": "<image_1/>", "task": "Identify the square's color."},
                            "call_visual_1",
                        )
                    ],
                ),
                finish_reason="tool_calls",
            )
        assert main_requests[-1]["messages"][-1] == {
            "role": "tool",
            "content": visual_chat_proxy._format_visual_tool_result("The square is green."),
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

    traces = decode_hidden_traces(response.choices[0].message.reasoning, runtime.trace_cipher)
    assert [(trace.image, trace.response) for trace in traces] == [
        (
            "<image_1/>",
            visual_chat_proxy._format_visual_tool_result("The square is green."),
        )
    ]


def test_missing_reader_task_falls_back_to_latest_user_text(monkeypatch):
    main_requests = []
    remote_calls = []

    async def fake_main(request, raw_request):
        main_requests.append(request)
        if len(main_requests) == 1:
            return _response(
                ChatMessage(
                    role="assistant",
                    content="",
                    tool_calls=[_call("vision_reader", {"image": "image_1"}, "call_missing_task")],
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
    assert "What color is the square?" in remote_calls[0]["task"]


def test_external_tools_stay_public_when_mixed_with_builtin(monkeypatch):
    main_requests = []

    async def fake_main(request, raw_request):
        main_requests.append(request.model_dump(exclude_none=True))
        if len(main_requests) > 1:
            assert main_requests[-1]["messages"][-1]["content"] == (
                visual_chat_proxy._format_visual_tool_result("The image says Shanghai.")
            )
            return _response(
                ChatMessage(
                    role="assistant",
                    content="",
                    tool_calls=[_call("get_weather", {"city": "Shanghai"}, "external_2")],
                ),
                finish_reason="tool_calls",
            )
        return _response(
            ChatMessage(
                role="assistant",
                content="",
                tool_calls=[
                    _call("vision_reader", {"image": "<image_1/>", "task": "Read the city."}, "builtin_1"),
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
    assert len(main_requests) == 2
    traces = decode_hidden_traces(message.reasoning, runtime.trace_cipher)
    assert len(traces) == 1
    expected_visual_result = visual_chat_proxy._format_visual_tool_result("The image says Shanghai.")
    assert traces[0].response == expected_visual_result

    replay_registry = ImageRegistry()
    replay_registry.add("data:image/png;base64,AAAA", "user", byte_size=3)
    replayed = expand_hidden_traces(
        [
            {
                "role": "assistant",
                "content": message.content,
                "reasoning": message.reasoning,
                "tool_calls": [call.model_dump(exclude_none=True) for call in message.tool_calls],
            }
        ],
        runtime.trace_cipher,
        registry=replay_registry,
        model="agent",
    )
    assert [item["role"] for item in replayed] == ["assistant", "tool", "assistant"]
    assert replayed[0]["tool_calls"][0]["function"]["name"] == "vision_reader"
    assert replayed[1]["content"] == expected_visual_result
    assert replayed[2]["tool_calls"][0]["function"]["name"] == "get_weather"

    legacy_replayed = expand_hidden_traces(
        [
            {
                "role": "assistant",
                "content": message.content,
                "reasoning_content": message.reasoning,
                "tool_calls": [call.model_dump(exclude_none=True) for call in message.tool_calls],
            }
        ],
        runtime.trace_cipher,
        registry=replay_registry,
        model="agent",
    )
    assert [item["role"] for item in legacy_replayed] == ["assistant", "tool", "assistant"]

    wrong_registry = ImageRegistry()
    wrong_registry.add("data:image/png;base64,BBBB", "user", byte_size=3)
    with pytest.raises(ValueError, match="image does not match"):
        expand_hidden_traces(
            [
                {
                    "role": "assistant",
                    "reasoning": message.reasoning,
                    "tool_calls": [call.model_dump(exclude_none=True) for call in message.tool_calls],
                }
            ],
            runtime.trace_cipher,
            registry=wrong_registry,
            model="agent",
        )
    with pytest.raises(ValueError, match="model does not match"):
        expand_hidden_traces(
            [{"role": "assistant", "reasoning": message.reasoning}],
            runtime.trace_cipher,
            registry=replay_registry,
            model="different-agent",
        )


def test_trace_is_encrypted_authenticated_and_supports_key_rotation():
    old_runtime = _runtime(client=object())
    trace = VisionTrace(
        call_id="call_secret",
        image="<image_1/>",
        task="Read a secret",
        response="sensitive visual result",
        reasoning="private reasoning",
    )
    encoded = visual_chat_proxy._encode_trace(trace, old_runtime.trace_cipher)

    assert "sensitive visual result" not in encoded
    assert decode_hidden_traces(encoded, old_runtime.trace_cipher) == [trace]

    marker_start = encoded.index(">") + 1
    tampered_character = "A" if encoded[marker_start] != "A" else "B"
    tampered = encoded[:marker_start] + tampered_character + encoded[marker_start + 1 :]
    with pytest.raises(ValueError, match="trace signature|Malformed|unsigned"):
        decode_hidden_traces(tampered, old_runtime.trace_cipher)

    rotated_runtime = _runtime(
        client=object(),
        trace_secrets=(b"new-production-trace-secret-32-bytes-minimum", _TEST_TRACE_SECRET),
    )
    assert decode_hidden_traces(encoded, rotated_runtime.trace_cipher) == [trace]

    unsigned_payload = base64.urlsafe_b64encode(b'{"response":"forged"}').decode().rstrip("=")
    unsigned = (
        f"{visual_chat_proxy._TRACE_OPEN}{unsigned_payload}{visual_chat_proxy._TRACE_CLOSE}"
    )
    with pytest.raises(ValueError, match="unsigned|Malformed"):
        decode_hidden_traces(unsigned, old_runtime.trace_cipher)

    expiring_runtime = _runtime(client=object(), trace_ttl_seconds=10)
    expired_trace = replace(trace, issued_at=int(visual_chat_proxy.time.time()) - 11)
    expired = visual_chat_proxy._encode_trace(expired_trace, expiring_runtime.trace_cipher)
    with pytest.raises(ValueError, match="expired"):
        decode_hidden_traces(expired, expiring_runtime.trace_cipher)


def test_startup_requires_trace_secret(monkeypatch):
    args = SimpleNamespace(visual_remote_url="https://vision.test/v1")
    monkeypatch.delenv("LIGHTLLM_VISUAL_TRACE_SECRET", raising=False)
    with pytest.raises(ValueError, match="at least 32 bytes"):
        VisualProxySettings.from_args(args)

    monkeypatch.setenv("LIGHTLLM_VISUAL_TRACE_SECRET", _TEST_TRACE_SECRET.decode())
    monkeypatch.setenv("LIGHTLLM_VISUAL_REMOTE_API_KEY", "upstream-token")
    monkeypatch.setenv("LIGHTLLM_VISUAL_REMOTE_HEADERS", '{"X-Tenant":"tenant-a"}')
    settings = VisualProxySettings.from_args(args)
    assert settings.remote_url == "https://vision.test/v1/chat/completions"
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


def test_image_sources_are_bounded_and_local_files_are_rooted(tmp_path):
    settings = _runtime(client=object()).settings
    assert visual_chat_proxy._remote_image_url("data:image/png;base64,AAAA", settings).startswith(
        "data:image/png;base64,"
    )
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
    assert visual_chat_proxy._remote_image_url(image.as_uri(), local_settings).startswith(
        "data:image/png;base64,"
    )
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

    runtime = _runtime(
        client=SlowClient(), remote_max_retries=0, remote_timeout=0.01
    )
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
        first = asyncio.create_task(runtime.post_json({"model": "vision"}, None, "first"))
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


def test_reserved_vision_reader_tool_name_is_rejected():
    vision_reader = Tool(
        type="function",
        function=Function(
            name="vision_reader",
            description="collision",
            parameters={"type": "object", "properties": {}},
        ),
    )

    async def should_not_run(request, raw_request):
        raise AssertionError("main model must not run for a reserved tool collision")

    with pytest.raises(ValueError, match="reserved"):
        asyncio.run(
            visual_chat_completions_impl(
                request=_multimodal_request(tools=[vision_reader.model_dump()]),
                raw_request=_raw_request(),
                runtime=_runtime(client=object()),
                main_chat_handler=should_not_run,
            )
        )


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
            request=_multimodal_request(stream=True),
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
