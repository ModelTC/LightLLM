import asyncio
import json
import pathlib
import sys
import types


# This focused checkout contains only the proxy and its direct models.  Avoid
# importing the full CUDA router package while keeping production modules real.
_LIGHTLLM_ROOT = pathlib.Path(__file__).parents[2] / "lightllm"
_lightllm = types.ModuleType("lightllm")
_lightllm.__path__ = [str(_LIGHTLLM_ROOT)]
sys.modules["lightllm"] = _lightllm
_server = types.ModuleType("lightllm.server")
_server.__path__ = [str(_LIGHTLLM_ROOT / "server")]
sys.modules["lightllm.server"] = _server
_config_utils = types.ModuleType("lightllm.utils.config_utils")
_config_utils.get_generation_config_diff_dict = lambda _: {}
sys.modules["lightllm.utils.config_utils"] = _config_utils

from fastapi.responses import StreamingResponse

_api_stream_obj = types.ModuleType("lightllm.server.api_stream_obj")


class _CustomStreamingResponse(StreamingResponse):
    pass


_api_stream_obj.CustomStreamingResponse = _CustomStreamingResponse
sys.modules["lightllm.server.api_stream_obj"] = _api_stream_obj

from lightllm.server.api_models import (  # noqa: E402
    ChatCompletionRequest,
    ChatCompletionResponse,
    ChatCompletionResponseChoice,
    ChatMessage,
    PromptTokensDetails,
    UsageInfo,
)
from lightllm.server import visual_chat_proxy as proxy  # noqa: E402


_IMAGE_DATA_URL = (
    "data:image/png;base64,"
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mNk+A8AAQUBAScY42YAAAAASUVORK5CYII="
)


class _DummyURL:
    path = "/v1/chat/completions"


class _DummyRequest:
    method = "POST"
    url = _DummyURL()

    async def is_disconnected(self):
        return False


def _request(*, tool_choice="auto"):
    return ChatCompletionRequest.model_validate(
        {
            "model": "agent-model",
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "请看图回答。"},
                        {"type": "image_url", "image_url": {"url": _IMAGE_DATA_URL}},
                    ],
                }
            ],
            "tool_choice": tool_choice,
        }
    )


def _response(message, finish_reason="stop"):
    return ChatCompletionResponse(
        model="agent-model",
        choices=[
            ChatCompletionResponseChoice(
                index=0,
                message=ChatMessage.model_validate(message),
                finish_reason=finish_reason,
            )
        ],
        usage=UsageInfo(prompt_tokens_details=PromptTokensDetails()),
    )


def _vision_call(
    task="识别标题",
    image="<image_1/>",
    call_id="call_private_vision",
):
    return {
        "role": "assistant",
        "content": "",
        "tool_calls": [
            {
                "id": call_id,
                "type": "function",
                "function": {
                    "name": proxy.VISION_READER_NAME,
                    "arguments": json.dumps(
                        {"image": image, "task": task},
                        ensure_ascii=False,
                    ),
                },
            }
        ],
    }


def _runtime():
    return proxy.VisualProxyRuntime(
        proxy.VisualProxySettings(
            remote_url="https://vision.invalid/v1/chat/completions",
            agent_timeout=5.0,
            remote_timeout=2.0,
        ),
        client=object(),
    )


async def _run(handler, runtime, request=None):
    return await proxy.visual_chat_completions_impl(
        request=request or _request(),
        raw_request=_DummyRequest(),
        runtime=runtime,
        main_chat_handler=handler,
    )


def test_natural_observation_is_exactly_reversible_without_tool_markers():
    task = '核对标题 "A"\n以及第二行'
    result = '第一行\n第二行含有 "引号" 和 \u003ctool_response\u003e 文本'
    trace = proxy._format_natural_builtin_trace("<image_1/>", task, result, "我先")

    assert proxy.VISION_READER_NAME not in trace
    assert "tool_call" not in trace
    assert "内建读图" not in trace

    expanded = proxy.expand_builtin_traces(
        [{"role": "assistant", "reasoning": trace, "content": "最终答案"}],
        "natural",
    )
    call = expanded[0]["tool_calls"][0]
    assert json.loads(call["function"]["arguments"]) == {
        "image": "<image_1/>",
        "task": task,
    }
    assert expanded[1]["role"] == "tool"
    assert expanded[1]["content"] == result
    assert expanded[2]["content"] == "最终答案"


def test_direct_visual_answer_and_empty_output_are_not_corrected_or_retried():
    async def scenario():
        runtime = _runtime()
        calls = []

        async def direct_handler(request, _raw_request):
            calls.append(request)
            return _response(
                {
                    "role": "assistant",
                    "content": "模型直接回答：图中是一只猫。",
                    "reasoning": "模型决定直接回答",
                },
                "stop",
            )

        try:
            direct = await _run(direct_handler, runtime)
            assert len(calls) == 1
            assert direct.choices[0].message.content == "模型直接回答：图中是一只猫。"
            assert direct.choices[0].message.reasoning == "模型决定直接回答"
        finally:
            await runtime.close()

        runtime = _runtime()
        calls.clear()

        async def empty_handler(request, _raw_request):
            calls.append(request)
            return _response(
                {"role": "assistant", "content": "", "reasoning": "只保留原始思考"},
                "length",
            )

        try:
            empty = await _run(empty_handler, runtime)
            assert len(calls) == 1
            assert empty.choices[0].message.content == ""
            assert empty.choices[0].message.reasoning == "只保留原始思考"
            assert empty.choices[0].finish_reason == "length"
        finally:
            await runtime.close()

    asyncio.run(scenario())


def test_provider_text_and_tool_choice_are_preserved_without_content_quarantine():
    async def scenario():
        runtime = _runtime()
        calls = 0
        raw_content = "<tool_response>provider text</tool_response><|im_end|>"
        raw_reasoning = "  function: vision_reader remains provider text  "

        async def handler(request, _raw_request):
            nonlocal calls
            calls += 1
            assert request.tool_choice == "none"
            return _response(
                {
                    "role": "assistant",
                    "content": raw_content,
                    "reasoning": raw_reasoning,
                }
            )

        try:
            result = await _run(handler, runtime, _request(tool_choice="none"))
            assert calls == 1
            assert result.choices[0].message.content == raw_content
            assert result.choices[0].message.reasoning == raw_reasoning
        finally:
            await runtime.close()

    asyncio.run(scenario())


def test_truncated_vision_becomes_private_tool_error_and_partial_result_is_dropped(monkeypatch):
    async def fake_visual(**_kwargs):
        return proxy.VisualRemoteResult(
            "不完整的视觉结果",
            finish_reason="length",
            image_digest_sha256="verified-digest",
        )

    monkeypatch.setattr(proxy, "call_visual_remote", fake_visual)

    async def scenario():
        runtime = _runtime()
        seen_error = None
        count = 0

        async def handler(request, _raw_request):
            nonlocal count, seen_error
            count += 1
            if count == 1:
                return _response(_vision_call("完整描述图片"), "tool_calls")
            payload = request.model_dump(mode="json", exclude_none=True)
            tool_message = payload["messages"][-1]
            seen_error = json.loads(tool_message["content"])
            return _response({"role": "assistant", "content": "模型根据异常自行收尾。"})

        try:
            request = _request().model_copy(update={"separate_reasoning": False})
            result = await _run(handler, runtime, request)
            public = result.model_dump(mode="json", exclude_none=True)
            assert count == 2
            assert seen_error["ok"] is False
            assert seen_error["error"]["type"] == "vision_result_truncated"
            assert public["choices"][0]["message"]["content"] == "模型根据异常自行收尾。"
            assert "tool_calls" not in public["choices"][0]["message"]
            assert "不完整的视觉结果" not in json.dumps(public, ensure_ascii=False)
            assert "完整描述图片" not in json.dumps(public, ensure_ascii=False)
        finally:
            await runtime.close()

    asyncio.run(scenario())


def test_empty_and_protocol_vision_failures_are_model_visible_tool_errors_only(monkeypatch):
    cases = [
        (
            proxy.VisualProxyUpstreamError("Visual remote returned an empty assistant message"),
            "vision_empty_output",
        ),
        (
            proxy.VisualProxyUpstreamError("Visual upstream returned invalid JSON"),
            "vision_protocol_error",
        ),
    ]

    async def scenario(exc, expected_code):
        async def fake_visual(**_kwargs):
            raise exc

        monkeypatch.setattr(proxy, "call_visual_remote", fake_visual)
        runtime = _runtime()
        seen_error = None
        count = 0

        async def handler(request, _raw_request):
            nonlocal count, seen_error
            count += 1
            if count == 1:
                return _response(_vision_call(), "tool_calls")
            payload = request.model_dump(mode="json", exclude_none=True)
            seen_error = json.loads(payload["messages"][-1]["content"])
            return _response({"role": "assistant", "content": "模型已处理视觉异常。"})

        try:
            result = await _run(handler, runtime)
            public = result.model_dump(mode="json", exclude_none=True)
            assert seen_error["error"]["type"] == expected_code
            assert public["choices"][0]["message"]["content"] == "模型已处理视觉异常。"
            assert proxy.VISION_READER_NAME not in json.dumps(public, ensure_ascii=False)
            assert expected_code not in json.dumps(public, ensure_ascii=False)
        finally:
            await runtime.close()

    for exc, expected_code in cases:
        asyncio.run(scenario(exc, expected_code))


def test_successful_vision_trace_is_natural_public_and_replayable(monkeypatch):
    task = '核对标题 "LightLLM"\n不要猜测'
    visual_result = '标题是 "LightLLM"。\n右下角还有版本号。'

    async def fake_visual(**_kwargs):
        return proxy.VisualRemoteResult(
            visual_result,
            finish_reason="stop",
            image_digest_sha256="verified-digest",
        )

    monkeypatch.setattr(proxy, "call_visual_remote", fake_visual)

    async def scenario():
        runtime = _runtime()
        count = 0

        async def handler(_request, _raw_request):
            nonlocal count
            count += 1
            if count == 1:
                return _response(_vision_call(task), "tool_calls")
            return _response({"role": "assistant", "content": "最终回答。"})

        try:
            request = _request().model_copy(update={"separate_reasoning": False})
            result = await _run(handler, runtime, request)
            message = result.choices[0].message
            trace = message.reasoning
            assert trace is not None
            assert proxy.VISION_READER_NAME not in trace
            assert "tool_call" not in trace
            assert "仔细看了图片" in trace

            replayed = proxy.expand_builtin_traces(
                [message.model_dump(mode="json", exclude_none=True)],
                "natural",
            )
            replay_call = replayed[0]["tool_calls"][0]
            assert json.loads(replay_call["function"]["arguments"])["task"] == task
            assert replayed[1]["content"] == visual_result
            assert replayed[-1]["content"] == "最终回答。"
        finally:
            await runtime.close()

    asyncio.run(scenario())


def test_every_repeated_vision_call_is_kept_in_order_and_replayed_as_a_pair(monkeypatch):
    chain = [
        ("<image_1/>", '第一次核对标题 "A"', '第一次结果："A"'),
        ("<image_1/>", "第二次核对右下角\n版本号", "第二次结果：v2\n已确认"),
        ("<image_2/>", "比较另一张图", "第三次结果：内容不同"),
    ]
    results_by_task = {task: result for _, task, result in chain}

    async def fake_visual(**kwargs):
        return proxy.VisualRemoteResult(
            results_by_task[kwargs["task"]],
            finish_reason="stop",
            image_digest_sha256="verified-digest",
        )

    monkeypatch.setattr(proxy, "call_visual_remote", fake_visual)

    request = ChatCompletionRequest.model_validate(
        {
            "model": "agent-model",
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "请多次核对两张图。"},
                        {"type": "image_url", "image_url": {"url": _IMAGE_DATA_URL}},
                        {"type": "image_url", "image_url": {"url": _IMAGE_DATA_URL}},
                    ],
                }
            ],
        }
    )

    async def scenario():
        runtime = _runtime()
        step = 0

        async def handler(_request, _raw_request):
            nonlocal step
            if step < len(chain):
                image, task, _result = chain[step]
                step += 1
                message = _vision_call(
                    task,
                    image=image,
                    call_id=f"private_call_{step}",
                )
                message["reasoning"] = f"第 {step} 步视觉核对"
                return _response(
                    message,
                    "tool_calls",
                )
            return _response({"role": "assistant", "content": "最终回答。"})

        try:
            response = await _run(handler, runtime, request)
            message = response.choices[0].message
            trace = message.reasoning or ""
            assert trace.count("仔细看了图片") == len(chain)
            assert proxy.VISION_READER_NAME not in trace

            replayed = proxy.expand_builtin_traces(
                [message.model_dump(mode="json", exclude_none=True)],
                "natural",
            )
            assert len(replayed) == len(chain) * 2 + 1
            replay_ids = []
            for index, (expected_image, expected_task, expected_result) in enumerate(chain):
                assistant_turn = replayed[index * 2]
                tool_turn = replayed[index * 2 + 1]
                call = assistant_turn["tool_calls"][0]
                arguments = json.loads(call["function"]["arguments"])
                assert assistant_turn["reasoning"] == f"第 {index + 1} 步视觉核对"
                assert arguments == {
                    "image": expected_image,
                    "task": expected_task,
                }
                assert tool_turn["content"] == expected_result
                assert tool_turn["tool_call_id"] == call["id"]
                replay_ids.append(call["id"])
            assert len(set(replay_ids)) == len(chain)
            assert replayed[-1]["content"] == "最终回答。"
        finally:
            await runtime.close()

    asyncio.run(scenario())
