import asyncio
import json
import pickle
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import pytest

from lightllm.server import api_openai
from lightllm.server.api_models import (
    ChatCompletionRequest,
    ChatCompletionStreamResponse,
    ChatCompletionStreamResponseChoice,
    CompletionRequest,
    CompletionStreamChoice,
    CompletionStreamResponse,
    DeltaMessage,
    Function,
    Tool,
)
from lightllm.server.core.objs import FinishStatus, SamplingParams, StartArgs
from lightllm.server.httpserver.async_queue import AsyncQueue
from lightllm.server.httpserver.manager import HttpServerManager
from lightllm.utils.error_utils import InvalidRequestError


@pytest.mark.parametrize(
    "delta,choice_nulls,response_nulls,finish",
    [
        ({"role": "assistant", "content": ""}, ("logprobs", "finish_reason"), ("prompt_token_ids",), None),
        ({"content": "你好"}, ("logprobs", "token_ids", "finish_reason"), (), None),
        ({"reasoning": "think"}, ("logprobs", "token_ids", "finish_reason"), (), None),
        ({"content": "done"}, ("logprobs", "token_ids", "stop_reason"), (), "stop"),
        ({}, ("logprobs", "token_ids", "stop_reason"), (), "length"),
    ],
)
def test_fast_chat_serializer_matches_existing_models(delta, choice_nulls, response_nulls, finish):
    old = ChatCompletionStreamResponse(
        id="chatcmpl-test",
        created=123,
        model="test",
        choices=[ChatCompletionStreamResponseChoice(index=1, delta=DeltaMessage(**delta), finish_reason=finish)],
    )
    expected = json.loads(api_openai._serialize_sse_chunk(old, choice_nulls, response_nulls))
    actual = json.loads(
        api_openai._serialize_chat_sse_chunk(
            "chatcmpl-test", 123, "test", 1, delta, choice_nulls, response_nulls, finish
        )
    )
    assert actual == expected


def _install_manager(monkeypatch, rows, failure=None, reasoning_parser=None):
    from lightllm.server.api_http import g_objs

    async def generate(*_args, **_kwargs):
        for text, metadata, finish in rows:
            yield 0, text, {"prompt_tokens": 10, "prompt_cache_len": 2, **metadata}, FinishStatus(finish)
        if failure:
            raise failure

    monkeypatch.setattr(g_objs, "httpserver_manager", SimpleNamespace(tokenizer=Mock(), generate=generate))
    monkeypatch.setattr(g_objs, "args", StartArgs(tool_call_parser="qwen3_coder"))
    monkeypatch.setattr(api_openai, "get_env_start_args", lambda: StartArgs(reasoning_parser=reasoning_parser))
    monkeypatch.setattr(api_openai, "build_prompt", AsyncMock(return_value="prompt"))
    monkeypatch.setattr(SamplingParams, "init", lambda *_a, **_kw: None)
    monkeypatch.setattr(SamplingParams, "verify", lambda *_a: None)


def _chat_response(monkeypatch, rows, failure=None, tools=None):
    _install_manager(monkeypatch, rows, failure)
    request = ChatCompletionRequest(
        model="test",
        messages=[{"role": "user", "content": "hi"}],
        stream=True,
        tools=tools,
        tool_choice="auto" if tools else "none",
    )

    async def run():
        response = await api_openai.chat_completions_impl(request, Mock())
        return [chunk async for chunk in response.body_iterator]

    return asyncio.run(run())


def _events(chunks):
    text = "".join(chunk.decode() if isinstance(chunk, bytes) else chunk for chunk in chunks)
    return [
        json.loads(event[6:]) for event in text.split("\n\n") if event.startswith("data: ") and event != "data: [DONE]"
    ]


def test_pd_batch_coalesces_sse_without_changing_events_or_usage(monkeypatch):
    rows = [
        ("a", {"_pd_stream_batch_end": False}, FinishStatus.NO_FINISH),
        ("b", {"_pd_stream_batch_end": True}, FinishStatus.FINISHED_STOP),
    ]
    batched = _chat_response(monkeypatch, rows)
    baseline = _chat_response(monkeypatch, [(text, {}, finish) for text, _, finish in rows])
    actual, expected = _events(batched), _events(baseline)
    for event in actual + expected:
        event.pop("id")
        event.pop("created")
    assert actual == expected
    assert len(batched) < len(baseline)
    assert actual[0]["choices"][0]["delta"] == {"role": "assistant", "content": ""}
    assert actual[-1]["usage"]["completion_tokens"] == 2
    assert "_pd_stream_batch_end" not in json.dumps(actual)


def test_batch_end_flushes_before_waiting_for_more_model_output(monkeypatch):
    from lightllm.server.api_http import g_objs

    _install_manager(monkeypatch, [])
    second_token_started = []

    async def generate(*_args, **_kwargs):
        yield 0, "first", {"prompt_tokens": 10, "_pd_stream_batch_end": True}, FinishStatus()
        second_token_started.append(True)
        await asyncio.Event().wait()

    monkeypatch.setattr(g_objs.httpserver_manager, "generate", generate)
    request = ChatCompletionRequest(model="test", messages=[{"role": "user", "content": "hi"}], stream=True)

    async def run():
        response = await api_openai.chat_completions_impl(request, Mock())
        chunk = await asyncio.wait_for(response.body_iterator.__anext__(), 1)
        assert "first" in chunk
        assert not second_token_started
        await response.body_iterator.aclose()

    asyncio.run(run())


def test_buffered_events_are_flushed_before_stream_error(monkeypatch):
    chunks = _chat_response(
        monkeypatch,
        [("kept", {"_pd_stream_batch_end": False}, FinishStatus.NO_FINISH)],
        failure=ValueError("generation failed"),
    )
    events = _events(chunks)
    assert events[1]["choices"][0]["delta"]["content"] == "kept"
    assert events[2]["error"]["message"] == "generation failed"


def test_tool_calls_keep_id_arguments_and_finish_reason(monkeypatch):
    tool = Tool(
        function=Function(
            name="write_file", parameters={"type": "object", "properties": {"content": {"type": "string"}}}
        )
    )
    chunks = _chat_response(
        monkeypatch,
        [
            (
                "<tool_call>\n<function=write_file>\n<parameter=content>hello",
                {"_pd_stream_batch_end": False},
                FinishStatus.NO_FINISH,
            ),
            ("</parameter>\n</function>\n</tool_call>", {"_pd_stream_batch_end": True}, FinishStatus.FINISHED_STOP),
        ],
        tools=[tool],
    )
    events = _events(chunks)
    choices = [event["choices"][0] for event in events if event.get("choices")]
    calls = [call for choice in choices for call in choice["delta"].get("tool_calls", [])]
    assert calls[0]["id"].startswith("call_")
    assert calls[0]["function"]["name"] == "write_file"
    assert json.loads("".join(call["function"].get("arguments", "") for call in calls)) == {"content": "hello"}
    assert choices[-1]["finish_reason"] == "tool_calls"


@pytest.mark.parametrize("logprobs", [None, 1])
def test_completion_stream_preserves_logprobs_payload_and_echo(monkeypatch, logprobs):
    _install_manager(monkeypatch, [("answer", {"is_first_token": True}, FinishStatus.FINISHED_STOP)])
    request = CompletionRequest(model="test", prompt="prompt", stream=True, echo=True, logprobs=logprobs)

    async def run():
        response = await api_openai._handle_streaming_completion(
            "prompt", SamplingParams(), Mock(), Mock(), request, 123
        )
        return [chunk async for chunk in response.body_iterator]

    actual = _events(asyncio.run(run()))[0]
    old = CompletionStreamResponse(
        id=0,
        created=123,
        model="test",
        choices=[
            CompletionStreamChoice(
                index=0, text="promptanswer", finish_reason="stop", logprobs=None if logprobs is None else {}
            )
        ],
    ).model_dump()
    assert actual == old


@pytest.mark.parametrize("prompt_tokens,output_tokens,expected", [(10, 5, 5), (60, 10, 4), (1_000_000, 5, 5)])
def test_length_validation_uses_counts_and_preserves_repair(prompt_tokens, output_tokens, expected):
    manager = HttpServerManager.__new__(HttpServerManager)
    manager.max_req_total_len = 1_048_576 if prompt_tokens > 100 else 100
    manager.get_real_supported_max_req_total_len = lambda: 1_048_540 if prompt_tokens > 100 else 64
    params = SimpleNamespace(max_new_tokens=output_tokens)
    manager._check_and_repair_length(prompt_tokens, params)
    assert params.max_new_tokens == expected


@pytest.mark.parametrize("prompt_tokens", [0, 64, 65])
def test_length_validation_rejects_empty_or_exhausted_context(prompt_tokens):
    manager = HttpServerManager.__new__(HttpServerManager)
    manager.max_req_total_len = 100
    manager.get_real_supported_max_req_total_len = lambda: 64
    with pytest.raises(InvalidRequestError):
        manager._check_and_repair_length(prompt_tokens, SimpleNamespace(max_new_tokens=5))


def test_async_queue_drain_does_not_lose_concurrent_producers():
    async def run():
        queue = AsyncQueue()
        await asyncio.gather(*(queue.put(index) for index in range(100)))
        assert queue.event.is_set()
        assert sorted(await queue.get_all_data()) == list(range(100))
        assert not queue.event.is_set()
        await queue.put(100)
        assert await queue.wait_to_get_all_data() == [100]

    asyncio.run(run())


def test_pd_batch_end_is_last_emitted_token_not_filtered_duplicate():
    from lightllm.server.httpserver_for_pd_master.manager import HttpServerManagerForPDMaster
    from lightllm.server.pd_io_struct import PD_Client_Obj

    async def run():
        manager = HttpServerManagerForPDMaster.__new__(HttpServerManagerForPDMaster)
        manager.args = StartArgs(pd_node_id=1)
        manager.metric_client = Mock()
        manager.req_id_to_out_inf = {}
        manager._wait_for_prefill_token_if_needed = AsyncMock(return_value=0)
        nodes = [
            PD_Client_Obj(
                node_id=index,
                client_ip_port=f"node-{index}:8000",
                mode=mode,
                start_args={},
                websocket=SimpleNamespace(send_bytes=AsyncMock()),
            )
            for index, mode in enumerate(["prefill", "decode"])
        ]

        async def wait(event, _request, **kwargs):
            if kwargs["stage"] == "prefill":
                event.prompt_ids = [1, 2, 3]
            else:
                event.upkv_status = SimpleNamespace(pd_kv_trans_params=pickle.dumps(SimpleNamespace(ready_kv_len=2)))
                status = manager.req_id_to_out_inf[0]
                status.out_token_info_list = [
                    (0, "first", {"count_output_tokens": 1, "node_mode": "prefill"}, FinishStatus()),
                    (0, "second", {"count_output_tokens": 2}, FinishStatus()),
                    (0, "duplicate", {"count_output_tokens": 1, "node_mode": "decode"}, FinishStatus()),
                ]
                status.event.set()

        manager._wait_for_event_or_disconnect = wait
        params = SamplingParams()
        params.group_request_id = 0
        params.max_new_tokens = 5
        request = SimpleNamespace(is_disconnected=AsyncMock(return_value=False))
        stream = manager.fetch_pd_stream(*nodes, "prompt", params, SimpleNamespace(), request)
        first = await asyncio.wait_for(stream.__anext__(), 1)
        second = await asyncio.wait_for(stream.__anext__(), 1)
        assert first[1] == "first" and first[2]["_pd_stream_batch_end"] is False
        assert second[1] == "second" and second[2]["_pd_stream_batch_end"] is True
        await stream.aclose()

    asyncio.run(run())


@pytest.mark.parametrize("timeout", [False, True])
def test_pd_receive_stays_in_handler_task_and_preserves_heartbeat_timeout(monkeypatch, timeout):
    import anyio
    from fastapi import WebSocketDisconnect
    from lightllm.server import api_http_pd
    from lightllm.server.api_http import g_objs
    from lightllm.server.pd_io_struct import ObjType

    manager = SimpleNamespace(register_pd=AsyncMock(), put_to_handle_queue=AsyncMock(), remove_pd=AsyncMock())
    monkeypatch.setattr(g_objs, "httpserver_manager", manager)
    monkeypatch.setattr(api_http_pd, "get_lightllm_websocket_max_message_size", lambda: 1024)
    original_fail_after = anyio.fail_after
    deadlines = []

    def fail_after(seconds):
        deadlines.append(seconds)
        return original_fail_after(0.01 if timeout else seconds)

    monkeypatch.setattr(api_http_pd.anyio, "fail_after", fail_after)

    async def run():
        handler_task = asyncio.current_task()
        receives = []

        async def receive():
            assert asyncio.current_task() is handler_task
            receives.append(True)
            if timeout:
                await asyncio.Event().wait()
            if len(receives) == 1:
                return pickle.dumps((ObjType.HEARTBEAT,))
            raise WebSocketDisconnect()

        websocket = SimpleNamespace(
            accept=AsyncMock(),
            client=("127.0.0.1", 8000),
            receive_text=AsyncMock(return_value="{}"),
            receive_bytes=receive,
            close=AsyncMock(),
        )
        await api_http_pd.register_and_keep_alive(websocket)
        manager.remove_pd.assert_awaited_once()
        manager.put_to_handle_queue.assert_not_awaited()
        assert deadlines and all(seconds == 30 for seconds in deadlines)
        if timeout:
            websocket.close.assert_awaited_once_with(code=1011, reason="PD heartbeat timed out")

    asyncio.run(run())
