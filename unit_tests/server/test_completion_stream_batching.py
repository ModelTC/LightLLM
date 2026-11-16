import asyncio
import json
import pickle
from contextlib import aclosing
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

from lightllm.server import api_openai
from lightllm.server.api_models import CompletionRequest
from lightllm.server.core.objs import FinishStatus, SamplingParams, StartArgs


def _collect_chunks(response):
    async def collect():
        return [chunk async for chunk in response.body_iterator]

    return asyncio.run(collect())


def _finish_status(reason):
    return FinishStatus(
        {
            None: FinishStatus.NO_FINISH,
            "stop": FinishStatus.FINISHED_STOP,
            "length": FinishStatus.FINISHED_LENGTH,
        }[reason]
    )


def _completion_stream(monkeypatch, results, *, echo=False, logprobs=None, include_batch_end=True):
    from lightllm.server.api_http import g_objs

    async def generate(*_args, **_kwargs):
        for choice, text, batch_end, reason in results:
            metadata = {
                "prompt_tokens": 4,
                "prompt_cache_len": 2,
                "is_first_token": text == "A",
            }
            if include_batch_end:
                metadata["_pd_stream_batch_end"] = batch_end
            yield 80 + choice, text, metadata, _finish_status(reason)

    monkeypatch.setattr(g_objs, "httpserver_manager", SimpleNamespace(generate=generate), raising=False)
    request = CompletionRequest(model="test", prompt="Prompt", stream=True, echo=echo, logprobs=logprobs)
    return asyncio.run(api_openai._handle_streaming_completion("Prompt", None, None, None, request, 123))


def _completion_payloads(response):
    return [
        json.loads(event[6:])
        for chunk in _collect_chunks(response)
        for event in (chunk.decode() if isinstance(chunk, bytes) else chunk).split("\n\n")
        if event.startswith("data: ") and event != "data: [DONE]"
    ]


def test_completion_stream_merges_ready_text_and_counts_tokens(monkeypatch):
    response = _completion_stream(
        monkeypatch,
        [
            (0, "A", False, None),
            (0, "你", False, None),
            (0, "好", True, "length"),
        ],
        echo=True,
    )
    payloads = _completion_payloads(response)
    assert [p["choices"][0]["text"] for p in payloads[:-1]] == ["PromptA", "你好"]
    assert payloads[-2]["choices"][0]["finish_reason"] == "length"
    assert payloads[-1]["usage"]["completion_tokens"] == 3
    assert payloads[-1]["usage"]["prompt_tokens_details"]["cached_tokens"] == 2


def test_completion_stream_keeps_choices_and_flushes_final_partial_batch(monkeypatch):
    response = _completion_stream(
        monkeypatch,
        [
            (0, "A", False, None),
            (1, "X", False, None),
            (0, "B", False, None),
            (1, "Y", False, None),
        ],
    )
    payloads = _completion_payloads(response)
    assert payloads[-2]["choices"] == [
        {"text": "B", "index": 0, "logprobs": None, "finish_reason": None},
        {"text": "Y", "index": 1, "logprobs": None, "finish_reason": None},
    ]
    assert payloads[-1]["usage"]["completion_tokens"] == 4


def test_completion_stream_bounds_ready_batch_and_flushes_finish(monkeypatch):
    results = [(0, str(i % 10), False, "stop" if i == 99 else None) for i in range(100)]
    payloads = _completion_payloads(_completion_stream(monkeypatch, results))
    texts = [p["choices"][0]["text"] for p in payloads[:-1]]
    assert len(texts[0]) == 1
    assert all(len(text) <= 32 for text in texts)
    assert "".join(texts) == "".join(str(i % 10) for i in range(100))
    assert payloads[-2]["choices"][0]["finish_reason"] == "stop"
    assert payloads[-1]["usage"]["completion_tokens"] == 100


def test_completion_without_batch_metadata_flushes_each_token(monkeypatch):
    response = _completion_stream(
        monkeypatch,
        [(0, "A", False, None), (0, "B", False, None), (0, "C", False, "stop")],
        include_batch_end=False,
    )
    payloads = _completion_payloads(response)
    assert [p["choices"][0]["text"] for p in payloads[:-1]] == ["A", "B", "C"]
    assert payloads[-2]["choices"][0]["logprobs"] is None
    assert payloads[-1]["usage"]["completion_tokens"] == 3


def test_completion_stream_preserves_requested_logprobs(monkeypatch):
    response = _completion_stream(monkeypatch, [(0, "A", True, "stop")], logprobs=1)
    choice = _completion_payloads(response)[0]["choices"][0]
    assert choice["text"] == "A"
    assert choice["logprobs"] == {}
    assert choice["finish_reason"] == "stop"


def test_completion_first_token_does_not_wait_for_batch(monkeypatch):
    from lightllm.server.api_http import g_objs

    async def generate(*_args, **_kwargs):
        yield 80, "A", {"prompt_tokens": 4, "_pd_stream_batch_end": False}, FinishStatus()
        raise AssertionError("The next token must not be fetched before the first chunk is emitted")

    monkeypatch.setattr(g_objs, "httpserver_manager", SimpleNamespace(generate=generate), raising=False)

    async def first_chunk():
        request = CompletionRequest(model="test", prompt="Prompt", stream=True)
        response = await api_openai._handle_streaming_completion("Prompt", None, None, None, request, 123)
        async with aclosing(response.body_iterator) as stream:
            return await stream.__anext__()

    assert json.loads(asyncio.run(first_chunk())[6:])["choices"][0]["text"] == "A"


def test_completion_stream_flushes_pending_text_before_error(monkeypatch):
    from lightllm.server.api_http import g_objs

    async def generate(*_args, **_kwargs):
        for text in ["A", "B"]:
            yield 80, text, {"prompt_tokens": 4, "_pd_stream_batch_end": False}, FinishStatus()
        raise ValueError("generation failed")

    monkeypatch.setattr(g_objs, "httpserver_manager", SimpleNamespace(generate=generate), raising=False)
    request = CompletionRequest(model="test", prompt="Prompt", stream=True)
    response = asyncio.run(api_openai._handle_streaming_completion("Prompt", None, None, None, request, 123))
    payloads = _completion_payloads(response)
    assert [p["choices"][0]["text"] for p in payloads[:-1]] == ["A", "B"]
    assert payloads[-1]["error"]["message"] == "generation failed"


def test_completion_batch_end_flushes_before_waiting_for_next_batch(monkeypatch):
    from lightllm.server.api_http import g_objs

    async def generate(*_args, **_kwargs):
        for text, batch_end in [("A", False), ("B", False), ("C", True)]:
            yield 80, text, {"prompt_tokens": 4, "_pd_stream_batch_end": batch_end}, FinishStatus()
        raise AssertionError("The completed batch must be emitted before fetching another token")

    monkeypatch.setattr(g_objs, "httpserver_manager", SimpleNamespace(generate=generate), raising=False)

    async def run():
        request = CompletionRequest(model="test", prompt="Prompt", stream=True)
        response = await api_openai._handle_streaming_completion("Prompt", None, None, None, request, 123)
        async with aclosing(response.body_iterator) as stream:
            assert json.loads((await stream.__anext__())[6:])["choices"][0]["text"] == "A"
            assert json.loads((await stream.__anext__())[6:])["choices"][0]["text"] == "BC"

    asyncio.run(run())


def test_completion_stream_flushes_at_character_limit(monkeypatch):
    response = _completion_stream(
        monkeypatch,
        [(0, "A", False, None), (0, "B" * 32768, False, None), (0, "C" * 32768, False, None), (0, "D", True, "stop")],
    )
    payloads = _completion_payloads(response)
    assert [p["choices"][0]["text"] for p in payloads[:-1]] == ["A", "B" * 32768 + "C" * 32768, "D"]
    assert payloads[-1]["usage"]["completion_tokens"] == 4


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
        assert first[2]["needs_prefill_first_token"] is False
        assert second[1] == "second" and second[2]["_pd_stream_batch_end"] is True
        await stream.aclose()

    asyncio.run(run())
