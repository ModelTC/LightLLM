import asyncio
import json
from contextlib import aclosing
from types import SimpleNamespace

from lightllm.server import api_openai
from lightllm.server.api_models import CompletionRequest
from lightllm.server.core.objs import FinishStatus


def _collect_chunks(response):
    async def collect():
        return [chunk async for chunk in response.body_iterator]

    return asyncio.run(collect())


def _completion_stream(monkeypatch, results, *, echo=False):
    from lightllm.server.api_http import g_objs

    async def generate(*_args, **_kwargs):
        for choice, text, batch_end, reason in results:
            yield 80 + choice, text, {
                "prompt_tokens": 4,
                "prompt_cache_len": 2,
                "is_first_token": text == "A",
                "_pd_stream_batch_end": batch_end,
            }, FinishStatus(
                {
                    None: FinishStatus.NO_FINISH,
                    "stop": FinishStatus.FINISHED_STOP,
                    "length": FinishStatus.FINISHED_LENGTH,
                }[reason]
            )

    monkeypatch.setattr(g_objs, "httpserver_manager", SimpleNamespace(generate=generate), raising=False)
    request = CompletionRequest(model="test", prompt="Prompt", stream=True, echo=echo)
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


def test_pd_single_choice_closes_source_on_client_stop():
    from lightllm.server.httpserver_for_pd_master.manager import HttpServerManagerForPDMaster

    closed = []
    consumer_tasks = []

    async def generate():
        try:
            assert asyncio.current_task() is consumer_tasks[0]
            yield "first"
            yield "second"
        finally:
            closed.append(True)

    async def consume_first():
        consumer_tasks.append(asyncio.current_task())
        manager = HttpServerManagerForPDMaster.__new__(HttpServerManagerForPDMaster)
        async with aclosing(manager._merge_choice_generators([generate()])) as stream:
            assert await stream.__anext__() == "first"

    asyncio.run(consume_first())
    assert closed == [True]


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
