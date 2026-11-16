import asyncio
import json
from contextlib import aclosing
from types import SimpleNamespace

from lightllm.server import api_openai
from lightllm.server.api_cli import make_argument_parser
from lightllm.server.api_models import CompletionRequest
from lightllm.server.core.objs import FinishStatus, StartArgs


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


def _completion_stream(monkeypatch, results, *, echo=False, logprobs=None, interval=1):
    from lightllm.server.api_http import g_objs

    async def generate(*_args, **_kwargs):
        for choice, text, reason in results:
            yield 80 + choice, text, {
                "prompt_tokens": 4,
                "prompt_cache_len": 2,
                "is_first_token": text == "A",
            }, _finish_status(reason)

    monkeypatch.setattr(g_objs, "httpserver_manager", SimpleNamespace(generate=generate), raising=False)
    monkeypatch.setattr(api_openai, "get_env_start_args", lambda: SimpleNamespace(stream_interval=interval))
    request = CompletionRequest(model="test", prompt="Prompt", stream=True, echo=echo, logprobs=logprobs)
    return asyncio.run(api_openai._handle_streaming_completion("Prompt", None, None, None, request, 123))


def _completion_payloads(response):
    return [
        json.loads(event[6:])
        for chunk in _collect_chunks(response)
        for event in (chunk.decode() if isinstance(chunk, bytes) else chunk).split("\n\n")
        if event.startswith("data: ") and event != "data: [DONE]"
    ]


def _texts(response):
    payloads = _completion_payloads(response)
    return [choice["text"] for payload in payloads[:-1] for choice in payload["choices"]], payloads


def test_stream_interval_defaults_to_one_token():
    assert StartArgs().stream_interval == make_argument_parser().parse_args([]).stream_interval == 1


def test_interval_one_sends_each_token(monkeypatch):
    response = _completion_stream(
        monkeypatch,
        [(0, "A", None), (0, "B", None), (0, "C", "stop")],
        echo=True,
    )
    texts, payloads = _texts(response)
    assert texts == ["PromptA", "B", "C"]
    assert payloads[-2]["choices"][0]["finish_reason"] == "stop"
    assert payloads[-2]["choices"][0]["logprobs"] is None
    assert payloads[-1]["usage"]["completion_tokens"] == 3
    assert payloads[-1]["usage"]["prompt_tokens_details"]["cached_tokens"] == 2


def test_interval_batches_after_the_first_token_and_flushes_finish(monkeypatch):
    tokens = [(0, str(i), "length" if i == 9 else None) for i in range(10)]
    texts, payloads = _texts(_completion_stream(monkeypatch, tokens, interval=4))
    assert texts == ["0", "1234", "5678", "9"]
    assert payloads[-2]["choices"][0]["finish_reason"] == "length"
    assert payloads[-1]["usage"]["completion_tokens"] == 10


def test_choices_keep_independent_intervals(monkeypatch):
    texts, _payloads = _texts(
        _completion_stream(
            monkeypatch,
            [
                (0, "A", None),
                (1, "X", None),
                (0, "B", None),
                (0, "C", None),
                (0, "D", None),
                (0, "E", None),
                (1, "Y", "stop"),
            ],
            interval=4,
        )
    )
    assert texts == ["A", "X", "BCDE", "Y"]


def test_requested_logprobs_stay_on_each_chunk(monkeypatch):
    texts, payloads = _texts(_completion_stream(monkeypatch, [(0, "A", "stop")], logprobs=1))
    assert texts == ["A"]
    assert payloads[0]["choices"][0]["logprobs"] == {}


def test_first_token_does_not_wait_for_the_interval(monkeypatch):
    from lightllm.server.api_http import g_objs

    async def generate(*_args, **_kwargs):
        yield 80, "A", {"prompt_tokens": 4}, FinishStatus()
        raise AssertionError("The next token must not be fetched before the first chunk is emitted")

    monkeypatch.setattr(g_objs, "httpserver_manager", SimpleNamespace(generate=generate), raising=False)
    monkeypatch.setattr(api_openai, "get_env_start_args", lambda: SimpleNamespace(stream_interval=8))

    async def first_chunk():
        request = CompletionRequest(model="test", prompt="Prompt", stream=True)
        response = await api_openai._handle_streaming_completion("Prompt", None, None, None, request, 123)
        async with aclosing(response.body_iterator) as stream:
            return await stream.__anext__()

    assert json.loads(asyncio.run(first_chunk())[6:])["choices"][0]["text"] == "A"
