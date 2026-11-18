import asyncio
import json
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest
from starlette.requests import Request

from lightllm.server import api_http, api_openai
from lightllm.server.api_models import CompletionRequest
from lightllm.server.core.objs import FinishStatus
from lightllm.server.httpserver.manager import HttpServerManager
from lightllm.server.httpserver_for_pd_master.manager import HttpServerManagerForPDMaster
from lightllm.utils.error_utils import GenerationError


@pytest.fixture
def anyio_backend():
    return "asyncio"


@pytest.fixture
def pd_completion(monkeypatch):
    manager = HttpServerManagerForPDMaster.__new__(HttpServerManagerForPDMaster)
    manager.running_request_count = 0
    manager.tokenizer = SimpleNamespace(encode=lambda text, **kwargs: list(text.encode()))
    manager.args = SimpleNamespace(enable_rl=False, enable_return_routed_experts=False, reasoning_parser=None)
    manager.active_req_ids = set()
    manager.tokens = lambda *args: 4
    manager.id_gen = SimpleNamespace(generate_id=lambda: 800)
    manager.metric_client = MagicMock()
    manager.output_grammar_compiler = None
    manager._log_req_header = AsyncMock()
    monkeypatch.setattr(HttpServerManager, "_check_and_repair_length", AsyncMock())
    monkeypatch.setattr(api_http.g_objs, "httpserver_manager", manager)

    async def open_stream(source, **options):
        async def generate_one(prompt, params, multimodal, request, start_time, sub_id, input_token_num):
            async for result in source(sub_id):
                yield result

        manager._generate_one = generate_one
        request = CompletionRequest(model="glm53", prompt="test", stream=True, **options)
        http = Request({"type": "http", "method": "POST", "path": "/v1/completions", "headers": []})
        response = await api_openai.completions_impl(request, http)
        return response.body_iterator

    return open_stream


def token(sub_id, text, *, first=False, end=FinishStatus.NO_FINISH):
    return sub_id, text, {"id": 1, "prompt_tokens": 4, "prompt_cache_len": 2, "is_first_token": first}, FinishStatus(end)


def event(frame):
    assert frame.startswith("data: ") and frame.endswith("\n\n")
    return json.loads(frame[6:])


@pytest.mark.anyio
@pytest.mark.parametrize("logprobs", [None, 0])
async def test_flush_available_text_before_next_decode_step(pd_completion, logprobs):
    next_step = asyncio.Event()

    async def source(sub_id):
        for i, text in enumerate(["A", "中", "C"]):
            yield token(sub_id, text, first=i == 0)
        await next_step.wait()
        yield token(sub_id, "D")
        yield token(sub_id, "E", end=FinishStatus.FINISHED_STOP)

    frames = await pd_completion(source, echo=True, logprobs=logprobs)
    expected = ["testA", "中C"] if logprobs is None else ["testA", "中", "C"]
    for text in expected:
        frame = event(await asyncio.wait_for(anext(frames), timeout=1))
        assert frame["choices"] == [
            {"index": 0, "text": text, "finish_reason": None, "logprobs": None if logprobs is None else {}}
        ]
        assert frame["id"] == "800" and frame["object"] == "text_completion"
    next_step.set()
    rest = [f async for f in frames]
    assert rest[-1] == "data: [DONE]\n\n"
    chunks = [event(f) for f in rest[:-1]]
    assert [c["choices"][0]["text"] for c in chunks[:-1]] == (["DE"] if logprobs is None else ["D", "E"])
    assert chunks[-2]["choices"][0]["finish_reason"] == "stop"
    usage = chunks[-1]["usage"]
    assert usage["completion_tokens"] == 5 and usage["prompt_tokens"] == 4 and usage["total_tokens"] == 9
    assert usage["prompt_tokens_details"]["cached_tokens"] == 2


@pytest.mark.anyio
async def test_stop_match_does_not_delay_already_visible_text(pd_completion):
    next_step = asyncio.Event()

    async def source(sub_id):
        for text in ("one", "two", "E"):
            yield token(sub_id, text)
        await next_step.wait()
        yield token(sub_id, "NDdiscarded", end=FinishStatus.FINISHED_STOP)

    frames = await pd_completion(source, stop="END")
    for text in ("one", "two"):
        out = event(await asyncio.wait_for(anext(frames), timeout=1))
        assert out["choices"][0]["text"] == text
    next_step.set()
    wire = [frame async for frame in frames]
    choices = [choice for frame in wire[:-1] for choice in event(frame)["choices"]]
    assert "".join(choice["text"] for choice in choices) == ""
    assert choices[-1]["finish_reason"] == "stop"


@pytest.mark.anyio
@pytest.mark.parametrize("width", [1, 4096])
async def test_text_batches_bound_work_and_keep_choices_independent(pd_completion, width):
    async def source(sub_id):
        for i in range(71):
            yield token(sub_id, str(sub_id - 800) * width, end=FinishStatus.FINISHED_LENGTH if i == 70 else 0)

    frames = await pd_completion(source, n=2, best_of=2)
    wire = [f async for f in frames]
    chunks = [event(f) for f in wire[:-1]]
    out = {0: [], 1: []}
    ends = []
    for chunk in chunks[:-1]:
        assert sum(len(c["text"]) for c in chunk["choices"]) <= min(32 * width, 64 * 1024)
        for choice in chunk["choices"]:
            out[choice["index"]].append(choice["text"])
            if choice["finish_reason"] is not None:
                ends.append((choice["index"], choice["finish_reason"]))
    for index, pieces in out.items():
        assert pieces[0] == str(index) * width
        assert "".join(pieces) == str(index) * width * 71
    assert sorted(ends) == [(0, "length"), (1, "length")]
    assert chunks[-1]["usage"]["completion_tokens"] == 142 and wire[-1] == "data: [DONE]\n\n"


@pytest.mark.anyio
async def test_ready_output_precedes_stream_error(pd_completion):
    async def source(sub_id):
        yield token(sub_id, "first")
        yield token(sub_id, "ready")
        raise GenerationError("decode failed")

    wire = [frame async for frame in await pd_completion(source)]
    assert [event(f)["choices"][0]["text"] for f in wire[:2]] == ["first", "ready"]
    assert event(wire[2])["error"]["code"] == "stream_error"
    assert wire[3:] == ["data: [DONE]\n\n"]
