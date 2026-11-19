import json
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import pytest

from fastapi import FastAPI
from fastapi.testclient import TestClient

from lightllm.server import api_openai, api_stream_obj
from lightllm.server.api_models import ChatCompletionRequest, CompletionRequest
from lightllm.server.core.objs import FinishStatus, SamplingParams, StartArgs
from lightllm.server.api_http import g_objs
from lightllm.server.api_http import generation_exception_handler
from lightllm.utils.error_utils import GenerationError


def test_generation_failure_before_first_chunk_returns_http_500(monkeypatch):
    monkeypatch.setattr(g_objs, "metric_client", Mock())
    monkeypatch.setattr(api_stream_obj, "get_env_start_args", lambda: StartArgs())
    app = FastAPI()
    app.add_exception_handler(GenerationError, generation_exception_handler)

    @app.get("/generate")
    async def generate():
        async def failed_stream():
            raise GenerationError("Generation failed before producing output")
            yield

        return api_stream_obj.CustomStreamingResponse(failed_stream(), media_type="text/event-stream")

    response = TestClient(app).get("/generate")
    assert response.status_code == 500
    assert "Generation failed" in response.json()["error"]["message"]


def _openai_stream_app(monkeypatch, args, generate):
    monkeypatch.setattr(g_objs, "metric_client", Mock())
    monkeypatch.setattr(g_objs, "args", args)
    monkeypatch.setattr(
        g_objs,
        "httpserver_manager",
        SimpleNamespace(tokenizer=Mock(), output_grammar_compiler=None, generate=generate),
    )
    monkeypatch.setattr(api_openai, "get_env_start_args", lambda: args)
    monkeypatch.setattr(api_stream_obj, "get_env_start_args", lambda: args)
    monkeypatch.setattr(api_openai, "build_prompt", AsyncMock(return_value="prompt"))
    monkeypatch.setattr(SamplingParams, "init", lambda *_args, **_kwargs: None)
    app = FastAPI()
    app.add_exception_handler(GenerationError, generation_exception_handler)

    @app.post("/v1/chat/completions")
    async def chat(request: ChatCompletionRequest):
        return await api_openai.chat_completions_impl(request, Mock())

    @app.post("/v1/completions")
    async def completion(request: CompletionRequest):
        return await api_openai.completions_impl(request, Mock())

    return app


@pytest.mark.parametrize(
    "early_headers, finish_status, expected_status",
    [
        (False, FinishStatus.FINISHED_ERROR, 500),
        (True, FinishStatus.FINISHED_ERROR, 200),
        (False, FinishStatus.FINISHED_STOP, 200),
        (True, FinishStatus.FINISHED_STOP, 200),
    ],
)
def test_chat_first_result_sets_http_status(monkeypatch, early_headers, finish_status, expected_status):
    args = StartArgs(disable_delay_response_start=early_headers)

    async def generate(*_args, **_kwargs):
        yield 0, "answer", {"prompt_tokens": 2, "prompt_cache_len": 0}, FinishStatus(finish_status)

    response = TestClient(_openai_stream_app(monkeypatch, args, generate)).post(
        "/v1/chat/completions",
        json={"model": "test", "messages": [{"role": "user", "content": "hi"}], "stream": True},
    )
    assert response.status_code == expected_status
    if expected_status == 500:
        assert "Generation failed" in response.json()["error"]["message"]
    elif finish_status == FinishStatus.FINISHED_ERROR:
        assert response.headers["content-type"].startswith("text/event-stream")
        events = response.text.strip().split("\n\n")
        assert len(events) == 2
        assert json.loads(events[0].removeprefix("data: ")) == {
            "error": {
                "message": "Generation failed before producing output",
                "type": "server_error",
                "code": "stream_error",
            }
        }
        assert events[1] == "data: [DONE]"
    else:
        assert "answer" in response.text
        assert "data: [DONE]" in response.text


@pytest.mark.parametrize("chat", [True, False], ids=["chat", "completions"])
def test_openai_stream_flushes_pending_deltas_before_generation_error(monkeypatch, chat):
    async def generate(*_args, **_kwargs):
        metadata = {"prompt_tokens": 2, "prompt_cache_len": 0, "_pd_stream_batch_end": False}
        yield 0, "answer", metadata, FinishStatus(FinishStatus.NO_FINISH)
        yield 0, " queued", metadata, FinishStatus(FinishStatus.NO_FINISH)
        yield 0, "error marker", metadata, FinishStatus(FinishStatus.FINISHED_ERROR)

    client = TestClient(_openai_stream_app(monkeypatch, StartArgs(), generate))
    if chat:
        path = "/v1/chat/completions"
        request = {"messages": [{"role": "user", "content": "hi"}]}
    else:
        path = "/v1/completions"
        request = {"prompt": "hi"}
    response = client.post(path, json={"model": "test", "stream": True, **request})

    assert response.status_code == 200
    events = response.text.strip().split("\n\n")
    assert events[-1] == "data: [DONE]"
    chunks = [json.loads(event.removeprefix("data: ")) for event in events[:-1]]
    assert chunks[-1] == {
        "error": {
            "message": "Generation failed during streaming",
            "type": "server_error",
            "code": "stream_error",
        }
    }
    choices = [choice for chunk in chunks[:-1] for choice in chunk["choices"]]
    assert all(choice["finish_reason"] is None for choice in choices)
    text = "".join((choice["delta"].get("content") or "") if chat else choice["text"] for choice in choices)
    assert text == "answer queued"
