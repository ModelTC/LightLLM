from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import pytest

from fastapi import FastAPI
from fastapi.testclient import TestClient

from lightllm.server import api_openai, api_stream_obj
from lightllm.server.api_models import ChatCompletionRequest
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


@pytest.mark.parametrize(
    "finish_status, expected_status",
    [(FinishStatus.FINISHED_ERROR, 500), (FinishStatus.FINISHED_STOP, 200)],
)
def test_chat_first_result_sets_http_status(monkeypatch, finish_status, expected_status):
    args = StartArgs()

    async def generate(*_args, **_kwargs):
        yield 0, "answer", {"prompt_tokens": 2, "prompt_cache_len": 0}, FinishStatus(finish_status)

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

    response = TestClient(app).post(
        "/v1/chat/completions",
        json={"model": "test", "messages": [{"role": "user", "content": "hi"}], "stream": True},
    )
    assert response.status_code == expected_status
    if expected_status == 500:
        assert "Generation failed" in response.json()["error"]["message"]
    else:
        assert "answer" in response.text
        assert "data: [DONE]" in response.text
