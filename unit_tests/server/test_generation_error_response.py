from unittest.mock import Mock

from fastapi import FastAPI
from fastapi.testclient import TestClient

from lightllm.server import api_stream_obj
from lightllm.server.core.objs import StartArgs
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
