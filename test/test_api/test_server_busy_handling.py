import asyncio
import json
from types import SimpleNamespace

import httpx
import pytest
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

from lightllm.server import api_anthropic, api_errors, api_http, api_openai, api_stream_obj
from lightllm.server.api_stream_obj import CustomStreamingResponse
from lightllm.utils.error_utils import ServerBusyError


class _MetricClient:
    def __init__(self):
        self.counters = []

    def counter_inc(self, name):
        self.counters.append(name)


def test_api_modules_share_error_response_factory():
    assert api_http.create_error_response is api_errors.create_error_response
    assert api_openai.create_error_response is api_errors.create_error_response


def test_server_busy_response_is_rate_limited(monkeypatch):
    metric_client = _MetricClient()
    monkeypatch.setattr(api_http.g_objs, "metric_client", metric_client)

    response = api_errors.create_server_busy_response(ServerBusyError())
    body = json.loads(response.body)

    assert response.status_code == 429
    assert body["error"]["type"] == "RateLimitError"
    assert body["error"]["code"] == 429
    assert metric_client.counters == ["lightllm_request_failure"]


@pytest.mark.parametrize(
    ("error", "expected"),
    [
        ({"code": 429, "type": "server_error"}, "rate_limit_error"),
        ({"type": "RateLimitError"}, "rate_limit_error"),
        ({"type": "invalid_request_error"}, "invalid_request_error"),
        ({"type": "server_error"}, "api_error"),
    ],
)
def test_anthropic_error_type(error, expected):
    assert api_anthropic._anthropic_error_type(error) == expected


def test_pd_master_stream_starts_response_after_first_chunk(monkeypatch):
    monkeypatch.setattr(api_stream_obj, "get_env_start_args", lambda: SimpleNamespace(run_mode="pd_master"))

    async def run():
        response = None

        async def generate():
            response.status_code = 201
            yield "first"
            yield "second"

        messages = []

        async def send(message):
            messages.append(message)

        response = CustomStreamingResponse(generate())
        await response.stream_response(send)
        return messages

    messages = asyncio.run(run())
    assert [message["type"] for message in messages] == [
        "http.response.start",
        "http.response.body",
        "http.response.body",
        "http.response.body",
    ]
    assert [message.get("body") for message in messages[1:]] == [b"first", b"second", b""]
    assert messages[0]["status"] == 201


def test_pd_master_stream_propagates_busy_error_before_response_start(monkeypatch):
    monkeypatch.setattr(api_stream_obj, "get_env_start_args", lambda: SimpleNamespace(run_mode="pd_master"))

    async def run():
        async def generate():
            if False:
                yield
            raise ServerBusyError()

        messages = []

        async def send(message):
            messages.append(message)

        with pytest.raises(ServerBusyError):
            await CustomStreamingResponse(generate()).stream_response(send)

        return messages

    assert asyncio.run(run()) == []


def test_pd_master_stream_can_return_http_429(monkeypatch):
    monkeypatch.setattr(api_stream_obj, "get_env_start_args", lambda: SimpleNamespace(run_mode="pd_master"))
    app = FastAPI()

    @app.exception_handler(ServerBusyError)
    async def handle_server_busy(_request: Request, _error: ServerBusyError):
        return JSONResponse({"error": "server busy"}, status_code=429)

    @app.get("/")
    async def generate_stream():
        async def generate():
            if False:
                yield
            raise ServerBusyError()

        return CustomStreamingResponse(generate())

    async def run():
        transport = httpx.ASGITransport(app=app, raise_app_exceptions=False)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            return await client.get("/")

    response = asyncio.run(run())
    assert response.status_code == 429
    assert response.json() == {"error": "server busy"}


def test_pd_master_anthropic_stream_preserves_error_envelope(monkeypatch):
    metric_client = _MetricClient()
    monkeypatch.setattr(api_http.g_objs, "metric_client", metric_client)
    monkeypatch.setattr(api_http, "get_env_start_args", lambda: SimpleNamespace(run_mode="normal"))
    monkeypatch.setattr(api_stream_obj, "get_env_start_args", lambda: SimpleNamespace(run_mode="pd_master"))

    async def anthropic_messages_impl(_request):
        async def generate():
            if False:
                yield
            raise ServerBusyError()

        return CustomStreamingResponse(generate(), media_type="text/event-stream")

    monkeypatch.setattr(api_anthropic, "anthropic_messages_impl", anthropic_messages_impl)

    async def run():
        transport = httpx.ASGITransport(app=api_http.app, raise_app_exceptions=False)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            return await client.post("/v1/messages", json={})

    response = asyncio.run(run())
    assert response.status_code == 429
    assert response.json() == {
        "type": "error",
        "error": {
            "type": "rate_limit_error",
            "message": "Server is busy, please try again later (Status code: 429)",
        },
    }
    assert metric_client.counters == ["lightllm_request_failure"]


def test_safe_stream_reports_busy_error_after_first_chunk():
    async def run():
        async def generate():
            yield "first"
            raise ServerBusyError()

        return [item async for item in api_openai._safe_stream_wrapper(generate())]

    chunks = asyncio.run(run())
    error = json.loads(chunks[1].removeprefix("data: "))

    assert chunks[0] == "first"
    assert error["error"]["type"] == "server_error"
    assert error["error"]["code"] == "stream_error"


def test_safe_stream_propagates_busy_error_before_first_chunk():
    async def run():
        async def generate():
            if False:
                yield
            raise ServerBusyError()

        with pytest.raises(ServerBusyError):
            async for _ in api_openai._safe_stream_wrapper(generate()):
                pass

    asyncio.run(run())
