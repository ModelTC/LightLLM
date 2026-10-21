import asyncio
import json
from types import SimpleNamespace

import httpx
import pytest
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

from lightllm.server import api_anthropic, api_errors, api_http, api_openai, api_stream_obj
from lightllm.server.api_cli import make_argument_parser
from lightllm.server.api_stream_obj import CustomStreamingResponse
from lightllm.server.core.objs.start_args_type import StartArgs
from lightllm.utils.error_utils import InvalidRequestError, ServerBusyError


class _MetricClient:
    def __init__(self):
        self.counters = []

    def counter_inc(self, name):
        self.counters.append(name)


def test_delay_response_start_cli_defaults_to_enabled_and_can_be_disabled():
    parser = make_argument_parser()

    assert parser.parse_args([]).disable_delay_response_start is False
    assert parser.parse_args(["--disable_delay_response_start"]).disable_delay_response_start is True
    assert StartArgs().disable_delay_response_start is False


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
    assert body["error"]["message"] == "Server is busy, please try again later"
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


def test_stream_starts_response_after_first_chunk_by_default(monkeypatch):
    monkeypatch.setattr(
        api_stream_obj,
        "get_env_start_args",
        lambda: SimpleNamespace(disable_delay_response_start=False),
    )

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


def test_disable_delay_response_start_sends_status_immediately(monkeypatch):
    monkeypatch.setattr(
        api_stream_obj,
        "get_env_start_args",
        lambda: SimpleNamespace(disable_delay_response_start=True),
    )

    async def run():
        response = None

        async def generate():
            response.status_code = 201
            yield "first"

        messages = []

        async def send(message):
            messages.append(message)

        response = CustomStreamingResponse(generate())
        await response.stream_response(send)
        return messages

    messages = asyncio.run(run())
    assert messages[0]["type"] == "http.response.start"
    assert messages[0]["status"] == 200
    assert messages[1]["body"] == b"first"


def test_stream_propagates_error_before_response_start(monkeypatch):
    monkeypatch.setattr(
        api_stream_obj,
        "get_env_start_args",
        lambda: SimpleNamespace(disable_delay_response_start=False),
    )

    async def run():
        async def generate():
            if False:
                yield
            raise InvalidRequestError("prompt is too long")

        messages = []

        async def send(message):
            messages.append(message)

        with pytest.raises(InvalidRequestError, match="prompt is too long"):
            await CustomStreamingResponse(generate()).stream_response(send)

        return messages

    assert asyncio.run(run()) == []


def test_delayed_stream_can_return_http_429(monkeypatch):
    monkeypatch.setattr(
        api_stream_obj,
        "get_env_start_args",
        lambda: SimpleNamespace(disable_delay_response_start=False),
    )

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


def test_delayed_stream_can_return_http_400_for_invalid_request(monkeypatch):
    metric_client = _MetricClient()
    monkeypatch.setattr(api_http.g_objs, "metric_client", metric_client)
    monkeypatch.setattr(
        api_stream_obj,
        "get_env_start_args",
        lambda: SimpleNamespace(disable_delay_response_start=False),
    )
    app = FastAPI()
    app.exception_handler(InvalidRequestError)(api_http.invalid_request_exception_handler)

    @app.get("/")
    async def generate_stream():
        async def generate():
            if False:
                yield
            raise InvalidRequestError("prompt is too long")

        return CustomStreamingResponse(api_openai._safe_stream_wrapper(generate()))

    async def run():
        transport = httpx.ASGITransport(app=app, raise_app_exceptions=False)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            return await client.get("/")

    response = asyncio.run(run())
    assert response.status_code == 400
    assert response.json()["error"]["message"] == "prompt is too long"
    assert metric_client.counters == ["lightllm_request_failure"]


def test_pd_master_anthropic_stream_preserves_error_envelope(monkeypatch):
    metric_client = _MetricClient()
    monkeypatch.setattr(api_http.g_objs, "metric_client", metric_client)
    monkeypatch.setattr(api_http, "get_env_start_args", lambda: SimpleNamespace(run_mode="normal"))
    monkeypatch.setattr(
        api_stream_obj,
        "get_env_start_args",
        lambda: SimpleNamespace(disable_delay_response_start=False),
    )

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
            "message": "Server is busy, please try again later",
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
    assert chunks[2] == "data: [DONE]\n\n"
    assert error["error"] == {
        "message": "Server is busy, please try again later",
        "type": "server_error",
        "code": "stream_error",
    }


def test_safe_stream_propagates_invalid_request_before_first_chunk():
    async def run():
        async def generate():
            if False:
                yield
            raise InvalidRequestError("prompt is too long")

        with pytest.raises(InvalidRequestError, match="prompt is too long"):
            async for _ in api_openai._safe_stream_wrapper(generate()):
                pass

    asyncio.run(run())


@pytest.mark.parametrize("error_type", [InvalidRequestError, ValueError])
def test_safe_stream_terminates_invalid_error_after_first_chunk(error_type):
    async def run():
        async def generate():
            yield "first"
            raise error_type("prompt is too long")

        return [item async for item in api_openai._safe_stream_wrapper(generate())]

    chunks = asyncio.run(run())
    error = json.loads(chunks[1].removeprefix("data: "))

    assert chunks[0] == "first"
    assert error["error"] == {"message": "prompt is too long", "type": "invalid_request_error"}
    assert chunks[2] == "data: [DONE]\n\n"


def test_safe_stream_converts_value_error_before_first_chunk():
    async def run():
        async def generate():
            if False:
                yield
            raise ValueError("prompt is too long")

        with pytest.raises(InvalidRequestError, match="prompt is too long"):
            async for _ in api_openai._safe_stream_wrapper(generate()):
                pass

    asyncio.run(run())


def test_delayed_stream_returns_http_400_for_value_error_before_first_chunk(monkeypatch):
    metric_client = _MetricClient()
    monkeypatch.setattr(api_http.g_objs, "metric_client", metric_client)
    monkeypatch.setattr(
        api_stream_obj,
        "get_env_start_args",
        lambda: SimpleNamespace(disable_delay_response_start=False),
    )
    app = FastAPI()
    app.exception_handler(InvalidRequestError)(api_http.invalid_request_exception_handler)

    @app.get("/")
    async def generate_stream():
        async def generate():
            if False:
                yield
            raise ValueError("invalid token id")

        return CustomStreamingResponse(api_openai._safe_stream_wrapper(generate()))

    async def run():
        transport = httpx.ASGITransport(app=app, raise_app_exceptions=False)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            return await client.get("/")

    response = asyncio.run(run())
    assert response.status_code == 400
    assert response.json()["error"]["message"] == "invalid token id"
    assert metric_client.counters == ["lightllm_request_failure"]


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
