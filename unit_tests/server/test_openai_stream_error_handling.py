import asyncio

import ujson as json

from lightllm.server import api_http
from lightllm.server.api_openai import _safe_stream_wrapper
from lightllm.utils.error_utils import ServerBusyError


class FakeMetricClient:
    def __init__(self):
        self.counters = []

    def counter_inc(self, name):
        self.counters.append(name)


def _decode_sse_payload(chunk):
    if isinstance(chunk, bytes):
        chunk = chunk.decode("utf-8")
    assert chunk.startswith("data: ")
    return json.loads(chunk.removeprefix("data: ").strip())


def _collect(gen):
    async def drain():
        return [c async for c in gen]

    return asyncio.run(drain())


def test_safe_stream_wrapper_converts_unexpected_exception_to_sse_error(monkeypatch):
    metric_client = FakeMetricClient()
    monkeypatch.setattr(api_http.g_objs, "metric_client", metric_client)

    async def failing_stream():
        if False:
            yield b""
        raise RuntimeError("backend failed")

    chunks = _collect(_safe_stream_wrapper(failing_stream()))

    assert len(chunks) == 1
    payload = _decode_sse_payload(chunks[0])
    assert payload["error"]["message"] == "backend failed"
    assert payload["error"]["type"] == "InternalServerError"
    assert metric_client.counters == ["lightllm_request_failure"]


def test_safe_stream_wrapper_maps_server_busy_error(monkeypatch):
    metric_client = FakeMetricClient()
    monkeypatch.setattr(api_http.g_objs, "metric_client", metric_client)

    async def busy_stream():
        if False:
            yield b""
        raise ServerBusyError("server overloaded")

    chunks = _collect(_safe_stream_wrapper(busy_stream()))

    assert len(chunks) == 1
    payload = _decode_sse_payload(chunks[0])
    assert payload["error"]["type"] == "ServerBusyError"
    assert payload["error"]["code"] == 503
    assert metric_client.counters == ["lightllm_request_failure"]


def test_safe_stream_wrapper_keeps_value_error_as_invalid_request(monkeypatch):
    metric_client = FakeMetricClient()
    monkeypatch.setattr(api_http.g_objs, "metric_client", metric_client)

    async def bad_stream():
        if False:
            yield b""
        raise ValueError("input too long")

    chunks = _collect(_safe_stream_wrapper(bad_stream()))

    payload = _decode_sse_payload(chunks[0])
    assert payload["error"]["type"] == "invalid_request_error"
    assert metric_client.counters == ["lightllm_request_failure"]


def test_safe_stream_wrapper_swallows_client_disconnect(monkeypatch):
    from lightllm.utils.error_utils import ClientDisconnected

    metric_client = FakeMetricClient()
    monkeypatch.setattr(api_http.g_objs, "metric_client", metric_client)

    async def gone_stream():
        if False:
            yield b""
        raise ClientDisconnected("client gone")

    chunks = _collect(_safe_stream_wrapper(gone_stream()))

    assert chunks == []
    assert metric_client.counters == []
