import json
from types import SimpleNamespace

import pytest

from lightllm.server import api_http
from lightllm.server.api_openai import _safe_stream_wrapper
from lightllm.utils.error_utils import ServerBusyError


class FakeMetricClient:
    def __init__(self):
        self.counters = []

    def counter_inc(self, name):
        self.counters.append(name)


@pytest.fixture()
def fake_metric_client(monkeypatch):
    metric_client = FakeMetricClient()
    monkeypatch.setattr(api_http.g_objs, "metric_client", metric_client)
    return metric_client


def _decode_sse_payload(chunk):
    if isinstance(chunk, bytes):
        chunk = chunk.decode("utf-8")
    assert chunk.startswith("data: ")
    return json.loads(chunk.removeprefix("data: ").strip())


@pytest.mark.asyncio
async def test_safe_stream_wrapper_converts_unexpected_exception_to_sse_error(fake_metric_client):
    async def failing_stream():
        if False:
            yield b""
        raise RuntimeError("backend failed")

    chunks = []
    async for chunk in _safe_stream_wrapper(failing_stream()):
        chunks.append(chunk)

    assert len(chunks) == 1
    payload = _decode_sse_payload(chunks[0])
    assert payload["error"]["message"] == "backend failed"
    assert payload["error"]["type"] == "InternalServerError"
    assert fake_metric_client.counters == ["lightllm_request_failure"]


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("handler_name", "impl_name"),
    [
        ("chat_completions", "chat_completions_impl"),
        ("completions", "completions_impl"),
    ],
)
@pytest.mark.parametrize(
    ("exc", "expected_status"),
    [
        (ServerBusyError("server overloaded"), 503),
        (RuntimeError("backend failed"), 417),
    ],
)
async def test_openai_handlers_return_structured_errors(
    monkeypatch,
    fake_metric_client,
    handler_name,
    impl_name,
    exc,
    expected_status,
):
    monkeypatch.setattr(api_http, "get_env_start_args", lambda: SimpleNamespace(run_mode="normal"))

    async def failing_impl(*args, **kwargs):
        raise exc

    monkeypatch.setattr(api_http, impl_name, failing_impl)

    response = await getattr(api_http, handler_name)(SimpleNamespace(), SimpleNamespace())

    assert response.status_code == expected_status
    payload = json.loads(response.body)
    assert payload["error"]["message"] == str(exc)
    assert fake_metric_client.counters == ["lightllm_request_failure"]
