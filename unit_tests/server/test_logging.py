"""Exercise actual HTTP logging and health-monitor boundaries without starting workers."""

import asyncio
from unittest.mock import Mock

import pytest

from lightllm.server import api_http
from lightllm.server.health_monitor import manager as health


@pytest.mark.parametrize("enabled", [False, True])
def test_access_middleware_preserves_response_and_respects_switch(monkeypatch, enabled):
    monkeypatch.setenv("LIGHTLLM_LOG_HTTP_ACCESS", "1" if enabled else "0")
    access = Mock()
    monkeypatch.setattr(api_http, "log_http_access", access)
    messages = []

    async def app(scope, receive, send):
        await send({"type": "http.response.start", "status": 201, "headers": []})
        await send({"type": "http.response.body", "body": b"ok"})

    async def send(message):
        messages.append(message)

    asyncio.run(api_http._AccessLogMiddleware(app)({"type": "http", "method": "POST", "path": "/generate"}, None, send))
    assert messages[0]["status"] == 201
    assert messages[1]["body"] == b"ok"
    assert access.call_args.args[:3] == ("POST", "/generate", 201)
    assert access.call_args.args[3] >= 0
    request_id = access.call_args.kwargs["http_request_id"]
    assert request_id
    assert dict(messages[0]["headers"])[b"x-request-id"] == request_id.encode("ascii")


def test_access_middleware_does_not_swallow_failure(monkeypatch):
    monkeypatch.setenv("LIGHTLLM_LOG_HTTP_ACCESS", "1")
    access = Mock()
    monkeypatch.setattr(api_http, "log_http_access", access)

    async def app(scope, receive, send):
        raise RuntimeError("request failed")

    with pytest.raises(RuntimeError, match="request failed"):
        asyncio.run(api_http._AccessLogMiddleware(app)({"type": "http", "method": "GET", "path": "/fail"}, None, None))
    assert access.call_args.args[:3] == ("GET", "/fail", 500)


def test_healthy_checks_are_silent_and_recovery_is_logged_once(monkeypatch):
    logger = Mock()
    monkeypatch.setattr(health, "logger", logger)
    monkeypatch.setattr(health, "consecutive_failures", 0)
    monkeypatch.setattr(health, "all_is_alive", lambda ids: True)
    monkeypatch.setattr(health.requests, "get", lambda *a, **k: Mock(status_code=200))
    health.health_monitor("http://unused/health", [])
    logger.info.assert_not_called()
    health.consecutive_failures = 2
    health.health_monitor("http://unused/health", [])
    assert logger.info.call_count == 1
    assert "health.recovered" in logger.info.call_args.args[0]
    assert health.consecutive_failures == 0
    health.health_monitor("http://unused/health", [])
    assert logger.info.call_count == 1


def test_health_failures_warn_until_termination(monkeypatch):
    logger = Mock()
    kill = Mock()
    monkeypatch.setattr(health, "logger", logger)
    monkeypatch.setattr(health, "consecutive_failures", 0)
    monkeypatch.setattr(health, "all_is_alive", lambda ids: True)
    monkeypatch.setattr(health.requests, "get", lambda *a, **k: Mock(status_code=500))
    monkeypatch.setattr(health.os, "kill", kill)
    health.health_monitor("http://unused/health", [100, 101])
    health.health_monitor("http://unused/health", [100, 101])
    logger.critical.assert_not_called()
    kill.assert_not_called()
    with pytest.raises(SystemExit):
        health.health_monitor("http://unused/health", [100, 101])
    assert logger.warning.call_count == 3
    assert logger.critical.call_count == 1
    assert kill.call_count == 2


@pytest.mark.parametrize("enabled", [False, True])
def test_concurrent_request_ids_propagate_and_context_is_reset(monkeypatch, enabled):
    from lightllm.utils.log_utils import get_http_request_id

    monkeypatch.setenv("LIGHTLLM_LOG_HTTP_ACCESS", "1" if enabled else "0")
    access = Mock()
    monkeypatch.setattr(api_http, "log_http_access", access)
    observed = {}

    async def app(scope, receive, send):
        request_id = dict(scope["headers"])[b"x-request-id"].decode()
        await asyncio.sleep(0.001)
        assert get_http_request_id() == request_id
        observed[scope["path"]] = request_id
        await send({"type": "http.response.start", "status": 200, "headers": []})
        await send({"type": "http.response.body", "body": b"ok"})

    async def send(message):
        pass

    async def run():
        middleware = api_http._AccessLogMiddleware(app)
        await asyncio.gather(
            middleware(
                {"type": "http", "method": "POST", "path": "/one", "headers": [(b"x-request-id", b"client-123")]},
                None,
                send,
            ),
            middleware({"type": "http", "method": "POST", "path": "/two", "headers": []}, None, send),
        )
        assert get_http_request_id() == "-"

    asyncio.run(run())
    assert observed["/one"] == "client-123"
    assert observed["/two"] != "client-123"
    assert {call.kwargs["http_request_id"] for call in access.call_args_list} == set(observed.values())


def test_request_id_context_reset_on_error(monkeypatch):
    from lightllm.utils.log_utils import get_http_request_id

    monkeypatch.setattr(api_http, "log_http_access", Mock())

    async def app(scope, receive, send):
        assert get_http_request_id() != "-"
        raise RuntimeError("failure")

    async def run():
        with pytest.raises(RuntimeError):
            await api_http._AccessLogMiddleware(app)({"type": "http", "method": "POST", "path": "/fail"}, None, None)
        assert get_http_request_id() == "-"

    asyncio.run(run())


@pytest.mark.parametrize("value", [b"", b"bad\r\nid", b"x" * 129, b"\xff"])
def test_unsafe_client_request_ids_are_replaced(value):
    from lightllm.utils.log_utils import new_http_request_id

    request_id = new_http_request_id([(b"x-request-id", value)])
    assert len(request_id) == 32
    int(request_id, 16)
