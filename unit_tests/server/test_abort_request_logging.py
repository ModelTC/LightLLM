import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest


@pytest.fixture
def script():
    path = Path(__file__).resolve().parents[2] / "test/test_api/test_abort_request.py"
    spec = importlib.util.spec_from_file_location("abort_request_script", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize(
    "line",
    [
        "received req X-Request-Id:client.1 X-Session-Id:abc lightllm_req_id:80",
        "DEBUG event=request.received request_id=80 http_request_id='client.1'",
        json.dumps({"event": "request.received", "request_id": 80, "http_request_id": "client.1"}),
    ],
)
def test_lookup_internal_id_from_legacy_and_current_logs(script, tmp_path, line):
    path = tmp_path / "server.log"
    path.write_text(line + "\n")
    assert script.lookup_lightllm_req_id_from_log(str(path), "client.1", timeout=0.1) == 80
    assert script.lookup_lightllm_req_id_from_log(str(path), "client.2", timeout=0.01) is None


@pytest.mark.parametrize("message", ["aborted group_request_id 80", "Request aborted: request_id=80"])
@pytest.mark.parametrize("stage", ["stage_c_abort_running", "stage_d_abort_by_id"])
def test_abort_stage_accepts_current_and_legacy_cancellation(script, monkeypatch, tmp_path, message, stage):
    path = tmp_path / "server.log"
    path.write_text("event=request.received request_id=80 http_request_id='client.1'\n")
    thread = SimpleNamespace(join=Mock(), is_alive=lambda: False)
    monkeypatch.setattr(script, "start_stream", lambda *a, **k: (thread, {"events": [{"_t": 0}]}, "client.1"))
    monkeypatch.setattr(script, "wait_for_first_token", lambda *a, **k: True)
    monkeypatch.setattr(script, "get_finish_reason", lambda out: "abort")

    def abort(*a, **k):
        with path.open("a") as stream:
            stream.write(json.dumps({"message": message}) + "\n")
        return 200, "ok"

    monkeypatch.setattr(script, "post_abort", abort)
    assert getattr(script, stage)("http://unused", str(path))


@pytest.mark.parametrize("headers,expected", [({"X-Request-Id": "client.1"}, "client.1"), ({}, "context.1")])
def test_http_request_header_output_is_consumed_by_abort_lookup(script, monkeypatch, tmp_path, headers, expected):
    import asyncio
    import io
    import logging
    from lightllm.server.httpserver import manager as module

    output = io.StringIO()
    logger = logging.getLogger("request-header-contract")
    monkeypatch.setattr(logger, "handlers", [logging.StreamHandler(output)])
    monkeypatch.setattr(logger, "propagate", False)
    monkeypatch.setattr(logger, "level", logging.DEBUG)
    logger.setLevel(logging.DEBUG)
    monkeypatch.setattr(module, "logger", logger)
    monkeypatch.setattr(module, "get_http_request_id", lambda: "context.1")
    manager = module.HttpServerManager.__new__(module.HttpServerManager)
    asyncio.run(manager._log_req_header(headers, 80))
    line = output.getvalue()
    assert "event=request.received" in line
    path = tmp_path / "server.log"
    path.write_text(line)
    assert script.lookup_lightllm_req_id_from_log(str(path), expected, timeout=0.1) == 80
    output.truncate(0)
    output.seek(0)
    logger.setLevel(logging.INFO)
    asyncio.run(manager._log_req_header(headers, 80))
    assert output.getvalue() == ""
