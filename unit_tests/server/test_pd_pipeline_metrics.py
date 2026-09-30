import asyncio
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from lightllm.server import api_stream_obj
from lightllm.server.core.objs import StartArgs
from lightllm.server.httpserver.async_queue import AsyncQueue
from lightllm.server.httpserver_for_pd_master.manager import ReqStatus
from lightllm.server.metrics.metrics import Monitor


def test_queue_age_tracks_first_pending_item_and_resets(monkeypatch):
    clock = [10.0]
    monkeypatch.setattr("lightllm.server.httpserver.async_queue.time.monotonic", lambda: clock[0])

    async def run():
        queue = AsyncQueue()
        assert queue.oldest_age() == 0
        await queue.put("first")
        clock[0] = 12.0
        await queue.put("second")
        assert queue.oldest_age() == 2.0
        assert await queue.get_all_data() == ["first", "second"]
        assert queue.oldest_age() == 0
        assert not queue.event.is_set()
        await queue.put("third")
        clock[0] = 13.0
        assert queue.oldest_age() == 1.0

    asyncio.run(run())


def test_request_queue_age_on_requeue_and_drain(monkeypatch):
    clock = [10.0]
    monkeypatch.setattr("lightllm.server.httpserver_for_pd_master.manager.time.monotonic", lambda: clock[0])

    async def run():
        status = ReqStatus(123, None, None)
        assert status.oldest_age() == 0
        await status.put_tokens_to_front(["first"])
        clock[0] = 12.0
        await status.put_tokens_to_front(["second"])
        assert status.oldest_age() == 2.0
        assert await status.pop_all_tokens() == ["second", "first"]
        assert status.oldest_age() == 0
        await status.put_tokens_to_front([])
        assert status.oldest_token_time is None

    asyncio.run(run())


@pytest.mark.parametrize("mode", ["normal", "pd_master"])
@pytest.mark.parametrize("immediate_headers", [False, True])
def test_http_metrics_only_measure_pd_body_sends(monkeypatch, mode, immediate_headers):
    from lightllm.server.api_http import g_objs

    clock = [10.0]
    metric_client = Mock()
    monkeypatch.setattr(g_objs, "httpserver_manager", SimpleNamespace(metric_client=metric_client))
    monkeypatch.setattr(
        api_stream_obj,
        "get_env_start_args",
        lambda: StartArgs(run_mode=mode, disable_delay_response_start=immediate_headers),
    )
    monkeypatch.setattr(api_stream_obj.time, "monotonic", lambda: clock[0])
    monkeypatch.setattr(api_stream_obj, "_pd_send_next_metric_time", 0.0)
    monkeypatch.setattr(api_stream_obj, "_pd_send_max_duration", 0.0)
    monkeypatch.setattr(api_stream_obj, "_pd_send_max_bytes", 0)
    messages = []

    async def body():
        clock[0] += 5.0  # Generation time must not be counted as send time.
        yield "你好"

    async def send(message):
        messages.append(message)
        if message["type"] == "http.response.body" and message["body"]:
            clock[0] += 0.25

    asyncio.run(api_stream_obj.CustomStreamingResponse(body()).stream_response(send))
    assert [m["type"] for m in messages] == ["http.response.start", "http.response.body", "http.response.body"]
    assert messages[1]["body"] == "你好".encode()
    assert messages[-1]["more_body"] is False
    if mode == "pd_master":
        metric_client.histogram_observe.assert_called_once_with("lightllm_pd_master_http_send_duration", 0.25)
        metric_client.gauge_set.assert_called_once_with("lightllm_pd_master_http_send_bytes", 6)
    else:
        metric_client.histogram_observe.assert_not_called()
        metric_client.gauge_set.assert_not_called()


def test_http_sample_window_reports_maxima(monkeypatch):
    from lightllm.server.api_http import g_objs

    clock = [10.0]
    metrics = Mock()
    monkeypatch.setattr(g_objs, "httpserver_manager", SimpleNamespace(metric_client=metrics))
    monkeypatch.setattr(api_stream_obj.time, "monotonic", lambda: clock[0])
    monkeypatch.setattr(api_stream_obj, "_pd_send_next_metric_time", 11.0)
    monkeypatch.setattr(api_stream_obj, "_pd_send_max_duration", 0.0)
    monkeypatch.setattr(api_stream_obj, "_pd_send_max_bytes", 0)
    api_stream_obj._record_pd_send_metrics(0.2, 100)
    api_stream_obj._record_pd_send_metrics(0.4, 50)
    metrics.histogram_observe.assert_not_called()
    clock[0] = 11.0
    api_stream_obj._record_pd_send_metrics(0.1, 80)
    metrics.histogram_observe.assert_called_once_with("lightllm_pd_master_http_send_duration", 0.4)
    metrics.gauge_set.assert_called_once_with("lightllm_pd_master_http_send_bytes", 100)


def test_metrics_are_registered_with_seconds_and_bytes_units():
    monitor = Monitor(StartArgs(model_name="test", max_req_total_len=1024))
    names = [
        "lightllm_pd_forward_queue_duration",
        "lightllm_pd_master_ingress_queue_duration",
        "lightllm_pd_master_request_queue_duration",
        "lightllm_pd_master_http_send_duration",
    ]
    for name in names:
        assert "(s)" in monitor.monitor_registry[name]._documentation
    assert "(bytes)" in monitor.monitor_registry["lightllm_pd_master_http_send_bytes"]._documentation
