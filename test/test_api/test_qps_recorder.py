from types import SimpleNamespace
from unittest.mock import patch

import pytest

from lightllm.server.httpserver.qps_recorder import QPSRecorder
from lightllm.utils.envs_utils import (
    get_pd_request_limit_max_allowed_request_count_seconds,
)


def _args(run_mode="decode", running_max_req_size=16):
    return SimpleNamespace(run_mode=run_mode, running_max_req_size=running_max_req_size)


def test_qps_recorder_waits_for_sixteen_finished_requests():
    recorder = QPSRecorder(_args())

    with patch("lightllm.server.httpserver.qps_recorder.time.monotonic", side_effect=range(15)):
        for _ in range(15):
            recorder.mark_one_req_finish()

    assert recorder.get_qps() == 0.0


def test_qps_recorder_calculates_qps_and_updates_ema():
    recorder = QPSRecorder(_args(), ema_alpha=0.25)

    with patch(
        "lightllm.server.httpserver.qps_recorder.time.monotonic",
        side_effect=[*range(16), 15, 15, 15.5, 15.5, 15.5],
    ):
        for _ in range(16):
            recorder.mark_one_req_finish()
        assert recorder.get_qps() == 1.0

        recorder.mark_one_req_finish()
        window_qps = 15 / 14.5
        expected_qps = 0.25 * window_qps + 0.75 * 1.0
        assert recorder.get_qps() == pytest.approx(expected_qps)


def test_get_qps_updates_ema_after_thirty_seconds_without_new_request():
    recorder = QPSRecorder(_args(), ema_alpha=0.25)

    with patch(
        "lightllm.server.httpserver.qps_recorder.time.monotonic",
        side_effect=[*range(16), 15, 45.1, 45.1, 46],
    ):
        for _ in range(16):
            recorder.mark_one_req_finish()

        stale_window_qps = 15 / 45.1
        expected_qps = 0.25 * stale_window_qps + 0.75 * 1.0
        assert recorder.get_qps() == pytest.approx(expected_qps)
        assert recorder.get_qps() == pytest.approx(expected_qps)


def test_max_allowed_request_count_uses_env(monkeypatch):
    monkeypatch.setenv("LIGHTLLM_PD_REQUEST_LIMIT_MAX_ALLOWED_REQUEST_COUNT_SECONDS", "12")
    get_pd_request_limit_max_allowed_request_count_seconds.cache_clear()
    recorder = QPSRecorder(_args(running_max_req_size=1))
    with patch("lightllm.server.httpserver.qps_recorder.time.monotonic", side_effect=range(17)):
        for _ in range(16):
            recorder.mark_one_req_finish()

    try:
        with patch.object(recorder, "get_qps", return_value=2.5):
            assert recorder.get_max_allowed_request_count() == 36
    finally:
        get_pd_request_limit_max_allowed_request_count_seconds.cache_clear()


def test_max_allowed_request_count_uses_running_capacity_during_warmup():
    recorder = QPSRecorder(_args(run_mode="prefill", running_max_req_size=32))

    with patch.object(recorder, "get_qps") as get_qps:
        assert recorder.get_max_allowed_request_count() == 32
        get_qps.assert_not_called()


def test_max_allowed_request_count_waits_until_qps_is_initialized():
    recorder = QPSRecorder(_args(run_mode="prefill", running_max_req_size=1))
    recorder.mark_one_req_finish()

    with patch.object(recorder, "get_qps") as get_qps:
        assert recorder.get_max_allowed_request_count() == 1
        get_qps.assert_not_called()


def test_max_allowed_request_count_keeps_six_probe_requests_at_zero_qps():
    recorder = QPSRecorder(_args(run_mode="decode", running_max_req_size=1))
    with patch(
        "lightllm.server.httpserver.qps_recorder.time.monotonic",
        side_effect=range(17),
    ):
        for _ in range(16):
            recorder.mark_one_req_finish()

    with patch.object(recorder, "get_qps", return_value=0.0):
        assert recorder.get_max_allowed_request_count() == 6


@pytest.mark.parametrize(("run_mode", "expected_seconds"), [("prefill", 20), ("decode", 60)])
def test_pd_request_limit_max_allowed_request_count_seconds_uses_node_default(monkeypatch, run_mode, expected_seconds):
    monkeypatch.delenv("LIGHTLLM_PD_REQUEST_LIMIT_MAX_ALLOWED_REQUEST_COUNT_SECONDS", raising=False)
    get_pd_request_limit_max_allowed_request_count_seconds.cache_clear()

    try:
        assert get_pd_request_limit_max_allowed_request_count_seconds(run_mode) == expected_seconds
    finally:
        get_pd_request_limit_max_allowed_request_count_seconds.cache_clear()


@pytest.mark.parametrize("ema_alpha", [0, -0.1, 1.1])
def test_qps_recorder_rejects_invalid_ema_alpha(ema_alpha):
    with pytest.raises(ValueError, match=r"ema_alpha must be in the range \(0, 1\]"):
        QPSRecorder(_args(), ema_alpha=ema_alpha)
