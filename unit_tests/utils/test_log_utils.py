import logging

import pytest

from lightllm.utils import log_utils


def _format_message(message: str) -> str:
    formatter = log_utils.NewLineFormatter("%(node_role)s%(message)s")
    record = logging.LogRecord("test", logging.INFO, __file__, 1, message, (), None)
    return formatter.format(record)


@pytest.mark.parametrize(
    ("run_mode", "expected_marker"),
    [
        ("prefill", "P"),
        ("decode", "D"),
        ("normal", ""),
        ("pd_master", ""),
    ],
)
def test_log_node_role_marker_is_limited_to_pd_workers(monkeypatch, run_mode, expected_marker):
    monkeypatch.delenv("LIGHTLLM_LOG_NODE_ROLE", raising=False)
    monkeypatch.setattr(log_utils, "_LOG_NODE_ROLE", "")

    log_utils.set_log_node_role(run_mode)

    formatted_marker = f"[{expected_marker}] " if expected_marker else ""
    assert _format_message("ready") == f"{formatted_marker}ready"
    if expected_marker:
        assert log_utils.os.environ["LIGHTLLM_LOG_NODE_ROLE"] == expected_marker
    else:
        assert "LIGHTLLM_LOG_NODE_ROLE" not in log_utils.os.environ


def test_log_node_role_marker_is_applied_to_every_line(monkeypatch):
    monkeypatch.delenv("LIGHTLLM_LOG_NODE_ROLE", raising=False)
    monkeypatch.setattr(log_utils, "_LOG_NODE_ROLE", "")
    log_utils.set_log_node_role("prefill")

    assert _format_message("first\nsecond") == "[P] first\r\n[P] second"
