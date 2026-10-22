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


# Configuration tests run in fresh interpreters: changing handlers in-process
# would interfere with pytest's capture and the rest of the test suite.
import json
import os
from pathlib import Path
import subprocess
import sys
import textwrap


def _run_logging(code, **settings):
    env = {key: value for key, value in os.environ.items() if not key.startswith("LIGHTLLM_LOG_")}
    env["PYTHONPATH"] = str(Path(__file__).resolve().parents[2]) + os.pathsep + env.get("PYTHONPATH", "")
    env["LIGHTLLM_LOG_REQUEST_SAMPLE_RATE"] = "1"
    env.update({key: str(value) for key, value in settings.items()})
    result = subprocess.run(
        [sys.executable, "-c", textwrap.dedent(code)],
        env=env,
        text=True,
        capture_output=True,
        timeout=30,
    )
    assert result.returncode == 0, result.stderr
    return result.stdout


def test_default_info_and_single_handler_for_module_loggers():
    output = _run_logging(
        """
        from lightllm.utils.log_utils import init_logger
        logger = init_logger("lightllm.server.router.test")
        assert not logger.handlers
        assert logger is init_logger(logger.name)
        logger.debug("hidden-debug")
        logger.info("visible-info")
    """
    )
    assert "hidden-debug" not in output
    assert output.count("visible-info") == 1
    assert "pid=" not in output and "tp=" not in output
    assert "[main]" in output
    assert "test_log_utils.py" not in output


def test_module_override_survives_global_error_and_reconfiguration():
    output = _run_logging(
        """
        from lightllm.utils.log_utils import init_logger, _setup_logger
        selected = init_logger("lightllm.server.router.test")
        other = init_logger("lightllm.server.httpserver.test")
        _setup_logger()
        _setup_logger()
        selected.debug("router-debug")
        other.info("hidden-info")
        other.error("visible-error")
    """,
        LIGHTLLM_LOG_LEVEL="ERROR",
        LIGHTLLM_LOG_MODULE_LEVELS='{"lightllm.server.router":"DEBUG"}',
    )
    assert output.count("router-debug") == 1
    assert "hidden-info" not in output
    assert output.count("visible-error") == 1


def test_json_preserves_context_multiline_and_traceback():
    output = _run_logging(
        """
        from lightllm.utils.log_utils import init_logger, set_log_node_role, set_log_context
        set_log_node_role("decode")
        set_log_context(component="model_infer", node_rank=2, tp_rank=3, dp_rank=1)
        try:
            raise ValueError("failure")
        except ValueError:
            init_logger("lightllm.test").exception("first\\nsecond", extra={"event":"test.failure", "request_id":42})
    """,
        LIGHTLLM_LOG_FORMAT="json",
    )
    lines = output.splitlines()
    assert len(lines) == 1
    event = json.loads(lines[0])
    assert event["level"] == "ERROR"
    assert event["message"] == "first\nsecond"
    assert event["role"] == "D"
    assert event["tp_rank"] == 3 and event["dp_rank"] == 1
    assert event["request_id"] == 42
    assert "ValueError: failure" in event["exception"]
    from datetime import datetime

    assert datetime.fromisoformat(event["timestamp"]).utcoffset() is not None


@pytest.mark.parametrize(
    "settings",
    [
        {"LIGHTLLM_LOG_LEVEL": "typo"},
        {"LIGHTLLM_LOG_FORMAT": "xml"},
        {"LIGHTLLM_LOG_MODULE_LEVELS": "[]"},
        {"LIGHTLLM_LOG_MODULE_LEVELS": '{"root":"DEBUG"}'},
        {"LIGHTLLM_LOG_MODULE_LEVELS": '{"lightllm.test":"bad"}'},
        {"LIGHTLLM_LOG_REQUEST_SAMPLE_RATE": "-0.1"},
        {"LIGHTLLM_LOG_REQUEST_SAMPLE_RATE": "nan"},
        {"LIGHTLLM_LOG_REQUEST_SAMPLE_RATE": "1.1"},
        {"LIGHTLLM_LOG_STATS_INTERVAL": "0"},
        {"LIGHTLLM_LOG_STATS_INTERVAL": "inf"},
    ],
)
def test_invalid_configuration_fails_explicitly(settings):
    env = {key: value for key, value in os.environ.items() if not key.startswith("LIGHTLLM_LOG_")}
    env.update(settings)
    env["PYTHONPATH"] = str(Path(__file__).resolve().parents[2]) + os.pathsep + env.get("PYTHONPATH", "")
    result = subprocess.run(
        [sys.executable, "-c", "from lightllm.utils import log_utils; log_utils._setup_logger()"],
        env=env,
        text=True,
        capture_output=True,
        timeout=30,
    )
    assert "using defaults until startup validation" in result.stdout
    assert result.returncode != 0
    assert "ValueError" in result.stderr


def test_cli_overrides_environment_and_inherits_unspecified_values():
    output = _run_logging(
        """
        import argparse
        import os
        from lightllm.server.api_cli import add_cli_args
        from lightllm.utils.log_utils import configure_logging_from_args, init_logger
        args = add_cli_args(argparse.ArgumentParser()).parse_args(["--log-level", "debug", "--run_mode", "prefill"])
        configure_logging_from_args(args)
        assert os.environ["LIGHTLLM_LOG_LEVEL"] == "debug"
        assert os.environ["LIGHTLLM_LOG_FORMAT"] == "json"
        assert os.environ["LIGHTLLM_LOG_NODE_ROLE"] == "P"
        init_logger("lightllm.test").debug("cli-debug")
    """,
        LIGHTLLM_LOG_LEVEL="ERROR",
        LIGHTLLM_LOG_FORMAT="json",
    )
    assert json.loads(output)["message"] == "cli-debug"


def test_request_and_access_categories_are_independent_and_exclude_probes():
    output = _run_logging(
        """
        from lightllm.utils.log_utils import init_logger, log_request_summary, log_http_access, get_http_logging_args
        init_logger("lightllm.server.router").info("hidden-service-info")
        log_request_summary(12, status="completed", output_tokens=4)
        log_http_access("GET", "/health", 200, 1)
        log_http_access("GET", "/metrics", 200, 1)
        log_http_access("POST", "/v1/chat/completions", 200, 3)
        assert "--access-logfile" not in get_http_logging_args(application_access=True)
        assert "--access-logfile" in get_http_logging_args()
    """,
        LIGHTLLM_LOG_LEVEL="ERROR",
        LIGHTLLM_LOG_FORMAT="json",
        LIGHTLLM_LOG_REQUEST_SAMPLE_RATE=1,
        LIGHTLLM_LOG_HTTP_ACCESS=1,
    )
    events = [json.loads(line) for line in output.splitlines()]
    assert [event["event"] for event in events] == ["request.completed", "http.access"]
    assert events[0]["request_id"] == 12
    assert events[1]["path"] == "/v1/chat/completions"


def test_request_logging_can_be_disabled_and_access_is_disabled_by_default():
    assert (
        _run_logging(
            """
        from lightllm.utils.log_utils import log_request_summary, log_http_access, get_http_logging_args
        log_request_summary(12, status="completed")
        log_http_access("POST", "/generate", 200, 1)
        assert "--access-logfile" not in get_http_logging_args()
    """,
            LIGHTLLM_LOG_REQUEST_SAMPLE_RATE=0,
        )
        == ""
    )


def test_sampling_is_stable_and_does_not_change_random_state(monkeypatch):
    import random

    monkeypatch.setattr(log_utils, "_request_sample_rate", 0.5)
    state = random.getstate()
    selected = [log_utils.should_log_request(i) for i in range(100)]
    assert selected == [log_utils.should_log_request(i) for i in range(100)]
    assert 20 < sum(selected) < 80
    assert random.getstate() == state


def test_periodic_stats_idle_resume_and_interval(monkeypatch):
    from unittest.mock import Mock

    clock = [0.0]
    monkeypatch.setattr(log_utils.time, "monotonic", lambda: clock[0])
    monkeypatch.setattr(log_utils, "_stats_interval", 10)
    logger = Mock()
    stats = log_utils.PeriodicStatsLogger(logger)
    assert not stats.should_log(False)
    clock[0] = 10
    assert stats.should_log(False)
    stats.log(running=0)
    assert logger.info.call_args.kwargs["extra"]["event"] == "engine.idle"
    clock[0] = 11
    assert not stats.should_log(True)
    clock[0] = 20
    assert stats.should_log(True)
    stats.log(running=2)
    assert logger.info.call_args.kwargs["extra"]["event"] == "engine.stats"
    clock[0] = 21
    assert not stats.should_log(False)
    clock[0] = 30
    assert not stats.should_log(False)
    clock[0] = 31
    assert stats.should_log(False)
    clock[0] = 100
    assert not stats.should_log(False)


def test_periodic_stats_flapping_never_bypasses_interval(monkeypatch):
    from unittest.mock import Mock

    clock = [0.0]
    monkeypatch.setattr(log_utils.time, "monotonic", lambda: clock[0])
    monkeypatch.setattr(log_utils, "_stats_interval", 10)
    stats = log_utils.PeriodicStatsLogger(Mock())
    emitted = []
    for tick in range(1000):
        clock[0] = tick / 10
        if stats.should_log(tick % 2 == 0):
            emitted.append(clock[0])
    assert emitted
    assert len(emitted) <= 10
    assert all(right - left >= 10 for left, right in zip(emitted, emitted[1:]))


def test_peer_configuration_is_debug_but_peer_errors_are_preserved():
    output = _run_logging(
        """
        from lightllm.utils.log_utils import init_logger, set_log_context, log_info_on_leader
        logger = init_logger("lightllm.test")
        set_log_context(tp_rank=0)
        log_info_on_leader(logger, "shared-config")
        set_log_context(tp_rank=1)
        log_info_on_leader(logger, "shared-config")
        logger.error("peer-failure")
    """
    )
    assert output.count("shared-config") == 1
    assert output.count("peer-failure") == 1


@pytest.mark.parametrize("method", ["fork", "spawn"])
def test_process_files_and_context_are_isolated(tmp_path, method):
    script = tmp_path / "logging_worker.py"
    script.write_text(
        """
import multiprocessing
from lightllm.utils.log_utils import init_logger, set_log_node_role, set_log_context
logger = init_logger("lightllm.worker")
def worker():
    logger.info("child-event")
if __name__ == "__main__":
    set_log_node_role("prefill")
    set_log_context(component="parent", tp_rank=7)
    logger.info("parent-event")
    child = multiprocessing.get_context(METHOD).Process(target=worker)
    child.start()
    child.join(20)
    assert child.exitcode == 0
""".replace(
            "METHOD", repr(method)
        )
    )
    env = {key: value for key, value in os.environ.items() if not key.startswith("LIGHTLLM_LOG_")}
    env.update(LIGHTLLM_LOG_FORMAT="json", LIGHTLLM_LOG_DIR=str(tmp_path / "logs"))
    env["PYTHONPATH"] = str(Path(__file__).resolve().parents[2]) + os.pathsep + env.get("PYTHONPATH", "")
    result = subprocess.run([sys.executable, str(script)], env=env, text=True, capture_output=True, timeout=40)
    assert result.returncode == 0, result.stderr
    files = sorted((tmp_path / "logs").glob("*.log"))
    assert len(files) == 2
    events = [json.loads(path.read_text()) for path in files]
    by_message = {event["message"]: event for event in events}
    assert by_message["parent-event"]["tp_rank"] == 7
    assert by_message["child-event"]["tp_rank"] == "-"
    assert by_message["child-event"]["component"] != "parent"
    assert all(event["role"] == "P" for event in events)
    assert all(path.name == f"process.{event['pid']}.log" for path, event in zip(files, events))


def test_file_rotation_is_bounded(tmp_path):
    _run_logging(
        """
        from lightllm.utils.log_utils import init_logger
        logger = init_logger("lightllm.test")
        for i in range(30):
            logger.info("rotation-entry-%d", i)
    """,
        LIGHTLLM_LOG_DIR=tmp_path,
        LIGHTLLM_LOG_MAX_BYTES=300,
        LIGHTLLM_LOG_BACKUP_COUNT=2,
    )
    files = list(tmp_path.iterdir())
    assert len(files) == 3
    current = next(path for path in files if path.name.endswith(".log"))
    assert "rotation-entry-29" in current.read_text()


def test_startup_summary_does_not_dump_secrets():
    output = _run_logging(
        """
        from types import SimpleNamespace
        from lightllm.utils.log_utils import init_logger, log_startup_config
        log_startup_config(init_logger("lightllm.test"), SimpleNamespace(
            run_mode="normal", tp=2, dp=1, api_key="never-log-this", model_dir="s3://secret@bucket"))
    """,
        LIGHTLLM_LOG_LEVEL="DEBUG",
    )
    assert "tp=2" in output
    assert "never-log-this" not in output and "secret@bucket" not in output


@pytest.mark.parametrize(
    "tty, log_format, level, expected",
    [
        (False, "text", logging.INFO, False),
        (True, "json", logging.INFO, False),
        (True, "text", logging.ERROR, False),
        (True, "text", logging.INFO, True),
    ],
)
def test_progress_only_enabled_for_interactive_text(monkeypatch, tty, log_format, level, expected):
    monkeypatch.setattr(log_utils.sys.stderr, "isatty", lambda: tty)
    monkeypatch.setenv("LIGHTLLM_LOG_FORMAT", log_format)
    monkeypatch.setattr(log_utils._root_logger, "isEnabledFor", lambda severity: level <= severity)
    assert log_utils.progress_enabled() is expected


def test_log_helpers_preserve_business_callsite():
    output = _run_logging(
        """
        import inspect
        import logging
        from lightllm.utils.log_utils import log_request_summary, log_http_access, PeriodicStatsLogger
        def business_call():
            line = inspect.currentframe().f_lineno + 1
            log_request_summary(11, status="completed", expected_line=line)
            return line
        expected = business_call()
        log_http_access("POST", "/generate", 200, 1)
        stats = PeriodicStatsLogger(logging.getLogger("lightllm.stats"))
        stats.active = True
        stats.log(running=1)
    """,
        LIGHTLLM_LOG_FORMAT="json",
        LIGHTLLM_LOG_REQUEST_SAMPLE_RATE=1,
        LIGHTLLM_LOG_HTTP_ACCESS=1,
    )
    events = [json.loads(line) for line in output.splitlines()]
    assert all(event["file"] == "<string>" for event in events)
    assert events[0]["line"] == events[0]["expected_line"]
    assert events[0]["line"] < events[1]["line"] < events[2]["line"]


def test_python_warnings_use_central_output():
    output = _run_logging(
        """
        import warnings
        from lightllm.utils import log_utils
        warnings.warn("dependency-warning", UserWarning)
    """,
        LIGHTLLM_LOG_FORMAT="json",
    )
    event = json.loads(output)
    assert event["level"] == "WARNING"
    assert event["logger"] == "py.warnings"
    assert "dependency-warning" in event["message"]


@pytest.mark.parametrize(
    "mode, tp_size, dp_size, nnodes, ranks, expected",
    [
        ("normal", 1, 1, 1, (0, 0, 0), "[infer]"),
        ("prefill", 2, 1, 1, (0, 1, 0), "[P/infer tp=1]"),
        ("decode", 1, 2, 1, (0, 0, 1), "[D/infer dp=1]"),
        ("normal", 2, 2, 2, (1, 0, 1), "[infer node=1 tp=0 dp=1]"),
        ("normal", 2, 2, 2, ("-", "-", "-"), "[infer]"),
    ],
)
def test_compact_text_context_only_shows_applicable_ranks(monkeypatch, mode, tp_size, dp_size, nnodes, ranks, expected):
    import re

    monkeypatch.setenv("LIGHTLLM_LOG_TP_SIZE", str(tp_size))
    monkeypatch.setenv("LIGHTLLM_LOG_DP_SIZE", str(dp_size))
    monkeypatch.setenv("LIGHTLLM_LOG_NNODES", str(nnodes))
    monkeypatch.setattr(log_utils, "_LOG_NODE_ROLE", {"prefill": "P", "decode": "D"}.get(mode, ""))
    monkeypatch.setattr(
        log_utils,
        "_context",
        {"component": "model_infer", "node_rank": ranks[0], "tp_rank": ranks[1], "dp_rank": ranks[2]},
    )
    monkeypatch.setattr(log_utils, "_context_pid", os.getpid())
    record = logging.LogRecord("lightllm.test", logging.INFO, __file__, 12, "ready", (), None)
    output = log_utils.NewLineFormatter(log_utils._FORMAT).format(record)
    assert re.fullmatch(r"\d{2}:\d{2}:\d{2}\.\d{3} INFO " + re.escape(expected) + " ready", output)
    assert "pid=" not in output and "global=" not in output


@pytest.mark.parametrize(
    "level, source", [(logging.DEBUG, True), (logging.INFO, False), (logging.WARNING, False), (logging.ERROR, True)]
)
def test_compact_text_retains_source_for_debug_and_error(level, source):
    record = logging.LogRecord("lightllm.test", level, "/example/business.py", 42, "event", (), None)
    output = log_utils.NewLineFormatter(log_utils._FORMAT).format(record)
    assert ("business.py:42" in output) is source


def test_compact_exception_prefixes_every_traceback_line():
    try:
        raise ValueError("failed")
    except ValueError:
        record = logging.LogRecord(
            "lightllm.test", logging.ERROR, "/example/business.py", 42, "operation failed", (), sys.exc_info()
        )
    output = log_utils.NewLineFormatter(log_utils._FORMAT).format(record)
    assert "ValueError: failed" in output
    assert all("business.py:42" in line for line in output.splitlines())


def test_pd_master_component_is_short(monkeypatch):
    monkeypatch.setenv("LIGHTLLM_LOG_RUN_MODE", "pd_master")
    monkeypatch.setattr(log_utils, "_context", {"component": "http"})
    monkeypatch.setattr(log_utils, "_context_pid", os.getpid())
    monkeypatch.setattr(log_utils, "_LOG_NODE_ROLE", "")
    record = logging.LogRecord("lightllm.test", logging.INFO, __file__, 12, "ready", (), None)
    assert "[master] ready" in log_utils.NewLineFormatter(log_utils._FORMAT).format(record)


def test_json_remains_complete_after_compact_text_formatting(monkeypatch):
    monkeypatch.setattr(log_utils, "_context", {"component": "model_infer", "tp_rank": 1, "dp_rank": 0})
    monkeypatch.setattr(log_utils, "_context_pid", os.getpid())
    record = logging.LogRecord("lightllm.test", logging.INFO, "/example/business.py", 42, "ready", (), None)
    log_utils.NewLineFormatter(log_utils._FORMAT).format(record)
    result = json.loads(log_utils.JsonFormatter().format(record))
    assert "T" in result["timestamp"]
    assert result["pid"] == os.getpid()
    assert result["hostname"]
    assert result["component"] == "model_infer"
    assert result["tp_rank"] == 1 and result["dp_rank"] == 0
    assert result["file"] == "/example/business.py" and result["line"] == 42
    assert "compact_context" not in result and "source" not in result


def test_request_summary_defaults_and_optional_fields():
    output = _run_logging(
        """
        from lightllm.utils.log_utils import log_request_summary
        log_request_summary(1, status="completed", timing_scope="decode", gpu_cache_tokens=0,
                            cpu_cache_tokens=12, mtp_verify_steps=2, mtp_enabled=False)
        log_request_summary(2, status="completed", mtp_verify_steps=2, mtp_enabled=True)
    """,
        LIGHTLLM_LOG_FORMAT="json",
        LIGHTLLM_LOG_LEVEL="DEBUG",
    )
    first, second = [json.loads(line) for line in output.splitlines()]
    assert first["timing_scope"] == "decode"
    assert "gpu_cache_tokens" not in first and "mtp_verify_steps" not in first
    assert first["cpu_cache_tokens"] == 12
    assert second["mtp_verify_steps"] == 2


def test_compiler_error_summary_preserves_root_cause():
    reason = log_utils.summarize_exception(
        RuntimeError(
            "Ninja build failed\n"
            + "nvcc "
            + "flags " * 1000
            + "\n/bin/sh: /usr/local/cuda-12.6/bin/nvcc: No such file or directory\nninja: build stopped."
        )
    )
    assert "nvcc: No such file or directory" in reason
    assert len(reason) <= 240
    assert len(log_utils.summarize_exception(RuntimeError("x" * 1000))) <= 240


def test_errors_bypass_sampling_and_pd_stages_are_debug():
    output = _run_logging(
        """
        from lightllm.utils.log_utils import log_request_summary
        log_request_summary(1, status="completed", timing_scope="end_to_end")
        log_request_summary(2, status="error", timing_scope="decode")
        log_request_summary(3, status="cancelled", timing_scope="end_to_end")
        log_request_summary(6, status="continued", timing_scope="end_to_end")
        """,
        LIGHTLLM_LOG_FORMAT="json",
        LIGHTLLM_LOG_LEVEL="DEBUG",
        LIGHTLLM_LOG_REQUEST_SAMPLE_RATE=0,
    )
    records = [json.loads(line) for line in output.splitlines()]
    assert [(r["request_id"], r["event"], r["level"]) for r in records] == [
        (2, "request.failed", "ERROR"),
        (3, "request.cancelled", "DEBUG"),
    ]
    output = _run_logging(
        """
        from lightllm.utils.log_utils import log_request_summary
        log_request_summary(4, status="completed", timing_scope="prefill")
        log_request_summary(5, status="completed", timing_scope="end_to_end")
        log_request_summary(7, status="completed", timing_scope="decode")
        """,
        LIGHTLLM_LOG_FORMAT="json",
        LIGHTLLM_LOG_LEVEL="DEBUG",
    )
    records = [json.loads(line) for line in output.splitlines()]
    assert [(r["request_id"], r["level"]) for r in records] == [(4, "DEBUG"), (5, "INFO"), (7, "DEBUG")]


def test_terminal_error_is_not_hidden_by_successful_sibling():
    assert log_utils.request_finish_fields(["stop", "error"])["status"] == "error"
    assert log_utils.request_finish_fields(["length", "abort"])["status"] == "abort"
    assert log_utils.request_finish_fields(["length"])["finish_reason"] == "length"


def test_compact_stats_text_retains_exact_json_window():
    code = """
        import logging
        from lightllm.utils.log_utils import PeriodicStatsLogger
        stats = PeriodicStatsLogger(logging.getLogger("lightllm.stats"))
        stats.active = True
        stats.log(prefill_tokens=20, output_tokens=40, completed_reqs=1, window_s=2,
                  output_tok_s=20, running=1, kv_active_pct_by_local_dp=[3.0])
    """
    text = _run_logging(code)
    assert "output_tok_s=20" in text and "running=1" in text
    assert "output_tokens=" not in text and "window_s=" not in text
    record = json.loads(_run_logging(code, LIGHTLLM_LOG_FORMAT="json"))
    assert record["output_tokens"] == 40 and record["window_s"] == 2


def test_python_warnings_survive_error_service_level_and_reconfiguration():
    output = _run_logging(
        """
        import warnings
        from lightllm.utils.log_utils import _setup_logger
        warnings.warn("before-reconfigure", RuntimeWarning)
        _setup_logger()
        warnings.warn("after-reconfigure", RuntimeWarning)
        """,
        LIGHTLLM_LOG_LEVEL="ERROR",
    )
    assert output.count("before-reconfigure") == 1
    assert output.count("after-reconfigure") == 1


def test_legacy_pd_detail_flag_explains_bounded_debug_replacement():
    output = _run_logging(
        """
        import argparse
        from lightllm.server.api_cli import add_cli_args
        from lightllm.utils.log_utils import configure_logging_from_args
        args = add_cli_args(argparse.ArgumentParser()).parse_args(["--pd_detail_log"])
        configure_logging_from_args(args)
        """
    )
    assert output.count("--pd_detail_log is deprecated") == 1
    assert "bounded summaries" in output


@pytest.mark.parametrize("level, expected", [("warn", 30), ("fatal", 50), ("notset", 0)])
def test_standard_logging_level_aliases(level, expected):
    _run_logging(
        f"from lightllm.utils import log_utils; assert log_utils._root_logger.level == {expected}",
        LIGHTLLM_LOG_LEVEL=level,
    )


def test_config_server_access_switch_enables_hypercorn_info(monkeypatch):
    from hypercorn.config import Config
    from hypercorn.logging import Logger

    monkeypatch.setenv("LIGHTLLM_LOG_HTTP_ACCESS", "1")
    monkeypatch.delenv("LIGHTLLM_LOG_HTTP_LEVEL", raising=False)
    args = log_utils.get_http_logging_args()
    config = Config()
    config.loglevel = args[args.index("--log-level") + 1]
    config.accesslog = "-"
    assert Logger(config).access_logger.isEnabledFor(logging.INFO)


def test_multiline_warning_text_is_preserved():
    record = logging.LogRecord("py.warnings", logging.WARNING, __file__, 1, "first\nsecond", (), None)
    formatted = log_utils.NewLineFormatter("%(message)s").format(record)
    assert "first" in formatted and "second" in formatted


@pytest.mark.parametrize("kind", ["vision", "audio"])
def test_multimodal_rank_context_uses_its_own_tp_size(monkeypatch, kind):
    from lightllm.utils import dist_utils

    context = {}
    for setter in ("set_dp_size", "set_dp_world_size", "set_current_rank_in_dp", "set_current_device_id"):
        monkeypatch.setattr(dist_utils, setter, lambda _value: None)
    monkeypatch.setattr(dist_utils, "set_log_context", lambda **fields: context.update(fields))

    class StopBeforeCuda(Exception):
        pass

    def stop(_device):
        raise StopBeforeCuda()

    monkeypatch.setattr(dist_utils.torch.cuda, "set_device", stop)
    init = dist_utils.init_vision_distributed_env if kind == "vision" else dist_utils.init_audio_distributed_env
    params = {"vit_tp": 4, "audio_tp": 4, "tp_rank_id": 3, "device_id": 0}
    with pytest.raises(StopBeforeCuda):
        init(params)
    assert context == {"tp_rank": 3, "tp_size": 4, "dp_rank": "-"}


def test_default_request_sampling_is_stable_and_bounded():
    _run_logging(
        """
        import os
        os.environ.pop("LIGHTLLM_LOG_REQUEST_SAMPLE_RATE", None)
        from lightllm.utils import log_utils
        assert log_utils._request_sample_rate == 0.01
        first = [log_utils.should_log_request(i) for i in range(2000)]
        assert first == [log_utils.should_log_request(i) for i in range(2000)]
        assert 5 <= sum(first) <= 40
        """
    )


@pytest.mark.parametrize(
    "status,level,event",
    [
        ("busy", logging.WARNING, "request.failed"),
        ("rejected", logging.WARNING, "request.rejected"),
        ("error", logging.ERROR, "request.failed"),
        ("timeout", logging.ERROR, "request.failed"),
        ("failed", logging.ERROR, "request.failed"),
        ("cancelled", logging.DEBUG, "request.cancelled"),
    ],
)
def test_failure_summary_level_and_sampling(monkeypatch, caplog, status, level, event):
    monkeypatch.setattr(log_utils, "should_log_request", lambda request_id: False)
    logger = logging.getLogger("lightllm.requests")
    monkeypatch.setattr(logger, "handlers", [caplog.handler])
    with caplog.at_level(logging.DEBUG, logger="lightllm.requests"):
        log_utils.log_request_summary(123, status=status, reason="bounded reason")
    record = caplog.records[-1]
    assert record.levelno == level
    assert record.event == event
    assert record.reason == "bounded reason"


@pytest.mark.parametrize("separator", ["-", "_"])
def test_all_logging_cli_options_survive_start_args_and_override_environment(separator):
    output = _run_logging(
        f"""
        import argparse
        import os
        from lightllm.server.api_cli import add_cli_args
        from lightllm.server.core.objs.start_args_type import StartArgs
        from lightllm.utils.log_utils import configure_logging_from_args
        flags = [
            ("log_level", "debug"), ("log_format", "json"),
            ("log_module_levels", '{{"lightllm.test": "ERROR"}}'),
            ("log_request_sample_rate", "0.5"), ("log_stats_interval", "2.5"),
            ("log_http_level", "error"),
        ]
        argv = []
        for name, value in flags:
            argv.extend(["--" + name.replace("_", {separator!r}), value])
        argv.append("--" + "enable_http_access_log".replace("_", {separator!r}))
        args = StartArgs(**vars(add_cli_args(argparse.ArgumentParser()).parse_args(argv)))
        configure_logging_from_args(args)
        for name, value in flags:
            assert os.environ["LIGHTLLM_" + name.upper()] == value
        assert os.environ["LIGHTLLM_LOG_HTTP_ACCESS"] == "1"
        print("configured")
        """,
        LIGHTLLM_LOG_LEVEL="critical",
        LIGHTLLM_LOG_HTTP_ACCESS="0",
    )
    assert output.strip() == "configured"


@pytest.mark.parametrize("setting", ["20", "Info,WARNING"])
def test_invalid_legacy_level_import_keeps_warning_capture(setting):
    output = _run_logging(
        """
        import warnings
        from lightllm.utils import log_utils
        warnings.warn("spawn-warning-regression", UserWarning)
        """,
        LIGHTLLM_LOG_LEVEL=setting,
        LIGHTLLM_LOG_FORMAT="json",
    )
    events = [json.loads(line) for line in output.splitlines()]
    assert any(event["logger"] == "py.warnings" and "spawn-warning-regression" in event["message"] for event in events)


def test_log_time_ready_uses_process_local_monotonic_windows(monkeypatch):
    clock = [0]
    pid = [100]
    monkeypatch.setattr(log_utils, "_log_time_mark_dict", {})
    monkeypatch.setattr(log_utils.time, "monotonic", lambda: clock[0])
    monkeypatch.setattr(log_utils.os, "getpid", lambda: pid[0])
    assert not log_utils.log_time_ready("window", 10)
    clock[0] = 10
    assert log_utils.log_time_ready("window", 10)
    assert not log_utils.log_time_ready("window", 10)
    pid[0] = 101
    assert not log_utils.log_time_ready("window", 10)
    clock[0] = 20
    assert log_utils.log_time_ready("window", 10)
