"""Verify debug volume bounds, aggregate accounting and exception evidence."""

from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor
import logging

from lightllm.utils import log_utils


def recording_logger():
    records = []

    class Handler(logging.Handler):
        def emit(self, record):
            records.append(record)

    logger = logging.Logger("debug-volume-test", logging.DEBUG)
    logger.addHandler(Handler())
    return logger, records


def test_debug_fields_do_not_expand_payloads_or_objects():
    logger, records = recording_logger()

    class Descriptor:
        def __repr__(self):
            raise AssertionError("Must not expand descriptors")

    recursive = []
    recursive.append(recursive)
    log_utils.log_debug_event(
        logger,
        "bounded",
        text="x" * 100000,
        token_ids=list(range(100000)),
        descriptor=Descriptor(),
        metadata=b"x" * 100000,
        nested=recursive,
    )
    record = records[0]
    assert len(record.getMessage()) < 1000
    assert record.token_ids["count"] == 100000
    assert len(record.token_ids["sample"]) == 8
    assert record.descriptor == {"type": "Descriptor"}
    assert record.metadata == {"type": "bytes", "bytes": 100000}
    assert record.nested == {"count": 1, "sample": [{"count": 1}]}


def test_debug_event_shared_tp_state_is_only_reported_by_leader(monkeypatch):
    logger, records = recording_logger()
    for rank in range(8):
        monkeypatch.setattr(log_utils, "_context", {"tp_rank": rank})
        log_utils.log_debug_event(logger, "cache.idle", leader_only=True, cached_tokens=32)
    assert len(records) == 1


def test_debug_stats_count_pages_without_per_page_lines(monkeypatch):
    now = [0.0]
    monkeypatch.setattr(log_utils.time, "monotonic", lambda: now[0])
    logger, records = recording_logger()
    stats = log_utils.DebugStatsLogger(logger, "kv.transfer", interval=10)
    for index in range(10000):
        stats.add(request_id=index, pages=1, bytes_total=4096, kv_tokens_total=16)
    assert len(records) == 1
    now[0] = 10
    stats.flush()
    assert len(records) == 2
    assert records[1].request_ids_truncated
    assert len(records[1].request_id_sample["sample"]) == 8
    assert sum(record.pages for record in records) == 10000
    assert sum(record.bytes_total for record in records) == 10000 * 4096
    assert sum(record.kv_tokens_total for record in records) == 160000
    now[0] = 100
    stats.flush()
    assert len(records) == 2  # No idle heartbeat.


def test_debug_stats_are_thread_safe_and_do_nothing_when_disabled(monkeypatch):
    now = [0.0]
    monkeypatch.setattr(log_utils.time, "monotonic", lambda: now[0])
    logger, records = recording_logger()
    stats = log_utils.DebugStatsLogger(logger, "kv.transfer", interval=10)

    def add(_):
        for _ in range(1000):
            stats.add(pages=1)

    with ThreadPoolExecutor(max_workers=4) as pool:
        list(pool.map(add, range(4)))
    now[0] = 10
    stats.flush()
    assert sum(record.pages for record in records) == 4000
    logger.disabled = True
    stats.add(unused=1)
    assert stats.count == 0 and stats.totals == {}


def test_debug_stats_reset_inherited_lock_after_fork(monkeypatch):
    logger, records = recording_logger()
    stats = log_utils.DebugStatsLogger(logger, "kv.transfer", interval=10)
    stats.add(pages=1)
    stats.add(pages=5)
    stats.lock.acquire()
    monkeypatch.setattr(stats, "pid", -1)
    stats.add(pages=2)
    assert records[-1].pages == 2  # No parent counts or inherited locked mutex.


def test_repeated_connection_failures_keep_first_traceback_and_distinct_reasons(monkeypatch):
    now = [0.0]
    monkeypatch.setattr(log_utils.time, "monotonic", lambda: now[0])
    monkeypatch.setattr(log_utils, "_exception_windows", OrderedDict())
    logger, records = recording_logger()

    def fail(reason):
        try:
            raise ConnectionError(reason)
        except ConnectionError:
            log_utils.log_repeated_exception(logger, "pd.connection_failed", master="localhost:1")

    for _ in range(100):
        fail("connection refused")
    assert len(records) == 1 and records[0].exc_info
    now[0] = 10
    fail("connection refused")
    assert len(records) == 2 and not records[1].exc_info
    assert records[1].repeated == 100
    fail("connection reset")
    assert len(records) == 3 and records[2].exc_info
    for index in range(200):
        fail(f"different reason {index}")
    assert len(log_utils._exception_windows) == 128


def test_latency_summaries_do_not_repeat_while_idle(monkeypatch):
    from lightllm.utils import statics_utils

    now = [0.0]
    monkeypatch.setattr(statics_utils.time, "time", lambda: now[0])
    logger, records = recording_logger()
    monkeypatch.setattr(statics_utils, "logger", logger)
    average = statics_utils.MovingAverage()
    now[0] = 30
    average.print_log("ttft")
    assert not records
    average.add(10)
    average.print_log("ttft")
    assert len(records) == 1
    now[0] = 60
    average.print_log("ttft")
    assert len(records) == 1
    average.add(30)
    average.print_log("ttft")
    assert len(records) == 2
    assert average.average() == 20


def test_debug_numpy_scalars_retain_ids_and_counters():
    import numpy as np

    logger, records = recording_logger()
    log_utils.log_debug_event(logger, "request.admitted", request_id=np.int64(42), ratio=np.float32(0.25))
    assert records[0].request_id == 42
    assert records[0].ratio == 0.25
    assert "request_id=42" in records[0].getMessage()


def test_debug_aggregate_points_to_business_caller():
    logger, records = recording_logger()
    stats = log_utils.DebugStatsLogger(logger, "test.aggregate")
    stats.add(request_id=42, pages=1)
    assert records[0].pathname == __file__
    assert records[0].funcName == "test_debug_aggregate_points_to_business_caller"
