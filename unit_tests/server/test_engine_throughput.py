"""Window accounting without GPU synchronization or request-completion spikes."""

from types import SimpleNamespace
from unittest.mock import Mock
import numpy as np
import pytest
import torch

from lightllm.server import api_http  # Initialize service imports before backend modules.
from lightllm.server.core.objs import FinishStatus
from lightllm.server.router import engine_stats
from lightllm.server.router.batch import Batch
from lightllm.server.router.model_infer import infer_batch
from lightllm.server.router.model_infer.mode_backend import base_backend
from lightllm.utils import log_utils


@pytest.fixture
def counters(monkeypatch):
    arrays = {}

    class Array:
        def __init__(self, name, shape, dtype):
            self.name, self.shape, self.dtype = name, shape, dtype

        def create_shm(self):
            self.arr = arrays.setdefault(self.name, np.zeros(self.shape, dtype=self.dtype))

        def link_shm(self):
            self.arr = arrays[self.name]

    monkeypatch.setattr(engine_stats, "ShmArray", Array)
    return engine_stats.EngineCounters(2, owner=True)


def test_shared_dp_rows_are_aggregated_without_tp_multiplier(counters):
    first = engine_stats.EngineCounters(2, 0)
    second = engine_stats.EngineCounters(2, 1)
    first.add(512, 4)
    second.add(256, 2)
    assert counters.totals() == (768, 6)
    assert counters.storage.arr.tolist() == [[512, 4], [256, 2]]


def test_running_request_rates_and_completed_count_use_window_deltas(monkeypatch, counters):
    now = [0.0]
    monkeypatch.setattr(engine_stats.time, "monotonic", lambda: now[0])
    throughput = engine_stats.EngineThroughput(counters)
    writer = engine_stats.EngineCounters(2, 0)
    writer.add(512, 64)
    now[0] = 10
    first = throughput.snapshot()
    assert first["prefill_tok_s"] == 51.2
    assert first["output_tok_s"] == 6.4
    assert first["completed_req_s"] == 0  # Still running.
    writer.add(128, 32)
    throughput.finish(SimpleNamespace(is_aborted=False, finish_status=FinishStatus(FinishStatus.FINISHED_LENGTH)))
    throughput.finish(SimpleNamespace(is_aborted=True, finish_status=FinishStatus(FinishStatus.FINISHED_ABORTED)))
    now[0] = 20
    second = throughput.snapshot()
    assert second["prefill_tokens"] == 128 and second["output_tokens"] == 32
    assert second["completed_reqs"] == 1 and second["completed_req_s"] == 0.1
    assert not throughput.has_activity()


def test_short_burst_after_idle_is_reported_once(monkeypatch, counters):
    now = [0.0]
    monkeypatch.setattr(engine_stats.time, "monotonic", lambda: now[0])
    monkeypatch.setattr(log_utils, "get_log_stats_interval", lambda: 10)
    throughput = engine_stats.EngineThroughput(counters)
    periodic = log_utils.PeriodicStatsLogger(Mock())
    assert not periodic.should_log(False)
    now[0] = 10
    assert periodic.should_log(False)
    throughput.snapshot()
    now[0] = 11
    engine_stats.EngineCounters(2, 0).add(100, 10)
    assert throughput.has_activity()
    assert not periodic.should_log(throughput.has_activity())
    now[0] = 20
    assert periodic.should_log(throughput.has_activity())
    assert throughput.snapshot()["output_tokens"] == 10
    assert not periodic.should_log(False)
    now[0] = 30
    assert periodic.should_log(False)
    assert throughput.snapshot()["output_tokens"] == 0
    now[0] = 60
    assert not periodic.should_log(False)


def request(input_len, cached=0, output_len=0, max_output=100):
    shm = SimpleNamespace(input_len=input_len, shm_cur_kv_len=cached, shm_cur_output_len=output_len)
    status = FinishStatus()
    req = SimpleNamespace(
        cur_kv_len=cached, cur_output_len=output_len, shm_req=shm, finish_status=status, output_constraint=None
    )
    req.get_cur_total_len = lambda: input_len + req.cur_output_len
    req.get_chuncked_input_token_len = lambda: min(req.cur_kv_len + 512, req.get_cur_total_len())
    req.set_next_gen_token_id = Mock()

    def update(eos_ids, output_len):
        if output_len >= max_output:
            status.set_status(FinishStatus.FINISHED_LENGTH)

    req.update_finish_status = update
    return req


def backend(monkeypatch, counters, leader=True):
    result = base_backend.ModeBackend.__new__(base_backend.ModeBackend)
    result.is_master_in_dp = leader
    result.engine_counters = engine_stats.EngineCounters(2, 0) if leader else None
    result.eos_id = [2]
    monkeypatch.setattr(
        base_backend,
        "g_infer_context",
        SimpleNamespace(
            req_manager=SimpleNamespace(req_sampling_params_manager=Mock()),
        ),
    )
    return result


def post(result, reqs, packs):
    count = len(reqs)
    result._post_handle(
        reqs, torch.zeros(count, dtype=torch.int64), torch.zeros(count), torch.zeros(count, dtype=torch.int64), packs
    )


def test_chunked_prefill_counts_computed_input_only_after_postprocessing(monkeypatch, counters):
    result = backend(monkeypatch, counters)
    req = request(1000, cached=200)
    first = result._pre_post_handle([req], is_chuncked_mode=True)
    assert first[0].prefill_tokens == 512
    assert counters.totals() == (0, 0)  # No work reported before forward synchronization.
    post(result, [req], first)
    assert counters.totals() == (512, 0)
    second = result._pre_post_handle([req], is_chuncked_mode=True)
    post(result, [req], second)
    assert counters.totals() == (800, 1)


def test_pd_received_kv_and_prefill_first_token_are_not_recounted(monkeypatch, counters):
    result = backend(monkeypatch, counters)
    req = request(1000, cached=1000, output_len=1)  # KV and first token received from P.
    packs = result._pre_post_handle([req], is_chuncked_mode=False)
    post(result, [req], packs)
    assert counters.totals() == (0, 1)


def test_mtp_counts_accepted_tokens_and_ignores_tokens_after_finish(monkeypatch, counters):
    result = backend(monkeypatch, counters)
    req = request(10, cached=10, output_len=1, max_output=3)
    packs = [infer_batch.InferReqUpdatePack(req, output_len=length) for length in [2, 3, 4]]
    post(result, [req] * 3, packs)
    assert counters.totals() == (0, 2)


def test_nonleader_does_not_write_engine_counters(monkeypatch, counters):
    result = backend(monkeypatch, counters, leader=False)
    req = request(10)
    post(result, [req], result._pre_post_handle([req], is_chuncked_mode=False))
    assert counters.totals() == (0, 0)


def test_batch_records_success_once_before_releasing_request(monkeypatch, counters):
    throughput = engine_stats.EngineThroughput(counters)
    req = SimpleNamespace(
        request_id=42,
        shm_infer_released=True,
        is_aborted=False,
        candetoken_out_len=10,
        finish_status=FinishStatus(FinishStatus.FINISHED_STOP),
    )
    batch = Batch(1, [req], 1)
    shm_manager, statics = Mock(), Mock()
    batch.filter_out_finished_req(shm_manager, statics, throughput)
    batch.filter_out_finished_req(shm_manager, statics, throughput)
    assert throughput.completed == 1
    shm_manager.put_back_req_obj.assert_called_once_with(req)


def test_resume_excludes_idle_time_without_discarding_short_burst(monkeypatch, counters):
    now = [0.0]
    monkeypatch.setattr(engine_stats.time, "monotonic", lambda: now[0])
    throughput = engine_stats.EngineThroughput(counters)
    now[0] = 1000
    engine_stats.EngineCounters(2, 0).add(40, 20)
    throughput.resume(since=999.99)
    now[0] = 1009.99
    snapshot = throughput.snapshot()
    assert snapshot["window_s"] == 10
    assert snapshot["output_tokens"] == 20
    assert snapshot["output_tok_s"] == 2
