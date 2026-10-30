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
    throughput.finish(
        SimpleNamespace(
            is_stop_matched=False, is_aborted=False, finish_status=FinishStatus(FinishStatus.FINISHED_LENGTH)
        )
    )
    throughput.finish(
        SimpleNamespace(
            is_stop_matched=False, is_aborted=True, finish_status=FinishStatus(FinishStatus.FINISHED_ABORTED)
        )
    )
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
        cur_kv_len=cached,
        prefill_tokens=0,
        cur_output_len=output_len,
        shm_req=shm,
        finish_status=status,
        output_constraint=None,
        req_idx=0,
        multimodal_params=None,
    )
    req.get_cur_total_len = lambda: input_len + req.cur_output_len
    req.get_chuncked_input_token_len = lambda: min(req.cur_kv_len + 512, req.get_cur_total_len())
    req.get_input_token_ids = lambda: np.zeros(req.get_cur_total_len(), dtype=np.int64)
    req.get_chuncked_input_token_ids = lambda: req.get_input_token_ids()[: req.get_chuncked_input_token_len()]
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


def prefill(result, reqs, chunked=True):
    from lightllm.server.router.model_infer.mode_backend.generic_pre_process import prepare_prefill_inputs

    prepare_prefill_inputs(reqs, is_chuncked_mode=chunked)
    return result._pre_post_handle(reqs, is_chuncked_mode=chunked)


def test_chunked_prefill_counts_computed_input_only_after_postprocessing(monkeypatch, counters):
    result = backend(monkeypatch, counters)
    req = request(1000, cached=200)
    first = prefill(result, [req])
    assert first[0].prefill_tokens == 512
    assert counters.totals() == (0, 0)  # No work reported before forward synchronization.
    post(result, [req], first)
    assert counters.totals() == (512, 0)
    second = prefill(result, [req])
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
        is_stop_matched=False,
        candetoken_out_len=10,
        finish_status=FinishStatus(FinishStatus.FINISHED_STOP),
    )
    batch = Batch(1, [req], 1)
    shm_manager, statics = Mock(), Mock()

    def release(released):
        assert throughput.completed == 1
        released.is_aborted = True
        released.finish_status = FinishStatus()

    shm_manager.put_back_req_obj.side_effect = release
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


@pytest.mark.parametrize("node_rank,dp_size", [(0, 1), (1, 1), (1, 2)])
@pytest.mark.parametrize("readiness", ["always", "never", "drop"])
def test_router_first_burst_excludes_initial_idle(monkeypatch, counters, node_rank, dp_size, readiness):
    import asyncio
    from lightllm.server.router import manager

    now = [0.0]
    monkeypatch.setattr(manager.time, "monotonic", lambda: now[0])
    monkeypatch.setattr(log_utils, "get_log_stats_interval", lambda: 10)
    logger = Mock()
    monkeypatch.setattr(manager, "init_logger", lambda name: logger)
    router = manager.RouterManager.__new__(manager.RouterManager)
    router.engine_counters = counters
    router.service_ready = SimpleNamespace(
        get_value=lambda: int(readiness != "never" and (readiness != "drop" or now[0] < 6))
    )
    router.req_queue = Mock()
    router.req_queue.get_wait_req_num.return_value = 0
    router.running_batch = None
    router.schedule_new_batch = None
    router.dp_size, router.node_rank, router.dp_size_in_node = dp_size, node_rank, 2
    router.radix_cache_client = None
    router.max_total_token_num = 100
    router.get_used_tokens = lambda index: (25, 50)[index]
    router._get_paused_req_num = lambda: 0
    router._get_schedule_time_interval = lambda: 0
    writer = engine_stats.EngineCounters(2, 0)
    ticks = iter([1, 5, 6, 16])

    class StopLoop(Exception):
        pass

    async def step():
        try:
            now[0] = next(ticks)
        except StopIteration:
            raise StopLoop()
        if now[0] == 6:
            writer.add(100, 10)

    router._step = step
    with pytest.raises(StopLoop):
        asyncio.run(router.loop_for_fwd())
    if readiness == "never" or (dp_size == 1 and node_rank != 0):
        logger.info.assert_not_called()
    else:
        logger.info.assert_called_once()
        fields = logger.info.call_args.kwargs["extra"]
        assert fields["event"] == "engine.stats"
        assert fields["window_s"] == 11
        assert fields["output_tokens"] == 10
        assert fields["output_tok_s"] == 0.91
        assert fields["kv_active_pct_by_local_dp"] == [25.0, 50.0]
        assert fields["kv_cached_pct_by_local_dp"] == [0.0, 0.0]


@pytest.mark.parametrize("leader", [True, False])
@pytest.mark.parametrize("cached", [0, 40, 100])
def test_reward_prefill_counts_uncached_tokens_only_on_leader(monkeypatch, counters, leader, cached):
    from lightllm.server.router.model_infer.mode_backend.chunked_prefill import impl_for_reward_model as reward

    worker = reward.RewardModelBackend.__new__(reward.RewardModelBackend)
    worker.disable_chunked_prefill = True
    worker.is_master_in_dp = leader
    worker.eos_id = []
    worker.engine_counters = engine_stats.EngineCounters(2, 1) if leader else None
    worker.model = Mock()
    worker.model.forward.return_value = SimpleNamespace(logits=torch.tensor([0.75]))
    req = request(100, cached=cached)
    monkeypatch.setattr(reward, "prepare_prefill_inputs", lambda *a, **kw: ("input", [req]))
    events = Mock()
    worker.reward_prefill(events, [req])
    assert counters.storage.arr.tolist() == [[0, 0], [100 - cached, 1] if leader else [0, 0]]
    assert req.cur_output_len == 1
    if leader:
        assert req.shm_req.reward_score == 0.75
        assert req.shm_req.shm_cur_output_len == 1
    events.notify_post_handle_and_wait_pre_post_handle.assert_called_once_with()
    events.notify_forward_and_wait_post_handle.assert_called_once_with()
    events.notify_pre_post_handle.assert_called_once_with()


@pytest.mark.parametrize("tp_rank", [0, 1])
def test_init_model_attaches_only_the_dp_leaders_counter_row(monkeypatch, counters, tp_rank):
    from lightllm.server.core.objs.start_args_type import StartArgs

    distributed = Mock()
    monkeypatch.setattr(base_backend, "init_distributed_env", distributed)
    ranks = {
        "get_node_world_size": 4,
        "get_current_rank_in_node": 2 + tp_rank,
        "get_current_device_id": 2 + tp_rank,
        "get_current_rank_in_dp": tp_rank,
        "get_global_dp_rank": 1,
        "get_dp_rank_in_node": 1,
        "get_dp_world_size": 2,
        "get_global_rank": 2 + tp_rank,
        "get_global_world_size": 4,
        "get_dp_size": 2,
    }
    for name, value in ranks.items():
        monkeypatch.setattr(base_backend, name, lambda value=value: value)
    monkeypatch.setattr(log_utils, "set_log_context", Mock())

    class ModelSetupBoundary(Exception):
        pass

    monkeypatch.setattr(base_backend.dist_group_manager, "create_groups", Mock(side_effect=ModelSetupBoundary()))
    worker = base_backend.ModeBackend.__new__(base_backend.ModeBackend)
    with pytest.raises(ModelSetupBoundary):
        worker.init_model(
            {
                "args": StartArgs(dp=2),
                "world_size": 4,
                "load_way": "HF",
                "weight_dir": "unused",
                "max_total_token_num": 100,
            }
        )
    distributed.assert_called_once()
    assert worker.dp_rank_in_node == 1
    if tp_rank == 0:
        worker.engine_counters.add(7, 3)
        assert counters.storage.arr.tolist() == [[0, 0], [7, 3]]
    else:
        assert worker.engine_counters is None
        assert counters.totals() == (0, 0)


@pytest.mark.parametrize("node_rank,dp_size", [(0, 1), (1, 1), (0, 2), (1, 2)])
@pytest.mark.parametrize("queued,staged", [(2, 0), (0, 5)])
def test_router_statistics_wait_for_readiness_and_emit_once_per_node(
    monkeypatch, counters, node_rank, dp_size, queued, staged
):
    import asyncio
    from lightllm.server.router import manager as router

    now = [0.0]
    clock = SimpleNamespace(monotonic=lambda: now[0])
    for module in (router, engine_stats, log_utils):
        monkeypatch.setattr(module, "time", clock)
    monkeypatch.setattr(log_utils, "get_log_stats_interval", lambda: 10)
    logger = Mock()
    monkeypatch.setattr(router, "init_logger", lambda name: logger)
    manager = router.RouterManager.__new__(router.RouterManager)
    manager.engine_counters = counters
    manager.running_batch = None
    manager.schedule_new_batch = SimpleNamespace(reqs=[Mock() for _ in range(staged)]) if staged else None
    manager.req_queue = Mock()
    manager.req_queue.get_wait_req_num.return_value = queued
    manager.node_rank, manager.dp_size, manager.dp_size_in_node = node_rank, dp_size, dp_size
    manager.max_total_token_num = 100
    manager.get_used_tokens = lambda index: 10 * (index + 1)
    manager.radix_cache_client = SimpleNamespace(get_unrefed_tokens_num=lambda index: 5 * (index + 1))
    manager._get_paused_req_num = lambda: 1
    manager._get_schedule_time_interval = lambda: 0
    manager.service_ready = SimpleNamespace(get_value=lambda: int(now[0] >= 20))
    writer = engine_stats.EngineCounters(2, 0)

    class StopLoop(Exception):
        pass

    async def step():
        now[0] += 10
        if now[0] > 30:
            raise StopLoop()
        if now[0] <= 20:
            logger.info.assert_not_called()
        writer.add(100, 10)

    manager._step = step
    with pytest.raises(StopLoop):
        asyncio.run(manager.loop_for_fwd())
    if dp_size == 1 and node_rank != 0:
        logger.info.assert_not_called()
        return
    assert logger.info.call_count == 2
    first, second = [call.kwargs["extra"] for call in logger.info.call_args_list]
    assert first["prefill_tokens"] == 200 and second["prefill_tokens"] == 100
    assert first["window_s"] == 20 and second["window_s"] == 10
    assert first["output_tok_s"] == second["output_tok_s"] == 1
    assert first["running"] == 0 and first["waiting"] == queued + staged and first["paused"] == 1
    assert first["kv_active_pct_by_local_dp"] == [10.0 * (i + 1) for i in range(dp_size)]
    assert first["kv_cached_pct_by_local_dp"] == [5.0 * (i + 1) for i in range(dp_size)]
    assert first["kv_capacity_tokens_per_dp"] == 100


def test_real_shared_counter_owner_and_worker_rows(monkeypatch):
    import uuid
    from lightllm.server.core.objs.shm_array import ShmArray
    from lightllm.utils import shm_utils

    name = "engine_counter_test_" + uuid.uuid4().hex
    monkeypatch.setattr(shm_utils, "get_unique_server_name", lambda: name)
    owner = engine_stats.EngineCounters(2, owner=True)
    workers = []
    try:
        for rank in range(2):
            worker = engine_stats.EngineCounters(2, rank)
            workers.append(worker)
            assert isinstance(worker.storage, ShmArray)
            assert worker.storage.shape == (2, 2)
            worker.add(10 + rank, 2 + rank)
        assert owner.storage.arr.tolist() == [[10, 2], [11, 3]]
        assert owner.totals() == (21, 5)
    finally:
        for worker in workers:
            worker.storage.detach_shm()
        owner.storage.close_shm()


@pytest.mark.parametrize("cached", [0, 128, 2175])
@pytest.mark.parametrize("chunked", [True, False])
@pytest.mark.parametrize("diverse", [False, True])
def test_recompute_generated_context(monkeypatch, counters, cached, chunked, diverse):
    result = backend(monkeypatch, counters)
    if diverse:
        from lightllm.server.router.model_infer.mode_backend.diverse_backend.impl import DiversehBackend

        monkeypatch.setattr(torch.cuda, "current_stream", lambda: Mock())
        result._pre_post_handle = DiversehBackend._diverse_pre_post_handle.__get__(result)
    req = request(128, cached=cached, output_len=2048, max_output=4096)
    req.is_master_req = lambda: True
    req.slave_reqs = []
    while req.cur_kv_len < 2176:
        packs = prefill(result, [req], chunked)
        if req.cur_kv_len < 2176:
            assert packs[0].prefill_tokens == 512
            assert packs[0].output_len == 0
        post(result, [req], packs)
    assert counters.totals() == (2176 - cached, 1)
    post(result, [req], result._pre_post_handle([req], is_chuncked_mode=False))
    assert counters.totals() == (2176 - cached, 2)


def test_mixed_prefill_excludes_decode_tokens(monkeypatch, counters):
    result = backend(monkeypatch, counters)
    prefill_req = request(128, cached=128, output_len=2048, max_output=4096)
    decode_req = request(128, cached=132, output_len=5)
    decode_req.is_decode_req_mixed_in_prefill = True
    reqs = [prefill_req, decode_req]
    packs = prefill(result, reqs)
    assert [pack.prefill_tokens for pack in packs] == [512, 0]
    post(result, reqs, packs)
    assert counters.totals() == (512, 1)
