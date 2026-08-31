import threading
from array import array
from types import SimpleNamespace

import pytest
import torch
from prometheus_client import generate_latest

from lightllm.common.basemodel.layer_weights.meta_weights.fused_moe.ep_balance import PrefillEPBalanceCounters
from lightllm.distributed import communication_op as communication_op_module
from lightllm.server.metrics.metrics import Monitor
from lightllm.server.router.model_infer.mode_backend import ep_balance_monitor as monitor_module
from lightllm.server.router.model_infer.mode_backend.ep_balance_monitor import (
    EP_BALANCE_PRESSURE_DRIFT_BUCKET_ROUNDS,
    calculate_prefill_placement_pressure_drift,
    calculate_prefill_balance_stats,
    classify_prefill_placement_pressure_drift,
    should_enable_ep_balance_monitor,
)


@pytest.fixture(autouse=True)
def _mock_non_sm100(monkeypatch):
    monkeypatch.setattr(monitor_module, "is_sm100_gpu", lambda: False)


def _stats(source_token_replication: int):
    return calculate_prefill_balance_stats(
        torch.tensor([[[[800, 40], [800, 20]], [[800, 20], [800, 40]]]], dtype=torch.int64),
        layer_routed_experts=torch.tensor([1, 1]),
        layer_flops_per_expert_token=torch.tensor([2.0, 4.0]),
        layer_topks=torch.tensor([2.0, 2.0]),
        source_token_replication=source_token_replication,
    )


def _monitor_args(**overrides):
    args = {
        "enable_ep_moe": True,
        "disable_ep_balance_monitor": False,
        "run_mode": "normal",
        "enable_prefill_cudagraph": False,
    }
    args.update(overrides)
    return SimpleNamespace(**args)


def _pressure_round_stats(bucket_rank_loads):
    """Build [round, layer=1, rank, route/compute] CPU samples for drift tests."""
    return torch.tensor([[[[0, load] for load in rank_loads]] for rank_loads in bucket_rank_loads], dtype=torch.int64)


def test_pressure_drift_is_zero_for_identical_pressure():
    drift, _ = calculate_prefill_placement_pressure_drift(_pressure_round_stats([[2, 1], [2, 1]]), bucket_rounds=1)
    assert drift == 0.0


def test_pressure_drift_is_one_for_complete_hot_rank_migration():
    drift, _ = calculate_prefill_placement_pressure_drift(_pressure_round_stats([[2, 0], [0, 2]]), bucket_rounds=1)
    assert drift == 1.0


def test_pressure_drift_tracks_same_rank_magnitude_change():
    drift, _ = calculate_prefill_placement_pressure_drift(_pressure_round_stats([[2, 1], [3, 1]]), bucket_rounds=1)
    assert drift == pytest.approx(0.2)


def test_pressure_drift_is_invariant_to_uniform_load_scale():
    drift, _ = calculate_prefill_placement_pressure_drift(_pressure_round_stats([[2, 1], [4, 2]]), bucket_rounds=1)
    assert drift == 0.0


def test_pressure_drift_is_zero_for_balanced_inputs():
    drift, _ = calculate_prefill_placement_pressure_drift(_pressure_round_stats([[8, 8], [16, 16]]), bucket_rounds=1)
    assert drift == 0.0


def test_pressure_drift_previous_signature_bridges_report_boundary():
    _, signature = calculate_prefill_placement_pressure_drift(_pressure_round_stats([[2, 0]]), bucket_rounds=1)
    drift, next_signature = calculate_prefill_placement_pressure_drift(
        _pressure_round_stats([[0, 2]]), previous_pressure_signature=signature, bucket_rounds=1
    )
    assert drift == 1.0
    assert torch.equal(next_signature, torch.tensor([[0.0, 1.0]], dtype=torch.float64))


def test_pressure_drift_default_bucket_handles_normal_report_window():
    round_stats = _pressure_round_stats([[4, 2]] * 100)
    drift, signature = calculate_prefill_placement_pressure_drift(round_stats)
    assert EP_BALANCE_PRESSURE_DRIFT_BUCKET_ROUNDS == 20
    assert drift == 0.0
    assert signature.shape == (1, 2)


@pytest.mark.parametrize(
    ("drift", "expected"),
    [
        (0.0, "stable"),
        (0.0999, "stable"),
        (0.10, "dynamic"),
        (0.2999, "dynamic"),
        (0.30, "dynamic"),
        (0.75, "dynamic"),
        (1.0, "dynamic"),
    ],
)
def test_pressure_drift_classification_boundaries(drift, expected):
    assert classify_prefill_placement_pressure_drift(drift) == expected


def test_monitor_log_stats_reports_pressure_drift_and_bridges_reports(monkeypatch):
    monitor = monitor_module.EPBalanceMonitor.__new__(monitor_module.EPBalanceMonitor)
    monitor.layer_routed_experts = torch.tensor([1], dtype=torch.int64)
    monitor.layer_flops_per_expert_token = torch.tensor([2.0], dtype=torch.float64)
    monitor.layer_topks = torch.tensor([2.0], dtype=torch.float64)
    monitor.source_token_replication = 1
    monitor._previous_pressure_signature = None
    metric_calls = []
    monitor.metric_client = SimpleNamespace(gauge_set=lambda name, value: metric_calls.append((name, value)))
    logs = []
    monkeypatch.setattr(monitor_module.logger, "info", logs.append)

    def report_for_hot_rank(hot_rank):
        round_stats = torch.zeros((100, 1, 2, 2), dtype=torch.int64)
        round_stats[:, :, :, monitor_module.ROUTE_LOAD] = 100
        round_stats[:, :, hot_rank, monitor_module.COMPUTE_LOAD] = 2
        monitor._log_stats(round_stats)

    report_for_hot_rank(0)
    first_signature = monitor._previous_pressure_signature.clone()
    report_for_hot_rank(1)

    assert "prefill_ep_placement_pressure_drift=0.0000" in logs[0]
    assert "prefill_ep_placement_pressure_state=stable" in logs[0]
    assert "prefill_ep_placement_pressure_drift=0.2000" in logs[1]
    assert "prefill_ep_placement_pressure_state=dynamic" in logs[1]
    assert torch.equal(first_signature, torch.tensor([[1.0, 0.0]], dtype=torch.float64))
    assert torch.equal(monitor._previous_pressure_signature, torch.tensor([[0.0, 1.0]], dtype=torch.float64))
    assert metric_calls == [
        ("lightllm_prefill_ep_critical_overhead_gflops_per_routed_token", pytest.approx(2e-11)),
        ("lightllm_prefill_ep_compute_critical_overhead_ratio", pytest.approx(1.0)),
        ("lightllm_prefill_ep_placement_pressure_drift", 0.0),
        ("lightllm_prefill_ep_critical_overhead_gflops_per_routed_token", pytest.approx(2e-11)),
        ("lightllm_prefill_ep_compute_critical_overhead_ratio", pytest.approx(1.0)),
        ("lightllm_prefill_ep_placement_pressure_drift", pytest.approx(0.2)),
    ]


def test_critical_overhead_preserves_per_layer_slowest_rank():
    stats = _stats(source_token_replication=1)
    assert stats is not None
    assert stats["critical_overhead_gflops_per_routed_token"] == pytest.approx(7.5e-11)
    assert stats["prefill_ep_compute_critical_overhead_ratio"] == pytest.approx(1 / 3)


def test_non_tpsp_tp_replication_scales_gflops_per_token_but_not_ratio():
    tpsp_stats = _stats(source_token_replication=1)
    non_tpsp_tp8_stats = _stats(source_token_replication=8)
    assert tpsp_stats is not None and non_tpsp_tp8_stats is not None
    assert non_tpsp_tp8_stats["critical_overhead_gflops_per_routed_token"] == pytest.approx(
        tpsp_stats["critical_overhead_gflops_per_routed_token"] * 8
    )
    assert non_tpsp_tp8_stats["prefill_ep_compute_critical_overhead_ratio"] == pytest.approx(
        tpsp_stats["prefill_ep_compute_critical_overhead_ratio"]
    )


def test_critical_overhead_is_zero_when_ranks_are_balanced():
    stats = calculate_prefill_balance_stats(
        torch.tensor([[[[100, 32], [100, 32]], [[100, 64], [100, 64]]]], dtype=torch.int64),
        layer_routed_experts=torch.tensor([1, 1]),
        layer_flops_per_expert_token=torch.tensor([2.0, 4.0]),
        layer_topks=torch.tensor([2.0, 2.0]),
        source_token_replication=1,
    )
    assert stats is not None
    assert stats["critical_overhead_gflops_per_routed_token"] == 0.0
    assert stats["prefill_ep_compute_critical_overhead_ratio"] == 0.0


@pytest.mark.parametrize(
    "round_stats",
    [
        torch.tensor([[[[1, 1], [1, 1]]]], dtype=torch.int64),
        torch.tensor([[[[100, 0], [100, 0]]]], dtype=torch.int64),
    ],
)
def test_critical_overhead_rejects_insufficient_or_zero_compute_samples(round_stats):
    assert (
        calculate_prefill_balance_stats(
            round_stats,
            layer_routed_experts=torch.tensor([1]),
            layer_flops_per_expert_token=torch.tensor([2.0]),
            layer_topks=torch.tensor([2.0]),
            source_token_replication=1,
        )
        is None
    )


def test_cpu_counter_accumulates_multiple_prefill_dispatches():
    counters = PrefillEPBalanceCounters()
    counters.accumulate(route_load=3, compute_load=128)
    counters.accumulate(route_load=4, compute_load=256)
    assert (counters.route_load, counters.compute_load) == (7, 384)


def test_monitor_reuses_manager_precreated_dedicated_gloo_group(monkeypatch):
    sentinel_group = object()
    impl = SimpleNamespace(ep_balance_counters=None)
    weight = SimpleNamespace(
        fuse_moe_impl=impl,
        n_routed_experts=8,
        hidden_size=16,
        moe_intermediate_size=32,
        num_experts_per_tok=2,
    )
    model = SimpleNamespace(args=SimpleNamespace(enable_tpsp_mix_mode=True), tp_world_size_=1)

    monkeypatch.setattr(monitor_module, "_find_fused_moe_weights", lambda _: [weight])
    monkeypatch.setattr(monitor_module, "get_global_rank", lambda: 0)
    monkeypatch.setattr(monitor_module, "get_global_world_size", lambda: 2)
    metric_ports = []
    monkeypatch.setattr(monitor_module, "get_shm_port_args", lambda: SimpleNamespace(metric_port=4321))
    monkeypatch.setattr(monitor_module, "MetricClient", lambda port: metric_ports.append(port) or object())
    monkeypatch.setattr(monitor_module, "dist_group_manager", SimpleNamespace(ep_balance_monitor_group=sentinel_group))
    monkeypatch.setattr(monitor_module.dist, "new_group", lambda *args, **kwargs: pytest.fail("unexpected new_group"))
    monkeypatch.setattr(
        monitor_module.threading,
        "Thread",
        lambda *args, **kwargs: SimpleNamespace(start=lambda: None),
    )

    monitor = monitor_module.EPBalanceMonitor(model)

    assert monitor.gloo_group is sentinel_group
    assert impl.ep_balance_counters is monitor.counters[0]
    assert metric_ports == [4321]


def test_nonzero_rank_monitor_does_not_create_metric_client(monkeypatch):
    sentinel_group = object()
    impl = SimpleNamespace(ep_balance_counters=None)
    weight = SimpleNamespace(
        fuse_moe_impl=impl,
        n_routed_experts=8,
        hidden_size=16,
        moe_intermediate_size=32,
        num_experts_per_tok=2,
    )
    model = SimpleNamespace(args=SimpleNamespace(enable_tpsp_mix_mode=True), tp_world_size_=1)
    metric_client_calls = []
    monkeypatch.setattr(monitor_module, "_find_fused_moe_weights", lambda _: [weight])
    monkeypatch.setattr(monitor_module, "get_global_rank", lambda: 1)
    monkeypatch.setattr(monitor_module, "get_global_world_size", lambda: 2)
    monkeypatch.setattr(monitor_module, "dist_group_manager", SimpleNamespace(ep_balance_monitor_group=sentinel_group))
    monkeypatch.setattr(monitor_module.dist, "new_group", lambda *args, **kwargs: pytest.fail("unexpected new_group"))
    monkeypatch.setattr(monitor_module, "get_shm_port_args", lambda: pytest.fail("unexpected port lookup"))
    monkeypatch.setattr(
        monitor_module,
        "MetricClient",
        lambda port: metric_client_calls.append(port) or pytest.fail("unexpected metric client"),
    )
    monkeypatch.setattr(
        monitor_module.threading,
        "Thread",
        lambda *args, **kwargs: SimpleNamespace(start=lambda: None),
    )

    monitor = monitor_module.EPBalanceMonitor(model)

    assert monitor.metric_client is None
    assert metric_client_calls == []


@pytest.mark.parametrize("disable_monitor", [False, True])
def test_group_manager_creates_monitor_gloo_group_only_when_enabled(monkeypatch, disable_monitor):
    monitor_group = object()
    custom_groups = []

    class FakeCustomProcessGroup:
        def init_symm_mem_reduce(self):
            pass

        def init_flashinfer_reduce(self):
            pass

    args = SimpleNamespace(
        enable_ep_moe=True,
        disable_ep_balance_monitor=disable_monitor,
        run_mode="normal",
        enable_prefill_cudagraph=False,
        disable_symm_mem_allreduce=True,
        disable_flashinfer_allreduce=True,
    )
    monkeypatch.setattr(communication_op_module, "get_env_start_args", lambda: args)
    monkeypatch.setattr(
        communication_op_module,
        "CustomProcessGroup",
        lambda: custom_groups.append(FakeCustomProcessGroup()) or custom_groups[-1],
    )
    monkeypatch.setattr(communication_op_module, "get_global_world_size", lambda: 2)
    monkeypatch.setattr(communication_op_module, "is_sm100_gpu", lambda: False)
    calls = []
    monkeypatch.setattr(
        communication_op_module.dist,
        "new_group",
        lambda *args, **kwargs: calls.append((args, kwargs)) or monitor_group,
    )

    manager = communication_op_module.DistributeGroupManager()
    manager.create_groups(group_size=2)

    assert len(manager.groups) == 2
    if disable_monitor:
        assert calls == []
        assert manager.ep_balance_monitor_group is None
    else:
        assert calls == [((), {"ranks": [0, 1], "backend": "gloo"})]
        assert manager.ep_balance_monitor_group is monitor_group


def test_monitor_registers_prefill_ep_gauges_with_model_label():
    monitor = Monitor(
        SimpleNamespace(
            metric_gateway=None,
            job_name="test",
            grouping_key=[],
            enable_monitor_auth=False,
            model_name="monitor-test-model",
            max_req_total_len=128,
            mtp_step=0,
        )
    )
    values = {
        "lightllm_prefill_ep_critical_overhead_gflops_per_routed_token": 1.25,
        "lightllm_prefill_ep_compute_critical_overhead_ratio": 0.3,
        "lightllm_prefill_ep_placement_pressure_drift": 0.125,
    }
    assert set(values).issubset(monitor.monitor_registry)
    for name, value in values.items():
        monitor.gauge_set(name, value)

    exposition = generate_latest(monitor.registry).decode()
    for name, value in values.items():
        assert f'{name}{{model_name="monitor-test-model"}} {value}' in exposition


def test_record_prefill_round_stores_cumulative_counter_deltas_in_ring_buffer():
    monitor = monitor_module.EPBalanceMonitor.__new__(monitor_module.EPBalanceMonitor)
    monitor.enabled = True
    monitor.counters = [PrefillEPBalanceCounters()]
    monitor._round_buffer_storage = array("q", [0]) * (monitor_module.EP_BALANCE_ROUND_BUFFER_CAPACITY * 2)
    monitor._round_buffer = torch.frombuffer(monitor._round_buffer_storage, dtype=torch.int64).view(
        monitor_module.EP_BALANCE_ROUND_BUFFER_CAPACITY, 1, 2
    )
    monitor._round_ready = threading.Event()
    monitor._written_round_count = 0
    monitor._processed_round_count = 0
    monitor._overflowed = False

    monitor.counters[0].accumulate(route_load=3, compute_load=128)
    monitor.record_prefill_round()
    assert (monitor.counters[0].route_load, monitor.counters[0].compute_load) == (0, 0)
    monitor.counters[0].accumulate(route_load=2, compute_load=256)
    monitor.record_prefill_round()

    assert torch.equal(
        monitor._copy_local_rounds(0, 2),
        torch.tensor([[[3, 128]], [[2, 256]]], dtype=torch.int64),
    )
    assert not hasattr(monitor, "_round_lock")


def test_spsc_ring_copy_wraps_without_a_lock():
    monitor = monitor_module.EPBalanceMonitor.__new__(monitor_module.EPBalanceMonitor)
    monitor.enabled = True
    monitor.counters = [PrefillEPBalanceCounters()]
    monitor._round_buffer_storage = array("q", [0]) * (monitor_module.EP_BALANCE_ROUND_BUFFER_CAPACITY * 2)
    monitor._round_buffer = torch.frombuffer(monitor._round_buffer_storage, dtype=torch.int64).view(
        monitor_module.EP_BALANCE_ROUND_BUFFER_CAPACITY, 1, 2
    )
    monitor._round_ready = threading.Event()
    monitor._written_round_count = monitor_module.EP_BALANCE_ROUND_BUFFER_CAPACITY - 2
    monitor._processed_round_count = monitor_module.EP_BALANCE_ROUND_BUFFER_CAPACITY - 2
    monitor._overflowed = False

    for value in (11, 12, 13):
        monitor.counters[0].accumulate(route_load=value, compute_load=value * 10)
        monitor.record_prefill_round()

    assert torch.equal(
        monitor._copy_local_rounds(monitor_module.EP_BALANCE_ROUND_BUFFER_CAPACITY - 2, monitor._written_round_count),
        torch.tensor([[[11, 110]], [[12, 120]], [[13, 130]]], dtype=torch.int64),
    )
    assert not hasattr(monitor, "_round_lock")


def test_spsc_ring_overflow_is_deferred_to_the_monitor_thread():
    monitor = monitor_module.EPBalanceMonitor.__new__(monitor_module.EPBalanceMonitor)
    monitor.enabled = True
    monitor.counters = [PrefillEPBalanceCounters(route_load=7, compute_load=70)]
    monitor._round_buffer_storage = array("q", [0]) * (monitor_module.EP_BALANCE_ROUND_BUFFER_CAPACITY * 2)
    monitor._round_buffer = torch.frombuffer(monitor._round_buffer_storage, dtype=torch.int64).view(
        monitor_module.EP_BALANCE_ROUND_BUFFER_CAPACITY, 1, 2
    )
    monitor._round_ready = threading.Event()
    monitor._written_round_count = monitor_module.EP_BALANCE_ROUND_BUFFER_CAPACITY
    monitor._processed_round_count = 0
    monitor._overflowed = False

    monitor.record_prefill_round()

    assert monitor._overflowed
    assert monitor._round_ready.is_set()
    assert monitor._written_round_count == monitor_module.EP_BALANCE_ROUND_BUFFER_CAPACITY
    assert (monitor.counters[0].route_load, monitor.counters[0].compute_load) == (7, 70)


def test_raise_buffer_overflow_always_reports_phase_and_ring_counts():
    monitor = monitor_module.EPBalanceMonitor.__new__(monitor_module.EPBalanceMonitor)
    monitor._written_round_count = 23
    monitor._processed_round_count = 7

    with pytest.raises(RuntimeError) as exc_info:
        monitor._raise_buffer_overflow("before_sync")

    assert str(exc_info.value) == (
        "EP balance prefill-round buffer overflowed "
        f"phase=before_sync written=23 processed=7 capacity={monitor_module.EP_BALANCE_ROUND_BUFFER_CAPACITY}"
    )


def test_raise_buffer_overflow_optionally_reports_common_round_end():
    monitor = monitor_module.EPBalanceMonitor.__new__(monitor_module.EPBalanceMonitor)
    monitor._written_round_count = 23
    monitor._processed_round_count = 7

    with pytest.raises(RuntimeError) as exc_info:
        monitor._raise_buffer_overflow("common_round_lag", common_round_end=19)

    assert str(exc_info.value) == (
        "EP balance prefill-round buffer overflowed "
        f"phase=common_round_lag written=23 processed=7 capacity={monitor_module.EP_BALANCE_ROUND_BUFFER_CAPACITY} "
        "common_round_end=19"
    )


def test_gather_round_stats_only_allocates_receive_buffers_on_rank_zero(monkeypatch):
    local_round_stats = torch.tensor([[[3, 128]]], dtype=torch.int64)
    sentinel_group = object()

    rank_zero_monitor = monitor_module.EPBalanceMonitor.__new__(monitor_module.EPBalanceMonitor)
    rank_zero_monitor.global_rank = 0
    rank_zero_monitor.world_size = 2
    rank_zero_monitor.gloo_group = sentinel_group

    def root_gather(input_tensor, gather_list, dst, group):
        assert dst == 0 and group is sentinel_group
        assert len(gather_list) == 2
        gather_list[0].copy_(input_tensor)
        gather_list[1].copy_(input_tensor + 1)

    monkeypatch.setattr(monitor_module.dist, "gather", root_gather)
    result = rank_zero_monitor._gather_round_stats(local_round_stats)
    assert torch.equal(result, torch.tensor([[[[3, 128], [4, 129]]]], dtype=torch.int64))

    nonzero_monitor = monitor_module.EPBalanceMonitor.__new__(monitor_module.EPBalanceMonitor)
    nonzero_monitor.global_rank = 1
    nonzero_monitor.world_size = 2
    nonzero_monitor.gloo_group = sentinel_group

    def nonroot_gather(input_tensor, gather_list, dst, group):
        assert input_tensor is local_round_stats
        assert gather_list is None
        assert dst == 0 and group is sentinel_group

    monkeypatch.setattr(monitor_module.dist, "gather", nonroot_gather)
    assert nonzero_monitor._gather_round_stats(local_round_stats) is None


def test_find_fused_moe_weights_discovers_any_layer_member_once_and_sorts(monkeypatch):
    class FakeFusedMoeWeight:
        def __init__(self, layer_num, enabled=True):
            self.layer_num_ = layer_num
            self.enable_ep_moe = enabled

    monkeypatch.setattr(monitor_module, "FusedMoeWeight", FakeFusedMoeWeight)
    first = FakeFusedMoeWeight(3)
    second = FakeFusedMoeWeight(1)
    disabled = FakeFusedMoeWeight(0, enabled=False)
    model = SimpleNamespace(
        trans_layers_weight=[
            SimpleNamespace(experts_=first, alias=first, ignored=disabled),
            SimpleNamespace(any_direct_member=second),
        ]
    )

    assert monitor_module._find_fused_moe_weights(model) == [second, first]


def test_monitor_disable_detaches_counters_from_all_impls():
    impl = SimpleNamespace(ep_balance_counters="unset")
    monitor = monitor_module.EPBalanceMonitor.__new__(monitor_module.EPBalanceMonitor)
    monitor.weights = [SimpleNamespace(fuse_moe_impl=impl)]
    monitor.enabled = True
    monitor._disable()
    assert impl.ep_balance_counters is None
    assert not monitor.enabled


def test_critical_overhead_requires_minimum_samples_for_every_layer():
    round_stats = torch.tensor([[[[200, 32], [200, 32]], [[1, 32], [1, 32]]]], dtype=torch.int64)
    assert (
        calculate_prefill_balance_stats(
            round_stats,
            layer_routed_experts=torch.tensor([1, 1]),
            layer_flops_per_expert_token=torch.tensor([2.0, 4.0]),
            layer_topks=torch.tensor([2.0, 2.0]),
            source_token_replication=1,
        )
        is None
    )


def test_ep_moe_normal_and_prefill_enable_monitor_by_default():
    assert should_enable_ep_balance_monitor(_monitor_args())
    assert should_enable_ep_balance_monitor(_monitor_args(run_mode="prefill"))


def test_disable_ep_balance_monitor_turns_monitor_off():
    assert not should_enable_ep_balance_monitor(_monitor_args(disable_ep_balance_monitor=True))


def test_non_ep_moe_and_decode_mode_do_not_enable_monitor():
    assert not should_enable_ep_balance_monitor(_monitor_args(enable_ep_moe=False))
    assert not should_enable_ep_balance_monitor(_monitor_args(run_mode="decode"))


def test_prefill_cudagraph_silently_disables_monitor():
    assert not should_enable_ep_balance_monitor(_monitor_args(run_mode="prefill", enable_prefill_cudagraph=True))


def test_sm100_silently_disables_monitor(monkeypatch):
    monkeypatch.setattr(monitor_module, "is_sm100_gpu", lambda: True)
    assert not should_enable_ep_balance_monitor(_monitor_args())
