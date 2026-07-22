import threading
from types import SimpleNamespace

import pytest
import torch

from lightllm.common.basemodel.layer_weights.meta_weights.fused_moe.ep_balance import PrefillEPBalanceCounters
from lightllm.server.router.model_infer.mode_backend import ep_balance_monitor as monitor_module
from lightllm.server.router.model_infer.mode_backend.ep_balance_monitor import (
    calculate_prefill_balance_stats,
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


def test_monitor_reuses_default_autotune_group_without_creating_a_new_group(monkeypatch):
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

    monkeypatch.setattr(monitor_module, "find_fused_moe_weights", lambda _: [weight])
    monkeypatch.setattr(monitor_module, "get_global_rank", lambda: 0)
    monkeypatch.setattr(monitor_module, "get_global_world_size", lambda: 2)
    monkeypatch.setattr(
        monitor_module.dist_group_manager,
        "get_default_group",
        lambda: SimpleNamespace(autotune_group=sentinel_group),
    )
    monkeypatch.setattr(monitor_module.dist, "new_group", lambda *args, **kwargs: pytest.fail("unexpected new_group"))
    monkeypatch.setattr(
        monitor_module.threading,
        "Thread",
        lambda *args, **kwargs: SimpleNamespace(start=lambda: None),
    )

    monitor = monitor_module.EPBalanceMonitor(model)

    assert monitor.gloo_group is sentinel_group
    assert impl.ep_balance_counters is monitor.counters[0]


def test_record_prefill_round_stores_cumulative_counter_deltas_in_ring_buffer():
    monitor = monitor_module.EPBalanceMonitor.__new__(monitor_module.EPBalanceMonitor)
    monitor.enabled = True
    monitor.counters = [PrefillEPBalanceCounters()]
    monitor._round_buffer = torch.zeros((monitor_module.EP_BALANCE_ROUND_BUFFER_CAPACITY, 1, 2), dtype=torch.int64)
    monitor._round_lock = threading.Lock()
    monitor._round_ready = threading.Event()
    monitor._written_round_count = 0
    monitor._processed_round_count = 0
    monitor._previous_prefill_totals = torch.zeros((1, 2), dtype=torch.int64)

    monitor.counters[0].accumulate(route_load=3, compute_load=128)
    monitor.record_prefill_round()
    monitor.counters[0].accumulate(route_load=2, compute_load=256)
    monitor.record_prefill_round()

    assert torch.equal(
        monitor._copy_local_rounds(0, 2),
        torch.tensor([[[3, 128]], [[2, 256]]], dtype=torch.int64),
    )


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
