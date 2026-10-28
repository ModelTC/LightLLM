import threading
from collections import deque
from types import SimpleNamespace

import pytest
import torch

from lightllm.server.router.model_infer.mode_backend import eplb_manager as manager_module
from lightllm.utils import envs_utils
from lightllm.common.basemodel.layer_weights.meta_weights.fused_moe import eplb_planner, eplb_placement


@pytest.fixture(autouse=True)
def clear_eplb_env_caches(monkeypatch):
    for name in (
        "LIGHTLLM_PREFILL_EPLB_STEP_INTERVAL",
        "LIGHTLLM_PREFILL_EPLB_STEADY_SAMPLE_STEPS",
        "LIGHTLLM_PREFILL_EPLB_MIN_REBALANCE_INTERVAL",
    ):
        monkeypatch.delenv(name, raising=False)
    for getter in (
        envs_utils.get_prefill_eplb_step_interval,
        envs_utils.get_prefill_eplb_steady_sample_steps,
        envs_utils.get_prefill_eplb_min_rebalance_interval,
    ):
        getter.cache_clear()
    yield
    for getter in (
        envs_utils.get_prefill_eplb_step_interval,
        envs_utils.get_prefill_eplb_steady_sample_steps,
        envs_utils.get_prefill_eplb_min_rebalance_interval,
    ):
        getter.cache_clear()


def test_eplb_sampling_getters_defaults_and_valid_values(monkeypatch):
    assert envs_utils.get_prefill_eplb_step_interval() == 20
    assert envs_utils.get_prefill_eplb_steady_sample_steps() == 4
    assert envs_utils.get_prefill_eplb_min_rebalance_interval() == 0
    monkeypatch.setenv("LIGHTLLM_PREFILL_EPLB_STEP_INTERVAL", "20")
    monkeypatch.setenv("LIGHTLLM_PREFILL_EPLB_STEADY_SAMPLE_STEPS", "20")
    monkeypatch.setenv("LIGHTLLM_PREFILL_EPLB_MIN_REBALANCE_INTERVAL", "80")
    for getter in (
        envs_utils.get_prefill_eplb_step_interval,
        envs_utils.get_prefill_eplb_steady_sample_steps,
        envs_utils.get_prefill_eplb_min_rebalance_interval,
    ):
        getter.cache_clear()
    assert envs_utils.get_prefill_eplb_steady_sample_steps() == 20
    assert envs_utils.get_prefill_eplb_min_rebalance_interval() == 80


@pytest.mark.parametrize(
    "name,value,match",
    [
        ("LIGHTLLM_PREFILL_EPLB_STEADY_SAMPLE_STEPS", "0", "greater than 0"),
        ("LIGHTLLM_PREFILL_EPLB_STEADY_SAMPLE_STEPS", "nope", "invalid literal"),
        ("LIGHTLLM_PREFILL_EPLB_MIN_REBALANCE_INTERVAL", "-1", "non-negative"),
        ("LIGHTLLM_PREFILL_EPLB_MIN_REBALANCE_INTERVAL", "19", "at least step interval"),
    ],
)
def test_eplb_sampling_getters_reject_invalid_values(monkeypatch, name, value, match):
    monkeypatch.setenv(name, value)
    envs_utils.get_prefill_eplb_steady_sample_steps.cache_clear()
    envs_utils.get_prefill_eplb_min_rebalance_interval.cache_clear()
    getter = (
        envs_utils.get_prefill_eplb_steady_sample_steps
        if "STEADY" in name
        else envs_utils.get_prefill_eplb_min_rebalance_interval
    )
    with pytest.raises(ValueError, match=match):
        getter()


def test_steady_window_clamps_to_configured_intervals():
    manager = manager_module.EPLBManager.__new__(manager_module.EPLBManager)
    manager.step_interval = 20
    manager.sampling_interval = 80
    manager.steady_sample_steps = 64
    assert manager._steady_sample_window_steps() == 20
    manager.sampling_interval = 3
    assert manager._steady_sample_window_steps() == 3


def _no_improvement_manager(interval, min_interval):
    manager = manager_module.EPLBManager.__new__(manager_module.EPLBManager)
    manager.evaluation_in_flight = True
    manager._evaluation_lock = threading.Lock()
    manager._evaluation_error = None
    manager._evaluation_thread = type("Done", (), {"join": lambda self: None})()
    manager._evaluation_result = {
        "kind": "no_improvement",
        "model_imbalance_ratio": 1.2,
        "candidate_model_imbalance_ratio": 1.1,
        "candidate_rebalance_gain": 0.01,
        "candidate_changed_layer_count": 1,
    }
    manager.global_rank = 1
    manager.step_interval = 20
    manager.sampling_interval = interval
    manager.min_rebalance_interval = min_interval
    manager.steady_sample_steps = 4
    manager.prefill_steps = 0
    manager.weights = []
    manager._eplb_states = []
    manager._last_evaluation_step = 0
    manager._set_recording = lambda _enabled: None
    manager._reset_recorded_samples = lambda: None
    return manager


def test_success_rebalance_uses_min80_interval():
    manager = manager_module.EPLBManager.__new__(manager_module.EPLBManager)
    manager.current_placement = torch.zeros((1, 1, 1), dtype=torch.int64)
    manager.num_logical_experts = manager.world_size = manager.node_world_size = 1
    manager.step_interval = 20
    manager.min_rebalance_interval = 80
    manager.sampling_interval = 320
    manager.global_rank = 1
    manager._reset_recorded_samples = lambda: None
    manager.transfer = type("Transfer", (), {"start": lambda *_args: None})()
    manager._start_rebalance(
        {
            "placement": torch.zeros((1, 1, 1), dtype=torch.int64),
            "metadata": [None],
            "layer_plans": [],
            "prepared_batches": [],
            "before": {"max": 1, "p95": 1},
            "after": {"max": 1, "p95": 1},
            "model_imbalance_ratio": 1,
            "candidate_model_imbalance_ratio": 1,
            "candidate_rebalance_gain": 1,
            "candidate_changed_layer_count": 0,
        }
    )
    assert manager.sampling_interval == 80


def test_eplb_diagnostics_dump_is_rank_zero_cpu_only(tmp_path):
    manager = manager_module.EPLBManager.__new__(manager_module.EPLBManager)
    manager.global_rank = 0
    manager._diagnostics_dir = str(tmp_path)
    manager._diagnostic_dump_sequence = 0
    manager.current_placement = torch.tensor([[1, 2]], dtype=torch.int64)
    manager._evaluation_snapshot = {"placement": manager.current_placement.clone(), "completed_step": 7}
    manager.prefill_steps = 9
    manager._dump_evaluation_samples(torch.ones((1, 1, 1, 1)), 4, 4)
    payload = torch.load(tmp_path / "evaluation-000001.pt", weights_only=True)
    assert set(payload) == {
        "global_load",
        "current_placement",
        "prefill_steps",
        "recorded_sample_count",
        "sample_window_steps",
    }
    assert payload["global_load"].device.type == "cpu"
    assert payload["prefill_steps"] == 7
    manager.global_rank = 1
    manager._dump_evaluation_samples(torch.ones((1, 1, 1, 1)), 4, 4)
    assert len(list(tmp_path.glob("*.pt"))) == 1


def test_eplb_diagnostics_dump_failure_disables_future_writes(monkeypatch, tmp_path):
    manager = manager_module.EPLBManager.__new__(manager_module.EPLBManager)
    manager.global_rank = 0
    manager._diagnostics_dir = str(tmp_path)
    manager._diagnostic_dump_sequence = 0
    manager.current_placement = torch.zeros(1)
    manager.prefill_steps = 1
    manager._evaluation_snapshot = {"placement": manager.current_placement, "completed_step": 1}
    monkeypatch.setattr(torch, "save", lambda *_args, **_kwargs: (_ for _ in ()).throw(OSError("nope")))
    assert manager._dump_evaluation_samples(torch.ones(1), 1, 1) >= 0
    assert manager._diagnostics_dir is None


def test_base_one_keeps_default_four_sample_window():
    manager = manager_module.EPLBManager.__new__(manager_module.EPLBManager)
    manager.step_interval = 1
    manager.sampling_interval = 4
    manager.steady_sample_steps = 4
    assert manager._steady_sample_window_steps() == 4


def manager_for_test(step=4, steady=4):
    manager = manager_module.EPLBManager.__new__(manager_module.EPLBManager)
    manager._use_full_window = True
    manager._initial_plan_complete = False
    manager.in_flight = manager.evaluation_in_flight = False
    manager.step_interval = manager.sampling_interval = step
    manager.steady_sample_steps = steady
    manager.min_rebalance_interval = 0
    manager.prefill_steps = 0
    manager._next_evaluation_step = step
    manager._last_evaluation_step = 0
    manager._sample_step_counts = deque([0], maxlen=max(step, 4) + 1)
    manager._eplb_states = [
        SimpleNamespace(
            route_counter=torch.zeros((2 * max(step, 4), 2, 4), dtype=torch.int64),
            recorded_sample_count=0,
            recording=True,
        )
        for _ in range(2)
    ]
    manager.current_placement = torch.tensor([[[0], [1]], [[0], [1]]])
    manager.starts = []
    manager._start_evaluation = lambda: manager.starts.append(manager.prefill_steps)
    return manager


def forward(manager, dispatches=1):
    manager.before_prefill()
    for _ in range(dispatches):
        for layer, state in enumerate(manager._eplb_states):
            index = state.recorded_sample_count % state.route_counter.shape[0]
            state.route_counter[index].add_(state.recorded_sample_count + 1 + layer * 1000)
            state.recorded_sample_count += 1
    manager.step()


def test_busy_work_counts_real_forwards_and_due_evaluation_uses_first_idle_forward():
    manager = manager_for_test()
    for _ in range(4):
        forward(manager)
    assert manager.starts == [4]
    manager.evaluation_in_flight = True
    for _ in range(3):
        forward(manager)
    manager.evaluation_in_flight = False
    manager.in_flight = True
    for _ in range(2):
        forward(manager)
    assert manager.prefill_steps == 9
    assert manager.starts == [4]
    manager.in_flight = False
    forward(manager)
    assert manager.starts == [4, 10]
    assert manager._next_evaluation_step == 12
    assert all(state.recording and state.recorded_sample_count == 10 for state in manager._eplb_states)


@pytest.mark.parametrize("pattern", [(1,) * 12, (2,) * 12, (1, 2, 1, 2, 2, 1, 2, 1, 1, 2, 1, 1)])
def test_wrapping_live_ring_preserves_exact_recent_forward_window(pattern):
    manager = manager_for_test()
    for dispatches in pattern:
        forward(manager, dispatches)
    manager._freeze_evaluation_samples()
    frozen = manager._evaluation_snapshot
    total = sum(pattern)
    count = sum(pattern[-4:])
    assert frozen["count"] == count
    assert frozen["forward_end_counts"][-1] == total
    assert frozen["completed_step"] == 12
    expected = torch.arange(total - count + 1, total + 1)
    assert torch.equal(frozen["samples"][:, 0, 0, 0], expected)
    assert torch.equal(frozen["samples"][:, 1, 0, 0], expected + 1000)
    # A new ring overwrite and a partial migration cannot change the frozen
    # logical routes or the complete base layout used by the CPU planner.
    saved = frozen["samples"].clone()
    forward(manager, 2)
    manager.current_placement.fill_(7)
    assert torch.equal(frozen["samples"], saved)
    assert not (frozen["placement"] == 7).any()


def test_background_copy_reads_frozen_samples_and_releases_gpu_ownership(monkeypatch):
    manager = manager_for_test()
    for _ in range(4):
        forward(manager)
    manager._freeze_evaluation_samples()
    expected = manager._evaluation_snapshot["samples"].clone()
    manager.evaluation_group = object()
    metadata = []
    monkeypatch.setattr(manager_module.dist, "all_reduce", lambda value, **kwargs: metadata.append(value.clone()))
    for state in manager._eplb_states:
        state.route_counter.fill_(-1)
        state.recorded_sample_count += 2
    copied = manager._collect_local_samples()
    assert torch.equal(copied, expected)
    assert manager._evaluation_snapshot["samples"] is None
    assert metadata[0].tolist() == [4, -4, 8, -8, 4, -4, 4, -4, 4, -4]
    assert metadata[1].tolist() == [0, 0, 1, -1, 2, -2, 3, -3, 4, -4]


def test_sampling_resume_keeps_history_and_honors_changed_min_interval():
    manager = manager_for_test()
    for _ in range(6):
        forward(manager)
    manager.sampling_interval = 16
    manager._prepare_next_sampling_window()
    assert manager._next_evaluation_step == 16
    assert manager._sample_step_counts[-1] == 6
    assert all(state.recorded_sample_count == 6 for state in manager._eplb_states)


def test_forward_count_mismatch_is_rejected_before_freezing():
    manager = manager_for_test()
    manager._eplb_states[0].recorded_sample_count = 1
    with pytest.raises(RuntimeError, match="matching dispatches"):
        manager.step()


def test_incremental_planner_bounds_migration_and_preserves_expert_coverage():
    current = eplb_placement.expand_redundant_placement(
        eplb_placement.build_initial_redundant_expert_ids(16, 4, 1), 16
    ).unsqueeze(0)
    load = torch.full((4, 1, 1, 16), 10, dtype=torch.int64)
    load[..., 0] = 2000
    load[..., 1] = 1500
    target = eplb_planner.plan_eplb_candidate(
        load, current, node_world_size=4, expert_alignment=128, stickiness=0.1, incremental=True
    )
    eplb_placement.validate_physical_placement(target, 16)
    old = eplb_placement._expert_locations(current, 16)
    new = eplb_placement._expert_locations(target, 16)
    assert 0 < int((new & ~old).sum()) <= 8
    before = eplb_placement._estimate_rank_load(load, current, 128, 4).max(-1).values.sum(0)
    after = eplb_placement._estimate_rank_load(load, target, 128, 4).max(-1).values.sum(0)
    assert torch.all(after < before)


@pytest.mark.parametrize("kind", ["insufficient", "no_improvement"])
@pytest.mark.parametrize("minimum_interval", [0, 80, 640])
def test_evaluation_keeps_sampling_and_uses_full_window_only_when_needed(kind, minimum_interval):
    manager = _no_improvement_manager(20, minimum_interval)
    manager._initial_plan_complete = False
    manager._use_full_window = False
    manager._evaluation_result["kind"] = kind
    assert not manager._poll_evaluation()
    assert manager._initial_plan_complete == (kind == "no_improvement")
    assert manager._use_full_window == (kind == "insufficient")
    expected = 20 if kind == "insufficient" else max(20, minimum_interval)
    assert manager.sampling_interval == expected
    assert manager._next_evaluation_step == expected
