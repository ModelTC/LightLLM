import threading

import pytest
import torch

from lightllm.server.router.model_infer.mode_backend import eplb_manager as manager_module
from lightllm.utils import envs_utils


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


def _window_manager(step=20, interval=80, steady=20):
    manager = manager_module.EPLBManager.__new__(manager_module.EPLBManager)
    manager.in_flight = False
    manager.evaluation_in_flight = False
    manager.prefill_steps = 0
    manager.step_interval = step
    manager.sampling_interval = interval
    manager.steady_sample_steps = steady
    manager.min_rebalance_interval = interval
    manager._continuous_collection_start_step = 0
    manager._continuous_collection_end_step = step
    manager._steady_collection_end_step = None
    manager._set_recording = lambda _enabled: None
    manager._reset_recorded_samples = lambda: None
    return manager


def test_initial_window_then_min80_sparse_window_arms_at_60_and_evaluates_at_80(monkeypatch):
    manager = _window_manager()
    starts = []
    monkeypatch.setattr(manager, "_start_evaluation", lambda: starts.append(manager.prefill_steps))
    for _ in range(20):
        manager.step()
    assert starts == [20]  # Initial continuous 20-step window is unchanged.
    manager._continuous_collection_start_step = None
    manager._continuous_collection_end_step = None
    manager.prefill_steps = 40
    for _ in range(20):
        manager.step()
    assert manager.prefill_steps == 60
    assert manager._steady_collection_end_step == 80
    for _ in range(20):
        manager.step()
    assert starts == [20, 80]


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
    manager._continuous_collection_start_step = None
    manager._continuous_collection_end_step = None
    manager._set_recording = lambda _enabled: None
    manager._reset_recorded_samples = lambda: None
    return manager


def test_skip_backoff_honors_min_interval_above_default_cap():
    manager = _no_improvement_manager(80, 640)
    for expected in (320, 640, 640):
        assert not manager._poll_evaluation()
        assert manager.sampling_interval == expected
        manager.evaluation_in_flight = True
        manager._evaluation_thread = type("Done", (), {"join": lambda self: None})()
        manager._evaluation_result = {
            "kind": "no_improvement",
            "model_imbalance_ratio": 1.2,
            "candidate_model_imbalance_ratio": 1.1,
            "candidate_rebalance_gain": 0.01,
            "candidate_changed_layer_count": 1,
        }


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
    manager.prefill_steps = 7
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
    monkeypatch.setattr(torch, "save", lambda *_args, **_kwargs: (_ for _ in ()).throw(OSError("nope")))
    assert manager._dump_evaluation_samples(torch.ones(1), 1, 1) >= 0
    assert manager._diagnostics_dir is None


def test_base_one_keeps_default_four_sample_window():
    manager = manager_module.EPLBManager.__new__(manager_module.EPLBManager)
    manager.step_interval = 1
    manager.sampling_interval = 4
    manager.steady_sample_steps = 4
    assert manager._steady_sample_window_steps() == 4
