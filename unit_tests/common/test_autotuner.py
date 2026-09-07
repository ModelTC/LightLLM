import json

import pytest
import torch
from frozendict import frozendict

from lightllm.common.triton_utils import autotuner as autotuner_module
from lightllm.common.triton_utils.autotuner import AutotuneKernelType, AutotuneLevel, Autotuner, autotune


@pytest.fixture(autouse=True)
def autotune_environment(monkeypatch):
    monkeypatch.setattr(Autotuner, "_autotune_warmup_kernel_type", None)
    monkeypatch.setattr(autotuner_module, "get_triton_autotune_level", lambda: AutotuneLevel.ADAPTIVE_AUTOTUNE)
    monkeypatch.setattr(autotuner_module.dist, "is_initialized", lambda: False)
    monkeypatch.setattr(autotuner_module.KernelConfigs, "get_config_file_name", lambda params: "configs.json")


def make_kernel(tmp_path, monkeypatch, name, kernel_type=None):
    calls = []
    benchmarks = []
    options = {} if kernel_type is None else {"kernel_type": kernel_type}

    @autotune(
        kernel_name=name,
        configs_gen_func=lambda: [{"block": 1}, {"block": 2}],
        static_key_func=lambda: {"dtype": "test"},
        run_key_func=lambda size: size,
        **options,
    )
    def kernel(size, run_config=None):
        calls.append((size, run_config))
        return run_config

    def bench(size, run_config):
        benchmarks.append((size, run_config))
        return 1.0 / run_config["block"]

    cache_dir = tmp_path / name
    cache_dir.mkdir()
    kernel._cache_dir = str(cache_dir)
    monkeypatch.setattr(kernel, "_bench", bench)
    return kernel, calls, benchmarks, cache_dir / "configs.json"


@pytest.mark.parametrize("level", [AutotuneLevel.ADAPTIVE_AUTOTUNE, AutotuneLevel.FORCE_AUTOTUNE])
def test_two_warmup_phases_only_persist_matching_kernel_configs(tmp_path, monkeypatch, level):
    monkeypatch.setattr(autotuner_module, "get_triton_autotune_level", lambda: level)
    general, _, general_benchmarks, general_cache = make_kernel(tmp_path, monkeypatch, "general")
    decode, decode_calls, decode_benchmarks, decode_cache = make_kernel(
        tmp_path, monkeypatch, "decode", AutotuneKernelType.DECODE_ATTENTION
    )

    with Autotuner.autotune_warmup():
        assert general(8) == {"block": 2}
        assert decode(8) is None
    assert len(general_benchmarks) == 2
    assert decode_benchmarks == []
    assert decode_calls == [(8, None)]
    assert not decode_cache.exists()
    general_cache_before = general_cache.read_bytes()

    with Autotuner.autotune_warmup(AutotuneKernelType.DECODE_ATTENTION):
        assert general(16) == {"block": 2}
        assert decode(16) == {"block": 2}
    assert len(general_benchmarks) == 2
    assert len(decode_benchmarks) == 2
    assert general_cache.read_bytes() == general_cache_before
    assert json.loads(general_cache.read_text()) == {"8": {"block": 2}}
    assert json.loads(decode_cache.read_text()) == {"16": {"block": 2}}


def test_other_phase_warms_history_without_tuning_and_later_uses_new_config(tmp_path, monkeypatch):
    decode, calls, benchmarks, cache_file = make_kernel(
        tmp_path, monkeypatch, "decode", AutotuneKernelType.DECODE_ATTENTION
    )
    cache_file.write_text(json.dumps({"8": {"block": 1}, "32": {"block": 2}}))
    cache_before = cache_file.read_bytes()

    with Autotuner.autotune_warmup():
        assert decode(16) == {"block": 1}
    # Historical configs are warmed in either phase, but only the matching phase searches new configs.
    assert calls == [(16, {"block": 1}), (16, {"block": 2}), (16, {"block": 1})]
    assert benchmarks == []
    assert cache_file.read_bytes() == cache_before

    with Autotuner.autotune_warmup(AutotuneKernelType.DECODE_ATTENTION):
        assert decode(16) == {"block": 2}
    assert len(benchmarks) == 2
    assert decode(16) == {"block": 2}
    assert json.loads(cache_file.read_text())["16"] == {"block": 2}


@pytest.mark.parametrize("level", [AutotuneLevel.ADAPTIVE_AUTOTUNE, AutotuneLevel.FORCE_AUTOTUNE])
def test_excluded_kernel_does_not_enter_distributed_tuning(tmp_path, monkeypatch, level):
    general, calls, benchmarks, cache_file = make_kernel(tmp_path, monkeypatch, "general")
    monkeypatch.setattr(autotuner_module, "get_triton_autotune_level", lambda: level)
    monkeypatch.setattr(autotuner_module.dist, "is_initialized", lambda: True)
    monkeypatch.setattr(autotuner_module, "get_global_rank", lambda: 0)
    monkeypatch.setattr(autotuner_module, "get_global_world_size", lambda: 2)

    def unexpected_collective(*args, **kwargs):
        pytest.fail("An excluded kernel must not enter autotuning collectives")

    monkeypatch.setattr(general, "_get_autotune_group", unexpected_collective)
    monkeypatch.setattr(autotuner_module.dist, "all_gather_object", unexpected_collective)
    with Autotuner.autotune_warmup(AutotuneKernelType.DECODE_ATTENTION):
        assert general(16) is None
    assert calls == [(16, None)]
    assert benchmarks == []
    assert not cache_file.exists()


@pytest.mark.parametrize("level", [0, 1, 2, 3])
def test_matching_phase_preserves_autotune_levels(tmp_path, monkeypatch, level):
    monkeypatch.setattr(autotuner_module, "get_triton_autotune_level", lambda: level)
    decode, _, benchmarks, cache_file = make_kernel(
        tmp_path, monkeypatch, "decode", AutotuneKernelType.DECODE_ATTENTION
    )
    cache_file.write_text(json.dumps({"16": {"block": 1}}))
    cache_before = cache_file.read_bytes()
    with Autotuner.autotune_warmup(AutotuneKernelType.DECODE_ATTENTION):
        result = decode(16)
    if level == AutotuneLevel.FORCE_AUTOTUNE:
        assert result == {"block": 2}
        assert len(benchmarks) == 2
        assert json.loads(cache_file.read_text()) == {"16": {"block": 2}}
    else:
        assert result == (None if level == AutotuneLevel.CLOSE_AUTOTUNE else {"block": 1})
        assert benchmarks == []
        assert cache_file.read_bytes() == cache_before


def test_history_is_warmed_on_load_or_during_autotune_warmup(tmp_path, monkeypatch):
    kernel, calls, benchmarks, cache_file = make_kernel(tmp_path, monkeypatch, "general")
    cache_file.write_text(json.dumps({"8": {"block": 1}, "32": {"block": 2}}))
    assert kernel(16) == {"block": 1}
    assert calls == [(16, {"block": 1}), (16, {"block": 2}), (16, {"block": 1})]
    calls.clear()
    with Autotuner.autotune_warmup(AutotuneKernelType.DECODE_ATTENTION):
        assert kernel(16) == {"block": 1}
    assert calls == [(16, {"block": 1})]
    calls.clear()
    assert kernel(16) == {"block": 1}
    assert calls == [(16, {"block": 1})]
    assert benchmarks == []


def test_repeated_configs_are_skipped_after_success(tmp_path, monkeypatch):
    kernel, calls, benchmarks, cache_file = make_kernel(tmp_path, monkeypatch, "general")
    config = {"block": 1, "warps": 4}
    cache_file.write_text(json.dumps({"8": config, "16": config, "32": {"block": 2}}))
    with Autotuner.autotune_warmup():
        assert kernel(16) == config
    assert calls == [(16, config), (16, {"block": 2}), (16, config)]

    calls.clear()
    with Autotuner.autotune_warmup():
        assert kernel(16) == config
    with Autotuner.autotune_warmup(AutotuneKernelType.DECODE_ATTENTION):
        assert kernel(16) == config
    assert kernel(16) == config
    assert calls == [(16, config)] * 3
    assert benchmarks == []


def test_failed_warmup_retries_only_during_autotune_warmup(tmp_path, monkeypatch):
    kernel, calls, _, cache_file = make_kernel(tmp_path, monkeypatch, "general")
    cache_file.write_text(json.dumps({"8": {"block": 1}, "32": {"block": 2}}))

    def shape_sensitive_kernel(size, run_config=None):
        calls.append((size, run_config))
        if size < 16 and run_config["block"] == 2:
            raise RuntimeError("This config requires a larger input")
        return run_config

    monkeypatch.setattr(kernel, "fn", shape_sensitive_kernel)
    with Autotuner.autotune_warmup(AutotuneKernelType.DECODE_ATTENTION):
        assert kernel(8) == {"block": 1}
    assert calls == [(8, {"block": 1}), (8, {"block": 2}), (8, {"block": 1})]
    calls.clear()
    assert kernel(16) == {"block": 1}
    assert calls == [(16, {"block": 1})]
    calls.clear()
    with Autotuner.autotune_warmup(AutotuneKernelType.DECODE_ATTENTION):
        assert kernel(16) == {"block": 1}
    assert calls == [(16, {"block": 2}), (16, {"block": 1})]
    calls.clear()
    with Autotuner.autotune_warmup(AutotuneKernelType.DECODE_ATTENTION):
        assert kernel(16) == {"block": 1}
    assert calls == [(16, {"block": 1})]


def test_new_configs_are_warmed_on_later_calls_and_state_is_not_persisted(tmp_path, monkeypatch):
    kernel, calls, benchmarks, cache_file = make_kernel(tmp_path, monkeypatch, "general")
    cache_file.write_text(json.dumps({"8": {"block": 1}}))
    with Autotuner.autotune_warmup():
        assert kernel(8) == {"block": 1}
    calls.clear()
    with Autotuner.autotune_warmup():
        assert kernel(16) == {"block": 2}
    assert calls == [(16, {"block": 2})]
    assert len(benchmarks) == 2
    assert json.loads(cache_file.read_text()) == {"8": {"block": 1}, "16": {"block": 2}}

    calls.clear()
    assert kernel(16) == {"block": 2}
    assert calls == [(16, {"block": 2})]
    calls.clear()
    with Autotuner.autotune_warmup():
        assert kernel(16) == {"block": 2}
    assert calls == [(16, {"block": 2}), (16, {"block": 2})]
    calls.clear()
    with Autotuner.autotune_warmup():
        assert kernel(16) == {"block": 2}
    assert calls == [(16, {"block": 2})]

    # A new autotuner loading the same file must warm its configs again.
    reloaded, reload_calls, _, _ = make_kernel(tmp_path, monkeypatch, "reloaded")
    reloaded._cache_dir = str(cache_file.parent)
    assert reloaded(16) == {"block": 2}
    assert len(reload_calls) == 3
    assert {call[1]["block"] for call in reload_calls[:2]} == {1, 2}
    reload_calls.clear()
    with Autotuner.autotune_warmup():
        assert reloaded(16) == {"block": 2}
    assert reload_calls == [(16, {"block": 2})]


def test_warmup_preserves_mutated_input_and_skips_repeated_execution():
    executions = []

    @autotune(
        kernel_name="mutating_kernel",
        configs_gen_func=lambda: [{"block": 1}],
        static_key_func=lambda: {},
        run_key_func=lambda state: state.numel(),
        mutates_args=["state"],
    )
    def kernel(state, run_config=None):
        executions.append(run_config)
        state.add_(run_config["block"])

    state = torch.zeros(4)
    static_key = frozendict({})
    kernel.kernel_warmup(static_key, state, run_config={"block": 1})
    torch.testing.assert_close(state, torch.zeros(4))

    kernel.kernel_warmup(static_key, state, run_config={"block": 1})
    assert executions == [{"block": 1}]
    torch.testing.assert_close(state, torch.zeros(4))


def test_explicit_config_bypasses_tuning(tmp_path, monkeypatch):
    kernel, calls, benchmarks, cache_file = make_kernel(tmp_path, monkeypatch, "general")
    with Autotuner.autotune_warmup():
        assert kernel(16, run_config={"block": 3}) == {"block": 3}
    assert calls == [(16, {"block": 3})]
    assert benchmarks == []
    assert not cache_file.exists()


def test_default_api_and_nested_phase_restore_after_exception():
    assert not Autotuner.is_autotune_warmup()
    assert not Autotuner.is_kernel_autotune_warmup(AutotuneKernelType.GENERAL)
    assert not Autotuner.is_kernel_autotune_warmup(AutotuneKernelType.DECODE_ATTENTION)
    Autotuner.start_autotune_warmup()
    assert Autotuner.is_kernel_autotune_warmup(AutotuneKernelType.GENERAL)
    assert not Autotuner.is_kernel_autotune_warmup(AutotuneKernelType.DECODE_ATTENTION)
    with pytest.raises(RuntimeError, match="test failure"):
        with Autotuner.autotune_warmup(AutotuneKernelType.DECODE_ATTENTION):
            assert Autotuner.is_autotune_warmup()
            assert Autotuner.is_kernel_autotune_warmup(AutotuneKernelType.DECODE_ATTENTION)
            assert not Autotuner.is_kernel_autotune_warmup(AutotuneKernelType.GENERAL)
            with Autotuner.autotune_warmup():
                assert Autotuner.is_kernel_autotune_warmup(AutotuneKernelType.GENERAL)
            assert Autotuner.is_kernel_autotune_warmup(AutotuneKernelType.DECODE_ATTENTION)
            raise RuntimeError("test failure")
    assert Autotuner.is_kernel_autotune_warmup(AutotuneKernelType.GENERAL)
    Autotuner.end_autotune_warmup()
    assert not Autotuner.is_autotune_warmup()
    assert not Autotuner.is_kernel_autotune_warmup(AutotuneKernelType.GENERAL)
    assert not Autotuner.is_kernel_autotune_warmup(AutotuneKernelType.DECODE_ATTENTION)
