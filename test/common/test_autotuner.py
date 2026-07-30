import collections

from frozendict import frozendict
import pytest

from lightllm.common.triton_utils import autotuner
from lightllm.common.triton_utils.autotuner import AutotuneLevel, Autotuner


def test_filter_valid_configs_drops_failed_tuning_results():
    configs = {
        "1024": {"num_warps": 4},
        "65536": None,
    }

    assert Autotuner._filter_valid_configs(configs) == {
        "1024": {"num_warps": 4},
    }


def test_failed_exact_config_fails_fast(monkeypatch):
    static_key = frozendict({"shape": 1})
    calls = []
    tuner = object.__new__(Autotuner)
    tuner.kernel_name = "test_kernel"
    tuner.fn = lambda *args, **kwargs: calls.append(kwargs) or "ok"
    tuner.cached_configs = {
        static_key: {
            "8192": {"num_warps": 4},
            "65536": None,
        }
    }
    tuner.fast_match_configs = collections.defaultdict(dict)
    tuner._static_key = lambda *args, **kwargs: static_key
    tuner._run_key = lambda *args, **kwargs: "65536"
    tuner.run_key_distance_func = lambda run_key, config_key: abs(int(run_key) - int(config_key))
    tuner._try_load_cache = lambda key: False
    tuner.kernel_warmup = lambda *args, **kwargs: None
    monkeypatch.setattr(autotuner, "get_triton_autotune_level", lambda: AutotuneLevel.USE_AUTOTUNE_HIS_CONFIG)

    with pytest.raises(RuntimeError, match="Cached autotuning failed.*65536"):
        tuner()
    assert calls == []


def test_cache_loader_preserves_failed_tuning_results(tmp_path, monkeypatch):
    static_key = frozendict({"shape": 1})
    tuner = object.__new__(Autotuner)
    tuner.kernel_name = "test_kernel"
    tuner.cache_dir = str(tmp_path)
    tuner.cached_configs = {}
    cache_file = tmp_path / "cache.json"
    cache_file.write_bytes(b'{"8192":{"num_warps":4},"65536":null}')
    monkeypatch.setattr(
        autotuner.KernelConfigs,
        "get_config_file_name",
        lambda key: cache_file.name,
    )

    assert tuner._try_load_cache(static_key)
    assert tuner.cached_configs[static_key] == {
        "8192": {"num_warps": 4},
        "65536": None,
    }


def test_warmup_executes_only_the_selected_config(monkeypatch):
    static_key = frozendict({"shape": 1})
    calls = []
    warmups = []
    tuner = object.__new__(Autotuner)
    tuner.fn = lambda *args, **kwargs: calls.append(kwargs) or "ok"
    tuner.cached_configs = {
        static_key: {
            "8192": {"num_warps": 4},
            "32768": {"num_warps": 8},
        }
    }
    tuner.fast_match_configs = collections.defaultdict(dict)
    tuner._static_key = lambda *args, **kwargs: static_key
    tuner._run_key = lambda *args, **kwargs: "32768"
    tuner.run_key_distance_func = lambda run_key, config_key: abs(int(run_key) - int(config_key))
    tuner._try_load_cache = lambda key: True
    tuner.kernel_warmup = lambda *args, **kwargs: warmups.append(kwargs)
    monkeypatch.setattr(autotuner, "get_triton_autotune_level", lambda: AutotuneLevel.USE_AUTOTUNE_HIS_CONFIG)

    assert tuner() == "ok"
    assert warmups == [{"run_config": {"num_warps": 8}}]
    assert calls == [{"run_config": {"num_warps": 8}}]
