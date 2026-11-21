import json
from pathlib import Path

from lightllm.common import kernel_config
from lightllm.common.kernel_config import KernelConfigs, get_kernel_config_dir
from lightllm.common.triton_utils import autotuner


def test_new_cache_uses_portable_directory(tmp_path):
    assert get_kernel_config_dir(tmp_path, "kernel:v1") == str(tmp_path / "kernel_v1")


def test_existing_legacy_cache_is_readable(tmp_path):
    legacy = tmp_path / "kernel:v1"
    legacy.mkdir()
    assert get_kernel_config_dir(tmp_path, "kernel:v1") == str(legacy)


def test_bundled_portable_cache_takes_precedence(tmp_path):
    (tmp_path / "kernel:v1").mkdir()
    portable = tmp_path / "kernel_v1"
    portable.mkdir()
    assert get_kernel_config_dir(tmp_path, "kernel:v1") == str(portable)


def test_kernel_config_reads_bundled_directory(monkeypatch):
    class Config(KernelConfigs):
        kernel_name = "_fwd_kernel_flash_decode_diverse_stage1:v2"

    directory = Path(kernel_config.__file__).parent / "all_kernel_configs" / Config.kernel_name.replace(":", "_")
    config_file = next(directory.glob("*.json"))
    monkeypatch.setattr(KernelConfigs, "get_config_file_name", lambda params: config_file.name)
    assert Config.get_the_config(None) == json.loads(config_file.read_text())


def test_autotuner_resolves_portable_directory(tmp_path, monkeypatch):
    monkeypatch.setattr(autotuner, "__file__", str(tmp_path / "autotuner.py"))
    monkeypatch.setattr(autotuner, "get_triton_version", lambda: "triton_test")
    monkeypatch.setattr(autotuner, "get_current_device_name", lambda: "test_gpu")
    kernel = autotuner.Autotuner.__new__(autotuner.Autotuner)
    kernel.kernel_name = "kernel:v1"
    assert kernel.cache_dir == str(tmp_path / "autotune_kernel_configs/triton_test/test_gpu/kernel_v1")
