from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from lightllm.utils import start_utils


@pytest.mark.parametrize("worker_alive", [False, True])
def test_shutdown_only_removes_instance_directory_after_workers_exit(tmp_path, monkeypatch, worker_alive):
    base = tmp_path / "cache"
    instance = base / "lightllm_disk_cache_current"
    other = base / "lightllm_disk_cache_other"
    instance.mkdir(parents=True)
    other.mkdir()
    manager = start_utils.SubmoduleManager()
    manager.register_disk_cache_dir(str(instance))
    worker = Mock(pid=12345)
    manager.processes = [worker]
    monkeypatch.setattr(start_utils, "kill_recursive", Mock())
    monkeypatch.setattr(
        start_utils.psutil, "wait_procs", lambda *_args, **_kwargs: ([], [worker] if worker_alive else [])
    )
    monkeypatch.setattr(start_utils, "is_process_active", lambda _pid: worker_alive)
    monkeypatch.setattr("lightllm.utils.envs_utils.get_env_start_args", lambda: SimpleNamespace(enable_mps=False))

    manager.terminate_all_processes()
    assert instance.exists() == worker_alive
    assert other.exists() and base.exists()
    if not worker_alive:
        manager.terminate_all_processes()


def test_disk_worker_uses_exact_instance_directory(tmp_path, monkeypatch):
    import torch
    from lightllm.server.multi_level_kv_cache import disk_cache_worker

    service = Mock(return_value=SimpleNamespace(_n=1))
    monkeypatch.setattr(disk_cache_worker, "PyLocalCacheService", service)
    directory = tmp_path / "lightllm_disk_cache_current"
    disk_cache_worker.DiskCacheWorker(1, SimpleNamespace(cpu_kv_cache_tensor=torch.zeros((2, 4))), str(directory))
    assert directory.is_dir()
    assert service.call_args.kwargs["file"] == str(directory / "cache_file")
