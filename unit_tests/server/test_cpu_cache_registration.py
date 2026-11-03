import threading
from types import SimpleNamespace

import pytest
import torch

from lightllm.server.multi_level_kv_cache import cpu_cache_client as module


@pytest.mark.parametrize("fail", [False, True])
def test_async_registration_overlaps_work_but_does_not_publish_early(monkeypatch, fail):
    started, release = threading.Event(), threading.Event()
    args = SimpleNamespace(cpu_kv_cache_shm_id=1234)
    meta = SimpleNamespace(
        page_num=8,
        layer_num=2,
        token_page_size=4,
        num_heads=1,
        data_type=torch.uint8,
        get_merged_head_dim=lambda: 8,
        calcu_size=lambda: 512,
    )
    monkeypatch.setattr(module, "get_env_start_args", lambda: args)
    monkeypatch.setattr(module, "calcu_cpu_cache_meta", lambda: meta)
    monkeypatch.setattr(module, "AtomicShmLock", lambda **kwargs: object())
    monkeypatch.setattr(module.CpuKvCacheClient, "_create_cpu_status_list", lambda *args: None)
    result = torch.empty(512, dtype=torch.uint8)

    def create_or_attach(self, *, init_shm_data, pin):
        assert not init_shm_data and pin
        started.set()
        assert release.wait(5)
        if fail:
            raise RuntimeError("registration failed")
        return result

    monkeypatch.setattr(module.CpuCacheCreator, "create_or_attach", create_or_attach)
    client = module.CpuKvCacheClient(False, False, defer_registration=True)
    try:
        assert started.wait(5)
        assert not hasattr(client, "cpu_kv_cache_tensor")
        release.set()
        if fail:
            with pytest.raises(RuntimeError, match="registration failed"):
                client.wait_registration()
        else:
            client.wait_registration()
            assert client.cpu_kv_cache_tensor is result
            client.wait_registration()
    finally:
        release.set()
        if client._registration_executor is not None:
            client._registration_executor.shutdown()
