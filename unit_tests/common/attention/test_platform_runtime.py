import pytest


import torch


from lightllm.platform import get_device_runtime


def test_runtime_resolves_host_platform_without_querying_cuda(monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: pytest.fail("queried CUDA for a CPU runtime"))
    assert get_device_runtime(platform="cpu").device == torch.device("cpu")
    with pytest.raises(ValueError, match="cannot operate"):
        get_device_runtime("cuda:0", platform="cpu")
    with pytest.raises(ValueError, match="cannot operate"):
        get_device_runtime("cpu", platform="cuda")


def test_runtime_rejects_unimplemented_hardware():
    with pytest.raises(ValueError, match="CUDA and CPU"):
        get_device_runtime(platform="musa")
    with pytest.raises(ValueError, match="CUDA and CPU"):
        get_device_runtime("npu:0")


def test_explicit_current_device_is_not_overridden_by_worker_default(monkeypatch):
    monkeypatch.setenv("LIGHTLLM_CURRENT_DEVICE_ID", "0")
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "current_device", lambda: 1)
    assert get_device_runtime("cuda").device == torch.device("cuda:1")
    assert get_device_runtime("cuda:2").device == torch.device("cuda:2")
    assert get_device_runtime().device == torch.device("cuda:0")


def test_runtime_registration_keeps_platform_validation_in_adapter(monkeypatch):
    import lightllm.platform as platform

    monkeypatch.setattr(platform, "_runtime_factories", {})
    calls = []
    adapter = object()
    platform.register_device_runtime("musa", lambda device: calls.append(device) or adapter)
    assert platform.get_device_runtime("musa:2", platform="musa") is adapter
    assert calls == ["musa:2"]
    with pytest.raises(ValueError, match="already registered"):
        platform.register_device_runtime("musa", lambda device: adapter)
