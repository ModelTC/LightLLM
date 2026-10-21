import pytest


@pytest.fixture
def cuda_platform(monkeypatch):
    """Resolve the LightLLM hardware platform as cuda.

    For tests that mock torch.cuda, so they keep exercising the cuda code path on hosts whose torch build
    targets another device (for example Intel XPU).
    """
    import lightllm.platform as platform
    from lightllm.platform.backends.cuda_like import CudaBackend

    def reset():
        for fn in (platform.get_platform_name, platform.get_device_type, platform.get_device_module):
            fn.cache_clear()

    reset()
    monkeypatch.setattr(platform, "detect_platform_name", lambda: "cuda")
    monkeypatch.setattr(platform, "_backend", CudaBackend())
    yield
    reset()
