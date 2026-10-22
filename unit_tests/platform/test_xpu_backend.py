"""XPU backend tests, they run on an Intel GPU and are skipped elsewhere."""
import json

import pytest
import torch

import lightllm.platform as platform

pytestmark = pytest.mark.skipif(not (hasattr(torch, "xpu") and torch.xpu.is_available()), reason="needs an Intel XPU")


@pytest.fixture(autouse=True)
def xpu_platform(monkeypatch):
    from lightllm.utils.envs_utils import get_env_start_args

    def reset():
        platform._backend = None
        for fn in (platform.get_platform_name, platform.get_device_type, platform.get_device_module):
            fn.cache_clear()
        get_env_start_args.cache_clear()

    monkeypatch.setenv(
        "LIGHTLLM_START_ARGS",
        json.dumps({"hardware_platform": "xpu", "enable_torch_fallback": False, "enable_triton_fallback": False}),
    )
    reset()
    yield
    reset()


def test_xpu_runtime_basics():
    backend = platform.get_hardware_backend()
    runtime = backend.runtime
    assert runtime.device_type == "xpu" and runtime.dist_backend == "xccl"
    assert runtime.device_count() >= 1
    runtime.set_device(0)
    assert runtime.current_device() == 0
    assert runtime.get_device_name(0) == torch.xpu.get_device_name(0)
    stream = runtime.create_stream()
    event = runtime.create_event(enable_timing=True)
    with runtime.stream(stream):
        x = torch.ones(1024, device="xpu") * 2
        event.record()
    event.synchronize()
    runtime.synchronize()
    assert x.sum().item() == 2048


def test_xpu_mem_get_info_counts_pooled_memory():
    runtime = platform.get_hardware_backend().runtime
    torch.xpu.empty_cache()
    free_before, total = runtime.mem_get_info(0)
    nbytes = 2 * 1024 ** 3
    x = torch.empty(nbytes, dtype=torch.uint8, device="xpu")
    torch.xpu.synchronize()
    free_alloc, _ = runtime.mem_get_info(0)
    assert free_before - free_alloc >= nbytes * 0.9
    del x
    torch.xpu.empty_cache()
    torch.xpu.synchronize()
    free_after, _ = runtime.mem_get_info(0)
    # The released memory is free again for this process even though Level Zero keeps it pooled.
    assert free_after >= free_before - 256 * 1024 ** 2
    assert free_after <= total - torch.xpu.memory_reserved(0)


def test_xpu_graph_capture_and_replay():
    graph_backend = platform.get_hardware_backend().graph
    x = torch.arange(16, dtype=torch.float32, device="xpu")
    y = x * 2
    torch.xpu.synchronize()
    g = graph_backend.create_graph()
    with graph_backend.graph(g, pool=graph_backend.graph_pool_handle()):
        assert graph_backend.is_capturing()
        y = x * 2
    assert not graph_backend.is_capturing()
    x.add_(1)
    g.replay()
    torch.xpu.synchronize()
    assert torch.equal(y, (torch.arange(16, device="xpu") + 1) * 2.0)


def test_no_ref_tensor_on_xpu():
    from lightllm.utils.tensor_utils import tensor_to_no_ref_tensor

    x = torch.randn(4, 8, dtype=torch.bfloat16, device="xpu")
    y = tensor_to_no_ref_tensor(x)
    assert y.data_ptr() == x.data_ptr() and y.shape == x.shape and y.dtype == x.dtype
    y[0, 0] = 7
    torch.xpu.synchronize()
    assert x[0, 0].item() == 7


def test_device_helpers_on_xpu():
    from lightllm.utils.device_utils import (
        get_current_device_name,
        is_4090,
        is_hopper,
        is_sm100_gpu,
        is_tesla,
        is_xpu,
    )

    for fn in (get_current_device_name, is_tesla, is_hopper, is_4090, is_sm100_gpu):
        fn.cache_clear()
    assert is_xpu()
    assert get_current_device_name() == torch.xpu.get_device_name(torch.xpu.current_device()).replace(" ", "_")
    assert not is_tesla() and not is_hopper() and not is_4090() and not is_sm100_gpu()


def test_platform_ops_route_to_xpu_forward(monkeypatch):
    from lightllm.common.basemodel.layer_weights.meta_weights.norm_weight import RMSNormWeight
    from lightllm.common.basemodel.layer_weights.meta_weights.embedding_weight import EmbeddingWeight, LMHeadWeight

    torch.xpu.set_device(0)
    monkeypatch.setenv("LIGHTLLM_CURRENT_DEVICE_ID", "0")
    norm = RMSNormWeight(dim=64, weight_name="w", data_type=torch.float16)
    assert norm._forward.__func__ is RMSNormWeight._xpu_forward
    norm.weight.fill_(1.0)
    x = torch.randn(3, 64, dtype=torch.float16, device="xpu")
    ref = (x.float() * torch.rsqrt(x.float().pow(2).mean(-1, keepdim=True) + 1e-6)).half()
    torch.testing.assert_close(norm(x, eps=1e-6), ref, atol=2e-3, rtol=2e-3)

    # LMHeadWeight must not inherit the embedding kernel from EmbeddingWeight._xpu_forward.
    assert LMHeadWeight._xpu_forward is not EmbeddingWeight._xpu_forward
