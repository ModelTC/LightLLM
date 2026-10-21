import pytest
import torch

from lightllm.common.basemodel.triton_kernel.linear_att import ssm_autotune as tuning
from lightllm.common.basemodel.triton_kernel.linear_att.replayssm import ReplaySSMBackend


@pytest.mark.parametrize("width", [1, 5])
def test_scratch_budget_is_independent_of_model_layer_count(width):
    state = torch.empty((48, 257, 48, 128, 128), dtype=torch.bfloat16, device="meta")
    ssm_backend = ReplaySSMBackend(state, 16, width, num_key_heads=16)
    tokens = 256 * width
    inputs = [
        torch.empty(shape, dtype=torch.bfloat16, device="meta")
        for shape in [(tokens, 16, 128), (tokens, 16, 128), (tokens, 48, 128), (tokens, 48), (tokens, 48), (48,), (48,)]
    ]
    required = tuning.scratch_size_bytes(ssm_backend, 256, 256, inputs)
    assert required < 1024 ** 3
    single_layer = ReplaySSMBackend(state[:1], 16, width, num_key_heads=16)
    assert required == tuning.scratch_size_bytes(single_layer, 256, 256, inputs)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.parametrize("width", [1, 5])
def test_tuning_budget_falls_back_without_touching_serving_state(monkeypatch, tmp_path, width):
    from lightllm.common.triton_utils import autotuner as autotuner_module
    from lightllm.common.triton_utils.autotuner import Autotuner, AutotuneKernelType

    state = torch.zeros((2, 4, 2, 32, 32), dtype=torch.bfloat16, device="cuda")
    ssm_backend = ReplaySSMBackend(state, 8, width, num_key_heads=2)
    for tensor in vars(ssm_backend).values():
        if isinstance(tensor, torch.Tensor):
            tensor.zero_()
    tokens = 3 * width
    q = torch.ones((tokens, 2, 32), dtype=torch.bfloat16, device="cuda")
    gates = torch.zeros((tokens, 2), dtype=torch.bfloat16, device="cuda")
    a_log = torch.zeros(2, device="cuda")
    bias = torch.zeros(2, device="cuda")
    cu = torch.arange(4, device="cuda", dtype=torch.int32) * width if width > 1 else None
    if cu is not None:
        q = q.unsqueeze(0)
    inputs = dict(
        ssm_backend=ssm_backend, mode="replay", q=q, k=q, v=q, a=gates, b=gates, a_log=a_log, bias=bias, cu_seqlens=cu
    )
    before = {name: x.clone() for name, x in vars(ssm_backend).items() if isinstance(x, torch.Tensor)}
    monkeypatch.setattr(torch.cuda, "mem_get_info", lambda device: (0, 1024 ** 3))
    monkeypatch.setattr(torch.cuda, "memory_reserved", lambda device: 0)
    monkeypatch.setattr(torch.cuda, "memory_allocated", lambda device: 0)
    tuner = tuning.select_config
    monkeypatch.setattr(tuner, "_cache_dir", str(tmp_path), raising=False)
    monkeypatch.setattr(tuner, "cached_configs", {})
    monkeypatch.setattr(tuner, "fast_match_configs", {})
    monkeypatch.setattr(tuner, "_oom_keys", set())
    monkeypatch.setattr(autotuner_module, "get_triton_autotune_level", lambda: 1)
    monkeypatch.setattr(
        tuner, "_bench", lambda *args, **kwargs: pytest.fail("Insufficient memory must skip benchmarks")
    )
    with Autotuner.autotune_warmup(AutotuneKernelType.DECODE_ATTENTION):
        config = tuning.resolve_run_config(**inputs)
        assert tuning.resolve_run_config(**inputs) is config
    assert config == {"BV": 32, "num_warps": 1}
    assert not list(tmp_path.iterdir())
    for name, expected in before.items():
        torch.testing.assert_close(getattr(ssm_backend, name), expected, atol=0, rtol=0)

    reqs = torch.arange(3, device="cuda", dtype=torch.int32)
    ssm_backend.prepare_decode(reqs, cu)
    ssm_backend.forward(0, q, q, q, gates, gates, a_log, bias, reqs, cu, run_config=config)
    torch.cuda.synchronize()
