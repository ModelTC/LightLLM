import pytest
import torch

from lightllm.common.basemodel.attention.linear import replay_gdn_decode as tuning
from unit_tests.common.basemodel.triton_kernel.linear_att.replay_history import (
    attach_replay_history as ReplaySSMExecutor,
)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.parametrize("width", [1, 5])
def test_rebuild_uses_one_layer_scratch(width):
    state = torch.zeros((4, 3, 2, 32, 32), dtype=torch.bfloat16, device="cuda")
    ssm_backend = ReplaySSMExecutor(state, 8, width, num_key_heads=2)
    tokens = 3 * width
    q = torch.ones((tokens, 2, 32), dtype=torch.bfloat16, device="cuda")
    gates = torch.zeros((tokens, 2), dtype=torch.bfloat16, device="cuda")
    a_log = torch.zeros(2, device="cuda")
    cu = torch.arange(4, device="cuda", dtype=torch.int32) * width
    reqs = torch.arange(3, device="cuda", dtype=torch.int32)
    _, kwargs = tuning.rebuild_inputs(ssm_backend, 0, q, q, q, gates, gates, a_log, a_log, reqs, cu)
    scratch = kwargs["state"]
    assert scratch is not ssm_backend
    assert scratch.ssm_state.shape[0] == 1
    assert scratch.capacity == ssm_backend.capacity
    assert torch.count_nonzero(ssm_backend.ssm_state) == 0


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.parametrize("width", [1, 5])
def test_rebuild_failure_leaves_serving_state_unchanged(monkeypatch, tmp_path, width):
    from lightllm.common.triton_utils import autotuner as autotuner_module
    from lightllm.common.triton_utils.autotuner import Autotuner, AutotuneKernelType

    state = torch.zeros((2, 4, 2, 32, 32), dtype=torch.bfloat16, device="cuda")
    ssm_backend = ReplaySSMExecutor(state, 8, width, num_key_heads=2)
    for tensor in vars(ssm_backend).values():
        if isinstance(tensor, torch.Tensor):
            tensor.zero_()
    tokens = 3 * width
    q = torch.ones((tokens, 2, 32), dtype=torch.bfloat16, device="cuda")
    gates = torch.zeros((tokens, 2), dtype=torch.bfloat16, device="cuda")
    a_log = torch.zeros(2, device="cuda")
    bias = torch.zeros(2, device="cuda")
    cu = torch.arange(4, device="cuda", dtype=torch.int32) * width
    reqs = torch.arange(3, device="cuda", dtype=torch.int32)
    before = {name: x.clone() for name, x in vars(ssm_backend).items() if isinstance(x, torch.Tensor)}

    def oom(*args, **kwargs):
        raise torch.cuda.OutOfMemoryError("scratch")

    tuner = tuning.replay
    monkeypatch.setattr(tuner, "rebuild_input_func", oom)
    monkeypatch.setattr(tuner, "_cache_dir", str(tmp_path), raising=False)
    monkeypatch.setattr(tuner, "cached_configs", {})
    monkeypatch.setattr(tuner, "fast_match_configs", {})
    monkeypatch.setattr(autotuner_module, "get_triton_autotune_level", lambda: 1)
    monkeypatch.setattr(
        tuner, "_bench", lambda *args, **kwargs: pytest.fail("Failed rebuild must not reach benchmarks")
    )
    with pytest.raises(torch.cuda.OutOfMemoryError, match="scratch"):
        with Autotuner.autotune_warmup(AutotuneKernelType.DECODE_ATTENTION):
            tuning.replay(ssm_backend, 0, q, q, q, gates, gates, a_log, bias, reqs, cu)
    assert not list(tmp_path.iterdir())
    config = {"BV": 32, "num_warps": 1, "num_stages": 3}
    for name, expected in before.items():
        torch.testing.assert_close(getattr(ssm_backend, name), expected, atol=0, rtol=0)

    tuning.replay(ssm_backend, 0, q, q, q, gates, gates, a_log, bias, reqs, cu, run_config=config)
    torch.cuda.synchronize()
