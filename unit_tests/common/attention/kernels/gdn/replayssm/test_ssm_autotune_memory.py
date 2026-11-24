import collections

import pytest
import torch

from lightllm.common.attention.kernels.gdn.replayssm import replay_gdn as tuning
from unit_tests.common.attention.kernels.gdn.replayssm.replay_history import (
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


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_tuning_with_the_callers_out_benchmarks_scratch(monkeypatch, tmp_path):
    """Backends pass out; tuning a missing config must accept it and leave it to the final run."""
    from lightllm.common.triton_utils import autotuner as autotuner_module
    from lightllm.common.triton_utils.autotuner import Autotuner, AutotuneKernelType

    width, tokens = 1, 3
    config = {"BV": 32, "num_warps": 1, "num_stages": 3}
    q = torch.randn((tokens, 2, 32), dtype=torch.bfloat16, device="cuda")
    gates = torch.randn((tokens, 2), dtype=torch.bfloat16, device="cuda")
    a_log = torch.zeros(2, device="cuda")
    cu = torch.arange(tokens + 1, device="cuda", dtype=torch.int32)
    reqs = torch.arange(tokens, device="cuda", dtype=torch.int32)

    def backend():
        state = torch.zeros((2, 4, 2, 32, 32), dtype=torch.bfloat16, device="cuda")
        return ReplaySSMExecutor(state, 8, width, num_key_heads=2)

    expected = tuning.replay(backend(), 0, q, q, q, gates, gates, a_log, a_log, reqs, cu, run_config=config)

    tuner = tuning.replay
    writes = []
    rebuild = tuner.rebuild_input_func

    def tracking_rebuild(*args, **kwargs):
        args, kwargs = rebuild(*args, **kwargs)
        writes.append(kwargs.get("out"))
        return args, kwargs

    monkeypatch.setattr(tuner, "rebuild_input_func", tracking_rebuild)
    monkeypatch.setattr(tuner, "configs_gen_func", lambda: [config])
    monkeypatch.setattr(tuner, "_cache_dir", str(tmp_path), raising=False)
    monkeypatch.setattr(tuner, "cached_configs", {})
    monkeypatch.setattr(tuner, "fast_match_configs", collections.defaultdict(dict))
    monkeypatch.setattr(autotuner_module, "get_triton_autotune_level", lambda: 1)
    out = torch.full_like(q, float("nan"))
    with Autotuner.autotune_warmup(AutotuneKernelType.DECODE_ATTENTION):
        result = tuning.replay(backend(), 0, q, q, q, gates, gates, a_log, a_log, reqs, cu, out=out)
    # The benchmark wrote its own scratch output; the final run wrote the caller's out.
    assert writes == [None]
    assert result is out and torch.equal(out, expected)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_kda_rebuild_accepts_the_callers_out():
    from lightllm.common.attention.kernels.kda import replay_kda

    slots, heads, dim, capacity, tokens = 3, 2, 32, 8, 2
    device = "cuda"
    q = torch.randn((tokens, heads, dim), dtype=torch.bfloat16, device=device)
    _, kwargs = replay_kda.rebuild_inputs(
        state_ssm=torch.zeros((slots, heads, dim, dim), dtype=torch.float32, device=device),
        state_keys=torch.zeros((slots, heads, capacity, dim), dtype=torch.bfloat16, device=device),
        state_deltas=torch.zeros((slots, heads, capacity, dim), dtype=torch.bfloat16, device=device),
        state_gates=torch.zeros((slots, heads, capacity, dim), dtype=torch.bfloat16, device=device),
        state_raw_keys=torch.zeros((slots, heads, capacity, dim), dtype=torch.bfloat16, device=device),
        state_raw_values=torch.zeros((slots, heads, capacity, dim), dtype=torch.bfloat16, device=device),
        state_betas=torch.zeros((slots, heads, capacity), dtype=torch.bfloat16, device=device),
        state_cursors=torch.zeros(slots, dtype=torch.int32, device=device),
        state_tile_sync_mark=torch.zeros((slots, heads), dtype=torch.int32, device=device),
        q=q,
        k=q,
        v=q,
        a=torch.zeros((tokens, heads * dim), dtype=torch.bfloat16, device=device),
        b=torch.zeros((tokens, heads), dtype=torch.bfloat16, device=device),
        a_log=torch.zeros(heads, device=device),
        bias=torch.zeros(heads * dim, device=device),
        reqs=torch.arange(tokens, dtype=torch.int32, device=device),
        cu_seqlens=torch.arange(tokens + 1, dtype=torch.int32, device=device),
        state_verify_width=1,
        state_compute_mode="fp32",
        state_gate_lower_bound=-5.0,
        out=torch.empty_like(q),
    )
    # The benchmark gets its own scratch output, allocated by the kernel.
    assert "out" not in kwargs
