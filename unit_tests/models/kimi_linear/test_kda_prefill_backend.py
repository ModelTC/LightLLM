import functools
import sys
from types import SimpleNamespace

import pytest
import torch
import torch.nn.functional as F

from lightllm.models.kimi_linear.triton_kernel import kda_prefill_backend
from lightllm.models.kimi_linear.triton_kernel.fla.ops.kda import (
    RCP_LN2,
    chunk_kda_with_fused_gate,
    fused_kda_gate,
    fused_kda_gate_chunk_cumsum,
)


@pytest.fixture(autouse=True)
def clear_backend_cache():
    kda_prefill_backend.get_kda_prefill_chunk_fn.cache_clear()
    yield
    kda_prefill_backend.get_kda_prefill_chunk_fn.cache_clear()


def test_fla_backend_can_be_selected_explicitly(monkeypatch):
    monkeypatch.setattr(
        kda_prefill_backend,
        "get_env_start_args",
        lambda: SimpleNamespace(kda_prefill_backend="fla"),
    )

    fn = kda_prefill_backend.get_kda_prefill_chunk_fn(128, torch.bfloat16)

    assert fn is kda_prefill_backend._triton_kda_chunk


def test_flashkda_falls_back_on_pre_hopper_gpu(monkeypatch):
    monkeypatch.setattr(
        kda_prefill_backend,
        "get_env_start_args",
        lambda: SimpleNamespace(kda_prefill_backend="flashkda"),
    )
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda: (8, 0))
    monkeypatch.setattr(kda_prefill_backend, "_flashkda_cuda_version_supported", lambda: True)

    fn = kda_prefill_backend.get_kda_prefill_chunk_fn(128, torch.bfloat16)

    assert fn is kda_prefill_backend._triton_kda_chunk


def test_flashkda_falls_back_below_cuda_12_9(monkeypatch):
    monkeypatch.setattr(
        kda_prefill_backend,
        "get_env_start_args",
        lambda: SimpleNamespace(kda_prefill_backend="flashkda"),
    )
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(kda_prefill_backend, "_flashkda_cuda_version_supported", lambda: False)

    fn = kda_prefill_backend.get_kda_prefill_chunk_fn(128, torch.bfloat16)

    assert fn is kda_prefill_backend._triton_kda_chunk


def test_flashkda_is_selected_when_requirements_are_met(monkeypatch):
    fake_flash_kda = SimpleNamespace(fwd=lambda *args, **kwargs: None)
    monkeypatch.setitem(sys.modules, "flash_kda", fake_flash_kda)
    monkeypatch.setattr(
        kda_prefill_backend,
        "get_env_start_args",
        lambda: SimpleNamespace(kda_prefill_backend="flashkda"),
    )
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda: (9, 0))
    monkeypatch.setattr(kda_prefill_backend, "_flashkda_cuda_version_supported", lambda: True)

    fn = kda_prefill_backend.get_kda_prefill_chunk_fn(128, torch.bfloat16)

    assert isinstance(fn, functools.partial)
    assert fn.func is kda_prefill_backend._flash_kda_chunk
    assert fn.keywords["flash_kda"] is fake_flash_kda


def test_auto_selects_flashkda_when_requirements_are_met(monkeypatch):
    fake_flash_kda = SimpleNamespace(fwd=lambda *args, **kwargs: None)
    monkeypatch.setitem(sys.modules, "flash_kda", fake_flash_kda)
    monkeypatch.setattr(
        kda_prefill_backend,
        "get_env_start_args",
        lambda: SimpleNamespace(kda_prefill_backend="auto"),
    )
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda: (9, 0))
    monkeypatch.setattr(kda_prefill_backend, "_flashkda_cuda_version_supported", lambda: True)

    fn = kda_prefill_backend.get_kda_prefill_chunk_fn(128, torch.bfloat16)

    assert isinstance(fn, functools.partial)
    assert fn.func is kda_prefill_backend._flash_kda_chunk


def test_flashkda_adapter_prepares_raw_contiguous_inputs():
    captured = {}

    def fake_fwd(q, k, v, raw_g, beta, scale, output, **kwargs):
        captured.update(
            q=q,
            k=k,
            v=v,
            raw_g=raw_g,
            beta=beta,
            scale=scale,
            output=output,
            **kwargs,
        )
        output.fill_(3)
        kwargs["final_state"].fill_(4)

    batch, tokens, heads, dim = 1, 4, 2, 4
    packed_qkv = torch.randn(batch, tokens, heads, 3 * dim, dtype=torch.bfloat16)
    q, k, v = packed_qkv.split(dim, dim=-1)
    raw_g = torch.randn(batch, tokens, heads, dim, dtype=torch.bfloat16)
    beta = torch.randn(batch, tokens, heads, dtype=torch.bfloat16)
    initial_state = torch.randn(1, heads, dim, dim, dtype=torch.float32)

    output, final_state = kda_prefill_backend._flash_kda_chunk(
        flash_kda=SimpleNamespace(fwd=fake_fwd),
        q=q,
        k=k,
        v=v.clone(),
        raw_g=raw_g,
        beta=beta,
        A_log=torch.randn(heads, dtype=torch.float32),
        g_bias=torch.randn(heads * dim, dtype=torch.float32),
        initial_state=initial_state.clone(),
        output_final_state=True,
        use_qk_l2norm_in_kernel=True,
        use_beta_sigmoid_in_kernel=True,
        safe_gate=True,
        lower_bound=-5.0,
        cu_seqlens=torch.tensor([0, tokens], dtype=torch.int32),
    )

    for name in ("q", "k", "v", "raw_g", "beta"):
        assert captured[name].is_contiguous()
    assert torch.equal(captured["beta"], beta)
    assert captured["dt_bias"].shape == (heads, dim)
    assert captured["cu_seqlens"].dtype == torch.long
    assert captured["lower_bound"] == -5.0
    assert captured["initial_state"].dtype == final_state.dtype == torch.float32
    assert torch.all(output == 3)
    assert torch.all(final_state == 4)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
@pytest.mark.parametrize("lower_bound", [None, -5.0])
def test_kda_gate_matches_torch_reference(lower_bound):
    torch.manual_seed(11)
    tokens, heads, dim = 17, 3, 32
    raw_g = torch.randn(tokens, heads * dim, device="cuda", dtype=torch.bfloat16)
    A_log = torch.randn(heads, device="cuda", dtype=torch.float32)
    bias = torch.randn(heads * dim, device="cuda", dtype=torch.float32)

    actual = fused_kda_gate(raw_g, A_log, dim, g_bias=bias, lower_bound=lower_bound)
    gate_input = raw_g.float().view(tokens, heads, dim) + bias.view(heads, dim)
    if lower_bound is None:
        expected = -A_log.exp().view(1, heads, 1) * F.softplus(gate_input)
    else:
        expected = lower_bound * torch.sigmoid(A_log.exp().view(1, heads, 1) * gate_input)

    torch.testing.assert_close(actual, expected, rtol=2e-3, atol=2e-3)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_lower_bound_gate_chunk_cumsum_matches_torch_reference():
    torch.manual_seed(12)
    tokens, heads, dim = 23, 2, 32
    raw_g = torch.randn(1, tokens, heads, dim, device="cuda", dtype=torch.bfloat16)
    A_log = torch.randn(heads, device="cuda", dtype=torch.float32)
    bias = torch.randn(heads * dim, device="cuda", dtype=torch.float32)

    actual = fused_kda_gate_chunk_cumsum(raw_g, A_log, g_bias=bias, lower_bound=-5.0)
    gate_input = raw_g.float() + bias.view(heads, dim)
    gate = -5.0 * torch.sigmoid(A_log.exp().view(1, 1, heads, 1) * gate_input)
    expected = gate.cumsum(dim=1) * RCP_LN2

    torch.testing.assert_close(actual, expected, rtol=2e-3, atol=2e-3)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_raw_beta_path_preserves_existing_triton_output():
    torch.manual_seed(13)
    batch, tokens, heads, dim = 1, 19, 2, 32
    q = torch.randn(batch, tokens, heads, dim, device="cuda", dtype=torch.bfloat16)
    k = torch.randn_like(q)
    v = torch.randn_like(q)
    raw_g = torch.randn_like(q)
    raw_beta = torch.randn(batch, tokens, heads, device="cuda", dtype=torch.bfloat16)
    A_log = torch.randn(heads, device="cuda", dtype=torch.float32)
    bias = torch.randn(heads * dim, device="cuda", dtype=torch.float32)
    initial_state = torch.randn(batch, heads, dim, dim, device="cuda", dtype=torch.float32)

    expected = chunk_kda_with_fused_gate(
        q=q,
        k=k,
        v=v.clone(),
        raw_g=raw_g,
        beta=raw_beta.float().sigmoid(),
        A_log=A_log,
        g_bias=bias,
        initial_state=initial_state.clone(),
        output_final_state=True,
        use_qk_l2norm_in_kernel=True,
    )
    actual = chunk_kda_with_fused_gate(
        q=q,
        k=k,
        v=v.clone(),
        raw_g=raw_g,
        beta=raw_beta,
        A_log=A_log,
        g_bias=bias,
        initial_state=initial_state.clone(),
        output_final_state=True,
        use_qk_l2norm_in_kernel=True,
        use_beta_sigmoid_in_kernel=True,
    )

    torch.testing.assert_close(actual[0], expected[0], rtol=0, atol=0)
    torch.testing.assert_close(actual[1], expected[1], rtol=0, atol=0)
