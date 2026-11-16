from unit_tests.common.attention.helpers import build, run_attention
import sys
from types import ModuleType, SimpleNamespace

import pytest
import torch

import lightllm.common.attention.backends.gdn.triton.kernels.chunk as fla_ops
import lightllm.utils.backend_validator as backend_validator
from lightllm.common.attention.api import AttentionPhase
from lightllm.common.attention.api.inputs import GDNInputs, GDNResources
from lightllm.common.attention.backends.gdn.triton import backend as gdn
from lightllm.common.attention.backends.gdn.triton.backend import TritonGDNDecodeBackend, TritonGDNPrefillBackend
from lightllm.models.qwen3next.config import Qwen3NextResourceConfig
from unit_tests.common.attention.helpers import make_backend, make_batch, make_context, make_layer_attention


@pytest.mark.parametrize("cache_dtype", [torch.bfloat16, torch.float32])
def test_prefill_casts_final_state_to_cache_dtype(monkeypatch, cache_dtype):
    ssm = torch.zeros((1, 1, 2, 2), dtype=cache_dtype)
    final = torch.ones((1, 1, 2, 2), dtype=torch.float32)
    q = torch.zeros((1, 1, 1, 1), dtype=cache_dtype)
    monkeypatch.setattr(
        "lightllm.common.attention.backends.gdn.triton.kernels.fused_gdn_gating.fused_gdn_gating",
        lambda _log, a, b, _bias: (a, b),
    )
    monkeypatch.setattr(
        "lightllm.common.attention.backends._shared.linear.causal_conv1d.causal_conv1d_fn",
        lambda mixed, *args, **kwargs: mixed,
    )
    backend = make_backend(
        TritonGDNDecodeBackend,
        mtp_step=0,
        activation="silu",
        _rearrange_mixed_qkv=lambda mixed: (q, q, q),
        prefill_kernel=lambda **kwargs: (q, final),
        _split_qkvzba=lambda mixed: (mixed, q, q, q),
    )
    batch = make_batch((1,), phase=AttentionPhase.PREFILL)
    context = make_context(backend, batch)
    build(backend, context)
    params = GDNResources(torch.zeros((1, 3)), ssm, None, None, None, None)
    attention = make_layer_attention(backend, context, params)
    result, z = run_attention(attention, context, GDNInputs(torch.zeros((1, 3), dtype=cache_dtype)))
    assert result is q and z is q
    assert ssm.dtype == cache_dtype
    torch.testing.assert_close(ssm, final.to(cache_dtype))


@pytest.mark.parametrize("provider", [TritonGDNDecodeBackend])
def test_decode_and_fixed_mtp_state_routes(provider):
    requests = torch.tensor([2, 2, 2, 4, 4, 4], dtype=torch.int32)
    cursor = torch.tensor([0, 0, 1, 0, 2, 0], dtype=torch.int32)
    backend = make_backend(
        provider,
        draft_step=2,
    )
    context = make_context(backend, make_batch((3,) * 6, b_req_idx=requests))
    build(backend, context)
    assert context.b_conv_buffer_idx.tolist() == [2, 4]
    assert context.accepted_tokens(cursor).tolist() == [2, 3]
    assert context.b_ssm_buffer_idx.tolist() == [[6, 7, 8], [12, 13, 14]]
    backend.draft_step = 0
    context = make_context(backend, make_batch((3, 3), b_req_idx=torch.tensor([2, 4])))
    build(backend, context)
    assert context.b_ssm_buffer_idx.tolist() == [2, 4]


def test_dynamic_mtp_layout_uses_bound_state_cursor(monkeypatch):
    calls = []

    def build_layout(**kwargs):
        calls.append(kwargs)
        return torch.tensor([0, 1, 3]), torch.tensor([2, 4])

    monkeypatch.setattr(
        "lightllm.common.attention.backends._shared.linear.mtp_state_params.build_dynamic_mtp_layout", build_layout
    )
    cursor = torch.zeros(8, dtype=torch.int32)
    backend = make_backend(
        TritonGDNDecodeBackend,
        draft_step=2,
        dynamic_verify=True,
    )
    batch = make_batch((3, 3, 3), b_req_idx=torch.tensor([2, 4, 4]), b_mtp_index=torch.tensor([0, 0, 1]))
    context = make_context(backend, batch)
    build(backend, context)
    assert context.accepted_tokens(cursor).tolist() == [1, 1]
    assert calls[0]["hold_req_id"] == 7
    assert context.b_ssm_buffer_idx.tolist() == [[6, 7, 8], [12, 13, 14]]


def test_triton_backend_supplies_chunk_kernel():
    backend = object.__new__(TritonGDNPrefillBackend)
    assert backend.get_prefill_kernel() is fla_ops.chunk_gated_delta_rule


def test_flashqla_validation_loads_linear_cache_config(monkeypatch):
    linear_config = SimpleNamespace(
        tp_world_size=2,
        global_linear_k_heads=4,
        global_linear_v_heads=8,
        head_linear_k_dim=8,
        head_linear_v_dim=16,
        conv_state_dtype=torch.bfloat16,
        ssm_state_dtype=torch.float32,
    )
    load_calls = []
    monkeypatch.setattr(
        Qwen3NextResourceConfig,
        "load_from_args",
        staticmethod(lambda: load_calls.append(True) or linear_config),
    )

    kernel_calls = []

    def kernel(**kwargs):
        kernel_calls.append(kwargs)
        return kwargs["q"], kwargs["initial_state"]

    flashqla = ModuleType("flash_qla")
    flashqla.chunk_gated_delta_rule = kernel
    monkeypatch.setitem(sys.modules, "flash_qla", flashqla)
    monkeypatch.setattr(fla_ops, "chunk_gated_delta_rule", kernel)

    original_randn = torch.randn
    original_rand = torch.rand
    original_tensor = torch.tensor

    def cpu_randn(*args, **kwargs):
        kwargs.pop("device", None)
        return original_randn(*args, **kwargs)

    def cpu_rand(*args, **kwargs):
        kwargs.pop("device", None)
        return original_rand(*args, **kwargs)

    def cpu_tensor(*args, **kwargs):
        kwargs.pop("device", None)
        return original_tensor(*args, **kwargs)

    monkeypatch.setattr(backend_validator.torch, "randn", cpu_randn)
    monkeypatch.setattr(backend_validator.torch, "rand", cpu_rand)
    monkeypatch.setattr(backend_validator.torch, "tensor", cpu_tensor)
    monkeypatch.setattr(backend_validator.torch.cuda, "synchronize", lambda: None)

    success, error = backend_validator._validate_flashqla()

    assert success is True
    assert error is None
    assert load_calls == [True]
    assert len(kernel_calls) == 2
    assert kernel_calls[0]["q"].shape == (1, 64, 2, 8)
    assert kernel_calls[0]["v"].shape == (1, 64, 4, 16)
    assert kernel_calls[0]["q"].dtype == torch.bfloat16
    assert kernel_calls[0]["initial_state"].shape == (1, 4, 8, 16)
    assert kernel_calls[0]["initial_state"].dtype == torch.float32
