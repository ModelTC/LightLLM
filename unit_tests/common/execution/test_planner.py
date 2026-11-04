from lightllm.core.formats.kv.spec import DenseKVStateSpec
from dataclasses import replace
from types import SimpleNamespace

import pytest
import torch

from lightllm.core.compute.attention.dense.spec import DenseAttentionSpec
from lightllm.core.contracts.device import DeviceCaps
from lightllm.core.contracts.device import ExecutionLimits
from lightllm.core.contracts.state import FormatSpec
from lightllm.core.planner.planner import Planner
from lightllm.core.contracts.state import StateSpec
from lightllm.core.contracts.state import StateView
from lightllm.core.compute.attention.dense.backend import DenseAttentionBackend, DenseStaticParams
from lightllm.core.contracts.provider import AttentionOffer, ProbeResult


def view(layer=0, role="main", heads=2, dim=64):
    spec = StateSpec(role, f"kv.{layer}", layer, (layer,), DenseKVStateSpec(kv_heads=heads, head_dim=dim))
    format = FormatSpec("plain", torch.float16, "token_head_dim", ("k", "v"), 1)
    k = torch.randn(8, heads, dim, dtype=torch.float16)
    return StateView(spec, format, (k, torch.randn_like(k)))


def planner(*backends, budget=8 * 1024 * 1024):
    return Planner(backends, DeviceCaps("cuda", 0, (9, 0)), ExecutionLimits(budget))


@pytest.mark.parametrize("implementation", ["triton", "fa3"])
@pytest.mark.parametrize("phase", ["prefill", "decode"])
def test_bound_execution_uses_native_views_without_manager(implementation, phase):
    state_view = view()
    backend = DenseAttentionBackend(implementation, phase)
    params = DenseStaticParams(DenseAttentionSpec(4, 2, 64), torch.empty)
    plan = planner(backend).bind(
        [(0, phase, params.spec, state_view, params, "auto")], lambda size: torch.empty(size, dtype=torch.uint8)
    )
    bound = plan.bindings[0].execution
    assert bound.k is state_view.fields[0]
    assert bound.v is state_view.fields[1]

    assert not hasattr(bound, "model")
    assert not hasattr(bound, "control")
    metadata = object()
    batch = SimpleNamespace(execution_metadata={bound.metadata_key: metadata})
    assert batch.execution_metadata[bound.metadata_key] is metadata
    assert plan.workspace_bytes >= 0


def test_heterogeneous_layers_and_roles_do_not_alias():
    backend = DenseAttentionBackend("triton", "decode")
    requests = []
    for layer_id, heads, dim in [(0, 2, 64), (1, 4, 128)]:
        state_view = view(layer_id, heads=heads, dim=dim)
        params = DenseStaticParams(DenseAttentionSpec(8, heads, dim), torch.empty)
        requests.append((layer_id, "decode", params.spec, state_view, params, "auto"))
    plan = planner(backend).bind(requests, lambda size: torch.empty(size, dtype=torch.uint8))
    assert [b.execution.k.shape[1:] for b in plan.bindings] == [(2, 64), (4, 128)]
    assert plan.bindings[0].execution.k.data_ptr() != plan.bindings[1].execution.k.data_ptr()
    assert view(role="main").state.key != view(role="draft0").state.key


@pytest.mark.parametrize("change", [{"softcap": 1}, {"causal": False}, {"window": (4096, 3)}, {"score_scale": 0.5}])
def test_unsupported_semantics_fail_explicitly(change):
    backend = DenseAttentionBackend("triton", "decode")
    with pytest.raises(ValueError, match="causal attention"):
        planner(backend).select(replace(DenseAttentionSpec(4, 2, 64), **change), view().format, "decode")


def test_missing_backend_phase_and_format_are_diagnostic():
    p = planner(DenseAttentionBackend("triton", "prefill"))
    spec, format = DenseAttentionSpec(4, 2, 64), view().format
    with pytest.raises(ValueError, match="not registered"):
        p.select(spec, format, "prefill", "missing")
    with pytest.raises(ValueError, match="compatible phase"):
        p.select(spec, format, "verify")
    with pytest.raises(ValueError, match="FP16/BF16"):
        p.select(spec, replace(format, encoding="int8"), "prefill")


def test_no_implicit_format_conversion_between_phases():
    p = planner(DenseAttentionBackend("triton", "decode"))
    state_view = view()
    state_view = replace(state_view, format=replace(state_view.format, layout="packed"))
    spec = DenseAttentionSpec(4, 2, 64)
    with pytest.raises(ValueError, match="layer 0"):
        p.bind(
            [(0, "decode", spec, state_view, DenseStaticParams(spec, torch.empty), "auto")],
            lambda size: torch.empty(size, dtype=torch.uint8),
        )


class WorkspaceBackend:
    implementation = "workspace"
    supports_auxiliary_states = False
    name = "workspace"

    def probe(self, spec, format, device, limits, *, auxiliary_formats=None):
        return ProbeResult(offers=(AttentionOffer(self.name, "decode", format, 8),))

    def bind(self, offer, view, params, workspace, *, auxiliary_views=None):
        return workspace


def test_aggregate_budget_is_checked_before_allocating():
    spec, state_view = DenseAttentionSpec(4, 2, 64), view()
    shared = replace(state_view.state, consumers=(0, 1))
    state_view = replace(state_view, state=shared)
    requests = [(i, "decode", spec, state_view, DenseStaticParams(spec, torch.empty), "auto") for i in (0, 1)]
    allocated = []
    with pytest.raises(ValueError, match="plan workspace 16 exceeds budget 8"):
        planner(WorkspaceBackend(), budget=8).bind(requests, allocated.append)
    assert allocated == []
    result = planner(WorkspaceBackend(), budget=16).bind(requests, lambda size: bytearray(size))
    assert result.bindings[0].execution is not result.bindings[1].execution


def test_shared_spec_does_not_authorize_undeclared_consumer():
    spec = DenseAttentionSpec(4, 2, 64)
    with pytest.raises(ValueError, match="not a declared consumer"):
        planner(WorkspaceBackend()).bind(
            [(1, "decode", spec, view(), None, "auto")], lambda size: torch.empty(size, dtype=torch.uint8)
        )


def test_binding_rejects_shape_mismatch():
    backend = DenseAttentionBackend("triton", "decode")
    spec = DenseAttentionSpec(4, 2, 128)
    with pytest.raises(ValueError, match="dimensions disagree"):
        planner(backend).bind(
            [(0, "decode", spec, view(), DenseStaticParams(spec, torch.empty), "auto")],
            lambda size: torch.empty(size, dtype=torch.uint8),
        )


@pytest.mark.parametrize("kwargs", [{"q_heads": 3}, {"head_dim": 0}, {"score_scale": float("nan")}, {"softcap": -1}])
def test_invalid_attention_spec(kwargs):
    with pytest.raises(ValueError):
        replace(DenseAttentionSpec(4, 2, 64), **kwargs)
