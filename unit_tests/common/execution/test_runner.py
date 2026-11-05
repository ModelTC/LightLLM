from lightllm.core.compute.attention.workspace import lane_workspace
from lightllm.core.formats.kv.spec import DenseKVStateSpec
from dataclasses import replace
from types import SimpleNamespace

import pytest
import torch

from lightllm.common.attention_state import AttentionStateManager
from lightllm.core.compute.attention.dense.backend import DenseAttentionBackend, DenseStaticParams
from lightllm.models.base.adapter import DenseModelAdapter
from lightllm.core.planner.planner import AttentionRequest, Planner
from lightllm.core.compute.attention.dense.spec import DenseAttentionSpec
from lightllm.core.contracts.device import DeviceCaps, ExecutionLimits
from lightllm.core.contracts.state import FormatSpec, StateSpec
from lightllm.core.storage.attention import DenseStateStore
from lightllm.models.llama.model import LlamaTpPartModel
from unit_tests.common.execution.test_model_binding import make_model


def test_state_declarations_precede_storage_and_handle_tp_replicas():
    config = dict(num_key_value_heads=2, num_attention_heads=16, num_hidden_layers=1, hidden_size=1024)
    adapter = DenseModelAdapter(config, role="main", dtype=torch.float16, tp_world_size=4)
    assert adapter.state_specs()[0].params == DenseKVStateSpec(kv_heads=2, head_dim=64)
    assert adapter.attention_specs()[0].kv_heads == 1
    config["num_attention_heads"] = 32
    assert adapter.attention_specs()[0].q_heads == 4


def test_state_store_allocates_heterogeneous_and_explicit_shared_resources():
    declarations = (
        StateSpec("main", "kv.a", 0, (0, 2), DenseKVStateSpec(kv_heads=2, head_dim=64)),
        StateSpec("main", "kv.b", 1, (1,), DenseKVStateSpec(kv_heads=4, head_dim=128)),
    )
    format = FormatSpec("plain", torch.float16, "token_head_dim", ("k", "v"), 4)
    manager = AttentionStateManager(request_capacity=2)
    store = DenseStateStore(declarations, format, manager)
    manager.finish_state_registration()
    assert len(manager.get_full_attention_states()) == 2
    for state in manager.get_full_attention_states():
        state.device = torch.device("cpu")
        state.allocate_storage(8)
    views = store.views()
    assert views["main", "kv.a"].fields[0].shape == (12, 2, 64)
    assert views["main", "kv.b"].fields[0].shape == (12, 4, 128)
    shared = manager.get_state("main", "gqa", 2)
    assert shared.get_layer_view(0)[0].data_ptr() == shared.get_layer_view(2)[0].data_ptr()
    assert {view.state.producer for view in views.values()} == {0, 1}


def test_workspace_is_reserved_once_for_serial_layers_and_separately_for_lanes():
    one, _ = make_model(LlamaTpPartModel)
    two, _ = make_model(LlamaTpPartModel, microbatches=2)
    one._init_execution_plan()
    two._init_execution_plan()
    assert two.execution_plan.workspace_bytes == one.execution_plan.workspace_bytes * 2
    first, second = [layer.bound_decode_attention for layer in two.layers_infer]
    assert first.workspace is second.workspace
    assert (
        lane_workspace(first.workspace, first.limits, 0).data_ptr()
        != lane_workspace(first.workspace, first.limits, 1).data_ptr()
    )
    assert (
        lane_workspace(first.workspace, first.limits, 0).numel()
        + lane_workspace(first.workspace, first.limits, 1).numel()
        == first.workspace.numel()
    )
    with pytest.raises(ValueError, match="execution lanes"):
        lane_workspace(first.workspace, first.limits, 2)


def test_failed_selection_never_allocates_workspace_or_state():
    backend = DenseAttentionBackend("triton", "decode")
    spec = DenseAttentionSpec(4, 2, 64)
    state = StateSpec("main", "kv.0", 0, (0,), DenseKVStateSpec(kv_heads=2, head_dim=64))
    format = FormatSpec("plain", torch.float16, "token_head_dim", ("k", "v"), 1)
    planner = Planner((backend,), DeviceCaps("cuda", 0, (9, 0)), ExecutionLimits(1))
    with pytest.raises(ValueError, match="workspace budget"):
        planner.prepare([AttentionRequest(0, "decode", spec, state, format, DenseStaticParams(spec, torch.empty))])


def test_conflicting_resource_formats_are_rejected_before_binding():
    backend = DenseAttentionBackend("triton", "prefill")
    spec = DenseAttentionSpec(4, 2, 64)
    state = StateSpec("main", "kv.0", 0, (0, 1), DenseKVStateSpec(kv_heads=2, head_dim=64))
    format = FormatSpec("plain", torch.float16, "token_head_dim", ("k", "v"), 1)
    planner = Planner((backend,), DeviceCaps("cuda", 0, (9, 0)), ExecutionLimits(0))
    with pytest.raises(ValueError, match="conflicting"):
        planner.prepare(
            [
                AttentionRequest(i, "prefill", spec, state, f, DenseStaticParams(spec, torch.empty))
                for i, f in enumerate((format, replace(format, page_size=4)))
            ]
        )


@pytest.mark.parametrize("implementation,expected", [("triton", False), ("fa3", True)])
def test_cpu_cache_honors_bound_backend_sync_requirement(implementation, expected):
    from lightllm.server.router.model_infer.mode_backend.multi_level_kv_cache import MultiLevelKvCacheModule

    model, _ = make_model(LlamaTpPartModel, implementation=implementation)
    model._init_execution_plan()
    cache = MultiLevelKvCacheModule.__new__(MultiLevelKvCacheModule)
    cache.backend = SimpleNamespace(model=model)
    assert cache.need_sync_compute_stream() is expected
    cache.need_sync_compute_stream.cache_clear()
