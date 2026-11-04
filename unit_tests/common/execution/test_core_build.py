import ast
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from lightllm.core import BuildConfig, ProviderRegistry, AttentionProvider, StateProvider, build
from lightllm.core.contracts.device import DeviceCaps, ExecutionLimits
from lightllm.core.contracts.provider import AttentionOffer, ProbeResult
from lightllm.core.contracts.state import FormatSpec, StateSpec, StateView


def _build_with_external_provider(retention):
    events = []
    declaration = StateSpec("offline", "counter", 3, (3,), (1,), retention=retention)
    format = FormatSpec("custom", torch.float32, "counter", ("value",), 1)
    layer = SimpleNamespace(layer_num_=3)
    adapter = SimpleNamespace(attention_specs=lambda: {3: "add-state"}, state_specs=lambda: (declaration,))

    class Store:
        def __init__(self, declarations, format, manager, device, specs):
            assert declarations == [declaration] and specs == {3: "add-state"}
            self.value = None
            events.append("declare")

        def views(self):
            assert self.value is not None
            return {declaration.key: StateView(declaration, format, (self.value,))}

    class Backend:
        implementation = "external"

        def __init__(self, phase):
            self.phase = phase
            self.name = f"external.{phase}"

        def probe(self, spec, format, device, limits):
            assert spec == "add-state" and device.device_type == "cpu"
            return ProbeResult((AttentionOffer(self.name, self.phase, format, 16, execution_limits=limits),))

        def bind(self, offer, view, params, workspace):
            events.append("bind")
            assert len(workspace) == 16
            return SimpleNamespace(
                metadata_key=self.name,
                phase=self.phase,
                view=view,
                run=lambda value: value + view.fields[0],
                create_metadata=lambda batch: batch,
            )

    def allocate_workspace(size):
        events.append("workspace")
        return bytearray(size)

    def allocate_storage(execution):
        events.append("allocate")
        execution.stores[retention].value = torch.tensor([4.0])

    providers = ProviderRegistry(
        (AttentionProvider(Backend),),
        (StateProvider("counter", retention, Store, lambda view: view),),
    )
    engine = build(
        model_adapter=adapter,
        providers=providers,
        device=SimpleNamespace(caps=DeviceCaps("cpu"), allocate_workspace=allocate_workspace),
        config=BuildConfig(
            [layer],
            None,
            ExecutionLimits(32),
            format if retention == "history" else None,
            {"prefill": "external", "decode": "external"} if retention == "history" else {},
            recurrent_format=format if retention == "recurrent" else None,
            recurrent_requested={"prefill": "external", "decode": "external"} if retention == "recurrent" else None,
            static_params={3: None},
            allocate_storage=allocate_storage,
        ),
    )
    assert events == ["workspace", "workspace", "declare", "allocate", "bind", "bind"]
    assert layer.bound_decode_attention.view.fields[0] is engine.stores[retention].value
    assert layer.bound_kv_writer.fields[0] is engine.stores[retention].value
    torch.testing.assert_close(layer.bound_decode_attention.run(torch.tensor([2.0])), torch.tensor([6.0]))
    assert engine.runner.plan is engine.plan
    assert set(engine.stores) == {retention}
    with pytest.raises(RuntimeError, match="cannot be replaced"):
        engine.bind()


def test_build_with_external_provider_and_native_state():
    _build_with_external_provider("history")


def test_recurrent_only_build_does_not_require_history_provider():
    _build_with_external_provider("recurrent")


def test_core_declarations_import_without_backend_or_service():
    script = """
import importlib.abc
import sys
import torch

class Blocked(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        roots = ("lightllm.models", "lightllm.server", "triton", "flashinfer", "sgl_kernel", "flash_attn_interface")
        if any(fullname == root or fullname.startswith(root + ".") for root in roots):
            raise AssertionError(fullname)

sys.meta_path.insert(0, Blocked())
from lightllm.core import build, BuildConfig, ProviderRegistry
from lightllm.core.contracts import ModelAdapter, StateSpec, FormatSpec, StateView
from lightllm.core.compute.attention.dense.spec import DenseAttentionSpec
from lightllm.core.compute.attention.mla.spec import MLAAttentionSpec
from lightllm.core.compute.attention.sparse_mla.spec import SparseMLAAttentionSpec
from lightllm.core.compute.attention.gdn.spec import GatedDeltaNetSpec
from lightllm.core.compute.attention.kda.spec import KimiDeltaAttentionSpec
from lightllm.core.compute.attention.pooled_nope.spec import PooledNoPEAttentionSpec
assert DenseAttentionSpec(4, 2, 64).head_dim == 64
assert not torch.cuda.is_initialized()
"""
    subprocess.run([sys.executable, "-c", script], check=True, capture_output=True, text=True)


def test_core_dependencies_and_compatibility_direction():
    root = Path(__file__).resolve().parents[3] / "lightllm"
    for path in (root / "core").rglob("*.py"):
        for node in ast.walk(ast.parse(path.read_text())):
            modules = []
            if isinstance(node, ast.ImportFrom):
                modules = [node.module or ""]
            elif isinstance(node, ast.Import):
                modules = [alias.name for alias in node.names]
            assert not any(
                module.startswith(("lightllm.models", "lightllm.server", "lightllm.common.execution"))
                for module in modules
            ), path
    for directory in (root / "common/execution", root / "common/transfer"):
        for path in directory.glob("*.py"):
            assert not any(
                isinstance(node, (ast.ClassDef, ast.FunctionDef)) for node in ast.parse(path.read_text()).body
            ), path


def test_legacy_paths_share_implementation_objects():
    from lightllm.common.execution.specs import DenseAttentionSpec as OldSpec
    from lightllm.core.compute.attention.dense.spec import DenseAttentionSpec
    from lightllm.common.execution.runner import Runner as OldRunner
    from lightllm.core.build import PreparedExecution
    from lightllm.common.execution.dense import DenseKVWriter as OldWriter
    from lightllm.core.formats.kv.dense import DenseKVWriter
    from lightllm.models.llama.model import LlamaModelAdapter as OldAdapter
    from lightllm.models.llama.adapter import LlamaModelAdapter
    from lightllm.common.basemodel.cuda_graph import CudaGraph as OldGraph
    from lightllm.core.runner.decode_graph import CudaGraph
    from lightllm.common.basemodel.prefill_cuda_graph import PrefillCudaGraph as OldPrefillGraph
    from lightllm.core.runner.prefill_graph import PrefillCudaGraph

    assert OldSpec is DenseAttentionSpec
    assert OldRunner is PreparedExecution
    assert OldWriter is DenseKVWriter
    assert OldAdapter is LlamaModelAdapter
    assert OldGraph is CudaGraph
    assert OldPrefillGraph is PrefillCudaGraph


def test_same_layout_providers_match_full_format_and_reject_ambiguity():
    from dataclasses import replace
    from lightllm.core.formats.providers import state_providers

    fmt = FormatSpec("custom", torch.float32, "token_head_dim", ("k", "v"), 1)
    custom = StateProvider(
        "token_head_dim",
        "history",
        object,
        supports=lambda value: value.encoding == "custom" and value.dtype == torch.float32 and value.page_size == 1,
    )
    providers = ProviderRegistry(storage=(*state_providers(), custom))
    assert providers.state_provider(fmt, "history") is custom
    assert providers.state_provider(replace(fmt, encoding="plain"), "history") is not custom
    with pytest.raises(ValueError, match="not registered"):
        providers.state_provider(replace(fmt, page_size=2), "history")
    with pytest.raises(ValueError, match="ambiguous"):
        ProviderRegistry(storage=(custom, custom)).state_provider(fmt, "history")


@pytest.mark.parametrize("requested", ["foo", "missing", "foo.prefill"])
def test_production_state_preparation_uses_injected_registry_and_device(requested):
    from argparse import ArgumentParser
    from contextlib import nullcontext
    from lightllm.common.basemodel.basemodel import TpPartBaseModel
    from lightllm.core.compute.attention.dense.spec import DenseAttentionSpec
    from lightllm.core.formats.providers import state_providers
    from lightllm.server.api_cli import add_cli_args

    args = add_cli_args(ArgumentParser()).parse_args(["--llm_decode_att_backend", requested])
    spec = DenseAttentionSpec(1, 1, 2, dtype=torch.float32)
    state = StateSpec("main", "kv", 0, (0,), (1, 2))
    fmt = FormatSpec("custom", torch.float32, "token_head_dim", ("k", "v"), 1)
    events = []

    class Backend:
        implementation = "foo"

        def __init__(self, phase):
            self.phase, self.name = phase, f"foo.{phase}"

        def probe(self, attention, format, device, limits):
            assert attention == spec and format == fmt and device == DeviceCaps("cpu")
            events.append("probe")
            return ProbeResult((AttentionOffer(self.name, self.phase, format, 0, execution_limits=limits),))

        def bind(self, offer, view, params, workspace):
            assert params == (spec, "static")
            events.append("bind")
            return SimpleNamespace(
                metadata_key=self.name,
                phase=self.phase,
                create_metadata=lambda batch: batch,
                run=lambda x: x + view.fields[0],
            )

    class Store:
        def __init__(self, declarations, format, manager, device, specs):
            assert declarations == [state] and manager is model.att_state_manager and device is runtime
            events.append("store")
            self.value = torch.tensor([3.0])

        def views(self):
            return {state.key: StateView(state, fmt, (self.value, self.value))}

    runtime = SimpleNamespace(
        caps=DeviceCaps("cpu"),
        workspace_budget=lambda fraction: 32,
        allocate_workspace=lambda size: pytest.fail("no workspace required"),
    )
    registry = ProviderRegistry(
        attention=(AttentionProvider(Backend, ("decode",)),),
        storage=(
            *state_providers(),
            StateProvider("token_head_dim", "history", Store, lambda view: view, supports=lambda value: value == fmt),
        ),
        static_params=lambda spec, device: (spec, "static"),
    )
    model = TpPartBaseModel.__new__(TpPartBaseModel)
    model.args, model.max_req_num, model.graph_max_batch_size, model.mem_fraction = args, 1, 2, 0.9
    model.max_seq_length, model.batch_max_tokens = 8, None
    model.layers_infer, model.trans_layers_weight = [SimpleNamespace(layer_num_=0)], [object()]
    model.model_adapter = SimpleNamespace(
        attention_phases=("decode",), attention_specs=lambda: {0: spec}, state_specs=lambda: (state,)
    )
    model.execution_providers, model.execution_device, model.execution_formats = registry, runtime, {"history": fmt}
    model.mtp_manager = SimpleNamespace(get_decode_tokens_per_request=lambda draft: 1)
    model.torch_memory_saver = SimpleNamespace(region=lambda **kwargs: nullcontext())
    if requested != "foo":
        with pytest.raises(ValueError, match="decode attention backend is not registered"):
            model.prepare_attention_states()
        assert events == []
        return
    model.prepare_attention_states()
    assert model.execution_runner.providers is registry
    assert model.execution_device is runtime
    model.execution_runner.bind()
    assert model.layers_infer[0].bound_kv_writer.fields[0] is model.execution_runner.store.value
    torch.testing.assert_close(
        model.layers_infer[0].bound_decode_attention.run(torch.tensor([2.0])), torch.tensor([5.0])
    )
    assert events == ["probe", "store", "bind"]


def test_storage_fixture_patches_the_production_lookup():
    from contextlib import nullcontext
    from lightllm.common.basemodel.basemodel import TpPartBaseModel
    from unit_tests.common.att_resources.model_helpers import prepare_model_states

    model = TpPartBaseModel.__new__(TpPartBaseModel)
    model.config, model.args = {"num_hidden_layers": 1}, SimpleNamespace()
    model.model_adapter = object()
    model.max_req_num, model.tp_rank_, model.tp_world_size_ = 1, 0, 1
    model.create_full_att_states = model.create_bounded_att_states = lambda **kwargs: ()
    model.torch_memory_saver = SimpleNamespace(region=lambda **kwargs: nullcontext())
    prepare_model_states(model)
    assert model.execution_runner is None


def test_native_weight_binding_uses_registered_compute_without_loader_objects():
    from lightllm.core.compute.linear.spec import LinearSpec
    from lightllm.core.formats.weights.view import WeightView
    from lightllm.core.planner.weights import bind_compute

    tensor = torch.eye(2)
    view = WeightView(FormatSpec("plain", torch.float32, "output_input", ("weight",), 1), {"weight": tensor})
    calls = []

    class ExternalLinear:
        name = "external.linear"

        def probe(self, spec, format, device):
            calls.append((spec, format, device))
            return ProbeResult((SimpleNamespace(backend=self.name),))

        def bind(self, offer, weight):
            assert weight is tensor
            return SimpleNamespace(run=lambda x: x @ weight)

    providers = ProviderRegistry(linear=(ExternalLinear,))
    spec, device = LinearSpec(2, 2, torch.float32), DeviceCaps("cpu")
    bound = bind_compute(spec, view, device, providers.linear, requested="external.linear")
    torch.testing.assert_close(bound.run(torch.tensor([[1.0, 2.0]])), torch.tensor([[1.0, 2.0]]))
    assert calls == [(spec, view.format, device)]
