from dataclasses import dataclass
from lightllm.common.execution.formats.kv.spec import DenseKVStateSpec
import ast
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from lightllm.common.execution import BuildConfig, ProviderRegistry, AttentionProvider, StateProvider, build, prepare
from lightllm.common.execution.contracts.device import DeviceCaps, ExecutionLimits
from lightllm.common.execution.contracts.provider import AttentionOffer, ProbeResult
from lightllm.common.execution.contracts.state import FormatSpec, StateSpec, StateView, LayerStates


@dataclass(frozen=True)
class CounterState:
    size: int = 1

    def validate(self):
        assert self.size > 0


@dataclass(frozen=True)
class CounterAttention:
    def validate_state(self, request):
        assert isinstance(request.state.params, CounterState)


def _build_with_external_provider(retention):
    events = []
    declaration = StateSpec("offline", "counter", 3, (3,), CounterState(), retention=retention)
    format = FormatSpec("custom", torch.float32, "counter", ("value",), 1)
    layer = SimpleNamespace(layer_num_=3)
    adapter = SimpleNamespace(
        attention_specs=lambda: {3: CounterAttention()},
        state_specs=lambda: (declaration,),
        attention_states=lambda: {3: LayerStates(declaration)},
    )

    class Store:
        def __init__(self, declarations, format, manager, device, specs):
            assert declarations == (declaration,) and specs == {3: CounterAttention()}
            self.value = None
            events.append("declare")

        def views(self):
            assert self.value is not None
            return {declaration.key: StateView(declaration, format, (self.value,))}

    class Backend:
        supports_auxiliary_states = False
        implementation = "external"

        def __init__(self, phase):
            self.phase = phase
            self.name = f"external.{phase}"

        def probe(self, spec, format, device, limits, *, auxiliary_formats=None):
            assert spec == CounterAttention() and device.device_type == "cpu"
            return ProbeResult((AttentionOffer(self.name, self.phase, format, 16, execution_limits=limits),))

        def bind(self, offer, view, params, workspace, *, auxiliary_views=None):
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

    providers = ProviderRegistry(
        (AttentionProvider(Backend),),
        (StateProvider("counter", retention, Store, lambda view: view),),
    )
    prepared = prepare(
        model_adapter=adapter,
        providers=providers,
        device=SimpleNamespace(caps=DeviceCaps("cpu")),
        config=BuildConfig(
            limits=ExecutionLimits(32),
            formats={retention: format},
            requested={retention: {"prefill": "external", "decode": "external"}},
            static_params={3: SimpleNamespace(spec=CounterAttention())},
        ),
    )
    assert events == []
    resources = prepared.allocate(manager=None, device=SimpleNamespace(allocate_workspace=allocate_workspace))
    assert events == ["workspace", "workspace", "declare"]
    resources.stores[retention].value = torch.tensor([4.0])
    engine = resources.bind()
    execution = engine.plan.bindings[1].execution
    assert not hasattr(layer, "bound_decode_attention")
    assert execution.view.fields[0] is resources.stores[retention].value
    assert engine.writers[declaration.key].fields[0] is resources.stores[retention].value
    torch.testing.assert_close(execution.run(torch.tensor([2.0])), torch.tensor([6.0]))
    assert engine.metadata.plan is engine.plan
    assert set(resources.stores) == {retention}
    assert events == ["workspace", "workspace", "declare", "bind", "bind"]
    with pytest.raises(RuntimeError, match="cannot be replaced"):
        resources.bind()


def test_build_with_external_provider_and_native_state():
    _build_with_external_provider("history")


def test_recurrent_only_build_does_not_require_history_provider():
    _build_with_external_provider("recurrent")


def test_common_execution_declarations_import_without_backend_or_service():
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
from lightllm.common.execution import build, BuildConfig, ProviderRegistry
from lightllm.common.execution.contracts import ModelAdapter, StateSpec, FormatSpec, StateView
from lightllm.common.execution.compute.attention.dense.spec import DenseAttentionSpec
from lightllm.common.execution.compute.attention.mla.spec import MLAAttentionSpec
from lightllm.common.execution.compute.attention.sparse_mla.spec import SparseMLAAttentionSpec
from lightllm.common.execution.compute.attention.gdn.spec import GatedDeltaNetSpec
from lightllm.common.execution.compute.attention.kda.spec import KimiDeltaAttentionSpec
from lightllm.common.execution.compute.attention.pooled_nope.spec import PooledNoPEAttentionSpec
assert DenseAttentionSpec(4, 2, 64).head_dim == 64
assert not torch.cuda.is_initialized()
"""
    subprocess.run([sys.executable, "-c", script], check=True, capture_output=True, text=True)


def test_common_execution_dependencies_and_layout():
    root = Path(__file__).resolve().parents[3] / "lightllm"
    assert not (root / "core").exists()
    for path in (root / "common/execution").rglob("*.py"):
        for node in ast.walk(ast.parse(path.read_text())):
            modules = []
            if isinstance(node, ast.ImportFrom):
                modules = [node.module or ""]
            elif isinstance(node, ast.Import):
                modules = [alias.name for alias in node.names]
            assert not any(module.startswith(("lightllm.models", "lightllm.server")) for module in modules), path


def test_existing_graph_paths_share_implementation_objects():
    from lightllm.common.basemodel.cuda_graph import CudaGraph as OldGraph
    from lightllm.common.execution.runner.decode_graph import CudaGraph
    from lightllm.common.basemodel.prefill_cuda_graph import PrefillCudaGraph as OldPrefillGraph
    from lightllm.common.execution.runner.prefill_graph import PrefillCudaGraph

    assert OldGraph is CudaGraph
    assert OldPrefillGraph is PrefillCudaGraph


def test_same_layout_providers_match_full_format_and_reject_ambiguity():
    from dataclasses import replace
    from lightllm.common.execution.formats.providers import state_providers

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
    from lightllm.common.execution.compute.attention.dense.spec import DenseAttentionSpec
    from lightllm.common.execution.formats.providers import state_providers
    from lightllm.server.api_cli import add_cli_args

    args = add_cli_args(ArgumentParser()).parse_args(["--llm_decode_att_backend", requested])
    spec = DenseAttentionSpec(1, 1, 2, dtype=torch.float32)
    state = StateSpec("main", "kv", 0, (0,), DenseKVStateSpec(kv_heads=1, head_dim=2))
    fmt = FormatSpec("custom", torch.float32, "token_head_dim", ("k", "v"), 1)
    events = []

    class Backend:
        supports_auxiliary_states = False
        implementation = "foo"

        def __init__(self, phase):
            self.phase, self.name = phase, f"foo.{phase}"

        def probe(self, attention, format, device, limits, *, auxiliary_formats=None):
            assert attention == spec and format == fmt and device == DeviceCaps("cpu")
            events.append("probe")
            return ProbeResult((AttentionOffer(self.name, self.phase, format, 0, execution_limits=limits),))

        def bind(self, offer, view, params, workspace, *, auxiliary_views=None):
            assert params.spec == spec
            events.append("bind")
            return SimpleNamespace(
                metadata_key=self.name,
                phase=self.phase,
                create_metadata=lambda batch: batch,
                run=lambda x: x + view.fields[0],
            )

    class Store:
        def __init__(self, declarations, format, manager, device, specs):
            assert declarations == (state,) and manager is model.att_state_manager and device is runtime
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
    )
    model = TpPartBaseModel.__new__(TpPartBaseModel)
    model.args, model.max_req_num, model.graph_max_batch_size, model.mem_fraction = args, 1, 2, 0.9
    model.max_seq_length, model.batch_max_tokens = 8, None
    model.tp_world_size_ = 1
    model.layers_infer, model.trans_layers_weight = [SimpleNamespace(layer_num_=0)], [object()]
    model.model_adapter = SimpleNamespace(
        attention_phases=("decode",),
        attention_specs=lambda: {0: spec},
        state_specs=lambda: (state,),
        attention_states=lambda: {0: LayerStates(state)},
        attention_params=lambda weights, device: {0: SimpleNamespace(spec=spec)},
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
    assert model.execution_providers is registry
    assert model.execution_device is runtime
    model._init_execution_plan()
    assert (
        model.layers_infer[0].bound_kv_writer.fields[0]
        is model.attention_execution.resources.stores.get("history").value
    )
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
    assert model.attention_execution is None


def test_native_weight_binding_uses_registered_compute_without_loader_objects():
    from lightllm.common.execution.compute.linear.spec import LinearSpec
    from lightllm.common.execution.formats.weights.view import WeightView
    from lightllm.common.execution.planner.weights import bind_compute

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


def test_named_state_inputs_share_history_and_keep_private_windows():
    from lightllm.common.execution.contracts.state import LayerStates

    kv = StateSpec("main", "kv", 0, (0, 1), CounterState())
    index = StateSpec("main", "index", 0, (0, 1), CounterState())
    windows = tuple(StateSpec("main", f"window{i}", i, (i,), CounterState(), "window") for i in range(2))
    declarations = (kv, index, *windows)
    inputs = {i: LayerStates(kv, (("index", index), ("window", windows[i]))) for i in range(2)}
    adapter = SimpleNamespace(
        attention_specs=lambda: {0: CounterAttention(), 1: CounterAttention()},
        state_specs=lambda: declarations,
        attention_states=lambda: inputs,
    )
    fmt = FormatSpec("plain", torch.float32, "value", ("value",), 1)

    class Store:
        def __init__(self, declarations, format, *args):
            self.native = {s.key: StateView(s, format, (torch.zeros(1),)) for s in declarations}

        def views(self):
            return self.native

    class Backend:
        implementation = "shared"
        name, phase, supports_auxiliary_states = "shared.decode", "decode", True

        def probe(self, spec, format, caps, limits, *, auxiliary_formats):
            assert set(auxiliary_formats) in ({"index", "window"}, {"kv", "index"})
            assert all(value == fmt for value in auxiliary_formats.values())
            return ProbeResult((AttentionOffer(self.name, self.phase, fmt, execution_limits=limits),))

        def bind(self, offer, view, params, workspace, *, auxiliary_views):
            return SimpleNamespace(metadata_key=self.name, phase=self.phase, view=view, auxiliary=auxiliary_views)

    providers = ProviderRegistry(
        attention=(AttentionProvider(lambda phase: Backend(), ("decode",)),),
        storage=tuple(StateProvider("value", retention, Store) for retention in ("history", "window")),
    )

    def build_engine():
        return build(
            model_adapter=adapter,
            providers=providers,
            device=SimpleNamespace(caps=DeviceCaps("cpu"), allocate_workspace=bytearray),
            config=BuildConfig(
                limits=ExecutionLimits(0),
                formats={"history": fmt, "window": fmt},
                requested={"history": {"decode": "auto"}, "window": {"decode": "auto"}},
                static_params={i: SimpleNamespace(spec=CounterAttention()) for i in range(2)},
            ),
        )

    engine = build_engine()
    first, second = [b.execution for b in engine.plan.bindings]
    first.view.fields[0].fill_(7)
    first.auxiliary["index"].fields[0].fill_(3)
    first.auxiliary["window"].fields[0].fill_(1)
    assert second.view.fields[0].item() == 7
    assert second.auxiliary["index"].fields[0].item() == 3
    assert second.auxiliary["window"].fields[0].item() == 0
    inputs.update({i: LayerStates(windows[i], (("kv", kv), ("index", index))) for i in range(2)})
    window_engine = build_engine()
    first, second = [b.execution for b in window_engine.plan.bindings]
    assert first.view.fields[0] is not second.view.fields[0]
    assert first.auxiliary["kv"].fields[0] is second.auxiliary["kv"].fields[0]
    inputs[1] = LayerStates(kv, (("index", StateSpec("main", "missing", 1, (1,), CounterState())),))
    with pytest.raises(ValueError, match="undeclared state"):
        build_engine()
