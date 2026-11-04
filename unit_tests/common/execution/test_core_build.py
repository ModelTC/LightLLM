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


def test_build_with_external_provider_and_native_state():
    events = []
    declaration = StateSpec("offline", "counter", 3, (3,), (1,))
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
        execution.store.value = torch.tensor([4.0])

    providers = ProviderRegistry(
        (AttentionProvider(Backend),),
        (StateProvider("counter", "history", Store, lambda view: view),),
    )
    engine = build(
        model_adapter=adapter,
        providers=providers,
        device=SimpleNamespace(caps=DeviceCaps("cpu"), allocate_workspace=allocate_workspace),
        config=BuildConfig(
            [layer],
            None,
            ExecutionLimits(32),
            format,
            {"prefill": "external", "decode": "external"},
            static_params={3: None},
            allocate_storage=allocate_storage,
        ),
    )
    assert events == ["workspace", "workspace", "declare", "allocate", "bind", "bind"]
    assert layer.bound_decode_attention.view.fields[0] is engine.store.value
    assert layer.bound_kv_writer.fields[0] is engine.store.value
    torch.testing.assert_close(layer.bound_decode_attention.run(torch.tensor([2.0])), torch.tensor([6.0]))
    assert engine.runner.plan is engine.plan
    with pytest.raises(RuntimeError, match="cannot be replaced"):
        engine.bind()


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
