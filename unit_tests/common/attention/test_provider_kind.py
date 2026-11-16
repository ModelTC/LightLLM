import pytest
import torch

from lightllm.common.attention.api import AttentionFamily, UnsupportedAttention
from lightllm.common.attention.registry import load_backend_class
from lightllm.platform import DeviceCaps, Platform
from unit_tests.common.attention.helpers import make_spec


@pytest.mark.parametrize(
    "reference,family",
    [
        ("backends.dense.fa3.backend:Fa3DenseDecodeBackend", AttentionFamily.DENSE),
        ("backends.mla.fa3.backend:Fa3MLADecodeBackend", AttentionFamily.MLA),
        ("backends.sparse_mla.flashmla.backend:FlashMLASparseMLADecodeBackend", AttentionFamily.SPARSE_MLA),
        ("backends.sparse_mla.flashmla.fp8:FlashMLASparseMLAFp8DecodeBackend", AttentionFamily.SPARSE_MLA),
        ("backends.gdn.triton.backend:TritonGDNDecodeBackend", AttentionFamily.GDN),
        ("backends.gdn.flashqla.backend:FlashQlaGDNPrefillBackend", AttentionFamily.GDN),
        ("backends.kda.triton.backend:TritonKDADecodeBackend", AttentionFamily.KDA),
    ],
)
def test_family_is_explicit_and_survives_adapter_relocation(reference, family):
    backend = load_backend_class(reference)
    relocated = type("ExternalAdapter", (backend,), {"__module__": "external.extension"})
    assert relocated.family is family


@pytest.mark.parametrize(
    "reference",
    [
        "backends.sparse_mla.flashmla.fp8:FlashMLASparseMLAFp8DecodeBackend",
        "lightllm.models.glm5_next.attention.fp8_sparse_mla:GlmSparseMLAFp8DecodeBackend",
        "lightllm.models.deepseek_v4.attention.sparse_mla:DeepseekV4SparseMLAFp8DecodeBackend",
    ],
)
def test_fp8_sparse_capabilities(monkeypatch, reference):
    from dataclasses import replace

    backend = load_backend_class(reference)
    caps = DeviceCaps(Platform.CUDA, torch.device("cuda:0"), capability=(8, 0))
    with pytest.raises(UnsupportedAttention, match="device capability"):
        backend.check_environment(caps)
    caps = replace(caps, capability=(9, 0))
    imported = []

    def import_provider(name):
        imported.append(name)
        raise ImportError("missing FlashMLA")

    monkeypatch.setattr("lightllm.common.attention._environment.importlib.import_module", import_provider)
    with pytest.raises(UnsupportedAttention, match="missing FlashMLA"):
        backend.check_environment(caps)
    assert imported == ["flash_mla"]

    spec = make_spec(AttentionFamily.SPARSE_MLA, variant=backend.variant)
    spec = replace(spec, cache=replace(spec.cache, encoding="fp8kv_dsa"))
    assert backend.check_support(spec) is None
    with pytest.raises(UnsupportedAttention, match="activation dtype"):
        backend.check_support(replace(spec, dtype=torch.float32))


@pytest.mark.parametrize(
    "reference",
    ["backends.dense.fa3.backend:Fa3DenseDecodeBackend", "backends.dense.fa3.varlen:Fa3VarlenAttentionOp"],
)
def test_environment_rejection_precedes_provider_import(monkeypatch, reference):
    from dataclasses import replace

    implementation = load_backend_class(reference)
    imported = []

    def import_provider(name):
        imported.append(name)
        raise OSError("provider library cannot load")

    monkeypatch.setattr("lightllm.common.attention._environment.importlib.import_module", import_provider)
    caps = DeviceCaps(Platform.CPU, torch.device("cpu"))
    with pytest.raises(UnsupportedAttention, match="platform cpu"):
        implementation.check_environment(caps)
    caps = DeviceCaps(Platform.CUDA, torch.device("cuda:0"), available=False, capability=(9, 0))
    with pytest.raises(UnsupportedAttention, match="unavailable"):
        implementation.check_environment(caps)
    with pytest.raises(UnsupportedAttention, match="device capability"):
        implementation.check_environment(replace(caps, available=True, capability=(8, 0)))
    assert imported == []
    with pytest.raises(UnsupportedAttention, match="provider library cannot load"):
        implementation.check_environment(replace(caps, available=True))
    assert imported == ["sgl_kernel.flash_attn"]


def test_selection_imports_only_backend_definitions():
    import subprocess
    import sys

    script = """
import importlib.abc
import sys
import torch

blocked = {"triton", "sgl_kernel", "flashinfer", "flash_mla", "flash_qla", "flash_attn_interface"}
class DefinitionsOnly(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if (fullname.split(".")[0] in blocked or ".kernels." in fullname
                or fullname.startswith("lightllm.common.basemodel")):
            raise AssertionError("selection imported execution dependency: " + fullname)

sys.meta_path.insert(0, DefinitionsOnly())
from lightllm.common.attention import selection
from lightllm.common.attention.api import (
    AttentionFamily, AttentionPhase, CacheEncoding, CacheLayout, LayerSpec, ServingConstraints
)
from lightllm.common.attention.backends.dense.triton.backend import TritonDenseDecodeBackend
from lightllm.platform import DeviceCaps, Platform
from lightllm.utils.backend_validator import ProbeResult, ProbeScope

TritonDenseDecodeBackend.check_environment = classmethod(lambda cls, caps: None)
selection.backend_validator.probe_backend = lambda *a, **kw: ProbeResult(True, ProbeScope.BACKEND_NUMERICS)
spec = LayerSpec(AttentionFamily.DENSE, AttentionPhase.DECODE, torch.bfloat16,
    CacheLayout(CacheEncoding.PLAIN, 1, torch.bfloat16, 2, 64, 64), 4, 64, 64)
assert selection.select_attention_backend(specs=(spec,), caps=DeviceCaps(Platform.CPU, torch.device("cpu")),
    execution=ServingConstraints(), requested="triton") is TritonDenseDecodeBackend
assert not torch.cuda.is_initialized()
"""
    result = subprocess.run([sys.executable, "-c", script], capture_output=True, text=True, timeout=30)
    assert result.returncode == 0, result.stdout + result.stderr
    assert "Neo FA3 is unavailable" not in result.stderr


def test_gdn_decode_constructor_does_not_load_prefill():
    import subprocess
    import sys

    script = """
import importlib.abc
import sys
import torch

class NoExecutionImports(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split(".")[0] in {"triton", "flash_qla"} or ".kernels." in fullname:
            raise AssertionError("decode construction imported execution code: " + fullname)

sys.meta_path.insert(0, NoExecutionImports())
from lightllm.common.attention.api import BackendEnv, AttentionFamily, AttentionPhase, CacheLayout, LayerSpec
from lightllm.common.attention.backends.gdn.triton.backend import TritonGDNDecodeBackend
from lightllm.common.attention.backends.gdn.replay.backend import TritonGDNReplayDecodeBackend
from lightllm.platform import DeviceRuntime

from lightllm.common.attention.api.spec import GDNConfig

env = BackendEnv(
    runtime=DeviceRuntime("cpu"), hold_req_id=7, draft_step=0, max_seq_len=64, tp_size=1, alloc_tensor=torch.empty
)
for cls in (TritonGDNDecodeBackend, TritonGDNReplayDecodeBackend):
    cache = CacheLayout("plain", 1, torch.bfloat16, 2, 8, 16,
        state_mode=cls.state_mode, state_dtype=torch.float32)
    spec = LayerSpec(AttentionFamily.GDN, AttentionPhase.DECODE, torch.bfloat16, cache, 2, 8, 16,
        config=GDNConfig(4, 4, "silu"))
    backend = cls(env, spec)
    assert backend.spec is spec
assert not torch.cuda.is_initialized()
"""
    result = subprocess.run([sys.executable, "-c", script], capture_output=True, text=True, timeout=30)
    assert result.returncode == 0, result.stdout + result.stderr


@pytest.mark.parametrize("bits", [4, 8])
def test_quantized_prefill_uses_layout_group(monkeypatch, bits):
    from importlib import import_module
    from types import SimpleNamespace

    from lightllm.common.attention.api import AttentionPhase, BackendEnv, CacheLayout
    from lightllm.common.attention.api.inputs import DenseInputs, DenseResources
    from lightllm.platform import DeviceRuntime
    from unit_tests.common.attention.helpers import make_batch, make_context, make_layer_attention, run_attention

    cls = load_backend_class(f"backends.dense.triton.int{bits}:TritonDenseInt{bits}PrefillBackend")
    monkeypatch.setattr(
        "lightllm.utils.envs_utils.get_env_start_args", lambda: SimpleNamespace(llm_kv_quant_group_size=128)
    )
    quantization = import_module(f"lightllm.common.basemodel.triton_kernel.kv_copy.ppl_int{bits}kv_copy_kv")
    prefill = import_module("lightllm.common.attention.backends._shared.prefill.context_flashattention_nopad")
    groups = []
    monkeypatch.setattr(
        quantization, f"dequantize_int{bits}kv", lambda **kwargs: groups.append(kwargs["quant_group_size"])
    )
    monkeypatch.setattr(prefill, "context_attention_fwd_contiguous_kv", lambda **kwargs: kwargs["o"].zero_())
    env = BackendEnv(
        runtime=DeviceRuntime("cpu"),
        hold_req_id=7,
        draft_step=0,
        max_seq_len=64,
        tp_size=1,
        alloc_tensor=torch.empty,
    )
    for group_size in (16, 32):
        spec = make_spec(
            phase=AttentionPhase.PREFILL,
            cache=CacheLayout(f"int{bits}kv", 1, torch.int8, 1, 64, 64, quant_group_size=group_size),
            num_q_heads=1,
            head_size=64,
            value_head_size=64,
        )
        backend = cls(env, spec)
        batch = make_batch((1,), phase=AttentionPhase.PREFILL)
        context = make_context(backend, batch)
        context.b_kv_start_loc = torch.zeros(1, dtype=torch.int32)
        cache = torch.empty(1, 1, 32 if bits == 4 else 64, dtype=torch.int8)
        scale = torch.ones(1, 1, 64 // group_size)
        layer = make_layer_attention(backend, context, DenseResources(cache, cache, key_scale=scale, value_scale=scale))
        output = run_attention(layer, context, DenseInputs(torch.ones(1, 1, 64)))
        assert output.eq(0).all()
    assert groups == [16, 32]


def test_neo_selection_fallback(monkeypatch):
    from dataclasses import replace

    from lightllm.common.attention import selection
    from lightllm.common.attention.api import AttentionPhase, ServingConstraints
    from lightllm.common.attention.backends.dense.fa3.backend import Fa3DensePrefillBackend
    from lightllm.common.attention.backends.dense.fa3 import neo
    from lightllm.common.attention.backends.dense.triton.neo import TritonDenseNeoPrefillBackend
    from lightllm.utils.backend_validator import ProbeResult, ProbeScope

    def missing():
        raise UnsupportedAttention("Neo extension missing")

    monkeypatch.setattr(neo, "get_neo_attention", missing)
    monkeypatch.setattr(Fa3DensePrefillBackend, "check_environment", classmethod(lambda cls, caps: None))
    monkeypatch.setattr(TritonDenseNeoPrefillBackend, "check_environment", classmethod(lambda cls, caps: None))
    monkeypatch.setattr(
        selection.backend_validator, "probe_backend", lambda *a, **kw: ProbeResult(True, ProbeScope.PROVIDER_SMOKE)
    )
    args = dict(
        specs=(replace(make_spec(phase=AttentionPhase.PREFILL), variant="neo"),),
        caps=DeviceCaps(Platform.CUDA, torch.device("cuda:0"), capability=(9, 0)),
        execution=ServingConstraints(),
        policy=selection.BackendPolicy(prefer=("fa3", "triton")),
    )
    assert selection.select_attention_backend(**args) is TritonDenseNeoPrefillBackend
    with pytest.raises(selection.BackendSelectionError, match="Neo extension missing"):
        selection.select_attention_backend(**args, requested="fa3")
