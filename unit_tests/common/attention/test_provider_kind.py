import pytest
import torch

from lightllm.common.attention.api import AttentionFamily, UnsupportedAttention
from lightllm.common.attention.selection import load_backend_class
from lightllm.platform import DeviceCaps, Platform
from unit_tests.common.attention.helpers import make_spec


@pytest.mark.parametrize(
    "reference,family",
    [
        ("backends.dense.fa3.backend:Fa3AttBackend", AttentionFamily.DENSE),
        ("backends.mla.fa3.backend:MlaFa3AttBackend", AttentionFamily.MLA),
        ("backends.sparse_mla.flashmla.backend:NsaFlashMlaSparseAttBackend", AttentionFamily.SPARSE_MLA),
        ("backends.sparse_mla.flashmla.fp8:NsaFlashMlaFp8SparseAttBackend", AttentionFamily.SPARSE_MLA),
        ("backends.gdn.triton.backend:TritonLinearAttBackend", AttentionFamily.GDN),
        ("backends.gdn.flashqla.backend:FlashQlaLinearAttBackend", AttentionFamily.GDN),
        ("backends.kda.triton.backend:KDALinearAttBackend", AttentionFamily.KDA),
    ],
)
def test_family_is_explicit_and_survives_adapter_relocation(reference, family):
    backend = load_backend_class(reference)
    relocated = type("ExternalAdapter", (backend,), {"__module__": "external.extension"})
    assert relocated.family is family


@pytest.mark.parametrize(
    "reference",
    [
        "backends.sparse_mla.flashmla.fp8:NsaFlashMlaFp8SparseAttBackend",
        "lightllm.models.glm5_next.attention.fp8_sparse_mla:Fp8Glm5NextSparseAttBackend",
        "lightllm.models.deepseek_v4.attention.sparse_mla:DeepseekV4FlashMlaFp8SparseAttBackend",
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

    spec = make_spec(AttentionFamily.SPARSE_MLA)
    spec = replace(spec, cache=replace(spec.cache, format="fp8kv_dsa"))
    assert backend.check_support(spec) is None
    with pytest.raises(UnsupportedAttention, match="activation dtype"):
        backend.check_support(replace(spec, dtype=torch.float32))


@pytest.mark.parametrize(
    "reference",
    ["backends.dense.fa3.backend:Fa3AttBackend", "backends.dense.fa3.varlen:Fa3VarlenAttentionOp"],
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
