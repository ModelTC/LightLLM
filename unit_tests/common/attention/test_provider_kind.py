import pytest
import torch

from lightllm.common.attention.api.spec import AttentionFamily
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
    ],
)
def test_fp8_sparse_capabilities(monkeypatch, reference):
    from dataclasses import replace

    backend = load_backend_class(reference)
    caps = DeviceCaps(Platform.CUDA, torch.device("cuda:0"), capability=(8, 0))
    assert not backend.check_environment(caps).supported
    caps = replace(caps, capability=(9, 0))
    imported = []

    def import_provider(name):
        imported.append(name)
        raise ImportError("missing FlashMLA")

    monkeypatch.setattr("lightllm.common.attention.backends.base.importlib.import_module", import_provider)
    result = backend.check_environment(caps)
    assert not result.supported
    assert imported == ["flash_mla"]
    assert "missing FlashMLA" in result.reason

    spec = make_spec(AttentionFamily.SPARSE_MLA)
    spec = replace(spec, cache=replace(spec.cache, format="fp8kv_dsa"))
    assert backend.check_support(spec).supported
    assert not backend.check_support(replace(spec, dtype=torch.float32)).supported
