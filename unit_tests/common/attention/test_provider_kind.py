import pytest
from lightllm.common.attention.api.spec import AttentionFamily
from lightllm.common.attention.selection import load_backend_class


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
