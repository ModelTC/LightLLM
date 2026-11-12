import torch

from lightllm.common.attention.api import AttentionPhase, ContextCapacity
from lightllm.common.attention.backends._workspace import AttentionWorkspace
from lightllm.models.glm5_next.attention.fp8_sparse_mla import (
    Fp8Glm5NextSparseAttBackend,
    Fp8Glm5NextSparsePrefillContext,
)


def test_fp8_prefill_context_is_owned_by_the_backend():
    backend = object.__new__(Fp8Glm5NextSparseAttBackend)
    backend.network_config = {"index_topk": 2048, "index_kpool": 4}
    workspace = AttentionWorkspace("cpu")
    context = backend.allocate_context(
        ContextCapacity(AttentionPhase.PREFILL, 2, 4, 32, 4), capture=False, workspace=workspace
    )
    assert isinstance(context, Fp8Glm5NextSparsePrefillContext)
    assert context.backend_identity is backend
    assert context.workspace is workspace


def test_fp8_backend_has_no_legacy_state_factory():
    assert not hasattr(Fp8Glm5NextSparseAttBackend, "create_att_prefill_state")
    assert not hasattr(Fp8Glm5NextSparseAttBackend, "create_att_decode_state")
