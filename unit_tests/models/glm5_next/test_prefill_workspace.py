import pytest
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


@pytest.mark.parametrize("decode_only", [False, True])
def test_fp8_workspace_reserved_per_microbatch_before_execution(decode_only):
    from types import SimpleNamespace

    from lightllm.common.attention.api import ExecutionRequirements
    from lightllm.common.basemodel.attention import AttentionGroup, reserve_attention_resources
    from lightllm.platform import get_device_runtime

    backend = object.__new__(Fp8Glm5NextSparseAttBackend)
    backend.network_config = {"index_topk": 2048, "index_kpool": 4}
    backend.prefill_token_capacity = 32
    backend.prefill_graph_enabled = True
    backend.decode_only = decode_only
    model = SimpleNamespace(
        attention_execution=ExecutionRequirements(microbatch_count=2, max_batch_size=2, max_seq_len=64),
        attention_groups=(AttentionGroup(backend, backend, None, None),),
        _attention_workspaces={},
        device_runtime=get_device_runtime("cpu"),
    )
    reserve_attention_resources(model)
    first, second = (model._attention_workspaces[(0, slot)] for slot in (0, 1))
    gathered = first.get("glm_gathered_kv", 1, torch.bfloat16)
    assert gathered.numel() == (32 if decode_only else 2 * 64 + 32) * 512
    assert gathered.data_ptr() != second.get("glm_gathered_kv", 1, torch.bfloat16).data_ptr()
    assert first.get("glm_chunk_indices", 1, torch.int32).numel() == 32 * 2176
    context = backend.allocate_context(
        ContextCapacity(AttentionPhase.PREFILL, 2, 4, 64, 4), capture=True, workspace=first
    )
    context.ragged_mem_index = torch.arange(100, dtype=torch.int32)
    q = torch.zeros(4, 64, 512, dtype=torch.bfloat16)
    packed = torch.empty(160, 1, 584, dtype=torch.uint8)
    topk = torch.zeros(4, 2176, dtype=torch.int32)
    buffers = backend.get_generic_prefill_workspace(context, q, packed, topk)
    assert buffers["gathered_kv"].data_ptr() == gathered.data_ptr()
    assert buffers["cu_seqlens_q"].tolist() == [0, 1, 2, 3, 4]
    first.freeze()
    assert (
        backend.get_generic_prefill_workspace(context, q, packed, topk)["gathered_kv"].data_ptr() == gathered.data_ptr()
    )
