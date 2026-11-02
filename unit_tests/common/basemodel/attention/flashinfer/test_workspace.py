from types import SimpleNamespace

import pytest
import torch

from lightllm.common.basemodel.attention import base_att
from lightllm.common.basemodel.attention.base_att import BaseAttBackend
from lightllm.common.basemodel.attention.flashinfer.fp import FlashInferAttBackend
from lightllm.common.basemodel.attention.flashinfer.fp8 import Fp8FlashInferAttBackend
from lightllm.common.basemodel.attention.flashinfer.mla import MlaFlashInferAttBackend
from lightllm.common.basemodel.attention.fa3.fp import Fa3AttBackend
from lightllm.common.basemodel.attention.nsa.flashmla_sparse import NsaFlashMlaSparseAttBackend
from lightllm.common.basemodel.attention.nsa.fp8_flashmla_sparse import NsaFlashMlaFp8SparseAttBackend
from lightllm.server.core.objs.start_args_type import StartArgs


def test_workspace_is_shared_by_backend_family_and_device(monkeypatch):
    allocations = []
    current_device_id = 0

    def fake_empty(size, dtype, device):
        workspace = object()
        allocations.append((workspace, size, dtype, device))
        return workspace

    monkeypatch.setattr(base_att, "get_current_device_id", lambda: current_device_id)
    monkeypatch.setattr(base_att.torch, "empty", fake_empty)
    monkeypatch.setattr(BaseAttBackend, "_workspace_buffers", {})

    def get_workspace(backend_cls):
        return backend_cls.get_gpu_workspace_buffer(
            key_name=backend_cls.workspace_buffer_key,
            workspace_size=backend_cls.workspace_buffer_size,
        )

    fp_workspace = get_workspace(FlashInferAttBackend)
    fp8_workspace = get_workspace(Fp8FlashInferAttBackend)
    mla_workspace = get_workspace(MlaFlashInferAttBackend)

    assert fp8_workspace is fp_workspace
    assert mla_workspace is not fp_workspace
    assert [allocation[1] for allocation in allocations] == [
        512 * 1024 * 1024,
        256 * 1024 * 1024,
    ]

    current_device_id = 1
    other_device_workspace = get_workspace(FlashInferAttBackend)

    assert other_device_workspace is not fp_workspace
    assert allocations[-1][3] == 1


def test_workspace_key_includes_size_and_dtype(monkeypatch):
    monkeypatch.setattr(base_att, "get_current_device_id", lambda: 0)
    monkeypatch.setattr(base_att.torch, "empty", lambda *args, **kwargs: object())
    monkeypatch.setattr(BaseAttBackend, "_workspace_buffers", {})

    small_workspace = BaseAttBackend.get_gpu_workspace_buffer("shared", 1024)
    large_workspace = BaseAttBackend.get_gpu_workspace_buffer("shared", 2048)
    fp16_workspace = BaseAttBackend.get_gpu_workspace_buffer("shared", 1024, dtype=base_att.torch.float16)

    assert BaseAttBackend.get_gpu_workspace_buffer("shared", 1024) is small_workspace
    assert large_workspace is not small_workspace
    assert fp16_workspace is not small_workspace


@pytest.mark.parametrize(
    "backend_class",
    [
        Fa3AttBackend,
        FlashInferAttBackend,
        MlaFlashInferAttBackend,
        NsaFlashMlaSparseAttBackend,
        NsaFlashMlaFp8SparseAttBackend,
    ],
)
@pytest.mark.parametrize("overlap", [False, True])
def test_provider_probe_accounts_for_all_fixed_buffers_without_allocating(monkeypatch, backend_class, overlap):
    monkeypatch.setenv("LIGHTLLM_DP_WORLD_SIZE", "1")
    monkeypatch.setattr(base_att, "get_current_device_id", lambda: 0)
    monkeypatch.setattr(BaseAttBackend, "_workspace_buffers", {})
    model = SimpleNamespace(
        config=dict(num_attention_heads=4, num_key_value_heads=2, hidden_size=256, rope_scaling=None),
        args=StartArgs(page_size=2, running_max_req_size=4, mtp_step=2, enable_decode_microbatch_overlap=overlap),
        att_state_role="main",
        graph_max_batch_size=8,
        max_seq_length=128,
        run_mode="normal",
        data_type=torch.bfloat16,
        qk_nope_head_dim=128,
        qk_rope_head_dim=64,
        kv_lora_rank=512,
        v_head_dim=128,
    )
    allocations = []

    def allocate(size, dtype, device):
        allocations.append(size * dtype.itemsize)
        return object()

    monkeypatch.setattr(torch, "empty", allocate)
    backend = object.__new__(backend_class)
    backend_class.__init__(backend, model)
    demands = backend.runtime_memory_demands()
    assert demands and not allocations
    backend.reserve_runtime_resources()
    assert sum(allocations) == sum(d.fixed_bytes for d in demands)
    assert backend.runtime_memory_demands() == ()
    backend.reserve_runtime_resources()
    assert sum(allocations) == sum(d.fixed_bytes for d in demands)
