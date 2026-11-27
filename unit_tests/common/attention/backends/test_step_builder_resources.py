"""Runtime initialization reads step capacities and shares workspace through the model's manager."""

from types import SimpleNamespace
from unittest.mock import patch
import weakref

import pytest
import torch

from lightllm.common.attention.backends.gqa import GQALayerConfig
from lightllm.common.attention.backends.gqa import fa3 as gqa_fa3
from lightllm.common.attention.backends.gqa.fa3 import FA3GQADecodeBackend, FA3GQAPrefillBackend
from lightllm.common.attention.backends.gqa.flashinfer import FlashInferGQADecodeBackend, FlashInferGQAPrefillBackend
from lightllm.common.attention.backends.mla import MLADecodeLayerConfig, MLAPrefillLayerConfig
from lightllm.common.attention.backends.mla.fa3 import FA3MLADecodeBackend
from lightllm.common.attention.backends.mla.flashinfer import FlashInferMLADecodeBackend, FlashInferMLAPrefillBackend
from lightllm.common.basemodel import attention_backend_manager
from lightllm.common.basemodel.attention_backend_manager import AttentionBackendManager
from unit_tests.common.attention.backends.groups_runner import init_backend
from unit_tests.common.attention.backends.step_fixture import make_infer_state
from lightllm.common.attention.backends.gqa import flashinfer as gqa_flashinfer


def context(monkeypatch, **overrides):
    monkeypatch.setattr(attention_backend_manager, "get_current_device_id", lambda: 0)
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda: (9, 0))
    monkeypatch.setattr(gqa_flashinfer, "get_env_start_args", lambda: SimpleNamespace(flashinfer_workspace_size=512))
    fields = dict(page_size=16, model_max_seq_len=64, max_cuda_graph_batch_size=2, max_decode_rows=2)
    return make_infer_state(b_seq_len=torch.tensor([], dtype=torch.int32), **{**fields, **overrides})


def gqa(**fields) -> GQALayerConfig:
    return GQALayerConfig(num_heads=8, num_kv_heads=2, head_dim=64, dtype=torch.bfloat16, **fields)


MLA_COMMON = dict(num_heads=16, qk_rope_head_dim=64, softmax_scale=0.1, dtype=torch.bfloat16)


def test_flashinfer_gqa_builders_share_one_workspace_and_mla_has_its_own(monkeypatch):
    allocations = []
    current_device_id = 0

    def fake_empty(size, dtype, device):
        workspace = object()
        allocations.append((workspace, size, dtype, device))
        return workspace

    monkeypatch.setattr(attention_backend_manager.torch, "empty", fake_empty)
    ctx = context(monkeypatch)
    # The manager shares workspaces within one device.
    monkeypatch.setattr(attention_backend_manager, "get_current_device_id", lambda: current_device_id)

    fp = init_backend(FlashInferGQAPrefillBackend, gqa(), ctx).workspace
    fp8 = init_backend(FlashInferGQAPrefillBackend, gqa(kv_quant="fp8_per_tensor"), ctx).workspace
    mla_config = MLAPrefillLayerConfig(qk_nope_head_dim=128, v_head_dim=128, **MLA_COMMON)
    mla = init_backend(FlashInferMLAPrefillBackend, mla_config, ctx).workspace

    assert fp8 is fp and mla is not fp
    assert [allocation[1] for allocation in allocations] == [512 * 1024 * 1024, 256 * 1024 * 1024]
    current_device_id = 1
    assert init_backend(FlashInferGQAPrefillBackend, gqa(), ctx).workspace is not fp
    assert allocations[-1][3] == 1


def test_workspace_requests_of_one_name_share_the_largest_storage(monkeypatch):
    monkeypatch.setattr(attention_backend_manager, "get_current_device_id", lambda: 0)
    monkeypatch.setattr(
        attention_backend_manager.torch, "empty", lambda size, dtype, device: torch.zeros(size, dtype=dtype)
    )
    manager = AttentionBackendManager()
    get = manager.get_workspace

    large = get("shared", 2048)
    assert get("shared", 2048) is large
    # A smaller request, as a draft model's, is a prefix view of the existing storage.
    small = get("shared", 1024)
    assert small.data_ptr() == large.data_ptr() and small.numel() == 1024
    assert get("shared", 1024) is small
    # A larger request cannot reuse it, and other names and dtypes never share.
    assert get("shared", 4096).data_ptr() != large.data_ptr()
    assert get("shared", 1024, dtype=torch.float16).data_ptr() != large.data_ptr()
    assert get("other", 1024).data_ptr() != large.data_ptr()
    assert AttentionBackendManager().get_workspace("shared", 2048).data_ptr() != large.data_ptr()
    assert get("shared", 2048) is large
    assert get("shared", 1024) is small


@pytest.mark.parametrize("backend", [FA3GQAPrefillBackend, FA3GQADecodeBackend])
def test_fa3_gqa_pages_like_the_cache_except_fp8_which_uses_single_token_pages(monkeypatch, backend):
    ctx = context(monkeypatch)
    assert init_backend(backend, gqa(), ctx).page_size == 16
    with patch.object(gqa_fa3.logger, "warning") as warning:
        assert init_backend(backend, gqa(kv_quant="fp8_per_head"), ctx).page_size == 1
    warning.assert_called_once()


def test_flashinfer_pages_like_the_cache_except_fp8_which_uses_single_token_pages(monkeypatch):
    ctx = context(monkeypatch)
    assert init_backend(FlashInferGQADecodeBackend, gqa(), ctx).page_size == 16
    assert init_backend(FlashInferGQADecodeBackend, gqa(kv_quant="fp8_per_tensor"), ctx).page_size == 1


def test_mla_decode_pages_like_the_cache(monkeypatch):
    ctx = context(monkeypatch)
    config = MLADecodeLayerConfig(kv_lora_rank=512, **MLA_COMMON)
    assert init_backend(FA3MLADecodeBackend, config, ctx).page_size == 16
    assert init_backend(FlashInferMLADecodeBackend, config, ctx).page_size == 16


@pytest.mark.parametrize("backend_cls", [FA3GQADecodeBackend, FlashInferGQADecodeBackend])
def test_init_runtime_does_not_retain_the_step(monkeypatch, backend_cls):
    step = context(monkeypatch)
    monkeypatch.setattr(step.attention_backend_manager, "get_workspace", lambda *args: torch.empty(1))
    step_ref = weakref.ref(step)
    backend = init_backend(backend_cls, gqa(), step)
    del step
    assert step_ref() is None
    assert backend.page_size == 16


@pytest.mark.parametrize("workspace_mb", [0, -1])
def test_flashinfer_rejects_invalid_workspace_size_before_allocation(monkeypatch, workspace_mb):
    step = context(monkeypatch)
    monkeypatch.setattr(
        gqa_flashinfer, "get_env_start_args", lambda: SimpleNamespace(flashinfer_workspace_size=workspace_mb)
    )
    with patch.object(step.attention_backend_manager, "get_workspace") as allocate:
        with pytest.raises(ValueError, match="must be positive"):
            init_backend(FlashInferGQAPrefillBackend, gqa(), step)
    allocate.assert_not_called()
