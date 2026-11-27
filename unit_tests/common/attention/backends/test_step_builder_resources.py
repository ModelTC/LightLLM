"""Backends take their page sizes and workspaces from the layer config and the model's resources."""

import dataclasses
from types import SimpleNamespace
from unittest.mock import patch

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
from lightllm.utils import workspace_utils
from lightllm.common.basemodel.attention_backend_manager import AttentionBackendManager
from unit_tests.common.attention.backends.groups_runner import set_up


def context(monkeypatch, **overrides):
    monkeypatch.setattr(attention_backend_manager, "get_current_device_id", lambda: 0)
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda: (9, 0))
    args = dict(
        mtp_step=0,
        page_size=16,
        running_max_req_size=2,
        enable_tpsp_mix_mode=False,
        enable_decode_microbatch_overlap=False,
        flashinfer_workspace_size=512,
        diverse_mode=False,
        enable_ep_moe=False,
    )
    model = SimpleNamespace(
        args=SimpleNamespace(**args),
        max_seq_length=64,
        graph_max_batch_size=2,
        graph_max_len_in_batch=64,
        tp_world_size_=1,
        is_mtp_draft_model=False,
        mtp_manager=SimpleNamespace(get_decode_draft_step=lambda _: 0),
        att_state_role="main",
        att_state_manager=SimpleNamespace(get_bounded_attention_states=lambda role: ()),
    )
    return dataclasses.replace(AttentionBackendManager(model).resources, device="cpu", **overrides)


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

    monkeypatch.setattr(workspace_utils.torch, "empty", fake_empty)
    monkeypatch.setattr(workspace_utils, "_workspace_buffers", {})
    ctx = context(monkeypatch)
    # The context's workspaces come from the process-wide store, keyed by the current device.
    monkeypatch.setattr(workspace_utils, "get_current_device_id", lambda: current_device_id)

    fp = set_up(FlashInferGQAPrefillBackend, gqa(), ctx).workspace
    fp8 = set_up(FlashInferGQAPrefillBackend, gqa(kv_quant="fp8_per_tensor"), ctx).workspace
    mla_config = MLAPrefillLayerConfig(qk_nope_head_dim=128, v_head_dim=128, **MLA_COMMON)
    mla = set_up(FlashInferMLAPrefillBackend, mla_config, ctx).workspace

    assert fp8 is fp and mla is not fp
    assert [allocation[1] for allocation in allocations] == [512 * 1024 * 1024, 256 * 1024 * 1024]
    current_device_id = 1
    assert set_up(FlashInferGQAPrefillBackend, gqa(), ctx).workspace is not fp
    assert allocations[-1][3] == 1


def test_workspace_requests_of_one_name_share_the_largest_storage(monkeypatch):
    monkeypatch.setattr(workspace_utils, "get_current_device_id", lambda: 0)
    monkeypatch.setattr(workspace_utils.torch, "empty", lambda size, dtype, device: torch.zeros(size, dtype=dtype))
    monkeypatch.setattr(workspace_utils, "_workspace_buffers", {})
    get = workspace_utils.get_gpu_workspace_buffer

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


@pytest.mark.parametrize("backend", [FA3GQAPrefillBackend, FA3GQADecodeBackend])
def test_fa3_gqa_pages_like_the_cache_except_fp8_which_uses_single_token_pages(monkeypatch, backend):
    ctx = context(monkeypatch)
    assert set_up(backend, gqa(), ctx).page_size == 16
    with patch.object(gqa_fa3.logger, "warning") as warning:
        assert set_up(backend, gqa(kv_quant="fp8_per_head"), ctx).page_size == 1
    warning.assert_called_once()


def test_flashinfer_pages_like_the_cache_except_fp8_which_uses_single_token_pages(monkeypatch):
    ctx = context(monkeypatch)
    assert set_up(FlashInferGQADecodeBackend, gqa(), ctx).page_size == 16
    assert set_up(FlashInferGQADecodeBackend, gqa(kv_quant="fp8_per_tensor"), ctx).page_size == 1


def test_mla_decode_pages_like_the_cache(monkeypatch):
    ctx = context(monkeypatch)
    config = MLADecodeLayerConfig(kv_lora_rank=512, **MLA_COMMON)
    assert set_up(FA3MLADecodeBackend, config, ctx).page_size == 16
    assert set_up(FlashInferMLADecodeBackend, config, ctx).page_size == 16
