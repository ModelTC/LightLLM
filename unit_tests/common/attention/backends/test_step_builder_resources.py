"""Step builders take their page sizes and workspaces from the layer config and the model's context."""
from typing import Any
from unit_tests.common.attention.backends.step_fixture import create_att_state

from dataclasses import replace
import sys
from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch

from lightllm.common.attention import base_backend as att_backend
from lightllm.common.attention.backends.gqa import GQAAttConfig
from lightllm.common.attention.backends.gqa import fa3 as gqa_fa3
from lightllm.common.attention.backends.gqa import flashinfer as gqa_flashinfer
from lightllm.common.attention.backends.gqa.fa3 import FA3GQADecodeBackend, FA3GQAPrefillBackend
from lightllm.common.attention.backends.gqa.flashinfer import FlashInferGQADecodeBackend, FlashInferGQAPrefillBackend
from lightllm.common.attention.backends.mla import MLADecodeAttConfig, MLAPrefillAttConfig
from lightllm.common.attention.backends.mla.fa3 import FA3MLADecodeBackend
from lightllm.common.attention.backends.mla.flashinfer import FlashInferMLADecodeBackend, FlashInferMLAPrefillBackend


def make_model(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("LIGHTLLM_CURRENT_DEVICE_ID", "0")
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda: (9, 0))
    args = dict(
        mtp_step=0,
        page_size=16,
        running_max_req_size=2,
        enable_tpsp_mix_mode=False,
        enable_decode_microbatch_overlap=False,
        flashinfer_workspace_size=512,
        diverse_mode=False,
    )
    return SimpleNamespace(args=SimpleNamespace(**args), max_seq_length=64, graph_max_batch_size=2, tp_world_size_=1)


def gqa(**fields) -> GQAAttConfig:
    return GQAAttConfig(
        num_heads=fields.get("num_heads", 8),
        num_kv_heads=fields.get("num_kv_heads", 2),
        head_dim=fields.get("head_dim", 64),
        dtype=fields.get("dtype", torch.bfloat16),
        sliding_window=fields.get("sliding_window", (-1, -1)),
        softmax_scale=fields.get("softmax_scale", None),
        causal=fields.get("causal", True),
        has_sinks=fields.get("has_sinks", False),
        kv_quant=fields.get("kv_quant", None),
        kv_quant_group_size=fields.get("kv_quant_group_size", None),
        has_q_scale=fields.get("has_q_scale", False),
        has_alibi=fields.get("has_alibi", False),
        bidirectional_image_spans=fields.get("bidirectional_image_spans", False),
    )


MLA_COMMON = dict(num_heads=16, qk_rope_head_dim=64, softmax_scale=0.1, dtype=torch.bfloat16)


@pytest.fixture
def fresh_workspace():
    get_workspace = att_backend.PrefillAttBackend.get_workspace
    with patch.dict(get_workspace.__dict__, clear=True):
        yield get_workspace


def test_flashinfer_gqa_backends_share_one_workspace_and_mla_has_its_own(
    monkeypatch: pytest.MonkeyPatch, fresh_workspace: Any
):
    model = make_model(monkeypatch=monkeypatch)
    from lightllm.common.basemodel.infer_struct import GraphMode

    att_backend_graph_mode = GraphMode.EAGER
    batch = SimpleNamespace(
        batch_size=1,
        microbatch_index=0,
        uses_cuda_graph=False,
        graph_mode=att_backend_graph_mode,
        b_seq_len=torch.tensor(data=[1]),
        b1_cu_q_seq_len=torch.tensor(data=[0, 1]),
        b1_cu_kv_seq_len=torch.tensor(data=[0, 1]),
    )
    empty = torch.empty
    allocations = []
    current_device_id = 0

    def fake_empty(size, dtype, device):
        if not isinstance(device, int):
            return empty(size=size, dtype=dtype, device=device)
        workspace = object()
        allocations.append((workspace, size[0], dtype, device))
        return workspace

    def wrapper(float_workspace_buffer: Any, *args: Any, **kwargs: Any):
        return SimpleNamespace(workspace=float_workspace_buffer, plan=lambda *args, **kwargs: None)

    monkeypatch.setitem(
        sys.modules,
        "flashinfer",
        SimpleNamespace(
            prefill=SimpleNamespace(
                BatchPrefillWithPagedKVCacheWrapper=wrapper, BatchPrefillWithRaggedKVCacheWrapper=wrapper
            ),
            decode=SimpleNamespace(BatchDecodeWithPagedKVCacheWrapper=wrapper),
        ),
    )
    monkeypatch.setattr(gqa_flashinfer.FlashInferGQAPrefillBackend, "_page_metadata", lambda **kwargs: (None, None))
    monkeypatch.setattr(torch, "empty", fake_empty)
    monkeypatch.setattr(att_backend, "get_current_device_id", lambda: current_device_id)

    def workspace(backend: Any, **config: Any):
        runtime_config = backend.create_runtime_config(model=model)
        if backend is FlashInferGQADecodeBackend:
            runtime_config = replace(runtime_config, device="cpu")
        impl = backend(gqa(kv_quant=config.get("kv_quant")), runtime_config=runtime_config)
        return create_att_state(backend=impl, infer_state=batch).wrapper.workspace

    fp = workspace(backend=FlashInferGQAPrefillBackend)
    fp8 = workspace(backend=FlashInferGQAPrefillBackend, kv_quant="fp8_per_tensor")
    decode = workspace(backend=FlashInferGQADecodeBackend)
    mla_config = MLAPrefillAttConfig(
        num_heads=MLA_COMMON["num_heads"],
        qk_nope_head_dim=MLA_COMMON.get("qk_nope_head_dim", 128),
        qk_rope_head_dim=MLA_COMMON["qk_rope_head_dim"],
        v_head_dim=MLA_COMMON.get("v_head_dim", 128),
        softmax_scale=MLA_COMMON["softmax_scale"],
        dtype=MLA_COMMON["dtype"],
        causal=MLA_COMMON.get("causal", True),
    )
    mla = create_att_state(
        backend=FlashInferMLAPrefillBackend(att_config=mla_config), infer_state=batch
    ).wrapper.workspace

    assert fp8 is fp and decode is fp and mla is not fp
    assert [allocation[1] for allocation in allocations] == [512 * 1024 * 1024, 256 * 1024 * 1024]
    current_device_id = 1
    assert workspace(backend=FlashInferGQAPrefillBackend) is not fp
    assert allocations[-1][3] == 1


def test_workspace_key_includes_size_and_dtype(monkeypatch: pytest.MonkeyPatch, fresh_workspace: Any):
    monkeypatch.setattr(att_backend, "get_current_device_id", lambda: 0)
    monkeypatch.setattr(torch, "empty", lambda *args, **kwargs: object())
    get = fresh_workspace

    small = get("shared", "main", 1024)
    assert get("shared", "main", 1024) is small
    assert get("shared", "main", 2048) is not small
    assert get("shared", "main", 1024, dtype=torch.float16) is not small


def test_workspace_reuses_larger_buffer_across_roles(monkeypatch: pytest.MonkeyPatch, fresh_workspace: Any):
    monkeypatch.setattr(att_backend, "get_current_device_id", lambda: 0)
    empty = torch.empty
    allocations = []

    def allocate(size: tuple[int, ...], *, dtype: torch.dtype, device: int):
        allocations.append(size[0])
        return empty(size=size, dtype=dtype)

    monkeypatch.setattr(torch, "empty", allocate)
    get = fresh_workspace

    main = get("shared", "main", 1024)
    draft = get("shared", "draft0", 256)
    assert draft.shape == (256,)
    assert draft.data_ptr() == main.data_ptr()
    assert get("shared", "draft1", 1024) is main
    assert allocations == [1024]

    larger = get("shared", "draft2", 2048)
    assert allocations == [1024, 2048]
    assert larger.data_ptr() != main.data_ptr()
    assert get("shared", "draft0", 256) is draft


@pytest.mark.parametrize("backend", [FA3GQAPrefillBackend, FA3GQADecodeBackend])
def test_fa3_gqa_pages_like_the_cache_except_fp8_which_uses_single_token_pages(monkeypatch, backend):
    runtime_config = backend.create_runtime_config(model=make_model(monkeypatch=monkeypatch))
    assert backend(gqa(), runtime_config=runtime_config)._page_size == 16
    with patch.object(gqa_fa3.logger, "warning") as warning:
        assert backend(gqa(kv_quant="fp8_per_head"), runtime_config=runtime_config)._page_size == 1
    warning.assert_called_once()


def test_flashinfer_pages_like_the_cache_except_fp8_which_uses_single_token_pages(monkeypatch):
    runtime_config = FlashInferGQADecodeBackend.create_runtime_config(model=make_model(monkeypatch=monkeypatch))
    assert FlashInferGQADecodeBackend(att_config=gqa(), runtime_config=runtime_config)._page_size == 16
    fp8 = FlashInferGQADecodeBackend(att_config=gqa(kv_quant="fp8_per_tensor"), runtime_config=runtime_config)
    assert fp8._page_size == 1


def test_mla_decode_pages_like_the_cache(monkeypatch):
    model = make_model(monkeypatch=monkeypatch)
    config = MLADecodeAttConfig(
        num_heads=MLA_COMMON["num_heads"],
        kv_lora_rank=MLA_COMMON.get("kv_lora_rank", 512),
        qk_rope_head_dim=MLA_COMMON["qk_rope_head_dim"],
        softmax_scale=MLA_COMMON["softmax_scale"],
        dtype=MLA_COMMON["dtype"],
        causal=MLA_COMMON.get("causal", True),
    )
    fa3 = FA3MLADecodeBackend.create_runtime_config(model=model)
    flashinfer = replace(FlashInferMLADecodeBackend.create_runtime_config(model=model), device="cpu")
    assert FA3MLADecodeBackend(att_config=config, runtime_config=fa3).runtime_config.page_size == 16
    assert FlashInferMLADecodeBackend(att_config=config, runtime_config=flashinfer)._page_size == 16
