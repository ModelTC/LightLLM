"""Real model initialization, compute writes, and profiled allocation."""

from __future__ import annotations

from unit_tests.common.att_resources.model_helpers import full_state_test_fields

from unit_tests.common.att_resources.model_helpers import full_state_test_buffers

from unit_tests.common.att_resources.model_helpers import read_kv_storage

from typing import TYPE_CHECKING
import json

from unit_tests.common.att_resources.model_helpers import prepare_model_states

import pytest
import torch

from lightllm.common.att_resources.full_att import PackedMLAFullAttentionState
from lightllm.utils.envs_utils import get_env_start_args
from unit_tests.common.att_resources.test_capacity import allocated_bytes
from unit_tests.server.router.model_infer.test_cpu_resource_layout import CASES, make_runtime, runtime

if TYPE_CHECKING:
    from lightllm.common.basemodel.basemodel import TpPartBaseModel
    from pathlib import Path


pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")


@pytest.mark.parametrize("kind,encoding,model_cls", CASES)
def test_model_storage_write_and_copy_preserve_all_fields(
    runtime: None,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    kind: str,
    encoding: str,
    model_cls: type[TpPartBaseModel],
):
    _, (resources,) = make_runtime(monkeypatch, tmp_path, kind, encoding, model_cls, 1)
    (state,) = resources.get_full_attention_states("main")
    config = state.config
    for storage_buffer in full_state_test_buffers(state).values():
        storage_buffer.zero_()
    ids = torch.tensor([0, 4], dtype=torch.int32, device="cuda")
    dim = config.kv_dim if isinstance(state, PackedMLAFullAttentionState) else config.head_dim
    kv = torch.randn(2, (1 if "kv" in full_state_test_buffers(state) else 2) * config.head_num, dim, device="cuda", dtype=torch.bfloat16)
    state.write_full_att_kv(config.layer_ids[-1], ids, kv)
    if len(config.layer_ids) > 1:
        assert not read_kv_storage(state)[0].any()
    resources.copy_full_att_tokens(ids, torch.tensor([1, 5], dtype=torch.int32, device="cuda"))
    for name, field in full_state_test_fields(state, full_state_test_buffers(state)).items():
        if "token" in field.axes:
            assert torch.equal(
                field.tensor[:, ids].contiguous().view(torch.uint8),
                field.tensor[:, [1, 5]].contiguous().view(torch.uint8),
            ), name
    assert state.capacity == resources.full_att_allocator.size == 16
    assert read_kv_storage(state).shape[1] == 20  # Includes the whole HOLD page.


@pytest.mark.parametrize("mode", ["normal", "prefill"])
@pytest.mark.parametrize("kind,encoding,model_cls", CASES[:8])
def test_auto_profile_fits_actual_gpu_storage_and_pd_pages(
    runtime: None,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    mode: str,
    kind: str,
    encoding: str,
    model_cls: type[TpPartBaseModel],
):
    _, (reference,) = make_runtime(monkeypatch, tmp_path, kind, encoding, model_cls, 1)
    (state,) = reference.get_full_attention_states("main")
    args = get_env_start_args()
    args.run_mode, args.pd_kv_page_size, args.pd_kv_page_num = mode, 8, 3
    transfer_bytes = reference.get_pd_page_bytes() * 3 if mode == "prefill" else 0
    storage_bytes = allocated_bytes(state)
    budget = storage_bytes + transfer_bytes
    monkeypatch.setattr(torch.cuda, "mem_get_info", lambda device: (budget, 1 << 30))
    monkeypatch.setattr(torch.distributed, "is_initialized", lambda: False)
    model = model_cls.__new__(model_cls)
    model.args, model.config = args, json.loads((tmp_path / "config.json").read_text())
    model.tp_rank_, model.tp_world_size_ = 0, 1
    model.max_total_token_num, model.max_req_num, model.mem_fraction = None, 1, 1.0
    prepare_model_states(model)
    model.att_resource_manager.initialize_resources(
        args, token_capacity=model.max_total_token_num, mem_fraction=model.mem_fraction
    )
    (state,) = model.att_resource_manager.get_full_attention_states("main")
    assert not hasattr(model, "full_att_state")
    assert state.capacity == 16
    assert allocated_bytes(state) == storage_bytes
    if mode == "prefill":
        assert model.att_resource_manager.get_pd_page_bytes() == reference.get_pd_page_bytes()
