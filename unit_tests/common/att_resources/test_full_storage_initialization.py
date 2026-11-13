"""Real model initialization, compute writes, and profiled allocation."""

from __future__ import annotations

from typing import TYPE_CHECKING
import json

from unit_tests.common.att_resources.model_helpers import prepare_model_states

import pytest
import torch

from lightllm.utils.envs_utils import get_env_start_args
from unit_tests.common.att_resources.test_capacity import allocated_bytes
from unit_tests.server.router.model_infer.test_cpu_resource_layout import CASES, make_runtime, runtime

if TYPE_CHECKING:
    from lightllm.common.basemodel.basemodel import TpPartBaseModel
    from pathlib import Path


pytestmark = [
    pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required"),
    pytest.mark.usefixtures("runtime"),
]


@pytest.mark.parametrize("mode", ["normal", "prefill"])
@pytest.mark.parametrize("kind,encoding,model_cls", CASES[:8])
def test_auto_profile_fits_actual_gpu_storage_and_pd_pages(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    mode: str,
    kind: str,
    encoding: str,
    model_cls: type[TpPartBaseModel],
):
    _, (reference,) = make_runtime(
        monkeypatch=monkeypatch,
        tmp_path=tmp_path,
        kind=kind,
        encoding=encoding,
        model_cls=model_cls,
        tp=1,
    )
    (state,) = reference.get_full_attention_states("main")
    args = get_env_start_args()
    args.run_mode, args.pd_kv_page_size, args.pd_kv_page_num = mode, 8, 3
    transfer_bytes = reference.get_pd_operator().get_page_bytes() * 3 if mode == "prefill" else 0
    storage_bytes = allocated_bytes(state)
    budget = storage_bytes + transfer_bytes
    monkeypatch.setattr(torch.cuda, "mem_get_info", lambda device: (budget, 1 << 30))
    monkeypatch.setattr(torch.distributed, "is_initialized", lambda: False)
    model = model_cls.__new__(model_cls)
    model.args, model.config = args, json.loads((tmp_path / "config.json").read_text())
    model.tp_rank_, model.tp_world_size_ = 0, 1
    model.max_total_token_num, model.max_req_num, model.mem_fraction = None, 1, 1.0
    prepare_model_states(model)
    model.att_state_manager.finish_state_registration()
    model.att_state_manager.load_full_att_calibration(args=args)
    model.att_state_manager.initialize_storage(
        args, token_capacity=model.max_total_token_num, mem_fraction=model.mem_fraction
    )
    (state,) = model.att_state_manager.get_full_attention_states("main")
    assert not hasattr(model, "full_att_state")
    assert state.capacity == 16
    assert allocated_bytes(state) == storage_bytes
    if mode == "prefill":
        assert (
            model.att_state_manager.get_pd_operator().get_page_bytes() == reference.get_pd_operator().get_page_bytes()
        )
