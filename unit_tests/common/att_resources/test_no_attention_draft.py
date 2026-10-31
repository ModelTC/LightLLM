from __future__ import annotations

from unit_tests.common.att_resources.model_helpers import read_kv_storage

from contextlib import nullcontext
from types import SimpleNamespace

import pytest
import torch

from lightllm.common.attention_state import AttentionStateManager
from lightllm.models.llama.model import LlamaTpPartModel
from lightllm.common.basemodel.infer_struct import InferStateInfo
from lightllm.server.core.objs.start_args_type import StartArgs
from unit_tests.common.att_resources.test_capacity import definition
from unit_tests.common.basemodel.test_overlap_utils import _make_decode_input, _make_prefill_input


def draft_model(mode: str):
    resources = AttentionStateManager(request_capacity=3)
    resources.register_state("main", definition("gqa"))
    draft = LlamaTpPartModel.__new__(LlamaTpPartModel)
    draft.is_mtp_draft_model = True
    draft.args = StartArgs(mtp_mode=mode, data_type="bfloat16")
    draft.max_req_num = 2
    draft.layers_infer = [SimpleNamespace(layer_num_=2)]
    draft.config = dict(
        num_attention_heads=2,
        num_key_value_heads=1,
        hidden_size=8,
        num_hidden_layers=1,
    )
    draft.tp_rank_, draft.tp_world_size_ = 0, 1
    main = LlamaTpPartModel.__new__(LlamaTpPartModel)
    main.args, main.config = draft.args, dict(draft.config, num_hidden_layers=2)
    main.layers_infer = [SimpleNamespace(layer_num_=i) for i in range(2)]
    main.att_state_manager = resources
    draft.torch_memory_saver = SimpleNamespace(region=lambda **_: nullcontext())
    draft.main_model, draft.mtp_previous_draft_models = main, []
    return draft, resources


@pytest.mark.parametrize("mode", ["vanilla_no_att", "eagle_no_att"])
@pytest.mark.parametrize("prefill", [False, True])
def test_ffn_only_draft_has_no_state_or_attention_backend(monkeypatch: pytest.MonkeyPatch, mode: str, prefill: bool):
    draft, resources = draft_model(mode)
    draft.prepare_attention_states()
    resources.finish_state_registration()
    draft._init_att_backend()
    draft._init_att_backend1()
    assert draft.att_state_manager is resources
    assert not hasattr(draft, "full_att_state")
    assert not resources.get_full_attention_states("draft0")
    assert not any(name.startswith("draft0_") and name.endswith("_state") for name in vars(resources))

    draft.infer_state_class = InferStateInfo
    draft.return_all_prompt_logics = False
    draft.hidden_collector_prototype = SimpleNamespace(new_instance=lambda: None)
    draft.req_manager = object()
    draft._select_mem_indexes = lambda value: torch.zeros(value.batch_size, dtype=torch.int32)
    monkeypatch.setattr("lightllm.common.basemodel.basemodel.dist_group_manager.get_group", lambda index: None)
    model_input = _make_prefill_input([10, 11], req_idx=0, is_decode_req=False) if prefill else _make_decode_input()
    state = draft._create_inferstate(model_input)
    assert not hasattr(state, "full_att_state")
    assert state.att_state_manager is resources
    assert state.is_draft_model
    assert state.prefill_att_state is None and state.decode_att_state is None
    assert state.req_manager is draft.req_manager


@pytest.mark.parametrize("mode", ["vanilla_with_att", "eagle_with_att"])
def test_attention_draft_prepares_its_own_state_before_shared_allocation(mode: str):
    draft, resources = draft_model(mode)
    draft.prepare_attention_states()
    assert draft.att_state_manager is resources
    resources.finish_state_registration()
    (state,) = resources.get_full_attention_states("draft0")
    assert not hasattr(draft, "full_att_state")
    assert state.state_name == "draft0_gqa_bf16_state"
    assert state.config.logical_layer_groups == ((2,),)
    assert read_kv_storage(state) is None
