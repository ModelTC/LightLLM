"""The model prepares attention inputs before dispatching to its role's backends."""

from types import SimpleNamespace

import pytest
import torch

from lightllm.common.attention.backends.gdn import GDNLayerConfig
from lightllm.common.attention.backends.gdn.triton import TritonGDNDecodeBackend
from lightllm.common.attention_state import AttentionStateManager
from lightllm.common.attention_state.bounded_att import ConvSSMStateConfig, GDNState
from lightllm.common.basemodel.basemodel import TpPartBaseModel
from lightllm.common.basemodel.attention_backend_manager import AttentionBackendManager
from lightllm.common.basemodel.infer_struct import GraphMode, InferStateInfo
from lightllm.common.basemodel.mtp_manager import MtpManager
from lightllm.server.core.objs.start_args_type import StartArgs
from unit_tests.common.basemodel.test_overlap_utils import _make_decode_input, _make_prefill_input


@pytest.fixture
def make_model(monkeypatch):
    monkeypatch.setattr(InferStateInfo, "init_some_extra_state", lambda self, model: None)
    monkeypatch.setattr("lightllm.common.basemodel.basemodel.dist_group_manager.get_group", lambda index: None)

    def create(*, role="main", mode="dflash2", mtp_step=2, dynamic=False, states=None, manager=None):
        model = TpPartBaseModel.__new__(TpPartBaseModel)
        model.args = StartArgs(mtp_mode=mode, mtp_step=mtp_step, mtp_dynamic_verify=dynamic)
        model.is_mtp_draft_model = role != "main"
        model.mtp_previous_draft_models = [None] * (int(role[5:]) if role != "main" else 0)
        model.mtp_manager = MtpManager.__new__(MtpManager)
        model.mtp_manager.args = model.args
        model.max_seq_length, model.graph_max_batch_size, model.graph_max_len_in_batch = 1024, 4, 512
        model.tp_world_size_ = 1
        model.layers_infer = []
        model.att_state_manager = states or SimpleNamespace(
            get_full_attention_states=lambda role: (object(),),
            get_bounded_attention_states=lambda role: (),
        )
        model.attention_backend_manager = manager or AttentionBackendManager()
        model.infer_state_class = InferStateInfo
        model.return_all_prompt_logics = False
        model.hidden_collector_prototype = SimpleNamespace(new_instance=lambda: None)
        model.req_manager = SimpleNamespace(
            req_to_token_indexs=torch.zeros((16, 16), dtype=torch.int32), HOLD_REQUEST_ID=15
        )
        model._select_mem_indexes = lambda value: torch.zeros_like(value.b_req_idx)
        model._init_attention()

        def record_inputs(infer, resources):
            return (
                infer.decode_rows_per_request,
                infer.dynamic_decode_rows,
                infer.req_to_mtp_state_index,
                infer.cuda_graph_max_kv_seq_len,
            )

        backend = SimpleNamespace(create_att_state=record_inputs)
        for phase in ("prefill", "decode"):
            model.attention_backend_manager.state_groups[role][phase] = {("inputs",): backend}
        return model

    return create


@pytest.mark.parametrize("prefill", [False, True])
@pytest.mark.parametrize("dynamic", [False, True])
@pytest.mark.parametrize(
    "mode,role,mtp_step,rows",
    [
        (None, "main", 0, 1),
        ("dspark", "main", 7, 8),
        ("dspark", "draft0", 7, 7),
        ("dflash", "main", 7, 8),
        ("dflash", "draft0", 7, 7),
        ("dflash2", "main", 7, 8),
        ("dflash2", "draft0", 7, 8),
        ("vanilla_with_att", "draft0", 7, 1),
        ("vanilla_with_att", "draft0", 0, 1),
        ("eagle3", "draft0", 0, 1),
        ("eagle3", "draft0", 7, 1),
        ("eagle3", "main", 7, 8),
    ],
)
def test_model_prepares_decode_layout(make_model, mode, role, mtp_step, rows, dynamic, prefill):
    model = make_model(mode=mode, role=role, mtp_step=mtp_step, dynamic=dynamic)
    model_input = _make_prefill_input([10, 11], 0, False) if prefill else _make_decode_input()

    infer = model._create_inferstate(model_input)

    expected_rows = 1 if prefill else rows
    expected_dynamic = not prefill and role == "main" and rows > 1 and dynamic
    assert infer.att_states[("inputs",)] == (expected_rows, expected_dynamic, None, None)


def test_recurrent_decode_reads_each_roles_live_acceptance_index(make_model):
    states = AttentionStateManager(request_capacity=16)
    config = ConvSSMStateConfig(
        logical_layer_ids=(3,),
        global_k_heads=1,
        global_v_heads=1,
        key_dim=2,
        value_dim=2,
        conv_width=2,
        conv_dtype=torch.float32,
        ssm_dtype=torch.float32,
    )
    recurrent = {}
    for role, accepted_index in (("main", 0), ("draft0", 1), ("draft1", 2)):
        state = GDNState(config, request_capacity=16, mtp_step=2, device="meta")
        state.mtp_state_index = torch.full((16,), accepted_index, dtype=torch.int32)
        states.register_state(role, state)
        recurrent[role] = state
    states.finish_state_registration()
    manager = AttentionBackendManager()
    models = [make_model(role=role, states=states, manager=manager) for role in recurrent]
    model_input = _make_decode_input()
    model_input.b_req_idx = torch.tensor([2, 2, 2, 4, 4, 4], dtype=torch.int32)
    for model in models:
        role = model.att_state_role
        backend = TritonGDNDecodeBackend(GDNLayerConfig(2, 4, 64, 64, 4, "silu", torch.bfloat16))
        manager.state_groups[role]["decode"] = {backend.state_key: backend}
        for _ in range(2):
            infer = model._create_inferstate(model_input)
            index = recurrent[role].mtp_state_index
            assert infer.req_to_mtp_state_index is index
            torch.testing.assert_close(
                infer.att_states[backend.state_key].num_accepted_tokens, index[torch.tensor([2, 4])] + 1
            )
            index.copy_((index + 1) % 3)
        prefill = model._create_inferstate(_make_prefill_input([10, 11], 0, False))
        assert prefill.att_states[("inputs",)] == (1, False, None, None)
        model.args.mtp_mode = None
        assert model._create_inferstate(model_input).req_to_mtp_state_index is None


def test_model_rejects_multiple_recurrent_states_at_initialization(make_model):
    state = SimpleNamespace(mtp_state_index=torch.zeros(16, dtype=torch.int32))
    states = SimpleNamespace(
        get_full_attention_states=lambda role: (),
        get_bounded_attention_states=lambda role: (state, state),
    )
    with pytest.raises(ValueError, match="at most one recurrent state for main, got 2"):
        make_model(states=states)


@pytest.mark.parametrize("prefill", [False, True])
@pytest.mark.parametrize("graph_mode", list(GraphMode))
def test_model_supplies_graph_capacity_before_building_attention(make_model, graph_mode, prefill):
    model = make_model()
    model_input = _make_prefill_input([10, 11], 0, False) if prefill else _make_decode_input()
    infer = model._create_inferstate(model_input, graph_mode=graph_mode)
    assert infer.att_states[("inputs",)][3] == (None if graph_mode is GraphMode.EAGER else 512)
    model.graph_max_len_in_batch = None
    if graph_mode is GraphMode.EAGER:
        assert model._create_inferstate(model_input).cuda_graph_max_kv_seq_len is None
    else:
        with pytest.raises(ValueError, match="kv length capacity"):
            model._create_inferstate(model_input, graph_mode=graph_mode)


@pytest.mark.parametrize("build_attention", [False, True])
def test_steps_without_attention_skip_preparation(make_model, build_attention):
    model = make_model()
    if build_attention:
        model.attention_backend_manager.state_groups["main"] = {"prefill": {}, "decode": {}}
    model.mtp_manager = None
    model.graph_max_len_in_batch = None
    infer = model._create_inferstate(
        _make_decode_input(), graph_mode=GraphMode.CAPTURE, build_attention=build_attention
    )
    assert infer.att_states == {}
    assert infer.decode_rows_per_request == 1
    assert not infer.dynamic_decode_rows
    assert infer.req_to_mtp_state_index is None
    assert infer.cuda_graph_max_kv_seq_len is None
