from types import SimpleNamespace

import pytest
import torch

from lightllm.common.basemodel.basemodel import TpPartBaseModel
from lightllm.common.mtp_config import FinalHiddenPolicy
from lightllm.common.basemodel.batch_objs import ModelInput
from lightllm.common.basemodel.infer_struct import InferStateInfo
from lightllm.common.basemodel.mtp_manager import MtpManager


def _create_model_input(*, is_prefill=False):
    batch_size = 2
    kwargs = dict(
        batch_size=batch_size,
        total_token_num=batch_size,
        max_q_seq_len=1,
        max_kv_seq_len=1,
        b_req_idx=torch.arange(batch_size, dtype=torch.int32),
        b_mtp_index=torch.zeros(batch_size, dtype=torch.int32),
        b_seq_len=torch.ones(batch_size, dtype=torch.int32),
        is_prefill=is_prefill,
        multimodal_params=[{"images": [], "audios": []} for _ in range(batch_size)],
    )
    if is_prefill:
        kwargs["max_cache_len"] = 0
        kwargs["input_ids"] = torch.ones(batch_size, dtype=torch.int64)
        kwargs["b_ready_cache_len"] = torch.zeros(batch_size, dtype=torch.int32)
        kwargs["b_prefill_start_loc"] = torch.arange(batch_size, dtype=torch.int32)
        kwargs["b_is_decode_req"] = torch.zeros(batch_size, dtype=torch.bool)
        kwargs["b_prefill_has_output_cpu"] = [False] * batch_size
    else:
        kwargs["b_position_delta"] = torch.zeros(batch_size, dtype=torch.int32)
        kwargs["b_shared_seq_len"] = torch.tensor([4, 4], dtype=torch.int32)
        kwargs["b_shared_radix_node_id"] = torch.tensor([10, 10], dtype=torch.int64)
    return ModelInput(**kwargs)


def test_decode_requires_shared_radix_metadata():
    with pytest.raises(AssertionError):
        ModelInput(
            batch_size=1,
            total_token_num=1,
            max_q_seq_len=1,
            max_kv_seq_len=1,
            b_req_idx=torch.zeros(1, dtype=torch.int32),
            b_mtp_index=torch.zeros(1, dtype=torch.int32),
            b_seq_len=torch.ones(1, dtype=torch.int32),
            b_position_delta=torch.zeros(1, dtype=torch.int32),
            is_prefill=False,
            multimodal_params=[{"images": [], "audios": []}],
        )


def test_decode_carries_raw_shared_radix_metadata():
    model_input = _create_model_input()

    assert torch.equal(model_input.b_shared_seq_len, torch.tensor([4, 4], dtype=torch.int32))
    assert torch.equal(model_input.b_shared_radix_node_id, torch.tensor([10, 10], dtype=torch.int64))


def test_prefill_does_not_require_shared_radix_metadata():
    model_input = _create_model_input(is_prefill=True)

    assert model_input.b_shared_seq_len is None
    assert model_input.b_shared_radix_node_id is None


def test_decode_requires_position_delta():
    model_input = _create_model_input()
    model_input.b_position_delta = None

    with pytest.raises(AssertionError):
        model_input.check_input()


def test_prefill_requires_prefill_metadata():
    model_input = _create_model_input(is_prefill=True)
    model_input.b_ready_cache_len = None

    with pytest.raises(AssertionError):
        model_input.check_input()


def test_prefill_requires_prefill_output_markers():
    model_input = _create_model_input(is_prefill=True)
    model_input.b_prefill_has_output_cpu = None

    with pytest.raises(AssertionError, match="prefill must provide b_prefill_has_output_cpu"):
        model_input.check_input()


def test_prefill_requires_decode_request_markers():
    model_input = _create_model_input(is_prefill=True)
    model_input.b_is_decode_req = None

    with pytest.raises(AssertionError):
        model_input.check_input()


def test_prefill_rejects_position_delta():
    model_input = _create_model_input(is_prefill=True)
    model_input.b_position_delta = torch.zeros(model_input.batch_size, dtype=torch.int32)

    with pytest.raises(AssertionError, match="prefill must not provide b_position_delta"):
        model_input.check_input()


def test_padded_prefill_adds_non_decode_request_marker():
    model_input = ModelInput(
        batch_size=1,
        total_token_num=2,
        max_q_seq_len=2,
        max_kv_seq_len=2,
        max_cache_len=0,
        input_ids=torch.ones(2, dtype=torch.int64),
        b_req_idx=torch.zeros(1, dtype=torch.int32),
        b_mtp_index=torch.zeros(1, dtype=torch.int32),
        b_seq_len=torch.full((1,), 2, dtype=torch.int32),
        b_is_decode_req=torch.ones(1, dtype=torch.bool),
        b_ready_cache_len=torch.zeros(1, dtype=torch.int32),
        b_prefill_start_loc=torch.zeros(1, dtype=torch.int32),
        is_prefill=True,
        b_prefill_has_output_cpu=[False],
        multimodal_params=[{"images": [], "audios": []}],
    )
    model = SimpleNamespace(
        mem_manager=SimpleNamespace(HOLD_TOKEN_MEMINDEXES=(-1,), page_size=1),
        req_manager=SimpleNamespace(HOLD_REQUEST_ID=-1),
    )

    padded_input = TpPartBaseModel._create_padded_prefill_model_input(
        model,
        model_input=model_input,
        new_handle_token_num=4,
    )

    assert padded_input.b_is_decode_req.dtype == torch.bool
    assert padded_input.b_is_decode_req.tolist() == [True, False]


def test_padded_prefill_builds_internal_request_for_empty_input():
    model_input = ModelInput(
        batch_size=0,
        total_token_num=0,
        max_q_seq_len=0,
        max_kv_seq_len=0,
        max_cache_len=0,
        input_ids=torch.empty((0,), dtype=torch.int64),
        b_req_idx=torch.empty((0,), dtype=torch.int32),
        b_mtp_index=torch.empty((0,), dtype=torch.int32),
        b_seq_len=torch.empty((0,), dtype=torch.int32),
        b_is_decode_req=torch.empty((0,), dtype=torch.bool),
        b_ready_cache_len=torch.empty((0,), dtype=torch.int32),
        b_prefill_start_loc=torch.empty((0,), dtype=torch.int32),
        is_prefill=True,
        b_prefill_has_output_cpu=[],
        multimodal_params=[],
        mtp_draft_input_hiddens=torch.empty((0, 4), dtype=torch.float32),
    )
    model = SimpleNamespace(
        mem_manager=SimpleNamespace(HOLD_TOKEN_MEMINDEXES=(99,), page_size=1),
        req_manager=SimpleNamespace(HOLD_REQUEST_ID=88),
    )

    padded_input = TpPartBaseModel._create_padded_prefill_model_input(
        model,
        model_input=model_input,
        new_handle_token_num=1,
    )

    assert model_input.batch_size == 0
    assert padded_input.batch_size == 1
    assert padded_input.input_ids.tolist() == [1]
    assert padded_input.b_req_idx.tolist() == [88]
    assert padded_input.b_seq_len.tolist() == [1]
    assert padded_input.b_prefill_has_output_cpu == [False]
    assert torch.equal(padded_input.mtp_draft_input_hiddens, torch.zeros((1, 4)))


def test_padded_decode_builds_internal_request_from_empty_token_tensor():
    model_input = ModelInput(
        batch_size=0,
        total_token_num=0,
        max_q_seq_len=1,
        max_kv_seq_len=0,
        input_ids=torch.empty((0,), dtype=torch.int64),
        b_req_idx=torch.empty((0,), dtype=torch.int32),
        b_mtp_index=torch.empty((0,), dtype=torch.int32),
        b_seq_len=torch.empty((0,), dtype=torch.int32),
        b_position_delta=torch.empty((0,), dtype=torch.int32),
        b_shared_seq_len=torch.empty((0,), dtype=torch.int32),
        b_shared_radix_node_id=torch.empty((0,), dtype=torch.int64),
        is_prefill=False,
        multimodal_params=[],
    )
    model = SimpleNamespace(
        mem_manager=SimpleNamespace(HOLD_TOKEN_MEMINDEXES=(99,), page_size=1),
        req_manager=SimpleNamespace(HOLD_REQUEST_ID=88),
    )

    padded_input = TpPartBaseModel._create_padded_decode_model_input(
        model,
        model_input=model_input,
        new_batch_size=1,
    )

    assert model_input.batch_size == 0
    assert padded_input.batch_size == 1
    assert padded_input.input_ids.tolist() == [1]
    assert padded_input.b_req_idx.tolist() == [88]
    assert padded_input.b_seq_len.tolist() == [2]
    assert padded_input.b_shared_seq_len.tolist() == [0]
    assert padded_input.b_shared_radix_node_id.tolist() == [-1]


@pytest.mark.parametrize("is_prefill", [False, True])
@pytest.mark.parametrize("policy", [FinalHiddenPolicy.PRE_NORM, FinalHiddenPolicy.POST_NORM])
@pytest.mark.parametrize(
    "mode,draft", [(None, False), ("vanilla_with_att", False), ("vanilla_with_att", True), ("dspark", True)]
)
def test_infer_state_carries_effective_final_hidden_policy(monkeypatch, is_prefill, policy, mode, draft):
    from lightllm.common.basemodel import basemodel

    model = TpPartBaseModel.__new__(TpPartBaseModel)
    model.infer_state_class = InferStateInfo
    model.is_mtp_draft_model = draft
    model.weight_dir_ = "/models/draft"
    model.mtp_manager = MtpManager.__new__(MtpManager)
    model.mtp_manager.args = SimpleNamespace(mtp_mode=mode, mtp_draft_model_dir=["/models/draft"])
    effective_policy = policy if mode == "vanilla_with_att" else FinalHiddenPolicy.NONE
    model.mtp_manager._get_pair_config = lambda _: SimpleNamespace(
        main_final_hidden_policy=effective_policy, draft_final_hidden_policy=effective_policy
    )
    model._init_hidden_collector()
    model.return_all_prompt_logics = False
    model.mem_manager = model.req_manager = None
    model._select_mem_indexes = lambda model_input: None
    model.prefill_att_backend = SimpleNamespace(create_att_prefill_state=lambda **kwargs: None)
    model.decode_att_backend = SimpleNamespace(create_att_decode_state=lambda **kwargs: None)
    model.prefill_att_backend1 = model.decode_att_backend1 = None
    monkeypatch.setattr(basemodel.dist_group_manager, "get_group", lambda _: None)

    state = model._create_inferstate(_create_model_input(is_prefill=is_prefill))

    expected = policy if mode == "vanilla_with_att" else FinalHiddenPolicy.NONE
    assert state.final_hidden_policy is expected
    assert not hasattr(state.hidden_collector, "final_hidden_policy")
