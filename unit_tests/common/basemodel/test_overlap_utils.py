from types import SimpleNamespace

import pytest
import torch

from lightllm.common.basemodel import basemodel
from lightllm.common.basemodel.basemodel import TpPartBaseModel
from lightllm.common.basemodel.batch_objs import ModelInput, ModelOutput


def _empty_multimodal_params(batch_size: int):
    return [{"images": [], "audios": []} for _ in range(batch_size)]


def _make_prefill_input(input_ids: list[int], req_idx: int, is_decode_req: bool) -> ModelInput:
    token_num = len(input_ids)
    return ModelInput(
        batch_size=1,
        total_token_num=token_num,
        max_q_seq_len=token_num,
        max_kv_seq_len=token_num,
        max_cache_len=0,
        input_ids=torch.tensor(input_ids, dtype=torch.int64),
        b_req_idx=torch.tensor([req_idx], dtype=torch.int32),
        b_mtp_index=torch.zeros(1, dtype=torch.int32),
        b_seq_len=torch.tensor([token_num], dtype=torch.int32),
        b_is_decode_req=torch.tensor([is_decode_req]),
        b_ready_cache_len=torch.zeros(1, dtype=torch.int32),
        b_prefill_start_loc=torch.zeros(1, dtype=torch.int32),
        mem_indexes_cpu=torch.arange(token_num, dtype=torch.int32),
        is_prefill=True,
        b_prefill_has_output_cpu=[True],
        multimodal_params=_empty_multimodal_params(1),
    )


def _make_decode_input() -> ModelInput:
    batch_size = 6
    return ModelInput(
        batch_size=batch_size,
        total_token_num=39,
        max_q_seq_len=1,
        max_kv_seq_len=9,
        input_ids=torch.arange(batch_size, dtype=torch.int64),
        b_req_idx=torch.tensor([10, 11, 12, 12, 12, 12], dtype=torch.int32),
        b_mtp_index=torch.tensor([0, 0, 0, 1, 2, 3], dtype=torch.int32),
        b_seq_len=torch.tensor([4, 5, 6, 7, 8, 9], dtype=torch.int32),
        b_position_delta=torch.arange(batch_size, dtype=torch.int32),
        b_shared_seq_len=torch.arange(10, 16, dtype=torch.int32),
        b_shared_radix_node_id=torch.arange(20, 26, dtype=torch.int64),
        mem_indexes_cpu=torch.arange(100, 106, dtype=torch.int32),
        is_prefill=False,
        multimodal_params=_empty_multimodal_params(batch_size),
    )


def _make_empty_decode_input() -> ModelInput:
    return ModelInput(
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
        mem_indexes_cpu=torch.empty((0,), dtype=torch.int32),
        is_prefill=False,
        multimodal_params=[],
    )


def test_base_model_prefill_accepts_two_prebuilt_inputs(monkeypatch):
    model_input0 = _make_prefill_input([10, 11], req_idx=20, is_decode_req=False)
    model_input1 = _make_prefill_input([12], req_idx=21, is_decode_req=True)
    events = []
    original_to_cuda = ModelInput.to_cuda

    def record_to_cuda(self):
        events.append("to_cuda")
        original_to_cuda(self)

    monkeypatch.setattr(ModelInput, "to_cuda", record_to_cuda)

    def fake_gather(**kwargs):
        events.append("gather")
        kwargs["input_ids"][-1] = 90 + events.count("gather")

    monkeypatch.setattr(basemodel, "gather_token_prefill_decode_mixed", fake_gather)

    model = TpPartBaseModel.__new__(TpPartBaseModel)
    model.args = SimpleNamespace(enable_prefill_decode_mixed=True)
    model.req_manager = SimpleNamespace(
        req_sampling_params_manager=SimpleNamespace(req_to_next_token_ids=object()),
    )
    captured_inputs = []

    def fake_overlap_forward(input0, input1):
        events.append("forward")
        captured_inputs.extend((input0, input1))
        return (
            ModelOutput(logits=torch.zeros((input0.batch_size, 1))),
            ModelOutput(logits=torch.zeros((input1.batch_size, 1))),
        )

    model._microbatch_overlap_prefill_cuda = fake_overlap_forward

    outputs = model.microbatch_overlap_prefill(model_input0, model_input1)

    assert events == ["to_cuda", "gather", "to_cuda", "gather", "forward"]
    assert captured_inputs == [model_input0, model_input1]
    assert captured_inputs[0].input_ids.tolist() == [10, 91]
    assert captured_inputs[1].input_ids.tolist() == [92]
    assert len(outputs) == 2


def test_base_model_decode_accepts_two_inputs_and_skips_empty_gather(monkeypatch):
    model_input0 = _make_decode_input()
    model_input0.input_ids = None
    model_input1 = _make_empty_decode_input()
    model_input1.input_ids = None
    events = []
    original_to_cuda = ModelInput.to_cuda

    def record_to_cuda(self):
        events.append("to_cuda")
        original_to_cuda(self)

    monkeypatch.setattr(ModelInput, "to_cuda", record_to_cuda)

    def fake_gather(**kwargs):
        events.append("gather")
        return torch.arange(40, 46, dtype=torch.int64, device="cuda")

    monkeypatch.setattr(basemodel, "gather_token", fake_gather)

    model = TpPartBaseModel.__new__(TpPartBaseModel)
    model.req_manager = SimpleNamespace(
        req_sampling_params_manager=SimpleNamespace(req_to_next_token_ids=object()),
    )
    captured_inputs = []

    def fake_overlap_forward(input0, input1):
        events.append("forward")
        captured_inputs.extend((input0, input1))
        return (
            ModelOutput(logits=torch.zeros((input0.batch_size, 1))),
            ModelOutput(logits=torch.zeros((input1.batch_size, 1))),
        )

    model._microbatch_overlap_decode_cuda = fake_overlap_forward

    model.microbatch_overlap_decode(model_input0, model_input1)

    assert events == ["to_cuda", "gather", "to_cuda", "forward"]
    assert captured_inputs == [model_input0, model_input1]
    assert captured_inputs[0].input_ids.tolist() == [40, 41, 42, 43, 44, 45]
    assert captured_inputs[1].input_ids.numel() == 0


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_overlap_decode_cuda_pads_empty_side_and_unpads_outputs(monkeypatch):
    model_input0 = _make_decode_input()
    model_input0.batch_size = 1
    model_input0.total_token_num = 4
    model_input0.max_kv_seq_len = 4
    model_input0.input_ids = model_input0.input_ids[:1]
    model_input0.b_req_idx = model_input0.b_req_idx[:1]
    model_input0.b_mtp_index = model_input0.b_mtp_index[:1]
    model_input0.b_seq_len = model_input0.b_seq_len[:1]
    model_input0.b_position_delta = model_input0.b_position_delta[:1]
    model_input0.b_shared_seq_len = model_input0.b_shared_seq_len[:1]
    model_input0.b_shared_radix_node_id = model_input0.b_shared_radix_node_id[:1]
    model_input0.mem_indexes_cpu = model_input0.mem_indexes_cpu[:1]
    model_input0.multimodal_params = model_input0.multimodal_params[:1]
    model_input0.check_input()
    model_input1 = _make_empty_decode_input()
    model_input0.to_cuda()
    model_input1.to_cuda()

    model = TpPartBaseModel.__new__(TpPartBaseModel)
    model.args = SimpleNamespace(enable_tpsp_mix_mode=True)
    model.tp_world_size_ = 2
    model.graph = None
    model.req_manager = SimpleNamespace(HOLD_REQUEST_ID=88, req_to_token_indexs=object())
    model.mem_manager = SimpleNamespace(HOLD_TOKEN_MEMINDEX=77)
    infer_batch_sizes = []

    def fake_create_inferstate(model_input, microbatch_index):
        infer_batch_sizes.append(model_input.batch_size)
        return SimpleNamespace(
            b_req_idx=model_input.b_req_idx,
            b_seq_len=model_input.b_seq_len,
            mem_index=model_input.mem_indexes,
            init_some_extra_state=lambda _: None,
            init_att_state=lambda: None,
        )

    model._create_inferstate = fake_create_inferstate
    model._overlap_tpsp_token_forward = lambda infer_state0, infer_state1: (
        ModelOutput(logits=torch.zeros((infer_state0.b_req_idx.shape[0], 1), device="cuda")),
        ModelOutput(logits=torch.zeros((infer_state1.b_req_idx.shape[0], 1), device="cuda")),
    )
    monkeypatch.setattr(basemodel, "copy_kv_index_to_req", lambda *args, **kwargs: None)

    output0, output1 = model._microbatch_overlap_decode_cuda(model_input0, model_input1)

    assert infer_batch_sizes == [2, 2]
    assert output0.logits.shape == (1, 1)
    assert output1.logits.shape == (0, 1)
