from types import SimpleNamespace
import weakref

import torch
import pytest

from lightllm.common.basemodel import basemodel
from lightllm.common.basemodel.attention.flashinfer.fp import FlashInferDecodeAttState
from lightllm.common.basemodel.attention.flashinfer.utils import should_init_decode_wrapper
from lightllm.common.basemodel.basemodel import TpPartBaseModel
from lightllm.common.basemodel.batch_objs import ModelInput, ModelMtpOutputCollector, ModelOutput
from lightllm.common.basemodel.infer_struct import InferStateInfo


def test_decode_unpad_slices_spec_output_with_logits():
    model = TpPartBaseModel.__new__(TpPartBaseModel)
    output = ModelOutput(
        logits=torch.arange(24).view(6, 4),
        mtp_collector=ModelMtpOutputCollector(spec_hidden=torch.arange(18).view(6, 3)),
    )

    unpadded = model._create_unpad_decode_model_output(output, origin_batch_size=4)

    assert unpadded.logits.shape == (4, 4)
    assert unpadded.mtp_collector.spec_hidden.shape == (4, 3)
    # Unpadding returns a shallow output copy and leaves the graph-owned
    # tensors on the original ModelOutput intact.
    assert output.logits.shape == (6, 4)
    assert output.mtp_collector.spec_hidden.shape == (6, 3)


def test_prefill_unpad_uses_token_rows_for_spec_hidden():
    model = TpPartBaseModel.__new__(TpPartBaseModel)
    output = ModelOutput(
        logits=torch.arange(20).view(5, 4),
        mtp_collector=ModelMtpOutputCollector(spec_hidden=torch.arange(24).view(8, 3)),
        prompt_logics=torch.arange(32).view(8, 4),
    )

    unpadded = model._create_unpad_prefill_model_output(
        output,
        origin_handle_token_num=6,
        origin_batch_size=3,
    )

    assert unpadded.logits.shape == (3, 4)
    assert unpadded.mtp_collector.spec_hidden.shape == (6, 3)
    assert unpadded.prompt_logics.shape == (6, 4)


def test_decode_unpad_restores_empty_output():
    model = TpPartBaseModel.__new__(TpPartBaseModel)
    output = ModelOutput(
        logits=torch.arange(4).view(1, 4),
        mtp_collector=ModelMtpOutputCollector(spec_hidden=torch.arange(3).view(1, 3)),
    )

    unpadded = model._create_unpad_decode_model_output(output, origin_batch_size=0)

    assert unpadded.logits.shape == (0, 4)
    assert unpadded.mtp_collector.spec_hidden.shape == (0, 3)


def test_prefill_unpad_restores_empty_output():
    model = TpPartBaseModel.__new__(TpPartBaseModel)
    output = ModelOutput(
        logits=torch.arange(4).view(1, 4),
        mtp_collector=ModelMtpOutputCollector(spec_hidden=torch.arange(6).view(2, 3)),
        prompt_logics=torch.arange(8).view(2, 4),
    )

    unpadded = model._create_unpad_prefill_model_output(
        output,
        origin_handle_token_num=0,
        origin_batch_size=0,
    )

    assert unpadded.logits.shape == (0, 4)
    assert unpadded.mtp_collector.spec_hidden.shape == (0, 3)
    assert unpadded.prompt_logics.shape == (0, 4)


def _create_empty_decode_input():
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
        is_prefill=False,
        multimodal_params=[],
    )


@torch.no_grad()
def test_decode_pads_only_once_after_selecting_execution_path(monkeypatch):
    execution_configs = (
        # eager 普通模式：空 batch 只补一个 dummy request。
        (None, False, 1, False, 1),
        # eager TPSP 模式：dummy request 仍需对齐到 TP world size。
        (None, True, 2, False, 2),
        # CUDA Graph replay：使用最终选中的 graph batch size。
        (4, False, 1, False, 4),
        # CUDA Graph capture：必须在初始化 attention state 前设置 graph 标记。
        (4, False, 1, True, 4),
    )
    for (
        graph_batch_size,
        enable_tpsp_mix_mode,
        tp_world_size,
        need_capture,
        expected_batch_size,
    ) in execution_configs:
        model = TpPartBaseModel.__new__(TpPartBaseModel)
        model.args = SimpleNamespace(enable_tpsp_mix_mode=enable_tpsp_mix_mode, page_size=1)
        model.tp_world_size_ = tp_world_size
        model.is_mtp_draft_model = False
        model.mtp_manager = SimpleNamespace(get_decode_batch_alignment=lambda _: 1)
        model.mem_manager = SimpleNamespace(HOLD_TOKEN_MEMINDEXES=(99,), page_size=1)
        model.req_manager = SimpleNamespace(HOLD_REQUEST_ID=88, req_to_token_indexs=object())

        pad_batch_sizes = []

        def pad_once(model_input, new_batch_size):
            pad_batch_sizes.append(new_batch_size)
            return TpPartBaseModel._create_padded_decode_model_input(model, model_input, new_batch_size)

        model._create_padded_decode_model_input = pad_once

        graph_flags_at_att_init = []
        executions = []

        def create_infer_state(model_input):
            infer_state = SimpleNamespace(
                b_req_idx=model_input.b_req_idx,
                b_seq_len=model_input.b_seq_len,
                mem_index=torch.empty((model_input.batch_size,), dtype=torch.int32),
                init_some_extra_state=lambda _: None,
                is_cuda_graph_capture=False,
                is_cuda_graph_replay=False,
            )
            infer_state.init_att_state = lambda: graph_flags_at_att_init.append(
                (infer_state.is_cuda_graph_capture, infer_state.is_cuda_graph_replay)
            )
            return infer_state

        def execute(mode, infer_state):
            executions.append((mode, infer_state.is_cuda_graph_capture, infer_state.is_cuda_graph_replay))
            return ModelOutput(logits=torch.ones((infer_state.b_req_idx.shape[0], 4)))

        model._create_inferstate = create_infer_state
        model._token_forward = lambda infer_state: execute("eager", infer_state)

        graph = None
        if graph_batch_size is not None:
            graph = SimpleNamespace()
            graph.can_run = lambda **kwargs: True
            graph.find_closest_graph_batch_size = lambda batch_size: graph_batch_size
            graph.need_capture = lambda batch_size: need_capture
            graph.capture_decode = lambda decode_func, infer_state: execute("capture", infer_state)
            graph.replay = lambda infer_state: execute("replay", infer_state)
        model.graph = graph

        output = model._decode(_create_empty_decode_input())

        expected_replay = graph_batch_size is not None and not need_capture
        expected_mode = "eager"
        if graph_batch_size is not None:
            expected_mode = "capture" if need_capture else "replay"
        assert pad_batch_sizes == [expected_batch_size]
        assert graph_flags_at_att_init == [(need_capture, expected_replay)]
        assert executions == [(expected_mode, need_capture, expected_replay)]
        assert output.logits.shape == (0, 4)


@pytest.mark.parametrize(
    "lengths,graph_mode,expected_mode",
    [
        pytest.param((1024, 2048), None, "eager", id="graph-disabled"),
        pytest.param((1024, 2048), "capture", "capture", id="capture"),
        pytest.param((1024, 2048), "replay", "replay", id="replay"),
        pytest.param((1024, 16384), "replay", "eager", id="long-second-microbatch"),
        pytest.param((16384, 1024), "capture", "eager", id="long-first-microbatch"),
    ],
)
def test_overlap_decode_initializes_attention_for_the_selected_execution_mode(lengths, graph_mode, expected_mode):
    model = TpPartBaseModel.__new__(TpPartBaseModel)
    model.args = SimpleNamespace(enable_tpsp_mix_mode=True, page_size=1)
    model.tp_world_size_ = 2
    model.is_mtp_draft_model = False
    model.mtp_manager = SimpleNamespace(get_decode_batch_alignment=lambda _: 1)
    model.mem_manager = SimpleNamespace(HOLD_TOKEN_MEMINDEXES=(99,), page_size=1)
    model.req_manager = SimpleNamespace(HOLD_REQUEST_ID=88)
    initializations = []
    executions = []

    def graph_flags(state):
        return state.is_cuda_graph_capture, state.is_cuda_graph_replay

    class DecodeState(FlashInferDecodeAttState):
        def init_state(self):
            infer_state = self.infer_state
            # Replace the external GPU wrapper allocation while preserving the
            # production decision and InferStateInfo's initialization entry point.
            if should_init_decode_wrapper(infer_state):
                self.decode_wrapper = object()
            initializations.append(
                (infer_state.microbatch_index, graph_flags(infer_state), self.decode_wrapper is not None)
            )

    def create_state(model_input, microbatch_index=0):
        state = InferStateInfo()
        state.is_prefill = False
        state.input_ids = model_input.input_ids
        state.batch_size = model_input.batch_size
        state.max_kv_seq_len = model_input.max_kv_seq_len
        state.b_req_idx = model_input.b_req_idx
        state.b_seq_len = model_input.b_seq_len
        state.microbatch_index = microbatch_index
        state.decode_att_state = DecodeState(_infer_state_weak_ref=weakref.ref(state))
        state.init_some_extra_state = lambda _: None
        return state

    def outputs(state, state1):
        return tuple(ModelOutput(logits=torch.stack((s.b_req_idx, s.b_seq_len), dim=1)) for s in (state, state1))

    def forward(state, infer_state1):
        # Eager and capture execute attention using these newly prepared states.
        assert state.decode_att_state.decode_wrapper is not None
        assert infer_state1.decode_att_state.decode_wrapper is not None
        executions.append(("forward", graph_flags(state), graph_flags(infer_state1)))
        return outputs(state, infer_state1)

    def capture(func, state, infer_state1):
        executions.append(("capture", graph_flags(state), graph_flags(infer_state1)))
        return func(state, infer_state1)

    def replay(state, infer_state1):
        # Replay uses graph-owned wrappers, so temporary states need none.
        assert state.decode_att_state.decode_wrapper is None
        assert infer_state1.decode_att_state.decode_wrapper is None
        executions.append(("replay", graph_flags(state), graph_flags(infer_state1)))
        return outputs(state, infer_state1)

    model._create_inferstate = create_state
    model._overlap_tpsp_token_forward = forward
    model.graph = None
    if graph_mode is not None:
        model.graph = SimpleNamespace(
            can_run=lambda batch_size, max_len_in_batch: batch_size <= 4 and max_len_in_batch <= 8192,
            find_closest_graph_batch_size=lambda batch_size: 4,
            need_capture=lambda batch_size: graph_mode == "capture",
            capture_decode=capture,
            replay=replay,
        )

    inputs = []
    for rows, length in zip((1, 3), lengths):
        model_input = model._create_padded_decode_model_input(_create_empty_decode_input(), rows)
        model_input.b_req_idx = torch.arange(rows, dtype=torch.int32)
        model_input.b_seq_len.fill_(length)
        model_input.max_kv_seq_len = length
        model_input.total_token_num = rows * length
        inputs.append(model_input)

    result = model._microbatch_overlap_decode_cuda(*inputs)

    expected_flags = {"eager": (False, False), "capture": (True, False), "replay": (False, True)}[expected_mode]
    assert initializations == [(index, expected_flags, expected_mode != "replay") for index in (0, 1)]
    expected_execution = {"eager": ["forward"], "capture": ["capture", "forward"], "replay": ["replay"]}[expected_mode]
    assert executions == [(mode, expected_flags, expected_flags) for mode in expected_execution]
    for output, model_input in zip(result, inputs):
        torch.testing.assert_close(output.logits, torch.stack((model_input.b_req_idx, model_input.b_seq_len), dim=1))


@pytest.mark.parametrize("overlap", [False, True])
@pytest.mark.parametrize("dynamic", [False, True])
@pytest.mark.parametrize("graph_mode", ["eager", "capture", "replay"])
@pytest.mark.parametrize("rows", [0, 3, 9, 24, 27])
def test_mtp_decode_preserves_groups_and_real_rows(overlap, dynamic, graph_mode, rows):
    model = TpPartBaseModel.__new__(TpPartBaseModel)
    model.args = SimpleNamespace(enable_tpsp_mix_mode=True)
    model.tp_world_size_ = 8
    model.is_mtp_draft_model = False
    model.mtp_manager = SimpleNamespace(get_decode_batch_alignment=lambda _: 1 if dynamic else 3)
    model.req_manager = SimpleNamespace(HOLD_REQUEST_ID=88)
    seen_sizes = []

    def create_state(model_input, microbatch_index=0):
        def check_layout():
            if not dynamic:
                assert model_input.batch_size % 3 == 0
            assert model_input.batch_size % 8 == 0
            seen_sizes.append(model_input.batch_size)

        return SimpleNamespace(
            b_req_idx=model_input.b_req_idx,
            b_seq_len=model_input.b_seq_len,
            init_some_extra_state=lambda _: None,
            init_att_state=check_layout,
        )

    def forward(state):
        return ModelOutput(logits=torch.stack((state.b_req_idx, state.b_seq_len), dim=1))

    model._create_inferstate = create_state
    model._token_forward = forward
    model._overlap_tpsp_token_forward = lambda state, infer_state1: (forward(state), forward(infer_state1))
    model.graph = None
    if graph_mode != "eager":
        model.graph = SimpleNamespace(
            can_run=lambda batch_size, max_len_in_batch: batch_size <= 24,
            find_closest_graph_batch_size=lambda batch_size: ((batch_size + 7) // 8 * 8 if dynamic else 24),
            need_capture=lambda batch_size: graph_mode == "capture",
            capture_decode=lambda func, state, **kwargs: func(state, **kwargs),
            replay=lambda state, infer_state1=None: (
                forward(state) if infer_state1 is None else (forward(state), forward(infer_state1))
            ),
        )

    model_input = model._create_padded_decode_model_input(_create_empty_decode_input(), rows)
    model_input.b_req_idx = torch.arange(rows, dtype=torch.int32) // 3
    model_input.b_seq_len = 10 + torch.arange(rows, dtype=torch.int32) % 3
    if dynamic:
        # 压缩后每请求分别保留 1、2、3 个 token，保持各请求内部的 token 顺序。
        request_rows = [(req, step) for req in range(rows) for step in range(req % 3 + 1)][:rows]
        model_input.b_req_idx = torch.tensor([req for req, _ in request_rows], dtype=torch.int32)
        model_input.b_mtp_index = torch.tensor([step for _, step in request_rows], dtype=torch.int32)
        model_input.b_seq_len = 10 + model_input.b_mtp_index
    expected = torch.stack((model_input.b_req_idx, model_input.b_seq_len), dim=1)
    if overlap:
        output, empty_output = model._microbatch_overlap_decode_cuda(model_input, _create_empty_decode_input())
        assert empty_output.logits.shape == (0, 2)
    else:
        output = model._decode(model_input)

    expected_size = (max(1, rows) + 7) // 8 * 8 if dynamic else (48 if rows > 24 else 24)
    assert seen_sizes == [expected_size] * (2 if overlap else 1)
    torch.testing.assert_close(output.logits, expected)
    torch.testing.assert_close(torch.stack((model_input.b_req_idx, model_input.b_seq_len), dim=1), expected)
