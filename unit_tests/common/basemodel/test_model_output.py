from types import SimpleNamespace

import torch

from lightllm.common.basemodel import basemodel
from lightllm.common.basemodel.basemodel import TpPartBaseModel
from lightllm.common.basemodel.batch_objs import ModelInput, ModelMtpOutputCollector, ModelOutput


def test_vocab_parallel_metadata_follows_row_operations():
    output = ModelOutput(
        logits=torch.tensor([[8.0, 1.0], [7.0, 2.0], [6.0, 3.0]]),
        logits_token_ids=torch.tensor([[4, 14], [9, 19], [2, 12]]),
    )

    selected = output.index_select_logits_rows(torch.tensor([2, 0]))
    combined = ModelOutput.concat_logits_rows([selected, output.index_select_logits_rows(torch.tensor([1]))])

    torch.testing.assert_close(combined.logits, torch.tensor([[6.0, 3.0], [8.0, 1.0], [7.0, 2.0]]))
    torch.testing.assert_close(combined.logits_token_ids, torch.tensor([[2, 12], [4, 14], [9, 19]]))


def test_decode_unpad_slices_spec_output_with_logits():
    model = TpPartBaseModel.__new__(TpPartBaseModel)
    output = ModelOutput(
        logits=torch.arange(24).view(6, 4),
        logits_token_ids=torch.arange(100, 124).view(6, 4),
        mtp_collector=ModelMtpOutputCollector(spec_hidden=torch.arange(18).view(6, 3)),
    )

    unpadded = model._create_unpad_decode_model_output(output, origin_batch_size=4)

    assert unpadded.logits.shape == (4, 4)
    assert unpadded.logits_token_ids.shape == (4, 4)
    assert unpadded.mtp_collector.spec_hidden.shape == (4, 3)
    # Unpadding returns a shallow output copy and leaves the graph-owned
    # tensors on the original ModelOutput intact.
    assert output.logits.shape == (6, 4)
    assert output.logits_token_ids.shape == (6, 4)
    assert output.mtp_collector.spec_hidden.shape == (6, 3)


def test_prefill_unpad_uses_token_rows_for_spec_hidden():
    model = TpPartBaseModel.__new__(TpPartBaseModel)
    output = ModelOutput(
        logits=torch.arange(20).view(5, 4),
        logits_token_ids=torch.arange(100, 120).view(5, 4),
        mtp_collector=ModelMtpOutputCollector(spec_hidden=torch.arange(24).view(8, 3)),
        prompt_logics=torch.arange(32).view(8, 4),
    )

    unpadded = model._create_unpad_prefill_model_output(
        output,
        origin_handle_token_num=6,
        origin_batch_size=3,
    )

    assert unpadded.logits.shape == (3, 4)
    assert unpadded.logits_token_ids.shape == (3, 4)
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
        mem_indexes=torch.empty((0,), dtype=torch.int32),
        is_prefill=False,
        multimodal_params=[],
    )


def test_infer_state_enables_vocab_parallel_topk_for_draft_or_requested_target(monkeypatch):
    model = TpPartBaseModel.__new__(TpPartBaseModel)
    model.infer_state_class = basemodel.InferStateInfo
    model.hidden_collector_prototype = SimpleNamespace(new_instance=lambda: object())
    model.is_token_healing = False
    model.return_all_prompt_logics = False
    model.is_mtp_draft_model = False
    model.mem_manager = object()
    model.req_manager = object()
    model.decode_att_backend = SimpleNamespace(create_att_decode_state=lambda infer_state: None)
    model.decode_att_backend1 = None
    monkeypatch.setattr(basemodel.dist_group_manager, "get_group", lambda _: None)

    model_input = _create_empty_decode_input()
    infer_state = model._create_inferstate(model_input)
    assert not infer_state.use_vocab_parallel_topk

    model_input.use_vocab_parallel_topk = True
    infer_state = model._create_inferstate(model_input)
    assert infer_state.use_vocab_parallel_topk

    model.is_mtp_draft_model = True
    model_input.use_vocab_parallel_topk = False
    infer_state = model._create_inferstate(model_input)
    assert infer_state.use_vocab_parallel_topk


def test_cuda_graph_contract_falls_back_for_dense_target_batch(monkeypatch):
    model = TpPartBaseModel.__new__(TpPartBaseModel)
    model.is_mtp_draft_model = False
    sparse_input = SimpleNamespace(use_vocab_parallel_topk=True)
    dense_input = SimpleNamespace(use_vocab_parallel_topk=False)

    monkeypatch.setattr(basemodel, "is_vocab_parallel_topk_enabled", lambda: True)
    assert model._is_cuda_graph_output_compatible(sparse_input)
    assert not model._is_cuda_graph_output_compatible(dense_input)
    assert not model._is_cuda_graph_output_compatible(sparse_input, dense_input)

    model.is_mtp_draft_model = True
    assert model._is_cuda_graph_output_compatible(dense_input)


@torch.no_grad()
def test_decode_pads_only_once_after_selecting_execution_path(monkeypatch):
    monkeypatch.setattr(basemodel, "copy_kv_index_to_req", lambda *args: None)

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
        model.args = SimpleNamespace(enable_tpsp_mix_mode=enable_tpsp_mix_mode)
        model.is_mtp_draft_model = False
        model.tp_world_size_ = tp_world_size
        model.mem_manager = SimpleNamespace(HOLD_TOKEN_MEMINDEX=99)
        model.req_manager = SimpleNamespace(HOLD_REQUEST_ID=88, req_to_token_indexs=object())

        pad_batch_sizes = []

        def pad_once(model_input, new_batch_size):
            pad_batch_sizes.append(new_batch_size)
            return TpPartBaseModel._create_padded_decode_model_input(model, model_input, new_batch_size)

        model._create_padded_decode_model_input = pad_once

        graph_flags_at_att_init = []

        def create_infer_state(model_input):
            infer_state = SimpleNamespace(
                b_req_idx=model_input.b_req_idx,
                b_seq_len=model_input.b_seq_len,
                mem_index=model_input.mem_indexes,
                init_some_extra_state=lambda _: None,
                is_cuda_graph=False,
            )
            infer_state.init_att_state = lambda: graph_flags_at_att_init.append(infer_state.is_cuda_graph)
            return infer_state

        model._create_inferstate = create_infer_state
        model._token_forward = lambda infer_state: ModelOutput(logits=torch.ones((infer_state.b_req_idx.shape[0], 4)))

        graph = None
        if graph_batch_size is not None:
            graph = SimpleNamespace()
            graph.can_run = lambda **kwargs: True
            graph.find_closest_graph_batch_size = lambda batch_size: graph_batch_size
            graph.need_capture = lambda batch_size: need_capture
            graph.capture_decode = lambda decode_func, infer_state: ModelOutput(
                logits=torch.ones((infer_state.b_req_idx.shape[0], 4))
            )
            graph.replay = lambda infer_state: ModelOutput(logits=torch.ones((infer_state.b_req_idx.shape[0], 4)))
        model.graph = graph

        output = model._decode(_create_empty_decode_input())

        assert pad_batch_sizes == [expected_batch_size]
        assert graph_flags_at_att_init == [need_capture]
        assert output.logits.shape == (0, 4)
