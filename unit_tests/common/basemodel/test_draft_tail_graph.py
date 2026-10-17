from types import SimpleNamespace

import pytest
import torch

from lightllm.common.basemodel import basemodel, cuda_graph
from lightllm.common.basemodel.basemodel import TpPartBaseModel
from lightllm.common.basemodel.batch_objs import ModelInput, ModelMtpOutputCollector, ModelOutput
from lightllm.common.basemodel.hidden_collector import NoopHiddenCollector
from lightllm.common.basemodel.infer_struct import InferStateInfo
from lightllm.server.core.objs.start_args_type import StartArgs


@pytest.fixture
def draft_model(monkeypatch):
    args = StartArgs(
        ssm_state_mode="replay",
        ssm_replay_config={"compute_mode": "parallel"},
        mtp_mode="eagle_with_att",
        mtp_step=4,
        running_max_req_size=264,
        graph_max_batch_size=256,
        max_req_total_len=128,
    )
    monkeypatch.setattr(basemodel, "get_env_start_args", lambda: args)
    monkeypatch.setattr(cuda_graph, "get_env_start_args", lambda: args)
    monkeypatch.setattr(basemodel.dist_group_manager, "get_group", lambda _: None)
    monkeypatch.setattr(InferStateInfo, "init_some_extra_state", lambda *args: None)
    warmups = []
    monkeypatch.setattr(cuda_graph.CudaGraph, "warmup", lambda graph, model, **kw: warmups.append((graph, kw)))

    # Execute CPU row selection while retaining the real graph lookup, input
    # copying, padding and output unpadding paths.
    def capture(graph, forward, state):
        output = forward(state)

        def replay():
            current = forward(state)
            output.logits.copy_(current.logits)
            output.mtp_collector.spec_hidden.copy_(current.mtp_collector.spec_hidden)

        graph.graph[state.batch_size] = (SimpleNamespace(replay=replay), state, output)
        return output

    monkeypatch.setattr(cuda_graph.CudaGraph, "capture_decode", capture)
    model = TpPartBaseModel.__new__(TpPartBaseModel)
    model.args = args
    model.is_mtp_draft_model = True
    model.tp_world_size_ = 1
    model.config = {"model_type": "qwen3_5_text"}
    model.disable_cudagraph = False
    model.graph_max_batch_size = args.graph_max_batch_size
    model.graph_max_len_in_batch = 128
    model.max_seq_length = 128
    model.mtp_manager = SimpleNamespace(
        get_decode_tokens_per_request=lambda _: 1, get_decode_batch_alignment=lambda _: 1
    )
    model.infer_state_class = InferStateInfo
    model.hidden_collector_prototype = NoopHiddenCollector()
    model.return_all_prompt_logics = False
    model.req_manager = SimpleNamespace(HOLD_REQUEST_ID=264)
    model.mem_manager = None
    model._select_mem_indexes = lambda _: None
    model.decode_att_backend = SimpleNamespace(
        create_att_decode_state=lambda **kw: SimpleNamespace(
            init_state=lambda: None, copy_for_decode_cuda_graph=lambda _: None
        )
    )
    model.decode_att_backend1 = None

    def forward(state):
        hidden = state.input_ids[:, None].float()
        logits = hidden if state.b_logits_rows is None else hidden.index_select(0, state.b_logits_rows)
        return ModelOutput(
            logits=logits,
            logits_row_stride=state.logits_row_stride,
            mtp_collector=ModelMtpOutputCollector(spec_hidden=hidden),
        )

    model._token_forward = forward
    model._init_cudagraph()
    assert warmups == [(model.draft_tail_graph, {"logits_row_stride": 5}), (model.graph, {})]
    return model


def make_input(rows, accepted=None):
    return ModelInput(
        batch_size=rows,
        total_token_num=rows * 10,
        max_q_seq_len=1,
        max_kv_seq_len=10,
        input_ids=torch.arange(rows),
        b_req_idx=torch.arange(rows, dtype=torch.int32) // (5 if accepted is not None else 1),
        b_mtp_index=torch.zeros(rows, dtype=torch.int32),
        b_seq_len=torch.full((rows,), 10, dtype=torch.int32),
        b_position_delta=torch.zeros(rows, dtype=torch.int32),
        b_shared_seq_len=torch.zeros(rows, dtype=torch.int32),
        b_shared_radix_node_id=torch.zeros(rows, dtype=torch.int64),
        multimodal_params=[{} for _ in range(rows)],
        b_logits_rows=accepted,
    )


@pytest.mark.parametrize("eager", [False, True])
@pytest.mark.parametrize("requests", [0, 3, 32, 51, 52, 53, 264])
def test_first_draft_selects_accepted_tail_across_padding(draft_model, eager, requests):
    model = draft_model
    if eager:
        model.graph = model.draft_tail_graph = None
    for step in range(2):
        accepted = torch.arange(requests) * 5 + (torch.arange(requests) + step) % 5
        model_input = make_input(requests * 5, accepted)
        output = model._decode(model_input)
        torch.testing.assert_close(output.logits[:, 0], accepted.float())
        torch.testing.assert_close(output.mtp_collector.spec_hidden[:, 0], model_input.input_ids.float())
        assert output.logits_row_stride == 5
        assert model_input.batch_size == requests * 5
        assert model_input.b_logits_rows is accepted
    if not eager and requests == 52:
        assert 320 in model.draft_tail_graph.graph
        assert not model.graph.graph


@pytest.mark.parametrize("rows", [0, 160, 256, 260, 263, 264])
def test_later_draft_keeps_one_output_per_request(draft_model, rows):
    model = draft_model
    # 160 physical rows occur in both layouts; their captured outputs must not mix.
    model._decode(make_input(160, torch.arange(32) * 5 + 1))
    for step in range(2):
        model_input = make_input(rows)
        model_input.input_ids += step
        output = model._decode(model_input)
        assert output.logits_row_stride == 1
        torch.testing.assert_close(output.logits[:, 0], model_input.input_ids.float())
        assert output.mtp_collector.spec_hidden.shape == (rows, 1)
    if rows == 160:
        assert 160 in model.graph.graph and 160 in model.draft_tail_graph.graph
        assert model.graph.graph[160][2].logits.shape == (160, 1)
        assert model.draft_tail_graph.graph[160][2].logits.shape == (32, 1)
    elif rows > 256:
        assert not model.graph.graph


def test_tail_graph_warmup_marks_selected_rows(monkeypatch):
    args = StartArgs()
    monkeypatch.setattr(cuda_graph, "get_env_start_args", lambda: args)

    def on_cpu(factory):
        def allocate(*args, **kwargs):
            kwargs["device"] = "cpu"
            return factory(*args, **kwargs)

        return allocate

    for name in ("tensor", "full", "zeros", "arange"):
        monkeypatch.setattr(torch, name, on_cpu(getattr(torch, name)))
    seen = []
    model = SimpleNamespace(
        req_manager=SimpleNamespace(HOLD_REQUEST_ID=8, free_all=lambda: None),
        mem_manager=SimpleNamespace(free_all=lambda: None),
        _gen_special_model_input=lambda _: {},
        forward=lambda model_input: seen.append(model_input),
    )
    graph = cuda_graph.CudaGraph(5, 5, 5, max_batch_size=10)
    graph.warmup(model, logits_row_stride=5)
    assert [x.batch_size for x in seen] == [10, 5]
    assert [x.b_logits_rows.tolist() for x in seen] == [[4, 9], [4]]
