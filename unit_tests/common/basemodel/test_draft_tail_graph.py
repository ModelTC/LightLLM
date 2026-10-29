from types import SimpleNamespace

import pytest
import torch

from lightllm.common.basemodel import basemodel, cuda_graph
from lightllm.common.basemodel.basemodel import TpPartBaseModel
from lightllm.common.basemodel.batch_objs import ModelInput, ModelMtpOutputCollector, ModelOutput
from lightllm.common.basemodel.hidden_collector import NoopHiddenCollector
from lightllm.common.basemodel.infer_struct import InferStateInfo
from lightllm.models.llama.layer_infer import post_layer_infer as llama_post
from lightllm.server.core.objs.start_args_type import StartArgs


@pytest.fixture
def draft_model(monkeypatch):
    args = StartArgs(
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
            init_state=lambda: None, copy_for_decode_cuda_graph=lambda _: None, check_no_infer_state=lambda: None
        )
    )
    model.decode_att_backend1 = None

    def forward(state):
        hidden = state.input_ids[:, None].float()
        logits = hidden if state.b_logits_row_idx is None else hidden.index_select(0, state.b_logits_row_idx)
        return ModelOutput(
            logits=logits,
            input_rows_per_logit=state.input_rows_per_logit,
            mtp_collector=ModelMtpOutputCollector(spec_hidden=hidden),
        )

    model._token_forward = forward
    model._init_cudagraph()
    assert warmups == [(model.draft_tail_graph, {"input_rows_per_logit": 5}), (model.graph, {})]
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
        b_logits_row_idx=accepted,
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
        assert output.input_rows_per_logit == 5
        assert model_input.batch_size == requests * 5
        assert model_input.b_logits_row_idx is accepted
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
        assert output.input_rows_per_logit == 1
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
    graph.warmup(model, input_rows_per_logit=5)
    assert [x.batch_size for x in seen] == [10, 5]
    assert [x.b_logits_row_idx.tolist() for x in seen] == [[4, 9], [4]]


@pytest.mark.parametrize("device", ["cpu", "cuda"])
@pytest.mark.parametrize("top_k", [None, 8])
def test_draft_head_projects_selected_rows_with_candidate_mapping(monkeypatch, device, top_k):
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA is required")
    dtype = torch.bfloat16 if device == "cuda" else torch.float32
    head, weight, hidden = make_projection_head(device, dtype)
    args = SimpleNamespace(draft_vocab_topk_sampling=top_k)
    monkeypatch.setattr(llama_post, "get_env_start_args", lambda: args)
    state = SimpleNamespace(b_logits_row_idx=None, dist_group=None)
    full = head._draft_lm_head_and_gather(hidden, hidden.shape[0], weight, state)
    accepted = torch.tensor([94, 1, 53, 1, 20, 7, 35, 99, 42, 8, 91, 14, 3, 77, 60, 6, 19, 25, 41, 88], device=device)
    state.b_logits_row_idx = accepted
    selected = head._draft_lm_head_and_gather(hidden, hidden.shape[0], weight, state)
    expected_logits = full.logits.index_select(0, accepted)
    if top_k is None:
        assert selected.logits_token_ids is None
        torch.testing.assert_close(selected.logits, expected_logits)
    else:
        # Compare candidate mappings independently of top-k's unspecified order.
        expected_ids, expected_order = full.logits_token_ids.index_select(0, accepted).sort(dim=1)
        selected_ids, selected_order = selected.logits_token_ids.sort(dim=1)
        torch.testing.assert_close(selected_ids, expected_ids)
        torch.testing.assert_close(selected.logits.gather(1, selected_order), expected_logits.gather(1, expected_order))
    assert weight.projected_rows == [100, accepted.numel()]
    assert state.b_logits_row_idx is accepted


def make_projection_head(device, dtype):
    head = llama_post.LlamaPostLayerInfer.__new__(llama_post.LlamaPostLayerInfer)
    head.tp_world_size_ = 1
    head.eps_ = 1e-6
    head.alloc_tensor = lambda shape, dtype, **kwargs: torch.empty(shape, dtype=dtype, device=device)
    matrix = torch.eye(128, dtype=dtype, device=device)
    projected_rows = []

    class Weight:
        vocab_size = 128
        tp_vocab_start_id = 0

        def __call__(self, input, **kwargs):
            projected_rows.append(input.shape[1])
            return torch.mm(matrix, input)

    weight = SimpleNamespace(
        lm_head_weight_=Weight(),
        final_norm_weight_=lambda input, eps, **kwargs: (
            input.float() * torch.rsqrt(input.float().square().mean(dim=1, keepdim=True) + eps)
        ).to(input.dtype),
        projected_rows=projected_rows,
    )
    values = torch.arange(1, 129, dtype=dtype, device=device)
    hidden = torch.stack([values.roll(row) for row in range(100)])
    return head, weight, hidden


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
@pytest.mark.parametrize("top_k", [None, 8])
def test_draft_head_cuda_graph_replays_changed_row_indices(monkeypatch, top_k):
    head, weight, hidden = make_projection_head("cuda", torch.bfloat16)
    monkeypatch.setattr(llama_post, "get_env_start_args", lambda: SimpleNamespace(draft_vocab_topk_sampling=top_k))
    rows = torch.arange(20, device="cuda", dtype=torch.int64) * 5
    state = SimpleNamespace(b_logits_row_idx=rows, dist_group=None)
    # Compile the real projection/top-k paths before capture.
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        for _ in range(3):
            head._draft_lm_head_and_gather(hidden, hidden.shape[0], weight, state)
    torch.cuda.current_stream().wait_stream(stream)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        selected = head._draft_lm_head_and_gather(hidden, hidden.shape[0], weight, state)
    for offset in (1, 4):
        rows.copy_(torch.arange(20, device="cuda", dtype=torch.int64) * 5 + offset)
        hidden.copy_(hidden.roll(3, dims=1))
        expected = head._draft_lm_head_and_gather(hidden, hidden.shape[0], weight, state)
        graph.replay()
        torch.testing.assert_close(selected.logits, expected.logits)
        if top_k is not None:
            torch.testing.assert_close(selected.logits_token_ids, expected.logits_token_ids)
