from types import SimpleNamespace

import pytest
import torch

from lightllm.common.basemodel.batch_objs import ModelInput
from lightllm.common.basemodel.cuda_graph import CudaGraph
from lightllm.common.basemodel.hidden_collector import FinalHiddenCollector
from lightllm.models.draft_registry import get_draft_model_class
from lightllm.models.qwen3next.model import Qwen3NextTpPartModel
from lightllm.models.qwen4_exp.mtp.model import Qwen4ExpMTPModel
from lightllm.models.qwen4_exp.mtp.infer import Qwen4ExpMTPPreLayerInfer
from .helpers import checkpoint, prepost_checkpoint, add_hyper, rand
from .runtime import state_for
from .mtp_reference import reference_fusion, reference_draft


def test_public_draft_registry():
    for model_type in ("qwen4_exp", "qwen4_exp_text"):
        assert get_draft_model_class({"model_type": model_type}, "eagle_with_att") is Qwen4ExpMTPModel


def build_draft(runtime, config):
    main = runtime.model
    source = prepost_checkpoint(config)
    main.pre_post_weight = main.pre_and_post_weight_class(torch.bfloat16, config)
    main.pre_post_weight.load_hf_weights(dict(source))
    main.pre_post_weight.verify_load()
    main.layers_infer = [None] * config["n_layer"]
    draft = Qwen4ExpMTPModel.__new__(Qwen4ExpMTPModel)
    draft.__dict__.update(
        main_model=main,
        mtp_previous_draft_models=[],
        args=main.args,
        weight_dir_=main.args.model_dir,
        finetune_config=None,
        tp_world_size_=1,
        tp_rank_=0,
        torch_memory_saver=main.torch_memory_saver,
        data_type=torch.bfloat16,
        quant_cfg=runtime.weight.quant_cfg,
        max_req_num=main.max_req_num,
        prefill_graph=None,
    )
    draft._init_config()
    draft._init_weights()
    from .resource_helpers import bind_states

    draft.bind_req_manager(main.req_manager)
    draft._init_infer_layer()
    main.att_state_manager._state_registration_finished = False
    draft.prepare_attention_states()
    runtime.mem = bind_states(main)
    main.att_state_manager.get_state.cache_clear()
    draft._init_custom()
    draft._init_infer_layer()
    draft._init_some_value()
    draft._init_att_backend()
    draft._init_att_backend1()
    draft.mtp_manager = SimpleNamespace(get_decode_draft_step=lambda is_draft: 0)
    for name, value in checkpoint(config, 3).items():
        source[name.replace("model.language_model.layers.3.", "mtp.layers.0.", 1)] = value
    add_hyper(source, "mtp.hyper_connection_mixer", config, False)
    source.update(
        {
            "mtp.pre_fc_norm_embedding.weight": rand(config["hidden_size"]),
            "mtp.pre_fc_norm_hidden.weight": rand(config["hidden_size"] * config["hc_count"]),
            "mtp.fc_embedding.weight": rand(config["hidden_size"], config["hidden_size"]),
            "mtp.fc_hidden.weight": rand(config["hidden_size"], config["hidden_size"]),
        }
    )

    for name in reversed(list(source)):
        draft.pre_post_weight.load_hf_weights({name: source[name]})
        draft.trans_layers_weight[0].load_hf_weights({name: source[name]})
    draft.pre_post_weight.verify_load()
    draft.trans_layers_weight[0].verify_load()
    return draft, source


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
@pytest.mark.parametrize(
    "runtime",
    [{"mtp_step": 2, "mtp_mode": "eagle_with_att"}, {"mtp_step": 2, "mtp_mode": "eagle_with_att", "page_size": 4}],
    indirect=True,
)
@pytest.mark.parametrize("graph_preparation", [False, True])
def test_native_complete_draft_prefill_cached_decode_and_full_hidden(runtime, config, graph_preparation):
    torch.manual_seed(2307)
    Qwen3NextTpPartModel._init_triton(None)
    draft, source = build_draft(runtime, config)
    assert draft.req_manager is runtime.req and draft.att_state_manager is runtime.model.att_state_manager
    assert draft.pre_post_weight.wte_weight_ is runtime.model.pre_post_weight.wte_weight_
    assert draft.pre_post_weight.lm_head_weight_ is runtime.model.pre_post_weight.lm_head_weight_
    assert draft.layers_infer[0].layer_num_ == 4
    assert not hasattr(draft.trans_layers_weight[0], "ple")
    for request in (1, 3):
        runtime.req.init_linear_att_state(SimpleNamespace(req_idx=request))
    owner = SimpleNamespace(model=draft, req=runtime.req, mem=runtime.mem)
    tokens = torch.tensor([7, 11, 13, 17, 19], device="cuda", dtype=torch.int64)
    hidden = torch.randn(5, 64, device="cuda", dtype=torch.bfloat16)
    hidden_before = hidden.clone()
    state = state_for(owner, [1, 3], [3, 2], tokens.tolist(), ready=[0, 0])
    state.mtp_draft_input_hiddens = hidden
    state.hidden_collector = FinalHiddenCollector()
    assert not state.qwen4_exp_ple
    actual = draft._context_forward(state)
    positions = torch.tensor([0, 1, 2, 0, 1], device="cuda")
    cos, sin = draft._cos_cached[positions].repeat(1, 2), draft._sin_cached[positions].repeat(1, 2)
    multi, logits = reference_draft(tokens, hidden, source, config, cos, sin, [3, 2])
    torch.testing.assert_close(actual.logits, logits[[2, 4]], atol=0.025, rtol=0.025)
    torch.testing.assert_close(actual.mtp_collector.spec_hidden, multi, atol=0.016, rtol=0.016)
    torch.testing.assert_close(hidden, hidden_before, atol=0, rtol=0)
    fusion = draft.pre_post_weight.fusion.component
    embeddings = source["model.language_model.embed_tokens.weight"].cuda()[tokens]
    torch.testing.assert_close(
        fusion(embeddings, hidden), reference_fusion(embeddings, hidden, source, config), atol=0.016, rtol=0.016
    )
    with pytest.raises(ValueError, match="full pre-mixer"):
        fusion(embeddings, hidden[:, :32].contiguous())

    image_pre = Qwen4ExpMTPPreLayerInfer(dict(config, image_token_id=60))
    virtual = torch.tensor([7, 1000000, 1000001], device="cuda", dtype=torch.int64)
    image_state = SimpleNamespace(
        multimodal_params=[{"images": [{"token_id": 1000000, "token_num": 2}]}],
        mtp_draft_input_hiddens=hidden[:3],
        dist_group=None,
    )
    canonical = torch.tensor([7, 60, 60], device="cuda")
    image_input = image_pre.context_forward(virtual, image_state, draft.pre_post_weight)
    image_embedding = source["model.language_model.embed_tokens.weight"].cuda()[canonical]
    torch.testing.assert_close(
        image_input, reference_fusion(image_embedding, hidden[:3], source, config), atol=0.016, rtol=0.016
    )
    assert virtual.tolist() == [7, 1000000, 1000001]

    state = state_for(owner, [3, 1], [3, 4], [23, 29], graph_mode="capture" if graph_preparation else None)
    new_hidden = torch.randn(2, 64, device="cuda", dtype=torch.bfloat16)
    state.mtp_draft_input_hiddens = new_hidden
    state.hidden_collector = FinalHiddenCollector()
    actual = draft._token_forward(state)
    if graph_preparation:
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            actual = draft._token_forward(state)
        graph.replay()
    full_tokens = torch.tensor([7, 11, 13, 29, 17, 19, 23], device="cuda")
    full_hidden = torch.cat([hidden[:3], new_hidden[1:2], hidden[3:], new_hidden[:1]])
    positions = torch.tensor([0, 1, 2, 3, 0, 1, 2], device="cuda")
    multi, logits = reference_draft(
        full_tokens,
        full_hidden,
        source,
        config,
        draft._cos_cached[positions].repeat(1, 2),
        draft._sin_cached[positions].repeat(1, 2),
        [4, 3],
    )
    torch.testing.assert_close(actual.logits, logits[[6, 3]], atol=0.025, rtol=0.025)
    torch.testing.assert_close(actual.mtp_collector.spec_hidden, multi[[6, 3]], atol=0.016, rtol=0.016)

    if graph_preparation:
        following = state_for(owner, [1, 3], [5, 4], [31, 37], graph_mode="replay")
        following_hidden = torch.randn(2, 64, device="cuda", dtype=torch.bfloat16)
        following.mtp_draft_input_hiddens = following_hidden
        following.hidden_collector = FinalHiddenCollector()
        state.copy_for_cuda_graph(following)
        graph.replay()
        full_tokens = torch.tensor([7, 11, 13, 29, 31, 17, 19, 23, 37], device="cuda")
        full_hidden = torch.cat([full_hidden[:4], following_hidden[:1], full_hidden[4:], following_hidden[1:]])
        positions = torch.tensor([0, 1, 2, 3, 4, 0, 1, 2, 3], device="cuda")
        multi, logits = reference_draft(
            full_tokens,
            full_hidden,
            source,
            config,
            draft._cos_cached[positions].repeat(1, 2),
            draft._sin_cached[positions].repeat(1, 2),
            [5, 4],
        )
        torch.testing.assert_close(actual.logits, logits[[4, 8]], atol=0.025, rtol=0.025)
        torch.testing.assert_close(actual.mtp_collector.spec_hidden, multi[[4, 8]], atol=0.016, rtol=0.016)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
@pytest.mark.parametrize("runtime", [{"mtp_step": 3, "mtp_mode": "eagle_with_att", "page_size": 4}], indirect=True)
@pytest.mark.parametrize("prewarm", [False, True])
def test_selected_draft_logits_keep_hidden_and_kv(runtime, config, monkeypatch, prewarm):
    import copy
    from lightllm.distributed import dist_group_manager

    torch.manual_seed(2309)
    Qwen3NextTpPartModel._init_triton(None)
    draft, _ = build_draft(runtime, config)
    draft.hidden_collector_prototype = FinalHiddenCollector()
    draft.return_all_prompt_logics = False
    draft.graph = None
    draft._align_decode_batch_size = lambda _: 8
    monkeypatch.setattr(dist_group_manager, "get_group", lambda _: None)
    graph = CudaGraph(8, 8, 8, max_batch_size=8, max_len_in_batch=64)
    if prewarm:
        draft.graph = graph
        graph.warmup(draft)
        assert set(graph.graph) == {8, (8, 1), (8, 2), (8, 4), (8, 8)}
        draft.graph = None

        def unexpected_capture(*args, **kwargs):
            raise AssertionError("Selected logits graph was not warmed before serving")

        monkeypatch.setattr(graph, "capture_decode", unexpected_capture)
    for request in (1, 3):
        runtime.req.init_linear_att_state(SimpleNamespace(req_idx=request))
    owner = SimpleNamespace(model=draft, req=runtime.req, mem=runtime.mem)
    prefix = state_for(owner, [1, 3], [3, 3], [7, 11, 13, 17, 19, 23], ready=[0, 0])
    prefix.mtp_draft_input_hiddens = torch.randn(6, 64, device="cuda", dtype=torch.bfloat16)
    prefix.hidden_collector = FinalHiddenCollector()
    draft._context_forward(prefix)
    state = state_for(owner, [1, 1, 3, 3], [4, 5, 4, 5], [29, 31, 37, 41])
    model_input = ModelInput(
        batch_size=4,
        total_token_num=18,
        max_q_seq_len=1,
        max_kv_seq_len=5,
        input_ids=state.input_ids,
        b_req_idx=state.b_req_idx,
        b_mtp_index=torch.tensor([0, 1, 0, 1], device="cuda", dtype=torch.int32),
        b_seq_len=state.b_seq_len,
        b_position_delta=state.b_position_delta,
        b_shared_seq_len=torch.zeros(4, device="cuda", dtype=torch.int32),
        b_shared_radix_node_id=torch.full((4,), -1, device="cuda", dtype=torch.int64),
        multimodal_params=[{"images": [], "audios": []} for _ in range(4)],
        mtp_draft_input_hiddens=torch.randn(4, 64, device="cuda", dtype=torch.bfloat16),
    )
    full = draft.forward(model_input)
    buffers = []
    for resource in draft.att_state_manager.get_full_attention_states(draft.att_state_role):
        buffers.extend([resource.k_storage, resource.v_storage, resource.indexer.k_storage])
        assert resource.indexer.v_storage is None
    kv = [buffer.index_select(1, state.mem_index.long()) for buffer in buffers]
    draft.graph = graph
    for selection in ([1, 3], [0, 2], [3], [0, 1, 3]):
        selected = copy.copy(model_input)
        selected.logits_indices = torch.tensor(selection, device="cuda", dtype=torch.int64)
        output = draft.forward(selected)
        torch.testing.assert_close(output.logits, full.logits[selection], atol=0.025, rtol=0.025)
        torch.testing.assert_close(output.mtp_collector.spec_hidden, full.mtp_collector.spec_hidden, atol=0, rtol=0)
        for buffer, values in zip(buffers, kv):
            torch.testing.assert_close(
                buffer.index_select(1, state.mem_index.long()), values, atol=0, rtol=0, equal_nan=True
            )
        assert selected.logits_indices.numel() == len(selection)
        assert model_input.logits_indices is None
    if not prewarm:
        assert set(graph.graph) == {(8, 1), (8, 2), (8, 4)}
    model_input.input_ids.add_(2)
    model_input.mtp_draft_input_hiddens.normal_()
    draft.graph = None
    expected = draft.forward(model_input)
    draft.graph = graph
    selected = copy.copy(model_input)
    selected.logits_indices = torch.tensor([0, 3], device="cuda", dtype=torch.int64)
    output = draft.forward(selected)
    torch.testing.assert_close(output.logits, expected.logits[[0, 3]], atol=0.025, rtol=0.025)
    torch.testing.assert_close(output.mtp_collector.spec_hidden, expected.mtp_collector.spec_hidden, atol=0, rtol=0)
    complete = draft.forward(model_input)
    torch.testing.assert_close(complete.logits, expected.logits, atol=0.025, rtol=0.025)
    if not prewarm:
        assert set(graph.graph) == {8, (8, 1), (8, 2), (8, 4)}
