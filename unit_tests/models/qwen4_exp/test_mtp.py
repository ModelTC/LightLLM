from types import SimpleNamespace

import pytest
import torch

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
@pytest.mark.parametrize("runtime", [{"mtp_step": 2, "mtp_mode": "eagle_with_att"}], indirect=True)
def test_native_complete_draft_prefill_cached_decode_and_full_hidden(runtime, config):
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

    state = state_for(owner, [3, 1], [3, 4], [23, 29])
    new_hidden = torch.randn(2, 64, device="cuda", dtype=torch.bfloat16)
    state.mtp_draft_input_hiddens = new_hidden
    state.hidden_collector = FinalHiddenCollector()
    actual = draft._token_forward(state)
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
