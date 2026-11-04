"""MLA drafts keep checkpoint semantics separate from shared execution layer IDs."""

from dataclasses import asdict
import json
from types import SimpleNamespace

import pytest
import torch
import torch.nn.functional as F

from lightllm.models.deepseek_mtp.model import Deepseek3MTPModel
from lightllm.models.glm4_moe_lite_mtp.model import Glm4MoeLiteMTPModel


@pytest.mark.parametrize("model_class", [Deepseek3MTPModel, Glm4MoeLiteMTPModel])
@pytest.mark.parametrize("previous_depth", [0, 2])
def test_mla_draft_declares_checkpoint_semantics_and_global_state(model_class, previous_depth):
    config = dict(
        num_hidden_layers=4 if model_class is Glm4MoeLiteMTPModel else 2,
        num_nextn_predict_layers=2,
        num_attention_heads=4,
        num_key_value_heads=4,
        hidden_size=256,
        kv_lora_rank=512,
        qk_nope_head_dim=128,
        qk_rope_head_dim=64,
        v_head_dim=128,
        n_routed_experts=4,
        n_shared_experts=1,
        first_k_dense_replace=1,
        num_experts_per_tok=2,
        moe_intermediate_size=256,
    )
    model = model_class.__new__(model_class)
    model.config = model_class.normalize_model_config(config)
    model.args = SimpleNamespace(mtp_mode="vanilla_with_att")
    model.data_type, model.tp_world_size_ = torch.float16, 1
    model.main_model = SimpleNamespace(
        layers_infer=[None] * 4, pre_post_weight=SimpleNamespace(wte_weight_=object(), lm_head_weight_=object())
    )
    model.mtp_previous_draft_models = [SimpleNamespace(layers_infer=[None] * previous_depth)] if previous_depth else []
    model._init_model_adapter()
    adapter = model.model_adapter
    start = 4 + previous_depth
    checkpoint = 4 if model_class is Glm4MoeLiteMTPModel else 0
    assert adapter.layer_ids == (start, start + 1)
    assert adapter.checkpoint_layer_ids == (checkpoint, checkpoint + 1)
    assert tuple(adapter.attention_specs()) == adapter.layer_ids
    assert [(s.role, s.producer, s.consumers) for s in adapter.state_specs()] == [
        (model.att_state_role, i, (i,)) for i in adapter.layer_ids
    ]
    assert set(adapter.moe_specs()) == ({start, start + 1} if checkpoint else {start + 1})
    assert all(spec.group_score_topk == 2 for spec in adapter.moe_specs().values())


@pytest.mark.parametrize(
    "method,default,expected_groups,expected_topk",
    [(None, 2, 2, 2), ("noaux_tc", 1, 2, 2), ("group_limited_greedy", 2, 2, 1), ("greedy", 2, 1, 1)],
)
def test_mla_routing_method_declares_group_score(method, default, expected_groups, expected_topk):
    from lightllm.common.execution.model_adapter import grouped_moe_specs

    config = dict(
        num_hidden_layers=1,
        hidden_size=128,
        moe_intermediate_size=128,
        n_routed_experts=4,
        first_k_dense_replace=0,
        n_group=2,
        topk_group=1,
        num_experts_per_tok=2,
        topk_method=method,
    )
    spec = grouped_moe_specs(config, torch.float16, group_score_topk=default)[0]
    assert spec.groups == expected_groups and spec.group_score_topk == expected_topk


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.parametrize("model_class", [Deepseek3MTPModel, Glm4MoeLiteMTPModel])
@pytest.mark.parametrize("backend", ["triton", "fa3", "flashinfer"])
@pytest.mark.parametrize("mode", ["vanilla_with_att", "eagle_with_att"])
@torch.no_grad()
def test_mla_draft_checkpoint_prefix_and_graph(model_class, backend, mode, monkeypatch, tmp_path):
    from safetensors.torch import save_file
    from transformers import DeepseekV3Config, DeepseekV3ForCausalLM
    from lightllm.common.basemodel.mtp_manager import MtpManager
    from lightllm.common.execution.decode_graph import CudaGraph
    from lightllm.common.execution.prefill_graph import PrefillCudaGraph
    from lightllm.common.quantization.no_quant import NoQuantization
    from lightllm.distributed.communication_op import dist_group_manager
    from lightllm.models.deepseek2.model import Deepseek2TpPartModel
    from lightllm.models.glm4_moe_lite.model import Glm4MoeLiteTpPartModel
    from lightllm.server.core.objs.start_args_type import StartArgs
    from lightllm.utils.envs_utils import get_env_start_args, get_llm_data_type
    from unit_tests.common.execution.test_model_cuda import prepare_model_batch

    torch.manual_seed(89)
    glm = model_class is Glm4MoeLiteMTPModel
    target_class = Glm4MoeLiteTpPartModel if glm else Deepseek2TpPartModel
    config = DeepseekV3Config(
        vocab_size=128,
        hidden_size=256,
        intermediate_size=512,
        num_hidden_layers=4 if glm else 2,
        num_attention_heads=4,
        num_key_value_heads=4,
        max_position_embeddings=32,
        rms_norm_eps=1e-6,
        rope_theta=10000.0,
        n_routed_experts=4,
        n_shared_experts=1,
        first_k_dense_replace=1,
        moe_intermediate_size=256,
        num_experts_per_tok=2,
        n_group=2,
        topk_group=1,
        norm_topk_prob=True,
        routed_scaling_factor=1.5,
        scoring_func="sigmoid",
        q_lora_rank=64,
        kv_lora_rank=512,
        qk_nope_head_dim=128,
        qk_rope_head_dim=64,
        v_head_dim=128,
    )
    reference = DeepseekV3ForCausalLM(config).half().eval()
    weights = {name: value.detach().contiguous() for name, value in reference.state_dict().items()}
    for name in tuple(weights):
        if name.endswith(".experts.gate_up_proj"):
            prefix = name.removesuffix("gate_up_proj")
            for i, value in enumerate(weights.pop(name)):
                gate, up = value.chunk(2)
                weights[f"{prefix}{i}.gate_proj.weight"] = gate.contiguous()
                weights[f"{prefix}{i}.up_proj.weight"] = up.contiguous()
        elif name.endswith(".experts.down_proj"):
            prefix = name.removesuffix("down_proj")
            for i, value in enumerate(weights.pop(name)):
                weights[f"{prefix}{i}.down_proj.weight"] = value.contiguous()
    target_dir, draft_dir = tmp_path / "target", tmp_path / "draft"
    target_dir.mkdir()
    draft_dir.mkdir()
    raw = config.to_dict()
    raw.update(model_type="glm4_moe_lite" if glm else "deepseek_v3", num_hidden_layers=2, num_nextn_predict_layers=2)
    (target_dir / "config.json").write_text(json.dumps(raw))
    save_file(weights, str(target_dir / "model.safetensors"))
    (draft_dir / "config.json").write_text(json.dumps(raw))
    checkpoint_start = 2 if glm else 0
    prefix = f"model.layers.{checkpoint_start}."
    matrix = lambda *shape: (torch.randn(*shape) * 0.02).half()
    extra = {prefix + "eh_proj.weight": matrix(256, 512)}
    for name in ("enorm", "hnorm", "shared_head.norm"):
        extra[prefix + name + ".weight"] = (1 + torch.randn(256) * 0.01).half()
    draft_weights = {name: value for name, value in weights.items() if name.startswith("model.layers.")}
    draft_weights.update(extra)
    save_file(draft_weights, str(draft_dir / "model.safetensors"))
    reference = reference.float().cuda()
    reference.model.layers = torch.nn.ModuleList(reference.model.layers[checkpoint_start : checkpoint_start + 2])
    reference.config.num_hidden_layers = 2
    reference.config._attn_implementation = "eager"
    reference.model.norm.weight.copy_(extra[prefix + "shared_head.norm.weight"].float().cuda())
    extra = {name: value.float().cuda() for name, value in extra.items()}

    for name in (
        "GLOBAL_RANK",
        "GLOBAL_DP_RANK",
        "DP_RANK_IN_NODE",
        "CURRENT_RANK_IN_DP",
        "CURRENT_RANK_IN_NODE",
        "CURRENT_DEVICE_ID",
    ):
        monkeypatch.setenv("LIGHTLLM_" + name, "0")
    for name in ("GLOBAL_WORLD_SIZE", "DP_SIZE", "DP_WORLD_SIZE", "NODE_WORLD_SIZE"):
        monkeypatch.setenv("LIGHTLLM_" + name, "1")
    draft_count = 2 if mode == "vanilla_with_att" else 1
    args = StartArgs(
        model_dir=str(target_dir),
        data_type="float16",
        page_size=1,
        running_max_req_size=2,
        llm_prefill_att_backend=[backend],
        llm_decode_att_backend=[backend],
        mtp_mode=mode,
        mtp_step=2,
        mtp_draft_model_dir=[str(draft_dir)] * draft_count,
        prefill_cudagraph_max_handle_token=8,
    )
    monkeypatch.setenv("LIGHTLLM_START_ARGS", json.dumps(asdict(args)))
    get_env_start_args.cache_clear()
    get_llm_data_type.cache_clear()
    monkeypatch.setattr(MtpManager, "_instance", None)
    monkeypatch.setattr(dist_group_manager, "groups", [SimpleNamespace()])
    for cls in (target_class, model_class):
        for name in (
            "pre_and_post_weight_class",
            "transformer_weight_class",
            "pre_layer_infer_class",
            "post_layer_infer_class",
            "transformer_layer_infer_class",
        ):
            monkeypatch.setattr(cls, name, lambda *a, **kw: pytest.fail("legacy model factory"))
    kwargs = dict(
        run_mode="normal",
        max_total_token_num=96,
        batch_max_tokens=16,
        max_req_num=2,
        max_seq_length=16,
        graph_max_batch_size=2,
        graph_max_len_in_batch=16,
        disable_cudagraph=True,
    )
    main = target_class(dict(kwargs, weight_dir=str(target_dir)))
    drafts = []
    for _ in range(draft_count):
        drafts.append(
            model_class(
                dict(kwargs, weight_dir=str(draft_dir), main_model=main, mtp_previous_draft_models=drafts.copy())
            )
        )
    for index, draft in enumerate(drafts):
        assert draft.model_adapter.layer_ids == (2 + index * 2, 3 + index * 2)
        assert draft.model_adapter.checkpoint_layer_ids == (checkpoint_start, checkpoint_start + 1)
        assert draft.pre_post_weight.wte_weight_ is main.pre_post_weight.wte_weight_
        assert draft.pre_post_weight.lm_head_weight_ is main.pre_post_weight.lm_head_weight_
        assert not draft.linear_plan.legacy and draft.linear_plan.bindings and not draft.moe_plan.legacy
        for layer, weight in zip(draft.layers_infer, draft.trans_layers_weight, strict=True):
            assert layer.is_moe == weight.is_moe
            if weight.is_moe:
                monkeypatch.setattr(weight.experts, "experts", lambda *a, **kw: pytest.fail("legacy MoE"))
    for model in (main, *drafts):
        model.prepare_attention_states()
    manager = main.att_state_manager
    manager.finish_state_registration()
    for state in manager.get_full_attention_states():
        state.device = torch.device("cuda")
        state.allocate_storage(96)
    for model in (main, *drafts):
        model.req_manager = SimpleNamespace(
            req_to_token_indexs=torch.arange(48, dtype=torch.int32, device="cuda").view(3, 16), HOLD_REQUEST_ID=2
        )
        model.prefill_att_backend = model.decode_att_backend = None
        model.prefill_att_backend1 = model.decode_att_backend1 = None
        model._init_execution_plan()
        model._init_hidden_collector()
        for view in model.execution_runner.store.views().values():
            for value in view.fields:
                value.zero_()
    assert {b.phase for b in main.execution_plan.bindings} == {"prefill", "verify"}
    main_fields = [value for view in main.execution_runner.store.views().values() for value in view.fields]
    monkeypatch.setattr(manager, "get_state", lambda *a, **kw: pytest.fail("runtime manager lookup"))
    monkeypatch.setattr(NoQuantization, "apply", lambda *a, **kw: pytest.fail("legacy weight compute"))
    histories = [torch.randn(16, 256, dtype=torch.float16, device="cuda") * 0.3 for _ in range(2)]

    def prepare(draft, sequences, cached, prefill):
        inp = prepare_model_batch(draft, sequences, cached, prefill, input_only=True)
        inp.mtp_draft_input_hiddens = torch.cat([h[n : len(s)] for h, n, s in zip(histories, cached, sequences)])
        state = draft._create_inferstate(inp)
        draft.batch_metadata(state)
        state.init_att_state()
        return state

    def check(output, sequences):
        expected = []
        norm = lambda x, name: F.rms_norm(x.float(), (256,), extra[prefix + name + ".weight"], 1e-6)
        for history, sequence in zip(histories, sequences, strict=True):
            embedding = reference.model.embed_tokens(torch.tensor(sequence, device="cuda"))
            hidden = F.linear(
                torch.cat((norm(embedding, "enorm"), norm(history[: len(sequence)], "hnorm")), -1),
                extra[prefix + "eh_proj.weight"],
            )
            expected.append(reference(inputs_embeds=hidden[None], use_cache=False).logits[:, -1])
        expected = torch.cat(expected)
        error = output.logits.float() - expected
        assert torch.all((error.square().mean(-1) / expected.square().mean(-1)).sqrt() <= 0.01)
        assert torch.all(error.abs().amax(-1) <= expected.abs().amax(-1) / 32)
        assert torch.equal(output.logits.argmax(-1), expected.argmax(-1))

    try:
        for draft in drafts:
            sequences = [[11, 12, 13], [21, 22]]
            batch = prepare(draft, sequences, [0, 0], True)
            source = batch.mtp_draft_input_hiddens.clone()
            check(draft._context_forward(batch), sequences)
            torch.testing.assert_close(batch.mtp_draft_input_hiddens, source, rtol=0, atol=0)
            sequences = [[11, 12, 13, 14], [21, 22, 23]]
            batch = prepare(draft, sequences, [3, 2], False)
            check(draft._token_forward(batch), sequences)
            batch = prepare(draft, sequences, [3, 2], False)
            source = batch.mtp_draft_input_hiddens.clone()
            draft.graph = CudaGraph(1, 2, 1, max_batch_size=2, max_len_in_batch=16)
            batch.is_cuda_graph_capture = True
            captured = draft.graph.capture_decode(draft._token_forward, batch)
            check(captured, sequences)
            torch.testing.assert_close(batch.mtp_draft_input_hiddens, source, rtol=0, atol=0)
            sequences = [[11, 12, 13, 14, 15], [21, 22, 23, 24]]
            batch = prepare(draft, sequences, [4, 3], False)
            batch.is_cuda_graph_replay = True
            draft.graph.replay(batch)
            check(captured, sequences)
            draft.prefill_graph = PrefillCudaGraph(draft.graph, 1)
            sequences = [[11, 12, 13, 14, 15, 16, 17], [21, 22, 23, 24, 25, 26]]
            check(draft._context_forward(prepare(draft, sequences, [5, 4], True)), sequences)
            check(draft._context_forward(prepare(draft, sequences, [5, 4], True)), sequences)
        assert all(torch.count_nonzero(value).item() == 0 for value in main_fields)
    finally:
        torch.cuda.synchronize()
        for model in (main, *drafts):
            model.graph_runner.close()
        get_env_start_args.cache_clear()
        get_llm_data_type.cache_clear()
