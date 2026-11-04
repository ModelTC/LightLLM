"""FFN-only drafts share target weights without creating attention resources."""

from dataclasses import asdict
import json
from types import SimpleNamespace

import pytest
import torch
import torch.nn.functional as F

from unit_tests.common.execution.test_model_cuda import prepare_model_batch


def test_ffn_draft_declarations_use_checkpoint_coordinates():
    from lightllm.models.qwen3_moe_mtp.model import Qwen3MoEMTPModelAdapter

    config = dict(
        num_hidden_layers=2,
        hidden_size=256,
        moe_intermediate_size=256,
        num_experts=4,
        num_experts_per_tok=2,
        mlp_only_layers=[2],
    )
    adapter = Qwen3MoEMTPModelAdapter(
        config, dtype=torch.float16, layer_ids=(4, 5), checkpoint_layer_ids=(1, 2), shared_weights=(None,) * 3
    )
    assert adapter.attention_specs() == {} and adapter.state_specs() == ()
    assert set(adapter.moe_specs()) == {4}
    assert "norm_topk_prob" not in config


@pytest.mark.parametrize("layer_ids,checkpoint_ids", [((4,), (1, 2)), ((4, 4), (1, 2)), ((4, 5), (-1, 0))])
def test_ffn_draft_rejects_incomplete_layer_mapping(layer_ids, checkpoint_ids):
    from lightllm.models.qwen3_moe_mtp.model import Qwen3MoEMTPModelAdapter

    with pytest.raises(ValueError, match="draft layer IDs"):
        Qwen3MoEMTPModelAdapter(
            {"num_hidden_layers": 2},
            dtype=torch.float16,
            layer_ids=layer_ids,
            checkpoint_layer_ids=checkpoint_ids,
            shared_weights=(None,) * 3,
        )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.parametrize("mode", ["vanilla_no_att", "eagle_no_att"])
@pytest.mark.parametrize("layout", ["all_moe", "mixed", "sparse_step"])
@torch.no_grad()
def test_ffn_draft_checkpoint_and_graph(mode, layout, monkeypatch, tmp_path):
    from safetensors.torch import save_file
    from transformers import Qwen3MoeConfig, Qwen3MoeForCausalLM
    from lightllm.common.basemodel.mtp_manager import MtpManager
    from lightllm.common.execution.decode_graph import CudaGraph
    from lightllm.common.execution.prefill_graph import PrefillCudaGraph
    from lightllm.common.quantization.no_quant import NoQuantization
    from lightllm.distributed.communication_op import dist_group_manager
    from lightllm.models.qwen3_moe.model import Qwen3MOEModel
    from lightllm.models.qwen3_moe_mtp.model import Qwen3MOEMTPModel
    from lightllm.server.core.objs.start_args_type import StartArgs
    from lightllm.utils.envs_utils import get_env_start_args, get_llm_data_type

    torch.manual_seed(67)
    config = Qwen3MoeConfig(
        vocab_size=128,
        hidden_size=256,
        intermediate_size=512,
        num_hidden_layers=3,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=64,
        max_position_embeddings=32,
        rms_norm_eps=1e-6,
        num_experts=4,
        moe_intermediate_size=256,
        num_experts_per_tok=2,
        norm_topk_prob=True,
        mlp_only_layers=[0, 2] if layout == "mixed" else [],
        decoder_sparse_step=2 if layout == "sparse_step" else 1,
    )
    reference = Qwen3MoeForCausalLM(config).half().eval()
    target_dir, draft_dir = tmp_path / "target", tmp_path / "draft"
    target_dir.mkdir()
    draft_dir.mkdir()
    config.save_pretrained(target_dir)
    weights = {name: value.detach().contiguous() for name, value in reference.state_dict().items()}
    save_file(weights, str(target_dir / "model.safetensors"))
    draft_config = dict(config.to_dict(), num_hidden_layers=2)
    (draft_dir / "config.json").write_text(json.dumps(draft_config))
    draft_weights = {
        name: value for name, value in weights.items() if ".mlp." in name or ".post_attention_layernorm." in name
    }
    for name in ("norm_after_embedding", "norm_before_output"):
        draft_weights[f"model.layers.0.{name}.weight"] = (1 + torch.randn(256) * 0.01).half()
    draft_weights["model.layers.0.proj.weight"] = (torch.randn(256, 512) * 0.02).half()
    save_file(draft_weights, str(draft_dir / "model.safetensors"))
    reference = reference.float().cuda()
    projection = {name: value.float().cuda() for name, value in draft_weights.items() if ".layers.0." in name}

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
    draft_count = 2 if mode == "vanilla_no_att" else 1
    args = StartArgs(
        model_dir=str(target_dir),
        data_type="float16",
        page_size=1,
        running_max_req_size=2,
        llm_prefill_att_backend=["triton"],
        llm_decode_att_backend=["triton"],
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
    for cls in (Qwen3MOEModel, Qwen3MOEMTPModel):
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
        max_total_token_num=64,
        batch_max_tokens=16,
        max_req_num=2,
        max_seq_length=16,
        graph_max_batch_size=2,
        graph_max_len_in_batch=16,
        disable_cudagraph=True,
    )
    main = Qwen3MOEModel(dict(kwargs, weight_dir=str(target_dir)))
    drafts = []
    for _ in range(draft_count):
        drafts.append(
            Qwen3MOEMTPModel(
                dict(kwargs, weight_dir=str(draft_dir), main_model=main, mtp_previous_draft_models=drafts.copy())
            )
        )
    main.prepare_attention_states()
    manager = main.att_state_manager
    for index, draft in enumerate(drafts):
        adapter = draft.model_adapter
        assert adapter.attention_specs() == {} and adapter.state_specs() == ()
        assert draft.linear_plan.bindings and not draft.linear_plan.legacy and not draft.moe_plan.legacy
        assert [weight.layer_num_ for weight in draft.trans_layers_weight] == [index, index + 1]
        assert [layer.layer_num_ for layer in draft.layers_infer] == [3 + index * 2, 4 + index * 2]
        assert len(draft.moe_plan.bindings) == len(adapter.moe_specs())
        for attr in ("wte_weight_", "lm_head_weight_", "final_norm_weight_"):
            assert getattr(draft.pre_post_weight, attr) is getattr(main.pre_post_weight, attr)
        draft.prepare_attention_states()
        assert draft.att_state_manager is manager
        for weight, layer in zip(draft.trans_layers_weight, draft.layers_infer, strict=True):
            assert weight.is_moe == layer.is_moe
            if weight.is_moe:
                monkeypatch.setattr(weight.experts, "experts", lambda *a, **kw: pytest.fail("legacy MoE"))
    manager.finish_state_registration()
    original_states = manager.get_full_attention_states(main.att_state_role)
    assert manager.get_full_attention_states() == original_states
    for draft in drafts:
        draft._init_att_backend()
        draft._init_att_backend1()
        assert not manager.get_full_attention_states(draft.att_state_role)
        assert not manager.get_bounded_attention_states(draft.att_state_role)
        assert draft.prefill_att_backend is draft.decode_att_backend is None
    for state in original_states:
        state.device = torch.device("cuda")
        state.allocate_storage(64)
    main._init_execution_plan()
    target_fields = [tensor for view in main.execution_runner.store.views().values() for tensor in view.fields]
    for tensor in target_fields:
        tensor.zero_()
    monkeypatch.setattr(manager, "get_state", lambda *a, **kw: pytest.fail("runtime manager lookup"))
    monkeypatch.setattr(NoQuantization, "apply", lambda *a, **kw: pytest.fail("legacy weight computation"))
    histories = [torch.randn(16, 256, dtype=torch.float16, device="cuda") * 0.3 for _ in range(2)]

    def make_input(draft, sequences, cached, prefill):
        inp = prepare_model_batch(draft, sequences, cached, prefill, input_only=True)
        inp.mtp_draft_input_hiddens = torch.cat([h[n : len(s)] for h, n, s in zip(histories, cached, sequences)])
        return inp

    def prepare(draft, sequences, cached):
        state = draft._create_inferstate(make_input(draft, sequences, cached, False))
        draft.batch_metadata(state)
        state.init_att_state()
        return state

    def check(draft, output, sequences):
        tokens = torch.tensor([s[-1] for s in sequences], device="cuda")
        embedding = reference.model.embed_tokens(tokens)
        hidden = torch.stack([h[len(s) - 1] for h, s in zip(histories, sequences)]).float()
        norm = lambda x, name: F.rms_norm(x, (256,), projection[f"model.layers.0.{name}.weight"], 1e-6)
        hidden = F.linear(
            torch.cat((norm(embedding, "norm_after_embedding"), norm(hidden, "norm_before_output")), -1),
            projection["model.layers.0.proj.weight"],
        )
        for checkpoint_id in draft.model_adapter.checkpoint_layer_ids:
            layer = reference.model.layers[checkpoint_id]
            transformed = layer.mlp(layer.post_attention_layernorm(hidden)[None])
            if isinstance(transformed, tuple):
                transformed = transformed[0]
            hidden = hidden + transformed[0]
        expected = reference.lm_head(reference.model.norm(hidden))
        delta = output.logits.float() - expected
        assert torch.all((delta.square().mean(-1) / expected.square().mean(-1)).sqrt() <= 0.01)
        assert torch.all(delta.abs().amax(-1) <= expected.abs().amax(-1) / 32)
        assert torch.equal(output.logits.argmax(-1), expected.argmax(-1))

    try:
        for draft in drafts:
            draft.req_manager = SimpleNamespace(
                req_to_token_indexs=torch.arange(48, dtype=torch.int32, device="cuda").view(3, 16),
                HOLD_REQUEST_ID=2,
            )
            draft._init_execution_plan()
            draft._init_hidden_collector()
            sequences = [[11, 12, 13], [21, 22]]
            check(draft, draft.forward(make_input(draft, sequences, [0, 0], True)), sequences)
            sequences = [[11, 12, 13, 14], [21, 22, 23]]
            check(draft, draft._token_forward(prepare(draft, sequences, [3, 2])), sequences)
            draft.graph = CudaGraph(1, 2, 1, max_batch_size=2, max_len_in_batch=16)
            state = prepare(draft, sequences, [3, 2])
            state.is_cuda_graph_capture = True
            captured = draft.graph.capture_decode(draft._token_forward, state)
            sequences = [[11, 12, 13, 14, 15], [21, 22, 23, 24]]
            state = prepare(draft, sequences, [4, 3])
            state.is_cuda_graph_replay = True
            draft.graph.replay(state)
            check(draft, captured, sequences)
            draft.prefill_graph = PrefillCudaGraph(draft.graph, 1)
            sequences = [[11, 12, 13, 14, 15, 16, 17], [21, 22, 23, 24, 25, 26]]
            check(draft, draft.forward(make_input(draft, sequences, [5, 4], True)), sequences)
            assert 4 in draft.prefill_graph.graph
            histories[0].mul_(0.7)
            histories[1].mul_(1.3)
            check(draft, draft.forward(make_input(draft, sequences, [5, 4], True)), sequences)
        assert all(torch.count_nonzero(tensor).item() == 0 for tensor in target_fields)
    finally:
        torch.cuda.synchronize()
        for model in (main, *drafts):
            model.graph_runner.close()
        get_env_start_args.cache_clear()
        get_llm_data_type.cache_clear()
