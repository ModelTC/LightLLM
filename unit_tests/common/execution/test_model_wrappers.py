"""Checkpoint wrappers retain their recipes while using native execution bindings."""

from dataclasses import asdict
from importlib import import_module
import json
from threading import Lock
from types import SimpleNamespace

import pytest
import torch

from unit_tests.common.execution.test_model_cuda import prepare_model_batch


WRAPPERS = {
    "reward": ("qwen2_reward", "Qwen2RewardTpPartModel", "qwen2", "Qwen2", None),
    "internvl_llama": ("internvl", "InternVLLlamaTpPartModel", "llama", "Llama", "llm_config"),
    "internvl_qwen2": ("internvl", "InternVLQwen2TpPartModel", "qwen2", "Qwen2", "llm_config"),
    "internvl_mla": ("internvl", "InternVLDeepSeek2TpPartModel", "deepseek_v2", "DeepseekV2", "llm_config"),
    "tarsier_llama": ("tarsier2", "Tarsier2LlamaTpPartModel", "llama", "Llama", "text_config"),
    "tarsier_qwen2": ("tarsier2", "Tarsier2Qwen2TpPartModel", "qwen2", "Qwen2", "text_config"),
    "tarsier_vl": ("tarsier2", "Tarsier2Qwen2VLTpPartModel", "qwen2_vl", "Qwen2VLText", "text_config"),
    "omni": (
        "qwen3_omni_moe_thinker",
        "Qwen3OmniMOETpPartModel",
        "qwen3_omni_moe",
        "Qwen3OmniMoeText",
        "thinker_config",
    ),
}

WRAPPERS["omni_dense"] = WRAPPERS["omni"]


def wrapper_class(family):
    module, name, *_ = WRAPPERS[family]
    return getattr(import_module(f"lightllm.models.{module}.model"), name)


def wrap_config(family, text):
    field = WRAPPERS[family][-1]
    if field == "thinker_config":
        return {field: {"text_config": text}}
    return {field: text} if field else text


@pytest.mark.parametrize("family", WRAPPERS)
def test_wrapper_startup_uses_normalized_semantics(family, tmp_path):
    model_cls = wrapper_class(family)
    text = dict(
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        hidden_size=256,
        vocab_size=128,
        num_local_experts=4,
        rope_parameters=dict(rope_type="default", rope_theta=10000.0, mrope_section=[11, 11, 10]),
    )
    (tmp_path / "config.json").write_text(json.dumps(wrap_config(family, text)))
    model = model_cls.__new__(model_cls)
    model.weight_dir_, model.finetune_config = str(tmp_path), None
    if family == "reward":
        model.max_total_token_num = 64
    model._init_config()
    if family != "reward":
        assert model.config == model_cls.normalize_model_config(wrap_config(family, text))
    assert model_cls.__dict__["model_adapter_class"] is not None
    assert model.config["rope_scaling"] == text["rope_parameters"]
    assert model.config["rope_theta"] == 10000.0
    if family.startswith("omni"):
        assert model.config["num_experts"] == 4
    assert "rope_scaling" not in text


@pytest.mark.parametrize("corruption", ["missing_bias", "up_rank", "down_width", "bias_width"])
def test_reward_checkpoint_rejects_incomplete_or_incompatible_head(corruption, monkeypatch):
    from lightllm.models.qwen2_reward.layer_weights import pre_and_post_layer_weight as module

    loader = module.Qwen2RewardPreAndPostLayerWeight.__new__(module.Qwen2RewardPreAndPostLayerWeight)
    loader.network_config_, loader.lock = {"hidden_size": 8}, Lock()
    loader.score_up_weight_ = loader.score_down_weight_ = None
    loader._score_weights = {}
    weights = {
        "score.0.weight": torch.empty(5, 8),
        "score.0.bias": torch.empty(5),
        "score.2.weight": torch.empty(1, 5),
        "score.2.bias": torch.empty(1),
    }
    monkeypatch.setattr(module, "ROWMMWeight", lambda **_: pytest.fail("allocated an invalid score head"))
    if corruption == "missing_bias":
        del weights["score.2.bias"]
        loader.load_hf_weights(weights)
        with pytest.raises(ValueError, match="missing score head"):
            loader.verify_load()
    else:
        field, shape = {
            "up_rank": ("score.0.weight", (8,)),
            "down_width": ("score.2.weight", (1, 4)),
            "bias_width": ("score.0.bias", (4,)),
        }[corruption]
        weights[field] = torch.empty(shape)
        with pytest.raises(ValueError, match="incompatible projection or bias"):
            loader.load_hf_weights(weights)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.parametrize("family", WRAPPERS)
@pytest.mark.parametrize("backend", ["triton", "fa3", "flashinfer"])
@torch.no_grad()
def test_wrapper_checkpoint_prefill_decode_prefix_and_graph(family, backend, monkeypatch, tmp_path):
    from safetensors.torch import save_file
    from lightllm.common.basemodel.mtp_manager import MtpManager
    from lightllm.common.execution.decode_graph import CudaGraph
    from lightllm.common.execution.prefill_graph import PrefillCudaGraph
    from lightllm.common.quantization.no_quant import NoQuantization
    from lightllm.distributed.communication_op import dist_group_manager
    from lightllm.server.core.objs.start_args_type import StartArgs
    from lightllm.server.router.model_infer.infer_batch import g_infer_context
    from lightllm.utils.envs_utils import get_env_start_args, get_llm_data_type

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
    args = StartArgs(
        model_dir=str(tmp_path),
        data_type="float16",
        page_size=1,
        llm_prefill_att_backend=[backend],
        llm_decode_att_backend=[backend],
        running_max_req_size=3,
        prefill_cudagraph_max_handle_token=16,
    )
    monkeypatch.setenv("LIGHTLLM_START_ARGS", json.dumps(asdict(args)))
    monkeypatch.setenv("LOADWORKER", "2")
    get_env_start_args.cache_clear()
    get_llm_data_type.cache_clear()
    monkeypatch.setattr(MtpManager, "_instance", None)
    monkeypatch.setattr(dist_group_manager, "groups", [SimpleNamespace()])
    torch.manual_seed(41)
    _, _, module, reference_name, _ = WRAPPERS[family]
    reference_module = import_module(f"transformers.models.{module}.modeling_{module}")
    config_cls = getattr(reference_module, reference_name + "Config")
    reference_cls = getattr(
        reference_module, "Qwen3OmniMoeThinkerTextModel" if family.startswith("omni") else reference_name + "Model"
    )
    mrope = family == "tarsier_vl" or family.startswith("omni")
    extra = dict(rope_parameters=dict(rope_type="default", rope_theta=10000.0))
    if mrope:
        extra["rope_parameters"]["mrope_section"] = [11, 11, 10]
    if family.startswith("omni"):
        extra.update(num_experts=4, num_experts_per_tok=2, moe_intermediate_size=192, norm_topk_prob=True)
    if family == "omni_dense":
        extra["mlp_only_layers"] = [0]
    if family == "internvl_mla":
        extra = dict(
            n_routed_experts=4,
            n_shared_experts=1,
            first_k_dense_replace=2,
            num_experts_per_tok=2,
            q_lora_rank=64,
            kv_lora_rank=512,
            qk_nope_head_dim=128,
            qk_rope_head_dim=64,
            v_head_dim=128,
            rope_scaling=dict(rope_type="yarn", factor=1.0, original_max_position_embeddings=32),
        )
    config = config_cls(
        vocab_size=128,
        hidden_size=256,
        intermediate_size=512,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=4 if family == "internvl_mla" else 2,
        head_dim=64,
        max_position_embeddings=32,
        **extra,
    )
    config._attn_implementation = "eager"
    reference = reference_cls(config).cuda().half().eval()
    if family == "reward":
        head = torch.nn.Sequential(torch.nn.Linear(256, 73), torch.nn.ReLU(), torch.nn.Linear(73, 1))
    else:
        head = torch.nn.Linear(256, 128, bias=False)
    head = head.cuda().half().eval()
    for parameter in head.parameters():
        torch.nn.init.normal_(parameter, std=0.02)
    if family != "reward" and getattr(config, "tie_word_embeddings", False):
        head.weight = reference.embed_tokens.weight
    (tmp_path / "config.json").write_text(json.dumps(wrap_config(family, config.to_dict())))
    prefix = "thinker." if family.startswith("omni") else "" if family == "reward" else "language_model."
    weights = {prefix + "model." + n: v.cpu().contiguous() for n, v in reference.state_dict().items()}
    if family == "reward":
        # Separate bias/weight files exercise unordered shard loading and arbitrary head width.
        for index, (name, value) in enumerate(head.state_dict().items()):
            save_file({"score." + name: value.cpu().contiguous()}, str(tmp_path / f"score-{index}.safetensors"))
    else:
        weights[prefix + "lm_head.weight"] = head.weight.cpu().contiguous().clone()
    save_file(weights, str(tmp_path / "language.safetensors"))
    features = (torch.randn(8, 3 if family.startswith("omni") else 1, 256) * 0.02).half().pin_memory()
    if family.startswith("omni"):
        features[6:, 1:] = 0  # Audio features have no visual deepstack contributions.
    model_cls = wrapper_class(family)
    for name in (
        "pre_and_post_weight_class",
        "transformer_weight_class",
        "pre_layer_infer_class",
        "post_layer_infer_class",
        "transformer_layer_infer_class",
    ):
        monkeypatch.setattr(model_cls, name, lambda *a, **kw: pytest.fail("legacy model factory"))
    model = model_cls(
        dict(
            run_mode="normal",
            weight_dir=str(tmp_path),
            max_total_token_num=64,
            batch_max_tokens=16,
            max_req_num=3,
            max_seq_length=16,
            graph_max_batch_size=2,
            graph_max_len_in_batch=16,
            disable_cudagraph=True,
            multimodal_embedding_cache=features,
        )
    )
    assert model.model_adapter is not None
    assert model.linear_plan.bindings and not model.linear_plan.legacy
    assert len(model.moe_plan.bindings) == (1 if family == "omni_dense" else 2 if family == "omni" else 0)
    assert not model.moe_plan.legacy
    if family == "reward":
        assert {"pre_post.score_up_weight_", "pre_post.score_down_weight_"} <= {
            name for name, _ in model.linear_plan.bindings
        }
        assert model.pre_post_weight.score_up_weight_.out_dims == [73]
    for layer_id, _ in model.moe_plan.bindings:
        monkeypatch.setattr(
            model.trans_layers_weight[layer_id].experts,
            "experts",
            lambda *a, **kw: pytest.fail("runtime weight-loader MoE"),
        )
    monkeypatch.setattr(NoQuantization, "apply", lambda *a, **kw: pytest.fail("runtime weight-loader compute"))
    model.prepare_attention_states()
    model.att_state_manager.finish_state_registration()
    for state in model.att_state_manager.get_full_attention_states():
        state.device = torch.device("cuda")
        state.allocate_storage(64)
    model._init_execution_plan()
    model._init_hidden_collector()
    model.prefill_att_backend = model.decode_att_backend = None
    model.prefill_att_backend1 = model.decode_att_backend1 = None
    model.req_manager = SimpleNamespace(
        req_to_token_indexs=torch.arange(48, dtype=torch.int32, device="cuda").view(3, 16),
        HOLD_REQUEST_ID=2,
    )
    monkeypatch.setattr(model.att_state_manager, "get_state", lambda **_: pytest.fail("runtime resource lookup"))
    monkeypatch.setattr(g_infer_context, "cpu_embed_cache_client", None)
    virtual = 2**62
    images = [
        dict(token_id=virtual, token_num=4, start_index_in_embed_cache=0, start_idx=2, grid_thwd=(1, 2, 2, -2)),
        dict(token_id=virtual + 8, token_num=2, start_index_in_embed_cache=4, start_idx=1, grid_thwd=(1, 1, 2, 0)),
    ]
    prompts = [[11, 12, *range(virtual, virtual + 4), 13], [21, virtual + 8, virtual + 9, 22]]
    params = [{"images": [img], "audios": []} for img in images]
    if family.startswith("omni"):
        prompts[1] = [21, virtual + 8, virtual + 9, virtual + 16, virtual + 17, 22]
        params[1]["audios"] = [dict(token_id=virtual + 16, token_num=2, start_index_in_embed_cache=6, start_idx=3)]
    elif family == "reward":
        prompts = [[11, 12, 13, 14, 15, 16, 17], [21, 22, 23, 24]]
        params = [{"images": [], "audios": []} for _ in prompts]

    def prepare(sequences, cached, prefill, *, input_only=False):
        inp = prepare_model_batch(model, sequences, cached, prefill, input_only=True)
        inp.multimodal_params = params
        if not prefill and mrope:
            inp.b_position_delta = torch.tensor([-2, 0], dtype=torch.int32, device="cuda")
        if input_only:
            return inp
        batch = model._create_inferstate(inp)
        model.batch_metadata(batch)
        batch.init_att_state()
        return batch

    def expected(sequences):
        outputs = []
        for sequence, items in zip(sequences, params):
            ids = torch.tensor(sequence, device="cuda")
            embeds = reference.embed_tokens(ids.masked_fill(ids >= virtual, 0))
            positions = torch.arange(len(sequence), device="cuda").expand(3, -1).clone()
            kwargs = {}
            for item in items["images"] + items["audios"]:
                mask = (ids >= item["token_id"]) & (ids < item["token_id"] + item["token_num"])
                offsets = ids[mask] - item["token_id"] + item["start_index_in_embed_cache"]
                embeds[mask] = features[:, 0].cuda()[offsets]
                if "grid_thwd" not in item:
                    continue
                start, size = item["start_idx"], item["token_num"]
                _, _, width, delta = item["grid_thwd"]
                for j in range(start, len(sequence)):
                    if j < start + size:
                        n = j - start
                        positions[:, j] = torch.tensor([start, start + n // width, start + n % width], device="cuda")
                    else:
                        positions[:, j] += delta
                if family.startswith("omni"):
                    kwargs.update(
                        visual_pos_masks=mask[None],
                        deepstack_visual_embeds=[features[:, i].cuda()[offsets] for i in (1, 2)],
                    )
            if mrope:
                kwargs["position_ids"] = positions[:, None]
            hidden = reference(inputs_embeds=embeds[None], use_cache=False, **kwargs).last_hidden_state
            outputs.append(head(hidden[:, -1]).float())
        return torch.cat(outputs)

    def check(output, sequences):
        target = expected(sequences)
        torch.testing.assert_close(output.logits.float(), target, rtol=1e-2, atol=4e-3)
        if family != "reward":
            assert torch.equal(output.logits.argmax(-1), target.argmax(-1))

    try:
        chunks = [prompts[0][:5], prompts[1][:2]]
        check(model._context_forward(prepare(chunks, [0, 0], True)), chunks)
        check(model._context_forward(prepare(prompts, [5, 2], True)), prompts)
        cached = list(map(len, prompts))
        sequences = [prompts[0] + [31], prompts[1] + [41]]
        batch = prepare(sequences, cached, False)
        check(model._token_forward(batch), sequences)
        model.graph = CudaGraph(1, 2, 1, max_batch_size=2, max_len_in_batch=16)
        batch.is_cuda_graph_capture = True
        captured = model.graph.capture_decode(model._token_forward, batch)
        cached = list(map(len, sequences))
        sequences = [sequences[0] + [32], sequences[1] + [42]]
        batch = prepare(sequences, cached, False)
        batch.is_cuda_graph_replay = True
        model.graph.replay(batch)
        check(captured, sequences)
        model.prefill_graph = PrefillCudaGraph(model.graph, 1)
        check(model.forward(prepare(prompts, [0, 0], True, input_only=True)), prompts)
        features.mul_(0.5)
        prompts[0][0], prompts[1][0] = 51, 61
        check(model.forward(prepare(prompts, [0, 0], True, input_only=True)), prompts)
        assert model.prefill_graph.graph
    finally:
        torch.cuda.synchronize()
        model.graph_runner.close()
        get_env_start_args.cache_clear()
        get_llm_data_type.cache_clear()
