"""VL language recipes with real image features and independent Transformers math."""

from dataclasses import asdict
import json
from types import SimpleNamespace

import pytest
import torch

from unit_tests.common.execution.test_model_cuda import prepare_model_batch

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")


@pytest.mark.parametrize("family", ["qwen2_vl", "qwen3_vl", "qwen3_vl_moe"])
@pytest.mark.parametrize("backend", ["triton", "fa3", "flashinfer"])
@torch.no_grad()
def test_image_features_prefill_decode_prefix_and_graph(family, backend, monkeypatch, tmp_path):
    from transformers.models.qwen2_vl.modeling_qwen2_vl import Qwen2VLTextConfig, Qwen2VLTextModel
    from transformers.models.qwen3_vl.modeling_qwen3_vl import Qwen3VLTextConfig, Qwen3VLTextModel
    from transformers.models.qwen3_vl_moe.modeling_qwen3_vl_moe import Qwen3VLMoeTextConfig, Qwen3VLMoeTextModel
    from lightllm.models.qwen2_vl.model import Qwen2VLTpPartModel
    from lightllm.models.qwen3_vl.model import Qwen3VLTpPartModel
    from lightllm.models.qwen3_vl_moe.model import Qwen3VLMOETpPartModel
    from lightllm.common.basemodel.mtp_manager import MtpManager
    from lightllm.common.execution.runner.decode_graph import CudaGraph
    from lightllm.common.execution.runner.prefill_graph import PrefillCudaGraph
    from lightllm.common.quantization.no_quant import NoQuantization
    from lightllm.distributed.communication_op import dist_group_manager
    from lightllm.server.core.objs.start_args_type import StartArgs
    from lightllm.server.router.model_infer.infer_batch import g_infer_context
    from lightllm.utils.envs_utils import get_env_start_args, get_llm_data_type

    config_cls, reference_cls, model_cls = {
        "qwen2_vl": (Qwen2VLTextConfig, Qwen2VLTextModel, Qwen2VLTpPartModel),
        "qwen3_vl": (Qwen3VLTextConfig, Qwen3VLTextModel, Qwen3VLTpPartModel),
        "qwen3_vl_moe": (Qwen3VLMoeTextConfig, Qwen3VLMoeTextModel, Qwen3VLMOETpPartModel),
    }[family]
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
        running_max_req_size=2,
        prefill_cudagraph_max_handle_token=11,
    )
    monkeypatch.setenv("LIGHTLLM_START_ARGS", json.dumps(asdict(args)))
    get_env_start_args.cache_clear()
    get_llm_data_type.cache_clear()
    monkeypatch.setattr(MtpManager, "_instance", None)
    monkeypatch.setattr(dist_group_manager, "groups", [SimpleNamespace()])
    torch.manual_seed(41)
    config = config_cls(
        vocab_size=128,
        hidden_size=256,
        intermediate_size=512,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=64,
        max_position_embeddings=32,
        num_experts=4,
        num_experts_per_tok=2,
        moe_intermediate_size=192,
        norm_topk_prob=True,
        rope_parameters=dict(rope_type="default", rope_theta=10000.0, mrope_section=[11, 11, 10]),
    )
    config._attn_implementation = "eager"
    reference = reference_cls(config).cuda().half().eval()
    head = torch.nn.Linear(256, 128, bias=False).cuda().half().eval()
    torch.nn.init.normal_(head.weight, std=0.02)
    if getattr(config, "tie_word_embeddings", False):
        head.weight = reference.embed_tokens.weight
    raw_config = config.to_dict()
    if family != "qwen2_vl":
        raw_config = {"text_config": raw_config, "vision_config": {}}
    (tmp_path / "config.json").write_text(json.dumps(raw_config))
    prefix = "model." if family == "qwen2_vl" else "model.language_model."
    weights = {prefix + name: value.cpu() for name, value in reference.state_dict().items()}
    weights["lm_head.weight"] = head.weight.cpu()
    if family == "qwen3_vl_moe":
        # Export the VL checkpoint's packed [E,H,2I] / [E,I,H] layout.
        for name in weights:
            if name.endswith((".experts.gate_up_proj", ".experts.down_proj")):
                weights[name] = weights[name].transpose(1, 2).contiguous()
    features = (torch.randn(6, 1 if family == "qwen2_vl" else 3, 256) * 0.02).half().pin_memory()
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
            max_req_num=2,
            max_seq_length=16,
            graph_max_batch_size=2,
            graph_max_len_in_batch=16,
            disable_cudagraph=True,
            weight_dict=weights,
            multimodal_embedding_cache=features,
        )
    )
    assert model.model_adapter is not None
    assert model.linear_plan.bindings and not model.linear_plan.legacy
    assert len(model.moe_plan.bindings) == (2 if family == "qwen3_vl_moe" else 0)
    assert not model.moe_plan.legacy
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
        req_to_token_indexs=torch.arange(32, dtype=torch.int32, device="cuda").view(2, 16)
    )
    monkeypatch.setattr(model.att_state_manager, "get_state", lambda **_: pytest.fail("runtime resource lookup"))
    monkeypatch.setattr(g_infer_context, "cpu_embed_cache_client", None)
    virtual = 2 ** 62
    images = [
        dict(token_id=virtual, token_num=4, start_index_in_embed_cache=0, start_idx=2, grid_thwd=(1, 2, 2, -2)),
        dict(token_id=virtual + 8, token_num=2, start_index_in_embed_cache=4, start_idx=1, grid_thwd=(1, 1, 2, 0)),
    ]
    prompts = [[11, 12, *range(virtual, virtual + 4), 13], [21, virtual + 8, virtual + 9, 22]]

    def prepare(sequences, cached, prefill):
        inp = prepare_model_batch(model, sequences, cached, prefill, input_only=True)
        inp.multimodal_params = [{"images": [dict(img)], "audios": []} for img in images]
        if not prefill:
            inp.b_position_delta = torch.tensor([-2, 0], dtype=torch.int32, device="cuda")
        batch = model._create_inferstate(inp)
        model.batch_metadata(batch)
        batch.init_att_state()
        return batch

    def expected(sequences):
        results = []
        for sequence, img in zip(sequences, images):
            ids = torch.tensor(sequence, device="cuda")
            mask = ids >= virtual
            embeds = reference.embed_tokens(ids.masked_fill(mask, 0))
            offsets = ids[mask] - img["token_id"] + img["start_index_in_embed_cache"]
            embeds[mask] = features[:, 0].cuda()[offsets]
            positions = torch.arange(len(sequence), device="cuda").expand(3, -1).clone()
            start, size = img["start_idx"], img["token_num"]
            _, _, width, delta = img["grid_thwd"]
            for j in range(start, len(sequence)):
                if j < start + size:
                    n = j - start
                    positions[:, j] = torch.tensor([start, start + n // width, start + n % width], device="cuda")
                else:
                    positions[:, j] += delta
            kwargs = {}
            if family != "qwen2_vl":
                kwargs.update(
                    visual_pos_masks=mask[None],
                    deepstack_visual_embeds=[features[:, i].cuda()[offsets] for i in (1, 2)],
                )
            hidden = reference(
                inputs_embeds=embeds[None], position_ids=positions[:, None], use_cache=False, **kwargs
            ).last_hidden_state
            results.append(head(hidden[:, -1]).float())
        return torch.cat(results)

    def check(output, sequences):
        target = expected(sequences)
        torch.testing.assert_close(output.logits, target, rtol=1e-2, atol=4e-3)
        assert torch.equal(output.logits.argmax(-1), target.argmax(-1))

    try:
        # First chunks stop inside both images. Later chunks reuse their KV prefix.
        chunks = [prompts[0][:5], prompts[1][:2]]
        check(model._context_forward(prepare(chunks, [0, 0], True)), chunks)
        check(model._context_forward(prepare(prompts, [5, 2], True)), prompts)
        sequences = [prompts[0] + [14], prompts[1] + [23]]
        batch = prepare(sequences, [7, 4], False)
        check(model._token_forward(batch), sequences)
        model.graph = CudaGraph(1, 2, 1, max_batch_size=2, max_len_in_batch=16)
        batch.is_cuda_graph_capture = True
        captured = model.graph.capture_decode(model._token_forward, batch)
        sequences = [sequences[0] + [15], sequences[1] + [24]]
        batch = prepare(sequences, [8, 5], False)
        batch.is_cuda_graph_replay = True
        model.graph.replay(batch)
        check(captured, sequences)
        # Capture and replay on new image contents and fresh request metadata.
        model.prefill_graph = PrefillCudaGraph(model.graph, 1)
        check(model._context_forward(prepare(prompts, [0, 0], True)), prompts)
        features.mul_(0.5).add_(0.003)
        prompts[0][0], prompts[1][0] = 31, 41
        check(model._context_forward(prepare(prompts, [0, 0], True)), prompts)
        assert model.prefill_graph.graph
    finally:
        torch.cuda.synchronize()
        model.graph_runner.close()
        get_env_start_args.cache_clear()
        get_llm_data_type.cache_clear()
