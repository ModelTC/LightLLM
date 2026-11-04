"""Draft checkpoint execution with shared target weights and an independent FP32 recipe."""

from dataclasses import asdict
import json
import os
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
import torch.nn.functional as F

from unit_tests.common.execution.test_model_cuda import prepare_model_batch


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.skipif(not os.environ.get("LIGHTLLM_TEST_QWEN35_CHECKPOINT"), reason="local Qwen3.5 checkpoint required")
@pytest.mark.parametrize("backend", ["triton", "fa3", "flashinfer"])
@torch.no_grad()
def test_qwen35_draft_checkpoint_prefix_and_graph(backend, monkeypatch, moe=False):
    from safetensors import safe_open
    from lightllm.common.basemodel.mtp_manager import MtpManager
    from lightllm.common.basemodel.batch_objs import ModelInput
    from lightllm.core.runner.decode_graph import CudaGraph
    from lightllm.core.runner.prefill_graph import PrefillCudaGraph
    from lightllm.common.quantization.no_quant import NoQuantization
    from lightllm.distributed.communication_op import dist_group_manager
    from lightllm.server.core.objs.start_args_type import StartArgs
    from lightllm.utils.envs_utils import get_env_start_args, get_llm_data_type

    if moe:
        from transformers.models.qwen3_5_moe import modeling_qwen3_5_moe as hf
        from lightllm.models.qwen3_5_moe.model import Qwen3_5MOETpPartModel as Qwen3_5TpPartModel
        from lightllm.models.qwen3_5_moe_mtp.model import Qwen3_5MoeMTPModel as Qwen3_5MTPModel
    else:
        from transformers.models.qwen3_5 import modeling_qwen3_5 as hf
        from lightllm.models.qwen3_5.model import Qwen3_5TpPartModel
        from lightllm.models.qwen3_5_mtp.model import Qwen3_5MTPModel

    directory = Path(os.environ["LIGHTLLM_TEST_QWEN35_CHECKPOINT"])
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
        model_dir=str(directory),
        data_type="float16",
        page_size=1,
        running_max_req_size=2,
        llm_prefill_att_backend=[backend, "triton"],
        llm_decode_att_backend=[backend, "triton"],
        linear_att_ssm_data_type="float32",
        mtp_mode="eagle_with_att",
        mtp_step=2,
        mtp_draft_model_dir=[str(directory)],
        prefill_cudagraph_max_handle_token=4,
    )
    monkeypatch.setenv("LIGHTLLM_START_ARGS", json.dumps(asdict(args)))
    get_env_start_args.cache_clear()
    get_llm_data_type.cache_clear()
    monkeypatch.setattr(MtpManager, "_instance", None)
    monkeypatch.setattr(dist_group_manager, "groups", [SimpleNamespace()])
    for cls in (Qwen3_5TpPartModel, Qwen3_5MTPModel):
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
        weight_dir=str(directory),
        max_total_token_num=64,
        batch_max_tokens=16,
        max_req_num=2,
        max_seq_length=16,
        graph_max_batch_size=2,
        graph_max_len_in_batch=16,
        disable_cudagraph=True,
    )
    main = Qwen3_5TpPartModel(kwargs.copy())
    draft = Qwen3_5MTPModel(dict(kwargs, main_model=main, mtp_previous_draft_models=[]))
    assert draft.model_adapter is not None and draft.linear_plan.bindings and not draft.linear_plan.legacy
    if moe:
        assert len(main.moe_plan.bindings) == len(main.layers_infer) and not main.moe_plan.legacy
        assert len(draft.moe_plan.bindings) == 1 and not draft.moe_plan.legacy
        monkeypatch.setattr(
            draft.trans_layers_weight[0].experts,
            "experts",
            lambda *a, **kw: pytest.fail("runtime weight-loader MoE"),
            raising=False,
        )
    assert draft.layers_infer[0].layer_num_ == len(main.layers_infer)
    assert draft.trans_layers_weight[0].layer_num_ == 0
    assert draft.pre_post_weight.wte_weight_ is main.pre_post_weight.wte_weight_
    assert draft.pre_post_weight.lm_head_weight_ is main.pre_post_weight.lm_head_weight_
    assert draft.pre_post_weight.main_norm_weight_ is main.pre_post_weight.final_norm_weight_
    main.prepare_attention_states()
    draft.prepare_attention_states()
    manager = main.att_state_manager
    manager.finish_state_registration()
    for state in manager.get_full_attention_states():
        state.device = torch.device("cuda")
        state.allocate_storage(64)
    for model in (main, draft):
        model.req_manager = SimpleNamespace(
            req_to_token_indexs=torch.arange(48, dtype=torch.int32, device="cuda").view(3, 16),
            HOLD_REQUEST_ID=2,
        )
        model.prefill_att_backend = model.decode_att_backend = None
        model.prefill_att_backend1 = model.decode_att_backend1 = None
        model._init_execution_plan()
        model._init_hidden_collector()
        for view in model.attention_execution.resources.stores.get("history").views().values():
            for tensor in view.fields:
                tensor.zero_()
    monkeypatch.setattr(manager, "get_state", lambda *a, **k: pytest.fail("runtime manager lookup"))
    monkeypatch.setattr(NoQuantization, "apply", lambda *a, **k: pytest.fail("runtime weight-loader compute"))

    data = {}
    names = {
        "lm_head.weight",
        "model.language_model.embed_tokens.weight",
        "model.language_model.norm.weight",
        "model.embed_tokens.weight",
        "model.norm.weight",
    }
    for path in directory.glob("*.safetensors"):
        with safe_open(path, framework="pt", device="cpu") as source:
            for name in source.keys():
                if name.startswith("mtp.") or name in names:
                    data[name] = source.get_tensor(name).float().cuda()
    prefix = "model.language_model." if "model.language_model.embed_tokens.weight" in data else "model."
    raw = json.loads((directory / "config.json").read_text())
    config_class = hf.Qwen3_5MoeTextConfig if moe else hf.Qwen3_5TextConfig
    reference_class = hf.Qwen3_5MoeForCausalLM if moe else hf.Qwen3_5ForCausalLM
    config = config_class(**dict(raw.get("text_config", raw), num_hidden_layers=1, layer_types=["full_attention"]))
    reference = reference_class(config).float().cuda().eval()
    reference.config._attn_implementation = "eager"
    weights = {
        name.replace("mtp.layers.0.", "model.layers.0."): value
        for name, value in data.items()
        if name.startswith("mtp.layers.0.")
    }
    weights.update(
        {
            "model.embed_tokens.weight": data[prefix + "embed_tokens.weight"],
            "model.norm.weight": data["mtp.norm.weight"],
            "lm_head.weight": data.get("lm_head.weight", data[prefix + "embed_tokens.weight"]),
        }
    )
    reference.load_state_dict(weights, strict=True)
    torch.manual_seed(61)
    histories = [torch.randn(16, config.hidden_size, dtype=torch.float16, device="cuda") * 0.3 for _ in range(2)]

    def prepare(sequences, cached, prefill):
        inp = prepare_model_batch(draft, sequences, cached, prefill, input_only=True)
        inp.mtp_draft_input_hiddens = torch.cat(
            [h[n : len(s)] for h, n, s in zip(histories, cached, sequences)]
        ).clone()
        state = draft._create_inferstate(inp)
        draft.batch_metadata(state)
        state.init_att_state()
        return state

    def norm(value, weight):
        return F.rms_norm(value, (config.hidden_size,), 1 + weight.float(), config.rms_norm_eps)

    def check(output, sequences, requests=(0, 1)):
        targets = []
        for request, sequence in zip(requests, sequences, strict=True):
            history = histories[request]
            embedding = F.embedding(torch.tensor(sequence, device="cuda"), weights["model.embed_tokens.weight"])
            hidden = norm(history[: len(sequence)].float(), data[prefix + "norm.weight"])
            fused = F.linear(
                torch.cat(
                    (
                        norm(embedding, data["mtp.pre_fc_norm_embedding.weight"]),
                        norm(hidden, data["mtp.pre_fc_norm_hidden.weight"]),
                    ),
                    dim=-1,
                ),
                data["mtp.fc.weight"],
            )
            targets.append(reference(inputs_embeds=fused[None], use_cache=False).logits[:, -1])
        target = torch.cat(targets)
        difference = output.logits.float() - target
        assert torch.all((difference.square().mean(-1) / target.square().mean(-1)).sqrt() <= 0.01)
        assert torch.all(difference.abs().amax(-1) <= target.abs().amax(-1) / 32)
        assert torch.equal(output.logits.argmax(-1), target.argmax(-1))

    def snapshot():
        return [
            (t, t.clone())
            for view in draft.attention_execution.resources.stores.get("history").views().values()
            for t in view.fields
        ]

    def restore(saved):
        for tensor, value in saved:
            tensor.copy_(value)

    try:
        sequences = [[11, 12, 13], [21, 22]]
        check(draft._context_forward(prepare(sequences, [0, 0], True)), sequences)
        sequences = [[11, 12, 13, 14], [21, 22, 23]]
        batch = prepare(sequences, [3, 2], False)
        check(draft._token_forward(batch), sequences)
        saved = snapshot()
        draft.graph = CudaGraph(1, 2, 1, max_batch_size=2, max_len_in_batch=16)
        batch.is_cuda_graph_capture = True
        captured = draft.graph.capture_decode(draft._token_forward, batch)
        restore(saved)
        sequences = [[11, 12, 13, 14, 15], [21, 22, 23, 24]]
        batch = prepare(sequences, [4, 3], False)
        batch.is_cuda_graph_replay = True
        draft.graph.replay(batch)
        check(captured, sequences)
        saved = snapshot()
        sequences = [[11, 12, 13, 14, 15, 16, 17], [21, 22, 23, 24, 25, 26]]
        check(draft._context_forward(prepare(sequences, [5, 4], True)), sequences)
        restore(saved)
        draft.prefill_graph = PrefillCudaGraph(draft.graph, 1)
        check(draft._context_forward(prepare(sequences, [5, 4], True)), sequences)

        # EAGLE first catches up all target verification rows before drafting one row per request.
        expanded = [
            prefix + suffix[:step]
            for prefix, suffix in zip(sequences, ([18, 19, 20], [27, 28, 29]))
            for step in (1, 2, 3)
        ]
        requests = [0, 0, 0, 1, 1, 1]
        lengths = [len(seq) for seq in expanded]
        tensor = lambda values: torch.tensor(values, dtype=torch.int32, device="cuda")
        inp = ModelInput(
            batch_size=6,
            total_token_num=sum(lengths),
            max_q_seq_len=1,
            max_kv_seq_len=max(lengths),
            max_cache_len=max(lengths) - 1,
            input_ids=tensor([seq[-1] for seq in expanded]).long(),
            b_req_idx=tensor(requests),
            b_seq_len=tensor(lengths),
            b_mtp_index=tensor([0, 1, 2, 0, 1, 2]),
            is_prefill=False,
            b_position_delta=tensor([0] * 6),
            b_shared_seq_len=tensor([0] * 6),
            b_shared_radix_node_id=tensor([0] * 6),
            multimodal_params=[{"images": [], "audios": []}] * 6,
            mtp_draft_input_hiddens=torch.stack([histories[r][n - 1] for r, n in zip(requests, lengths)]),
        )
        batch = draft._create_inferstate(inp)
        draft.batch_metadata(batch)
        batch.init_att_state()
        check(draft._token_forward(batch), expanded, requests)
        for view in main.attention_execution.resources.stores.get("history").views().values():
            assert all(torch.count_nonzero(t).item() == 0 for t in view.fields)
    finally:
        torch.cuda.synchronize()
        draft.graph_runner.close()
        main.graph_runner.close()
        get_env_start_args.cache_clear()
        get_llm_data_type.cache_clear()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.parametrize("backend", ["triton", "fa3", "flashinfer"])
def test_qwen35_moe_draft_packed_weights_prefix_and_graph(backend, monkeypatch, tmp_path):
    from safetensors.torch import save_file
    from transformers.models.qwen3_5_moe import modeling_qwen3_5_moe as hf

    torch.manual_seed(63)
    config = hf.Qwen3_5MoeTextConfig(
        vocab_size=128,
        hidden_size=256,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=64,
        layer_types=["linear_attention", "full_attention"],
        linear_num_key_heads=2,
        linear_num_value_heads=4,
        linear_key_head_dim=128,
        linear_value_head_dim=128,
        linear_conv_kernel_dim=4,
        moe_intermediate_size=256,
        shared_expert_intermediate_size=256,
        num_experts=4,
        num_experts_per_tok=2,
        rms_norm_eps=1e-6,
        max_position_embeddings=32,
        rope_parameters=dict(
            rope_type="default", rope_theta=10000.0, partial_rotary_factor=1.0, mrope_section=[11, 11, 10]
        ),
    )
    model = hf.Qwen3_5MoeForCausalLM(config)
    weights = {name: value.detach().contiguous() for name, value in model.state_dict().items()}
    weights.update(
        {
            name.replace("model.layers.1.", "mtp.layers.0."): value.clone()
            for name, value in tuple(weights.items())
            if name.startswith("model.layers.1.")
        }
    )
    for name in ("norm", "pre_fc_norm_hidden", "pre_fc_norm_embedding"):
        weights[f"mtp.{name}.weight"] = torch.randn(config.hidden_size) * 0.01
    weights["mtp.fc.weight"] = torch.randn(config.hidden_size, 2 * config.hidden_size) * 0.02
    config.save_pretrained(tmp_path)
    save_file(weights, str(tmp_path / "model.safetensors"))
    monkeypatch.setenv("LIGHTLLM_TEST_QWEN35_CHECKPOINT", str(tmp_path))
    test_qwen35_draft_checkpoint_prefix_and_graph(backend, monkeypatch, moe=True)
