"""EAGLE3 recipe, vocabulary mapping and target/recursive hidden Graph inputs."""

from dataclasses import asdict
import json
from types import SimpleNamespace

import pytest
import torch
import torch.nn.functional as F

from unit_tests.common.execution.test_model_cuda import prepare_model_batch


@pytest.mark.parametrize("compressed", [False, True])
def test_eagle_config_is_shared_with_cpu_storage(compressed, tmp_path):
    from lightllm.models.qwen3_eagle.model import Qwen3EagleModel

    raw = dict(vocab_size=128, num_hidden_layers=2, hidden_size=256, num_attention_heads=4, num_key_value_heads=2)
    if compressed:
        raw["draft_vocab_size"] = 64
    (tmp_path / "config.json").write_text(json.dumps(raw))
    model = Qwen3EagleModel.__new__(Qwen3EagleModel)
    model.weight_dir_, model.finetune_config = str(tmp_path), None
    model._init_config()
    config = Qwen3EagleModel.normalize_model_config(raw)
    assert model.config == config == Qwen3EagleModel.normalize_model_config(config)
    assert raw["vocab_size"] == 128 and "target_vocab_size" not in raw
    assert config["vocab_size"] == (64 if compressed else 128)
    if compressed:
        assert config["target_vocab_size"] == 128


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.parametrize("backend", ["triton", "fa3", "flashinfer"])
@pytest.mark.parametrize("compressed,head_dim", [(False, 64), (True, 128)])
@torch.no_grad()
def test_eagle_checkpoint_prefix_and_graph(backend, compressed, head_dim, monkeypatch, tmp_path):
    from safetensors.torch import save_file
    from transformers import Qwen3Config, Qwen3ForCausalLM
    from lightllm.common.basemodel.mtp_manager import MtpManager
    from lightllm.common.execution.decode_graph import CudaGraph
    from lightllm.common.execution.prefill_graph import PrefillCudaGraph
    from lightllm.common.quantization.no_quant import NoQuantization
    from lightllm.distributed.communication_op import dist_group_manager
    from lightllm.models.qwen3.model import Qwen3TpPartModel
    from lightllm.models.qwen3_eagle.model import Qwen3EagleModel
    from lightllm.server.core.objs.start_args_type import StartArgs
    from lightllm.utils.envs_utils import get_env_start_args, get_llm_data_type

    torch.manual_seed(71)
    config = Qwen3Config(
        vocab_size=128,
        hidden_size=256,
        intermediate_size=512,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=head_dim,
        max_position_embeddings=32,
        rms_norm_eps=1e-6,
        rope_theta=10000.0,
    )
    target = Qwen3ForCausalLM(config).half()
    target_dir, draft_dir = tmp_path / "target", tmp_path / "draft"
    target_dir.mkdir()
    draft_dir.mkdir()
    config.save_pretrained(target_dir)
    save_file(
        {k: v.detach().contiguous() for k, v in target.state_dict().items()}, str(target_dir / "model.safetensors")
    )
    draft_config = dict(config.to_dict(), target_layer_ids=[0, 1])
    vocab_size = 64 if compressed else 128
    if compressed:
        draft_config["draft_vocab_size"] = vocab_size
    (draft_dir / "config.json").write_text(json.dumps(draft_config))

    matrix = lambda *shape: (torch.randn(*shape) * 0.02).half()
    norm = lambda width=256: (1 + torch.randn(width) * 0.01).half()
    weights = {"fc.weight": matrix(256, 512), "norm.weight": norm(), "lm_head.weight": matrix(vocab_size, 256)}
    if compressed:
        weights["d2t"] = torch.arange(64, dtype=torch.int64)
        weights["t2d"] = torch.arange(128) % 2 == 0
        embedding = target.model.embed_tokens.weight.detach().float().cuda()
    else:
        weights["embed_tokens.weight"] = matrix(128, 256)
        embedding = weights["embed_tokens.weight"].float().cuda()
    for i in range(2):
        prefix = f"layers.{i}."
        for name in ("input_layernorm", "hidden_norm", "post_attention_layernorm"):
            weights[prefix + name + ".weight"] = norm()
        for name in ("q_norm", "k_norm"):
            weights[prefix + "self_attn." + name + ".weight"] = norm(head_dim)
        for name, width in (("q", 4 * head_dim), ("k", 2 * head_dim), ("v", 2 * head_dim)):
            weights[prefix + f"self_attn.{name}_proj.weight"] = matrix(width, 512)
        weights[prefix + "self_attn.o_proj.weight"] = matrix(256, 4 * head_dim)
        for name in ("gate", "up"):
            weights[prefix + f"mlp.{name}_proj.weight"] = matrix(512, 256)
        weights[prefix + "mlp.down_proj.weight"] = matrix(256, 512)
    save_file(weights, str(draft_dir / "model.safetensors"))
    reference = {name: value.float().cuda() for name, value in weights.items()}
    del target

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
        model_dir=str(target_dir),
        data_type="float16",
        page_size=1,
        running_max_req_size=2,
        llm_prefill_att_backend=[backend],
        llm_decode_att_backend=[backend],
        mtp_mode="eagle3",
        mtp_step=2,
        mtp_draft_model_dir=[str(draft_dir)],
        prefill_cudagraph_max_handle_token=8,
    )
    monkeypatch.setenv("LIGHTLLM_START_ARGS", json.dumps(asdict(args)))
    get_env_start_args.cache_clear()
    get_llm_data_type.cache_clear()
    monkeypatch.setattr(MtpManager, "_instance", None)
    monkeypatch.setattr(dist_group_manager, "groups", [SimpleNamespace()])
    for cls in (Qwen3TpPartModel, Qwen3EagleModel):
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
    main = Qwen3TpPartModel(dict(kwargs, weight_dir=str(target_dir)))
    draft = Qwen3EagleModel(dict(kwargs, weight_dir=str(draft_dir), main_model=main, mtp_previous_draft_models=[]))
    assert draft.model_adapter is not None and draft.linear_plan.bindings and not draft.linear_plan.legacy
    assert [layer.layer_num_ for layer in draft.layers_infer] == [2, 3]
    assert [weight.layer_num_ for weight in draft.trans_layers_weight] == [0, 1]
    assert (draft.pre_post_weight.wte_weight_ is main.pre_post_weight.wte_weight_) == compressed
    main.prepare_attention_states()
    draft.prepare_attention_states()
    manager = main.att_state_manager
    manager.finish_state_registration()
    for state in manager.get_full_attention_states():
        state.device = torch.device("cuda")
        state.allocate_storage(64)
    for model in (main, draft):
        model.req_manager = SimpleNamespace(
            req_to_token_indexs=torch.arange(48, dtype=torch.int32, device="cuda").view(3, 16), HOLD_REQUEST_ID=2
        )
        model.prefill_att_backend = model.decode_att_backend = None
        model.prefill_att_backend1 = model.decode_att_backend1 = None
        model._init_execution_plan()
        model._init_hidden_collector()
        for view in model.execution_runner.store.views().values():
            for tensor in view.fields:
                tensor.zero_()
    monkeypatch.setattr(manager, "get_state", lambda *a, **kw: pytest.fail("runtime manager lookup"))
    monkeypatch.setattr(NoQuantization, "apply", lambda *a, **kw: pytest.fail("legacy weight compute"))
    target_hidden = [torch.randn(16, 512, dtype=torch.float16, device="cuda") * 0.3 for _ in range(2)]
    recursive_hidden = [torch.randn(16, 256, dtype=torch.float16, device="cuda") * 0.3 for _ in range(2)]
    history = [torch.empty(16, 256, device="cuda") for _ in range(2)]

    def make_input(sequences, cached, prefill, projected=True):
        inp = prepare_model_batch(draft, sequences, cached, prefill, input_only=True)
        source = target_hidden if projected else recursive_hidden
        inp.mtp_draft_input_hiddens = torch.cat([h[n : len(s)] for h, n, s in zip(source, cached, sequences)])
        for h, values, n, seq in zip(history, source, cached, sequences):
            value = values[n : len(seq)].float()
            h[n : len(seq)] = F.linear(value, reference["fc.weight"]) if projected else value
        return inp

    def prepare(sequences, cached, projected=True):
        inp = make_input(sequences, cached, False, projected)
        state = draft._create_inferstate(inp)
        assert inp.mtp_draft_input_hiddens.shape[-1] == (512 if projected else 256)
        assert state.mtp_draft_input_hiddens.shape[-1] == 256
        draft.batch_metadata(state)
        state.init_att_state()
        return state

    def check(output, sequences):
        expected = []
        for request, sequence in enumerate(sequences):
            length = len(sequence)
            value = F.embedding(torch.tensor(sequence, device="cuda"), embedding)
            hidden = history[request][:length]
            position = torch.arange(length, device="cuda").float()
            inv = 10000.0 ** (-torch.arange(0, head_dim, 2, device="cuda").float() / head_dim)
            angle = position[:, None] * inv[None, :]
            cosine, sine = angle.cos()[:, None, :], angle.sin()[:, None, :]

            def rope(x):
                left, right = x.chunk(2, dim=-1)
                return torch.cat((left * cosine - right * sine, right * cosine + left * sine), dim=-1)

            for i in range(2):
                prefix = f"layers.{i}."
                norm = lambda x, name: F.rms_norm(x, (x.shape[-1],), reference[prefix + name + ".weight"], 1e-6)
                joined = torch.cat((norm(value, "input_layernorm"), norm(hidden, "hidden_norm")), -1)
                q = F.linear(joined, reference[prefix + "self_attn.q_proj.weight"]).view(length, 4, head_dim)
                k = F.linear(joined, reference[prefix + "self_attn.k_proj.weight"]).view(length, 2, head_dim)
                v = F.linear(joined, reference[prefix + "self_attn.v_proj.weight"]).view(length, 2, head_dim)
                q = rope(norm(q, "self_attn.q_norm")).transpose(0, 1)
                k = rope(norm(k, "self_attn.k_norm")).repeat_interleave(2, dim=1).transpose(0, 1)
                v = v.repeat_interleave(2, dim=1).transpose(0, 1)
                score = q @ k.transpose(-1, -2) / head_dim ** 0.5
                score.masked_fill_(torch.ones(length, length, dtype=torch.bool, device="cuda").triu(1), -float("inf"))
                attention = (score.softmax(-1) @ v).transpose(0, 1).reshape(length, 4 * head_dim)
                hidden = hidden + F.linear(attention, reference[prefix + "self_attn.o_proj.weight"])
                ffn_input = norm(hidden, "post_attention_layernorm")
                gate = F.linear(ffn_input, reference[prefix + "mlp.gate_proj.weight"])
                up = F.linear(ffn_input, reference[prefix + "mlp.up_proj.weight"])
                hidden = hidden + F.linear(F.silu(gate) * up, reference[prefix + "mlp.down_proj.weight"])
                value = hidden
            logits = F.linear(
                F.rms_norm(hidden[-1], (256,), reference["norm.weight"], 1e-6), reference["lm_head.weight"]
            )
            expected.append(logits)
        expected = torch.stack(expected)
        delta = output.logits.float() - expected
        assert torch.all((delta.square().mean(-1) / expected.square().mean(-1)).sqrt() <= 0.01)
        assert torch.all(delta.abs().amax(-1) <= expected.abs().amax(-1) / 32)
        assert torch.equal(output.logits.argmax(-1), expected.argmax(-1))
        tokens = output.logits.argmax(-1)
        assert torch.equal(draft.map_draft_vocab_to_main_vocab(tokens), tokens * (2 if compressed else 1))

    try:
        sequences = [[81, 82, 83], [101, 102]]
        check(draft.forward(make_input(sequences, [0, 0], True)), sequences)
        sequences = [[81, 82, 83, 84], [101, 102, 103]]
        check(draft._token_forward(prepare(sequences, [3, 2])), sequences)
        draft.graph = CudaGraph(1, 2, 1, max_batch_size=2, max_len_in_batch=16)
        state = prepare(sequences, [3, 2])
        state.is_cuda_graph_capture = True
        captured = draft.graph.capture_decode(draft._token_forward, state)
        sequences = [[81, 82, 83, 84, 85], [101, 102, 103, 104]]
        state = prepare(sequences, [4, 3], projected=False)
        state.is_cuda_graph_replay = True
        draft.graph.replay(state)
        check(captured, sequences)
        sequences = [[81, 82, 83, 84, 85, 86], [101, 102, 103, 104, 105]]
        state = prepare(sequences, [5, 4])
        state.is_cuda_graph_replay = True
        draft.graph.replay(state)
        check(captured, sequences)
        assert len(draft.graph.graph) == 1
        draft.prefill_graph = PrefillCudaGraph(draft.graph, 1)
        sequences = [[81, 82, 83, 84, 85, 86, 87, 88], [101, 102, 103, 104, 105, 106, 107]]
        check(draft.forward(make_input(sequences, [6, 5], True)), sequences)
        assert 4 in draft.prefill_graph.graph
        target_hidden[0][6:8].mul_(0.7)
        target_hidden[1][5:7].mul_(1.3)
        check(draft.forward(make_input(sequences, [6, 5], True)), sequences)
        for view in main.execution_runner.store.views().values():
            assert all(torch.count_nonzero(t).item() == 0 for t in view.fields)
    finally:
        torch.cuda.synchronize()
        draft.graph_runner.close()
        main.graph_runner.close()
        get_env_start_args.cache_clear()
        get_llm_data_type.cache_clear()
