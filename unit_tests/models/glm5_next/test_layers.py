import dataclasses
import json
from types import SimpleNamespace

import pytest
import torch
import torch.nn.functional as F

from lightllm.common.basemodel.attention.nsa.glm5_next import Glm5NextSparsePrefillState, Glm5NextSparseDecodeState
from lightllm.common.quantization import Quantcfg
from lightllm.models.glm5_next.layer_infer.transformer_layer_infer import Glm5NextTransformerLayerInfer
from lightllm.models.glm5_next.layer_weights.transformer_layer_weight import Glm5NextTransformerLayerWeight
from lightllm.models.glm5_next.model import Glm5NextTpPartModel
from lightllm.models.glm5_next_mtp.model import Glm5NextMTPModel
from lightllm.server.core.objs.start_args_type import StartArgs
from lightllm.utils.envs_utils import get_env_start_args


pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")


@pytest.fixture(autouse=True)
def runtime(monkeypatch):
    monkeypatch.setenv("LIGHTLLM_START_ARGS", json.dumps(dataclasses.asdict(StartArgs())))
    for name, value in {
        "GLOBAL_RANK": 0,
        "GLOBAL_WORLD_SIZE": 1,
        "DP_WORLD_SIZE": 1,
        "CURRENT_RANK_IN_DP": 0,
        "CURRENT_RANK_IN_NODE": 0,
        "CURRENT_DEVICE_ID": 0,
    }.items():
        monkeypatch.setenv("LIGHTLLM_" + name, str(value))
    get_env_start_args.cache_clear()
    torch.manual_seed(53)
    yield
    get_env_start_args.cache_clear()


@pytest.fixture
def config():
    return {
        "hidden_size": 128,
        "n_embed": 128,
        "intermediate_size": 128,
        "moe_intermediate_size": 128,
        "num_hidden_layers": 4,
        "n_layer": 4,
        "num_attention_heads": 4,
        "num_key_value_heads": 4,
        "vocab_size": 256,
        "rms_norm_eps": 1e-5,
        "q_lora_rank": 128,
        "kv_lora_rank": 512,
        "qk_nope_head_dim": 128,
        "qk_rope_head_dim": 0,
        "v_head_dim": 128,
        "index_n_heads": 4,
        "index_head_dim": 128,
        "index_topk": 2048,
        "index_kpool": 4,
        "linear_attn_config": {"num_heads": 4, "head_dim": 128, "short_conv_kernel_size": 4},
        "layer_types": ["linear_attention"] * 3 + ["deepseek_sparse_attention"],
        "mhc": False,
        "n_routed_experts": 4,
        "n_shared_experts": 1,
        "first_k_dense_replace": 1,
        "num_experts_per_tok": 2,
        "norm_topk_prob": True,
        "scoring_func": "sigmoid",
        "n_group": 1,
        "topk_group": 1,
        "swiglu_limit": 10.0,
        "num_nextn_predict_layers": 1,
    }


def random_weight(*shape):
    return torch.randn(*shape, device="cuda", dtype=torch.bfloat16) * 0.05


def rmsnorm(x, eps):
    return (x.float() * torch.rsqrt(x.float().square().mean(-1, keepdim=True) + eps)).to(x.dtype)


def test_kda_projections_need_no_sparse_attention_config(config):
    for key in ("q_lora_rank", "kv_lora_rank", "qk_nope_head_dim", "qk_rope_head_dim", "v_head_dim"):
        del config[key]
    for key in list(config):
        if key.startswith("index_"):
            del config[key]
    layer = Glm5NextTransformerLayerInfer(0, config)
    weight = Glm5NextTransformerLayerWeight(0, torch.bfloat16, config, Quantcfg(config))
    projections = {
        name: random_weight(size, 128)
        for name, size in {"q": 512, "k": 512, "v": 512, "b": 4, "f_a": 128, "g_a": 128, "f_b": 512, "g_b": 512}.items()
    }
    weight.load_hf_weights(
        {f"model.language_model.layers.0.self_attn.{name}_proj.weight": value for name, value in projections.items()}
    )
    hidden = random_weight(3, 128)
    actual = layer._kda_projections(hidden, SimpleNamespace(), weight)
    expected = (
        torch.cat([F.linear(hidden, projections[name]) for name in ("q", "k", "v")], dim=-1),
        F.linear(F.linear(hidden, projections["f_a"]), projections["f_b"]),
        F.linear(hidden, projections["b"]),
        F.linear(F.linear(hidden, projections["g_a"]), projections["g_b"]),
    )
    for result, reference in zip(actual, expected):
        torch.testing.assert_close(result, reference, atol=1e-3, rtol=0.01)


@pytest.mark.parametrize("rank", [0, 1])
def test_mla_loading_preserves_replicated_qkv_and_sharded_bmm(config, monkeypatch, rank):
    monkeypatch.setenv("LIGHTLLM_DP_WORLD_SIZE", "2")
    monkeypatch.setenv("LIGHTLLM_CURRENT_RANK_IN_DP", str(rank))
    layer = Glm5NextTransformerLayerInfer(3, config)
    weight = Glm5NextTransformerLayerWeight(3, torch.bfloat16, config, Quantcfg(config))
    q_a, kv_a = random_weight(128, 128), random_weight(512, 128)
    q_b, kv_b, o = random_weight(512, 128), random_weight(1024, 512), random_weight(128, 512)
    tensors = {
        "q_a_proj.weight": q_a,
        "kv_a_proj_with_mqa.weight": kv_a,
        "q_b_proj.weight": q_b,
        "kv_b_proj.weight": kv_b,
        "o_proj.weight": o,
        "q_a_layernorm.weight": torch.ones(128, device="cuda", dtype=torch.bfloat16),
        "kv_a_layernorm.weight": torch.ones(512, device="cuda", dtype=torch.bfloat16),
    }
    weight.load_hf_weights(
        {f"model.language_model.layers.3.self_attn.{name}": value for name, value in tensors.items()}
    )
    hidden = random_weight(3, 128)
    state = SimpleNamespace(need_dp_prefill_balance=False)
    q, kv = layer._get_qkv(hidden, state, weight)
    q_lora = rmsnorm(F.linear(hidden, q_a), config["rms_norm_eps"])
    expected_q = F.linear(q_lora, q_b).view(3, 4, 128)[:, rank * 2 : (rank + 1) * 2]
    expected_kv = rmsnorm(F.linear(hidden, kv_a), config["rms_norm_eps"]).unsqueeze(1)
    torch.testing.assert_close(q, expected_q, atol=0.015, rtol=0.015)
    torch.testing.assert_close(kv, expected_kv, atol=0.015, rtol=0.015)
    head_weights = kv_b.view(4, 256, 512)[rank * 2 : (rank + 1) * 2]
    projected = weight.k_b_proj_.bmm(q.transpose(0, 1))
    torch.testing.assert_close(projected, torch.bmm(q.transpose(0, 1), head_weights[:, :128]))
    latent_output = random_weight(3, 2, 512)
    monkeypatch.setattr(layer, "_tpsp_reduce", lambda input, infer_state: input)
    actual = layer._get_o(latent_output, state, weight)
    values = torch.bmm(latent_output.transpose(0, 1), head_weights[:, 128:].transpose(1, 2))
    expected = F.linear(values.transpose(0, 1).reshape(3, 256), o[:, rank * 256 : (rank + 1) * 256])
    torch.testing.assert_close(actual, expected)


def test_sparse_layer_prefill_and_decode_match_dense_nope_attention(config):
    if torch.cuda.get_device_capability()[0] != 9:
        pytest.skip("GLM sparse attention kernels require Hopper")
    layer = Glm5NextTransformerLayerInfer(3, config)
    weight = Glm5NextTransformerLayerWeight(3, torch.bfloat16, config, Quantcfg(config))
    kv_b = random_weight(1024, 512)
    weight.load_hf_weights({"model.layers.3.self_attn.kv_b_proj.weight": kv_b})
    q = random_weight(3, 4, 128)
    # Use noncontiguous token slots and a padded sparse index table.
    packed = random_weight(8, 1, 584)
    kv = packed[:, :, :512]
    slots = torch.tensor([7, 1, 5, 0, 3], device="cuda", dtype=torch.int32)
    lengths = torch.tensor([2, 3, 5], device="cuda", dtype=torch.int32)
    indices = torch.full((3, 128), -1, device="cuda", dtype=torch.int32)
    for row, length in enumerate(lengths.tolist()):
        indices[row, :length] = slots[:length]
    layer.indexer = SimpleNamespace(_get_indices=lambda **kwargs: (indices, indices))
    state = SimpleNamespace(
        get_topk_indices_params={"hidden_states": None, "q_lora": None},
        mem_manager=SimpleNamespace(get_att_input_params=lambda layer_index: kv),
        prefill_att_state=Glm5NextSparsePrefillState(),
        b1_cu_q_seq_len=torch.arange(4, device="cuda", dtype=torch.int32),
        max_q_seq_len=1,
    )
    state.decode_att_state = Glm5NextSparseDecodeState(
        infer_state=state,
        nsa_cache_seqlens=lengths,
        nsa_cu_seqlens_k_new=F.pad(lengths.cumsum(0, dtype=torch.int32), (1, 0)),
    )
    projected = torch.bmm(q.transpose(0, 1), kv_b.view(4, 256, 512)[:, :128]).transpose(0, 1)
    expected = []
    for row, length in enumerate(lengths.tolist()):
        keys = kv[slots[:length].long(), 0].float()
        scores = projected[row].float() @ keys.T * config["qk_nope_head_dim"] ** -0.5
        expected.append(scores.softmax(-1) @ keys)
    expected = torch.stack(expected)
    prefill = layer._context_attention_kernel(q, None, state, weight)
    state.get_topk_indices_params = {"hidden_states": None, "q_lora": None}
    decode = layer._token_attention_kernel(q, state, weight)
    for output in (prefill, decode):
        torch.testing.assert_close(output.float(), expected, atol=1e-3, rtol=0.015)


@pytest.mark.parametrize("mode", ["dense", "moe", "fused_moe"])
def test_ffn_dispatch_preserves_clamp_and_shared_expert(config, monkeypatch, mode):
    args = StartArgs(enable_fused_shared_experts=mode == "fused_moe")
    monkeypatch.setenv("LIGHTLLM_START_ARGS", json.dumps(dataclasses.asdict(args)))
    get_env_start_args.cache_clear()
    index = 0 if mode == "dense" else 3
    layer = Glm5NextTransformerLayerInfer(index, config)
    weight = Glm5NextTransformerLayerWeight(index, torch.bfloat16, config, Quantcfg(config))
    prefix = f"model.language_model.layers.{index}.mlp"
    eye = torch.eye(128, device="cuda", dtype=torch.bfloat16)
    tensors = {}

    def mlp_weights(name, gate_scale, up_scale):
        tensors[f"{name}.gate_proj.weight"] = eye * gate_scale
        tensors[f"{name}.up_proj.weight"] = eye * up_scale
        tensors[f"{name}.down_proj.weight"] = eye

    if mode == "dense":
        mlp_weights(prefix, 1.5, 0.75)
    else:
        mlp_weights(f"{prefix}.shared_experts", 1.5, 0.75)
        for expert in range(4):
            mlp_weights(f"{prefix}.experts.{expert}", 1 + expert / 4, 2 + expert / 8)
        router = torch.zeros(4, 128, device="cuda", dtype=torch.float32)
        router[:, :4] = torch.eye(4, device="cuda") * 0.2
        tensors[f"{prefix}.gate.weight"] = router
        tensors[f"{prefix}.gate.e_score_correction_bias"] = torch.zeros(4, device="cuda")
    weight.load_hf_weights(tensors)
    x = torch.linspace(-25, 25, 3 * 128, device="cuda", dtype=torch.bfloat16).view(3, 128)

    def reference_mlp(gate_scale, up_scale):
        gate = (x * gate_scale).float().clamp(max=10)
        up = (x * up_scale).float().clamp(-10, 10)
        return (F.silu(gate).bfloat16().float() * up).bfloat16()

    expected = reference_mlp(1.5, 0.75)
    if mode != "dense":
        scores = F.linear(x.float(), router).sigmoid()
        probabilities, ids = scores.topk(2, dim=-1)
        probabilities /= probabilities.sum(-1, keepdim=True)
        routed = torch.zeros_like(x)
        for expert in range(4):
            coefficient = (probabilities * (ids == expert)).sum(-1, keepdim=True)
            routed += (reference_mlp(1 + expert / 4, 2 + expert / 8).float() * coefficient).bfloat16()
        expected += routed
    actual = layer._ffn(x.clone(), SimpleNamespace(is_prefill=True), weight)
    torch.testing.assert_close(actual, expected, atol=0.25, rtol=0.015)


@pytest.mark.parametrize("is_prefill", [False, True])
def test_ep_ffn_uses_combined_output_and_replicated_shared_expert(config, monkeypatch, is_prefill):
    args = StartArgs(tp=2, enable_ep_moe=True, enable_fused_shared_experts=True)
    monkeypatch.setenv("LIGHTLLM_START_ARGS", json.dumps(dataclasses.asdict(args)))
    monkeypatch.setenv("LIGHTLLM_DP_WORLD_SIZE", "2")
    monkeypatch.setenv("LIGHTLLM_CURRENT_RANK_IN_DP", "1")
    monkeypatch.setenv("LIGHTLLM_GLOBAL_WORLD_SIZE", "2")
    monkeypatch.setenv("LIGHTLLM_GLOBAL_RANK", "1")
    get_env_start_args.cache_clear()
    layer = Glm5NextTransformerLayerInfer(3, config)
    weight = Glm5NextTransformerLayerWeight(3, torch.bfloat16, config, Quantcfg(config))
    assert weight.num_fused_shared_experts == 0
    prefix = "model.language_model.layers.3.mlp.shared_experts"
    eye = torch.eye(128, device="cuda", dtype=torch.bfloat16)
    weight.load_hf_weights({f"{prefix}.{name}.weight": eye for name in ("gate_proj", "up_proj", "down_proj")})
    x = torch.linspace(-25, 25, 3 * 128, device="cuda", dtype=torch.bfloat16).view(3, 128)
    original = x.clone()
    monkeypatch.setattr(weight.moe_gate, "mm", lambda x: torch.zeros(3, 4, device="cuda"))
    monkeypatch.setattr(layer, "_tpsp_reduce", lambda **kwargs: pytest.fail("EP output must not be TP-reduced again"))

    def combined_output(hidden_states, **kwargs):
        assert kwargs["is_prefill"] is is_prefill
        assert kwargs["alpha"] == 1.0 and kwargs["limit"] == 10.0
        assert kwargs["clamp_up_add_one"] is False
        return torch.full_like(hidden_states, 17)

    monkeypatch.setattr(weight.experts, "experts", combined_output)
    actual = layer._ffn(x, SimpleNamespace(is_prefill=is_prefill), weight)
    shared = (F.silu(x.float().clamp(max=10)).bfloat16().float() * x.float().clamp(-10, 10)).bfloat16()
    torch.testing.assert_close(actual, shared + 17, atol=0.25, rtol=0.01)
    torch.testing.assert_close(x, original, atol=0, rtol=0)


def test_fp8_shared_expert_scales_and_bf16_kv_b_load_together(config, monkeypatch):
    monkeypatch.setenv(
        "LIGHTLLM_START_ARGS", json.dumps(dataclasses.asdict(StartArgs(enable_fused_shared_experts=True)))
    )
    get_env_start_args.cache_clear()
    config["quantization_config"] = {"quant_method": "fp8", "weight_block_size": [128, 128]}
    weight = Glm5NextTransformerLayerWeight(3, torch.bfloat16, config, Quantcfg(config))
    prefix = "model.language_model.layers.3"
    tensors = {}
    for expert in range(5):
        name = f"experts.{expert}" if expert < 4 else "shared_experts"
        for projection in ("gate_proj", "up_proj", "down_proj"):
            tensors[f"{prefix}.mlp.{name}.{projection}.weight"] = random_weight(128, 128).to(torch.float8_e4m3fn)
            tensors[f"{prefix}.mlp.{name}.{projection}.weight_scale_inv"] = torch.ones(1, 1, device="cuda")
    tensors[f"{prefix}.mlp.gate.e_score_correction_bias"] = torch.zeros(4, device="cuda")
    kv_b = random_weight(1024, 512)
    tensors[f"{prefix}.self_attn.kv_b_proj.weight"] = kv_b
    # There is deliberately no kv_b scale: this matrix stays BF16 in the FP8 checkpoint.
    weight.load_hf_weights(tensors)
    assert weight.experts.verify_load()
    assert weight.k_b_proj_.verify_load() and weight.v_b_proj_.verify_load()
    q = random_weight(4, 2, 128)
    actual = weight.k_b_proj_.bmm(q)
    expected = torch.bmm(q, kv_b.view(4, 256, 512)[:, :128])
    torch.testing.assert_close(actual, expected)


@pytest.mark.parametrize("method", ["context_forward", "token_forward"])
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_mtp_pre_layer_applies_main_norm_before_hidden_fusion(config, monkeypatch, method, dtype):
    from lightllm.common.basemodel.layer_weights.meta_weights import RMSNormWeight
    from lightllm.models.glm5_next_mtp.layer_infer.pre_layer_infer import Glm5NextMTPPreLayerInfer
    from lightllm.models.qwen_vl.layer_infer.pre_layer_infer import LlamaMultimodalPreLayerInfer

    hidden_size, eps = config["hidden_size"], config["rms_norm_eps"]
    norms = [RMSNormWeight(hidden_size, name, dtype) for name in ("main_norm", "enorm", "hnorm")]
    for norm in norms:
        norm.weight.copy_(torch.randn_like(norm.weight))
    embeddings = torch.randn(5, hidden_size, device="cuda", dtype=dtype)
    hidden = torch.randn_like(embeddings)
    projection = torch.randn(2 * hidden_size, hidden_size, device="cuda", dtype=dtype) * 0.1
    weight = SimpleNamespace(
        main_norm_weight_=norms[0],
        enorm_weight_=norms[1],
        hnorm_weight_=norms[2],
        eh_proj_weight_=SimpleNamespace(mm=lambda x: x @ projection),
    )
    state = SimpleNamespace(mtp_draft_input_hiddens=hidden.clone())
    monkeypatch.setattr(LlamaMultimodalPreLayerInfer, method, lambda self, ids, state, weight: embeddings.clone())
    actual = getattr(Glm5NextMTPPreLayerInfer(config), method)(None, state, weight)

    def reference_norm(value, norm):
        return F.rms_norm(value.float(), (hidden_size,), norm.weight.float(), eps).to(dtype)

    normalized_hidden = reference_norm(reference_norm(hidden, norms[0]), norms[2])
    expected = torch.cat((reference_norm(embeddings, norms[1]), normalized_hidden), dim=-1) @ projection
    torch.testing.assert_close(actual, expected, atol=0.015 if dtype == torch.bfloat16 else 1e-5, rtol=0.015)


@pytest.mark.parametrize("model_class", [Glm5NextTpPartModel, Glm5NextMTPModel])
def test_glm_weight_loading_requires_hf(config, model_class):
    model = object.__new__(model_class)
    model.config, model.tp_world_size_ = config, 1
    model.args, model.load_way = StartArgs(), "HF"
    model._verify_params()
    model.load_way = "DS"
    with pytest.raises(AssertionError, match="only support HF format weights"):
        model._verify_params()


@pytest.mark.parametrize("tp_world_size", [1, 2])
def test_glm_rejects_tpsp(config, tp_world_size):
    model = object.__new__(Glm5NextTpPartModel)
    model.config, model.tp_world_size_ = config, tp_world_size
    model.args, model.load_way = StartArgs(), "HF"
    model._verify_params()

    model.args.enable_tpsp_mix_mode = True
    with pytest.raises(AssertionError, match="GLM-5.3 Flash does not support TP/SP mixed mode"):
        model._verify_params()


def test_native_drafts_share_caches_but_keep_config_and_layer_indices(config):
    main = object.__new__(Glm5NextTpPartModel)
    main.config = dict(config, mhc=True)
    main.tp_world_size_ = 1
    main.args, main.load_way = StartArgs(), "HF"
    main._verify_params()
    main._init_some_value()
    main.layers_infer = [object() for _ in range(config["num_hidden_layers"])]
    main.pre_post_weight = SimpleNamespace(wte_weight_=object(), lm_head_weight_=object(), final_norm_weight_=object())
    main.req_manager, main.mem_manager, main.linear_config = object(), object(), object()
    drafts = []
    for step in range(2):
        draft = object.__new__(Glm5NextMTPModel)
        draft.main_model, draft.mtp_previous_draft_models = main, list(drafts)
        draft.tp_world_size_, draft.data_type = 1, torch.bfloat16
        draft.load_way = "HF"
        draft._init_config()
        draft._verify_params()
        draft.quant_cfg = Quantcfg(draft.config)
        draft._init_weights()
        draft._init_infer_layer()
        draft._init_some_value()
        draft._init_req_manager()
        draft._init_mem_manager()
        draft._init_att_backend1()
        assert main.config["mhc"] and not draft.config["mhc"]
        assert draft.config["layer_types"] is not main.config["layer_types"]
        assert draft.layers_num == 1 and draft.head_dim_ == 512
        assert draft.layers_infer[0].layer_num_ == 4 + step
        assert draft.trans_layers_weight[0].layer_num_ == 4
        assert draft.pre_post_weight.wte_weight_ is main.pre_post_weight.wte_weight_
        assert draft.pre_post_weight.lm_head_weight_ is main.pre_post_weight.lm_head_weight_
        assert draft.pre_post_weight.main_norm_weight_ is main.pre_post_weight.final_norm_weight_
        assert draft.req_manager is main.req_manager and draft.mem_manager is main.mem_manager
        assert draft.prefill_att_backend1 is None and draft.decode_att_backend1 is None
        drafts.append(draft)

    # NoPE state initialization must not require a model-owned RoPE cache.
    for model in (main, *drafts):
        state = model.infer_state_class()
        state.is_prefill = True
        state.input_ids = torch.zeros(3, device="cuda", dtype=torch.int64)
        state.b_ready_cache_len = torch.tensor([2], device="cuda", dtype=torch.int32)
        state.b_seq_len = torch.tensor([5], device="cuda", dtype=torch.int32)
        state.init_some_extra_state(model)
        torch.testing.assert_close(
            state.position_ids, torch.arange(2, 5, device="cuda", dtype=state.position_ids.dtype)
        )
