import dataclasses
import json
from types import SimpleNamespace

import pytest
import torch
import torch.nn.functional as F

from lightllm.common.basemodel.layer_weights.meta_weights.fused_moe.impl.deepgemm_impl import FuseMoeDeepGEMM
from lightllm.common.basemodel.layer_weights.meta_weights.fused_moe.impl.triton_impl import FuseMoeTriton
from lightllm.common.quantization import Quantcfg
from lightllm.models.glm5_next.indexer import Glm5NextNsaInfer
from lightllm.models.glm5_next.layer_weights.transformer_layer_weight import Glm5NextTransformerLayerWeight
from lightllm.server.core.objs.start_args_type import StartArgs
from lightllm.utils.envs_utils import get_env_start_args, get_redundancy_expert_ids, get_redundancy_expert_num


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
    for getter in (get_env_start_args, get_redundancy_expert_ids, get_redundancy_expert_num):
        getter.cache_clear()
    yield
    for getter in (get_env_start_args, get_redundancy_expert_ids, get_redundancy_expert_num):
        getter.cache_clear()


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
        "rms_norm_eps": 1e-5,
        "q_lora_rank": 128,
        "kv_lora_rank": 512,
        "qk_nope_head_dim": 128,
        "qk_rope_head_dim": 0,
        "v_head_dim": 128,
        "index_n_heads": 32,
        "index_head_dim": 128,
        "index_topk": 2048,
        "index_kpool": 4,
        "index_kpool_always_select_tail": True,
        "linear_attn_config": {"num_heads": 4, "head_dim": 128, "short_conv_kernel_size": 4},
        "layer_types": ["linear_attention"] * 3 + ["deepseek_sparse_attention"],
        "mhc": False,
        "n_routed_experts": 4,
        "n_shared_experts": 1,
        "first_k_dense_replace": 4,
        "num_experts_per_tok": 2,
        "norm_topk_prob": True,
        "scoring_func": "sigmoid",
        "n_group": 1,
        "topk_group": 1,
    }


def test_indexer_layernorm_epsilon_preserves_pooled_topk_boundary(config, monkeypatch):
    pytest.importorskip("deep_gemm")
    if torch.cuda.get_device_capability()[0] < 9:
        pytest.skip("FP8 indexer requires Hopper or newer")
    # Exercise the real indexer and pooling kernels without requiring vLLM's
    # optional top-k extension. The LayerNorm and projections are not stubbed.
    monkeypatch.setattr("lightllm.models.glm5_next.indexer.HAS_VLLM", False)
    dim, heads, topk = config["index_head_dim"], config["index_n_heads"], config["index_topk"]
    weight = Glm5NextTransformerLayerWeight(3, torch.bfloat16, config, Quantcfg(config))
    indexer = Glm5NextNsaInfer(3, config, 1)
    columns = torch.arange(dim)
    a = torch.where(columns % 2 == 0, -1.0, 1.0)
    b = torch.where((columns // 2) % 2 == 0, -1.0, 1.0)
    query = 2 * a + b
    prefix = "model.language_model.layers.3.self_attn.indexer"
    eye = torch.eye(dim, dtype=torch.bfloat16)
    weight.load_hf_weights(
        {
            f"{prefix}.wk.weight": eye,
            f"{prefix}.k_norm.weight": torch.ones(dim, dtype=torch.bfloat16),
            f"{prefix}.k_norm.bias": torch.zeros(dim, dtype=torch.bfloat16),
            f"{prefix}.wq_b.weight": eye.repeat(heads, 1),
            f"{prefix}.weights_proj.weight": ((a + b) / dim).repeat(heads, 1),
            f"{prefix}.index_kpool_compress_gate": torch.zeros(dim, dim, dtype=torch.bfloat16),
            f"{prefix}.index_kpool_compress_ape": torch.zeros(4, dim),
        }
    )

    # At 2052 tokens, 513 complete pools compete for 512 slots. The first
    # 511 pools win under either epsilon. The last two have different K
    # variances: eps=1e-6 selects A, while the global RMSNorm eps selects B.
    hidden = torch.cat(
        (
            (query * 2 ** -4).repeat(topk - 4, 1),
            (a * 2 ** -10).repeat(4, 1),
            (b * 2 ** -4).repeat(4, 1),
        )
    ).to(device="cuda", dtype=torch.bfloat16)
    tokens = hidden.shape[0]
    q_lora = query.to(device="cuda", dtype=torch.bfloat16).repeat(tokens, 1)
    projected_k = weight.wk_proj_.mm(hidden)
    actual_k = weight.k_norm_(projected_k, eps=indexer.eps)
    expected_k = F.layer_norm(projected_k, (dim,), weight.k_norm_.weight, weight.k_norm_.bias, eps=1e-6)
    torch.testing.assert_close(actual_k, expected_k, atol=0, rtol=0)
    wrong_k = F.layer_norm(
        projected_k, (dim,), weight.k_norm_.weight, weight.k_norm_.bias, eps=config["rms_norm_eps"]
    )
    assert not torch.equal(expected_k, wrong_k)

    storage = torch.zeros(tokens + 1, 1, 584, dtype=torch.bfloat16, device="cuda")
    tail = torch.zeros(2, 4, 2 * dim, dtype=torch.bfloat16, device="cuda")
    locations = torch.arange(tokens, dtype=torch.int32, device="cuda")
    infer = SimpleNamespace(
        is_prefill=True,
        max_kv_seq_len=tokens,
        max_q_seq_len=tokens,
        b_req_idx=torch.zeros(1, dtype=torch.int32, device="cuda"),
        b_seq_len=torch.tensor([tokens], dtype=torch.int32, device="cuda"),
        b1_cu_q_seq_len=torch.tensor([0, tokens], dtype=torch.int32, device="cuda"),
        req_manager=SimpleNamespace(
            req_to_token_indexs=locations[None],
            get_indexer_tail_buffer=lambda _: tail,
        ),
        mem_manager=SimpleNamespace(get_indexer_k_buffer=lambda _: storage.view(torch.uint8)[:, :, -132:]),
    )
    state = SimpleNamespace(lengths=locations + 1, ks=torch.zeros_like(locations), ragged_mem_index=locations)
    _, actual = indexer._get_indices(hidden, q_lora, infer, state, weight)
    torch.testing.assert_close(actual[-1, :topk].sort().values, locations[:topk], atol=0, rtol=0)
    assert (actual[-1, topk:] == -1).all()

    # Verify this fixture detects the old behavior, including through the FP8
    # indexer path, rather than only checking a constructor constant.
    indexer.eps = config["rms_norm_eps"]
    _, wrong = indexer._get_indices(hidden, q_lora, infer, state, weight)
    wrong_selection = torch.cat((locations[: topk - 4], locations[topk:]))
    torch.testing.assert_close(wrong[-1, :topk].sort().values, wrong_selection, atol=0, rtol=0)


@pytest.mark.parametrize("enable_ep_moe", [False, True], ids=["tp", "ep"])
def test_checkpoint_correction_bias_preserves_fp32_routing(config, monkeypatch, enable_ep_moe):
    args = StartArgs(tp=2, enable_ep_moe=enable_ep_moe, enable_fused_shared_experts=False)
    monkeypatch.setenv("LIGHTLLM_START_ARGS", json.dumps(dataclasses.asdict(args)))
    # Construct rank 1's real weights on this GPU; routing itself needs no
    # distributed communication and must retain all experts' global biases.
    for name, value in {"GLOBAL_RANK": 1, "GLOBAL_WORLD_SIZE": 2, "DP_WORLD_SIZE": 2, "CURRENT_RANK_IN_DP": 1}.items():
        monkeypatch.setenv("LIGHTLLM_" + name, str(value))
    get_env_start_args.cache_clear()
    config.update(n_routed_experts=288, num_experts_per_tok=8, first_k_dense_replace=1)
    weight = Glm5NextTransformerLayerWeight(3, torch.bfloat16, config, Quantcfg(config))
    experts = weight.experts
    expected_impl = FuseMoeDeepGEMM if enable_ep_moe else FuseMoeTriton
    assert isinstance(experts.fuse_moe_impl, expected_impl)
    source_bias = torch.zeros(config["n_routed_experts"], dtype=torch.float32)
    source_bias[7:9] = torch.tensor([0.1002, 0.1001], dtype=torch.float32)
    weight.load_hf_weights({"model.language_model.layers.3.mlp.gate.e_score_correction_bias": source_bias})
    assert experts.e_score_correction_bias.dtype == torch.float32
    assert experts.e_score_correction_bias.load_ok
    torch.testing.assert_close(experts.e_score_correction_bias.cpu(), source_bias, atol=0, rtol=0)
    assert experts.w13.weight.dtype == experts.w2.weight.dtype == torch.bfloat16

    hidden = torch.zeros(1, config["hidden_size"], dtype=torch.bfloat16, device="cuda")
    logits = torch.full((1, config["n_routed_experts"]), -10.0, dtype=torch.float32, device="cuda")
    logits[:, :7] = 4.0
    logits[0, 7:9] = torch.tensor([0.0, 4e-5], dtype=torch.float32, device="cuda")

    def route(bias):
        # This is the actual common selection entry used by TP expert GEMMs,
        # EP low-latency dispatch, and EP select_experts_and_quant_input.
        return experts.fuse_moe_impl._select_experts(
            input_tensor=hidden,
            router_logits=logits,
            correction_bias=bias,
            top_k=8,
            renormalize=True,
            use_grouped_topk=True,
            topk_group=1,
            num_expert_group=1,
            scoring_func="sigmoid",
        )

    weights, ids, origin_ids = route(experts.e_score_correction_bias)
    expected_ids = torch.arange(8, dtype=ids.dtype, device="cuda")[None]
    torch.testing.assert_close(ids.sort(dim=-1).values, expected_ids, atol=0, rtol=0)
    torch.testing.assert_close(origin_ids, ids, atol=0, rtol=0)
    # Bias affects selection only; combination weights use the original
    # sigmoid scores and then normalize over the selected experts.
    expected_weights = logits.sigmoid().gather(1, ids.long())
    expected_weights /= expected_weights.sum(-1, keepdim=True)
    torch.testing.assert_close(weights, expected_weights, atol=1e-7, rtol=1e-6)

    # Both biases round to the same BF16 value. Reproducing the old storage
    # conversion therefore lets expert 8 displace expert 7 at the boundary.
    _, rounded_ids, _ = route(experts.e_score_correction_bias.to(torch.bfloat16))
    rounded_expected = torch.tensor([list(range(7)) + [8]], dtype=rounded_ids.dtype, device="cuda")
    torch.testing.assert_close(rounded_ids.sort(dim=-1).values, rounded_expected, atol=0, rtol=0)
