import json
from types import SimpleNamespace

import torch

from lightllm.common.basemodel.layer_weights.hf_load_utils import _select_weights
from lightllm.models.kimi_linear.layer_weights import transformer_layer_weight as kimi_layer_weights
from lightllm.models.kimi_linear.layer_weights.transformer_layer_weight import (
    KimiLinearTransformerLayerWeight,
)
from lightllm.models.kimi_linear.model import KimiLinearTpPartModel
from lightllm.models.registry import ModelRegistry


def test_kimi_k3_registry_and_nested_text_config(tmp_path):
    text_config = {
        "model_type": "kimi_linear",
        "num_attention_heads": 96,
        "hidden_size": 7168,
        "num_hidden_layers": 93,
        "num_experts": 896,
        "num_shared_experts": 2,
        "num_experts_per_token": 16,
        "moe_renormalize": True,
        "num_expert_group": 8,
        "moe_router_activation_func": "sigmoid",
        "routed_expert_hidden_size": 3584,
        "attn_res_block_size": 12,
        "hidden_act": "situ",
    }
    (tmp_path / "config.json").write_text(
        json.dumps({"model_type": "kimi_k3", "text_config": text_config}),
        encoding="utf-8",
    )
    model = object.__new__(KimiLinearTpPartModel)
    model.weight_dir_ = str(tmp_path)
    model.finetune_config = None

    model._init_config()

    assert ModelRegistry.get_model_class({"model_type": "kimi_k3"}) is KimiLinearTpPartModel
    assert model.weight_prefix == "language_model."
    assert model.config["model_type"] == "kimi_linear"
    assert model.config["n_head"] == 96
    assert model.config["n_embed"] == 7168
    assert model.config["n_layer"] == 93
    assert model.config["n_routed_experts"] == 896
    assert model.config["n_shared_experts"] == 2
    assert model.config["num_experts_per_tok"] == 16
    assert model.config["scoring_func"] == "sigmoid"
    assert model.config["routed_expert_hidden_size"] == 3584
    assert model.config["attn_res_block_size"] == 12
    assert model.config["hidden_act"] == "situ"


def test_weight_prefix_selects_language_model_and_strips_prefix():
    language_weight = torch.tensor([1.0])
    weights = {
        "language_model.model.embed_tokens.weight": language_weight,
        "vision_tower.encoder.weight": torch.tensor([2.0]),
    }

    selected = _select_weights(weights, "language_model.")

    assert list(selected) == ["model.embed_tokens.weight"]
    assert selected["model.embed_tokens.weight"] is language_weight


def test_shared_expert_width_accounts_for_all_kimi_k3_shared_experts(monkeypatch):
    calls = []

    class _Weight:
        def __init__(self, **kwargs):
            calls.append(kwargs)

    monkeypatch.setattr(kimi_layer_weights, "ROWMMWeight", _Weight)
    monkeypatch.setattr(kimi_layer_weights, "COLMMWeight", _Weight)
    monkeypatch.setattr(
        kimi_layer_weights,
        "get_env_start_args",
        lambda: SimpleNamespace(enable_ep_moe=False),
    )
    layer = object.__new__(KimiLinearTransformerLayerWeight)
    layer.n_embed = 7168
    layer.moe_inter = 3072
    layer.network_config_ = {"n_shared_experts": 2}
    layer.data_type_ = torch.bfloat16
    layer.get_quant_method = lambda name: name

    layer._init_shared_experts("model.layers.1.block_sparse_moe.shared_experts")

    assert calls[0]["out_dims"] == [6144, 6144]
    assert calls[1]["in_dim"] == 6144


def test_full_attention_layer_initializes_q_lora_norm(monkeypatch):
    calls = []

    class _NormWeight:
        def __init__(self, **kwargs):
            calls.append(kwargs)

    monkeypatch.setattr(kimi_layer_weights, "RMSNormWeight", _NormWeight)
    layer = object.__new__(KimiLinearTransformerLayerWeight)
    layer.layer_num_ = 3
    layer.n_embed = 7168
    layer.kv_lora_rank = 512
    layer.q_lora_rank = 1536
    layer.data_type_ = torch.bfloat16
    layer.is_linear_attention_layer = False

    layer._init_norm()

    assert calls[-1] == {
        "dim": 1536,
        "weight_name": "model.layers.3.self_attn.q_a_layernorm.weight",
        "data_type": torch.bfloat16,
    }
    assert hasattr(layer, "q_a_layernorm_")
