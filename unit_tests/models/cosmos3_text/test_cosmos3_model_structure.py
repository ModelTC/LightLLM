import json
from types import SimpleNamespace

import torch

from lightllm.models.cosmos3_text import model as model_module
from lightllm.models.cosmos3_text.layer_weights import (
    Cosmos3TextPreAndPostLayerWeight,
    Cosmos3TextTransformerLayerWeight,
)
from lightllm.models.cosmos3_text.layer_weights import (
    pre_and_post_layer_weight as prepost_module,
)
from lightllm.models.cosmos3_text.model import Cosmos3TextTpPartModel
from lightllm.models.qwen3.layer_infer.transformer_layer_infer import (
    Qwen3TransformerLayerInfer,
)
from lightllm.models.qwen3.model import Qwen3TpPartModel
from lightllm.models.registry import get_model_class


COSMOS3_TEXT_ARCHITECTURE = {
    "hidden_size": 4096,
    "num_hidden_layers": 36,
    "num_attention_heads": 32,
    "num_key_value_heads": 8,
    "head_dim": 128,
    "intermediate_size": 12288,
}


def _root_config(*, vocab_size=151_936, text_overrides=None):
    text_config = {
        "model_type": "qwen3_vl_text",
        **COSMOS3_TEXT_ARCHITECTURE,
        "vocab_size": vocab_size,
        "rms_norm_eps": 1e-6,
        "hidden_act": "silu",
        "attention_bias": False,
        "rope_theta": 5_000_000,
        "max_position_embeddings": 262_144,
        "rope_scaling": {
            "rope_type": "default",
            "mrope_section": [24, 20, 20],
            "mrope_interleaved": True,
        },
    }
    text_config.update(text_overrides or {})
    return {
        "model_type": "cosmos3_omni",
        "text_config": text_config,
        "tie_word_embeddings": False,
        "vision_config": {"hidden_size": 1152},
    }


def test_registry_and_qwen3_runtime_wiring():
    assert get_model_class(_root_config()) is Cosmos3TextTpPartModel
    assert issubclass(Cosmos3TextTpPartModel, Qwen3TpPartModel)
    assert (
        Cosmos3TextTpPartModel.pre_and_post_weight_class
        is Cosmos3TextPreAndPostLayerWeight
    )
    assert (
        Cosmos3TextTpPartModel.transformer_weight_class
        is Cosmos3TextTransformerLayerWeight
    )
    assert (
        Cosmos3TextTpPartModel.transformer_layer_infer_class
        is Qwen3TransformerLayerInfer
    )


def test_model_uses_nested_text_config_without_hardcoding_dimensions(tmp_path):
    root_config = _root_config(vocab_size=17)
    (tmp_path / "config.json").write_text(json.dumps(root_config), encoding="utf-8")
    model = object.__new__(Cosmos3TextTpPartModel)
    model.weight_dir_ = str(tmp_path)
    model.finetune_config = None

    model._init_config()

    assert model.config["model_type"] == "qwen3_vl_text"
    assert model.config["vocab_size"] == 17
    assert model.config["n_head"] == 32
    assert model.config["n_embed"] == 4096
    assert model.config["n_layer"] == 36


def test_complete_official_weight_name_mapping():
    layer_weight = object.__new__(Cosmos3TextTransformerLayerWeight)
    layer_weight.layer_num_ = 7
    layer_weight._init_weight_names()
    assert layer_weight._q_weight_name == "layers.7.self_attn.to_q.weight"
    assert layer_weight._k_weight_name == "layers.7.self_attn.to_k.weight"
    assert layer_weight._v_weight_name == "layers.7.self_attn.to_v.weight"
    assert layer_weight._o_weight_name == "layers.7.self_attn.to_out.weight"
    assert layer_weight._q_norm_name == "layers.7.self_attn.norm_q.weight"
    assert layer_weight._k_norm_name == "layers.7.self_attn.norm_k.weight"
    assert layer_weight._gate_weight_name == "layers.7.mlp.gate_proj.weight"
    assert layer_weight._up_weight_name == "layers.7.mlp.up_proj.weight"
    assert layer_weight._down_weight_name == "layers.7.mlp.down_proj.weight"
    assert layer_weight._att_norm_weight_name == "layers.7.input_layernorm.weight"
    assert (
        layer_weight._ffn_norm_weight_name == "layers.7.post_attention_layernorm.weight"
    )


def test_prepost_class_uses_checkpoint_dimensions_and_names(monkeypatch):
    monkeypatch.setenv("LIGHTLLM_CURRENT_RANK_IN_DP", "0")
    monkeypatch.setenv("LIGHTLLM_DP_WORLD_SIZE", "1")

    def fake_weight(**kwargs):
        return SimpleNamespace(**kwargs)

    monkeypatch.setattr(prepost_module, "EmbeddingWeight", fake_weight)
    monkeypatch.setattr(prepost_module, "LMHeadWeight", fake_weight)
    monkeypatch.setattr(prepost_module, "RMSNormWeight", fake_weight)

    config = _root_config(vocab_size=23)["text_config"]
    weights = Cosmos3TextPreAndPostLayerWeight(torch.bfloat16, config)

    assert weights.wte_weight_.weight_name == "embed_tokens.weight"
    assert weights.wte_weight_.vocab_size == 23
    assert weights.wte_weight_.dim == 4096
    assert weights.lm_head_weight_.weight_name == "lm_head.weight"
    assert weights.lm_head_weight_.embedding_weight is None
    assert weights.final_norm_weight_.weight_name == "norm.weight"


def test_loader_scans_only_official_transformer_directory(tmp_path, monkeypatch):
    transformer_dir = tmp_path / "transformer"
    transformer_dir.mkdir()
    captured = {}

    def fake_load_hf_weights(*args, **kwargs):
        captured["args"] = args
        captured["kwargs"] = kwargs

    class TrackedWeight:
        def __init__(self):
            self.verify_calls = 0

        def verify_load(self):
            self.verify_calls += 1

    monkeypatch.setattr(model_module, "load_hf_weights", fake_load_hf_weights)
    model = object.__new__(Cosmos3TextTpPartModel)
    model.weight_dir_ = str(tmp_path)
    model.data_type = torch.bfloat16
    model.weight_dict = None
    model.pre_post_weight = TrackedWeight()
    model.trans_layers_weight = [TrackedWeight(), TrackedWeight()]

    model._load_hf_weights()

    assert captured["args"] == (torch.bfloat16,)
    assert captured["kwargs"]["weight_dir"] == str(transformer_dir)
    assert captured["kwargs"]["pre_post_layer"] is model.pre_post_weight
    assert captured["kwargs"]["transformer_layer_list"] is model.trans_layers_weight
    assert model.pre_post_weight.verify_calls == 1
    assert [weight.verify_calls for weight in model.trans_layers_weight] == [1, 1]
