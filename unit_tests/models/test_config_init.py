"""Exercise real model config methods without weights or distributed initialization."""

import copy
import importlib
import json
from types import SimpleNamespace

import pytest

from lightllm.utils.model_config import load_model_config


@pytest.fixture
def text_config():
    return {
        "model_type": "llama",
        "hidden_size": 128,
        "num_attention_heads": 8,
        "num_key_value_heads": 4,
        "num_hidden_layers": 4,
        "vocab_size": 256,
        "sliding_window": 128,
        "partial_rotary_factor": 0.5,
        "full_attention_interval": 4,
        "rope_parameters": {"rope_type": "default", "rope_theta": 10000.0, "partial_rotary_factor": 0.5},
    }


def init_config(module, name, path, finetune=True):
    cls = getattr(importlib.import_module(f"lightllm.models.{module}.model"), name)
    model = cls.__new__(cls)
    model.args = SimpleNamespace(trust_remote_code=False)
    model.weight_dir_ = str(path)
    model.finetune_config = SimpleNamespace(vocab_size=512) if finetune else None
    model.tp_world_size_ = 2
    model.max_total_token_num = 4096
    model._init_config()
    return model


@pytest.mark.parametrize(
    "module,name,component,expected_vocab",
    [
        ("llama", "LlamaTpPartModel", (), 512),
        ("qwen2_vl", "Qwen2VLTpPartModel", ("text_config",), 512),
        ("internvl", "InternVLLlamaTpPartModel", ("llm_config",), 512),
        ("tarsier2", "Tarsier2LlamaTpPartModel", ("text_config",), 256),
        ("tarsier2", "Tarsier2Qwen2VLTpPartModel", ("text_config",), 256),
        ("qwen3_vl", "Qwen3VLTpPartModel", ("text_config",), 512),
        ("qwen3_vl_moe", "Qwen3VLMOETpPartModel", ("text_config",), 512),
        ("qwen3_omni_moe_thinker", "Qwen3OmniMOETpPartModel", ("thinker_config", "text_config"), 512),
        ("gemma3", "Gemma3TpPartModel", ("text_config",), 256),
        ("gemma4", "Gemma4TpPartModel", ("text_config",), 512),
    ],
)
def test_model_keeps_component_and_finetune_rules(tmp_path, text_config, module, name, component, expected_vocab):
    root = {"model_type": module, "hidden_size": 999}
    root["model_type"] = {
        "internvl": "internvl_chat",
        "tarsier2": "llava",
        "qwen3_omni_moe_thinker": "qwen3_omni_moe",
    }.get(module, module)
    if module == "tarsier2":
        root["architectures"] = ["TarsierForConditionalGeneration"]
    if module == "internvl":
        text_config["model_type"] = "qwen2" if "Qwen2" in name else "llama"
    elif module == "tarsier2":
        text_config["model_type"] = "qwen2_vl" if "Qwen2VL" in name else ("qwen2" if "Qwen2" in name else "llama")
    elif component:
        text_config["model_type"] = {"qwen3_omni_moe_thinker": "qwen3_omni_moe_text"}.get(module, module + "_text")
    if component:
        parent = root
        for key in component[:-1]:
            parent = parent.setdefault(key, {})
        parent[component[-1]] = text_config
    else:
        text_config["model_type"] = module
        root.update(text_config)
        # Root-reading models must not accidentally use a nested target config.
        root["text_config"] = {"hidden_size": 999}
    (tmp_path / "config.json").write_text(json.dumps(root))
    model = init_config(module, name, tmp_path)
    assert model.config["hidden_size"] == 128
    assert model.config["n_embed"] == 128
    assert model.config["n_head"] == 8
    assert model.config["n_layer"] == 4
    assert model.config["vocab_size"] == expected_vocab
    assert load_model_config(tmp_path).model_type == root["model_type"]


def test_qwen35_target_mtp_and_vision_configs_are_isolated(tmp_path, text_config):
    text_config["model_type"] = "qwen3_5_text"
    root = {
        "model_type": "qwen3_5",
        "text_config": text_config,
        "vision_config": {"hidden_size": 64},
    }
    (tmp_path / "config.json").write_text(json.dumps(root))
    target = init_config("qwen3_5", "Qwen3_5TpPartModel", tmp_path)
    mtp = init_config("qwen3_5_mtp", "Qwen3_5MTPModel", tmp_path)
    assert target.config["num_hidden_layers"] == 4
    assert target.config["full_attention_interval"] == 4
    assert mtp.config["num_hidden_layers"] == mtp.config["n_layer"] == 1
    assert mtp.config["full_attention_interval"] == 1
    assert target.config["rope_theta"] == 10000.0
    assert target.config["partial_rotary_factor"] == 0.5
    assert target.config["norm_topk_prob"] is True
    mtp.vision_config["hidden_size"] = 1
    assert target.vision_config["hidden_size"] == 64
    assert load_model_config(tmp_path).text_config.num_hidden_layers == 4


@pytest.mark.parametrize("family", ["qwen3_5", "qwen3_5_text"])
@pytest.mark.parametrize("mode", ["dflash", "dspark"])
def test_qwen35_block_draft_preserves_root_overlay_and_rope_rules(tmp_path, text_config, mode, family):
    root = copy.deepcopy(text_config)
    root.update(
        model_type=family,
        rope_scaling={"rope_type": "default"},
        dflash_config={"block_size": 16, "num_hidden_layers": 2},
        text_config={"hidden_size": 999, "num_hidden_layers": 32},
    )
    (tmp_path / "config.json").write_text(json.dumps(root))
    class_name = "Qwen3_5DFlashModel" if mode == "dflash" else "Qwen3_5DSparkModel"
    model = init_config(f"qwen3_5_{mode}", class_name, tmp_path)
    assert model.config["hidden_size"] == 128
    assert model.config["num_hidden_layers"] == 2
    assert model.config["block_size"] == 16
    assert model.config["partial_rotary_factor"] == 0.5
    assert model.config["rope_scaling"]["rope_type"] == "default"
    assert load_model_config(tmp_path).num_hidden_layers == 4


def test_eagle_keeps_finetune_then_draft_vocab_order(tmp_path, text_config):
    text_config["draft_vocab_size"] = 64
    (tmp_path / "config.json").write_text(json.dumps(text_config))
    model = init_config("qwen3_eagle", "Qwen3EagleModel", tmp_path)
    assert model.config["target_vocab_size"] == 512
    assert model.config["vocab_size"] == 64


def test_vit_keeps_vision_dimension_and_text_projection_inputs(tmp_path, text_config):
    root = {
        "model_type": "internvl_chat",
        "llm_config": text_config,
        "vision_config": {"hidden_size": 64, "num_attention_heads": 4, "num_hidden_layers": 2},
        "select_layer": -1,
        "downsample_ratio": 0.5,
    }
    (tmp_path / "config.json").write_text(json.dumps(root))
    model = init_config("vit", "VisionTransformer", tmp_path)
    assert model.config["hidden_size"] == 64
    assert model.config["llm_hidden_size"] == 128
    assert model.config["downsample_ratio"] == 0.5
    assert model.select_layer == -1
    assert model.layers_num == 2


def test_resource_config_readers_use_the_omni_text_component(tmp_path, text_config):
    from lightllm.common.basemodel.moe_route_info_manager import MoeRouteInfoManager
    from lightllm.utils.profile_max_tokens import load_config

    text_config["model_type"] = "qwen3_omni_moe_text"
    text_config.update(num_experts=16, num_experts_per_tok=2, mlp_only_layers=[0])
    root = {"model_type": "qwen3_omni_moe", "num_hidden_layers": 99, "thinker_config": {"text_config": text_config}}
    (tmp_path / "config.json").write_text(json.dumps(root))
    assert load_config(tmp_path)["num_hidden_layers"] == 4
    assert MoeRouteInfoManager.get_route_config_from_model_dir(tmp_path) == (3, 2, 1, {1: 0, 2: 1, 3: 2})


def test_linear_attention_cache_reads_normalized_text_dimensions(tmp_path, text_config, monkeypatch):
    from lightllm.common.state_cache_manager import linear_att

    text_config["model_type"] = "qwen3_5_text"
    text_config.update(
        head_dim=16,
        linear_num_key_heads=4,
        linear_num_value_heads=8,
        linear_key_head_dim=16,
        linear_value_head_dim=32,
        linear_conv_kernel_dim=4,
    )
    (tmp_path / "config.json").write_text(json.dumps({"model_type": "qwen3_5", "text_config": text_config}))
    args = SimpleNamespace(model_dir=str(tmp_path), data_type="bf16", linear_att_ssm_data_type="fp32", tp=2, dp=1)
    monkeypatch.setattr(linear_att, "get_env_start_args", lambda: args)
    monkeypatch.setattr(linear_att, "get_added_mtp_kv_layer_num", lambda: 2)
    config = linear_att.LinearAttCacheConfig.load_from_args()
    assert config.all_layer_num == 4
    assert config.linear_layer_num == 3
    assert config.full_att_num_kv_heads == 2
    assert config.num_linear_v_heads == 4
    assert config.get_full_att_kv_layer_num_with_draft_model() == 3


def test_llava_initializes_sparse_config(tmp_path):
    from lightllm.models.llava.model import LlavaTpPartModel

    config = {
        "model_type": "llava",
        "architectures": ["LlavaForConditionalGeneration"],
        "text_config": {"model_type": "llama", "vocab_size": 32064},
    }
    (tmp_path / "config.json").write_text(json.dumps(config))
    model = LlavaTpPartModel.__new__(LlavaTpPartModel)

    model.args = SimpleNamespace(trust_remote_code=False)
    model.weight_dir_ = str(tmp_path)
    model.finetune_config = None
    model.tp_world_size_ = 1
    model.load_way = "HF"
    model._init_config()
    model._verify_must()
    model._verify_params()
    assert model.config["hidden_size"] == 4096
    assert model.config["num_attention_heads"] == 32
    assert model.config["num_hidden_layers"] == 32
