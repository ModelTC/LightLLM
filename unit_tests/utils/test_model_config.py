"""HF objects, legacy boundaries and configuration ownership (no model weights)."""

import copy
import json
import os
from pathlib import Path
import subprocess
import sys

import pytest
from transformers import LlamaConfig, LlavaConfig, PretrainedConfig

from lightllm.utils import config_utils
from lightllm.utils.model_config import get_text_config, load_model_config, to_model_config_dict

ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture
def checkpoint(tmp_path):
    def write(config):
        (tmp_path / "config.json").write_text(json.dumps(config))
        return str(tmp_path)

    return write


def test_sparse_llava_and_public_getters_share_hf_defaults(checkpoint):
    path = checkpoint(
        {
            "model_type": "llava",
            "architectures": ["LlavaForConditionalGeneration"],
            "text_config": {"model_type": "llama", "vocab_size": 32064},
            "vision_config": {"image_size": 336},
        }
    )
    config = load_model_config(path)
    assert isinstance(config, LlavaConfig)
    assert isinstance(config.text_config, LlamaConfig)
    assert config.model_type == "llava"
    assert config.architectures == ["LlavaForConditionalGeneration"]
    assert config.vision_config.image_size == 336
    assert config_utils.get_hidden_size(path) == config.text_config.hidden_size == 4096
    assert config_utils.get_num_attention_heads(path) == config.text_config.num_attention_heads == 32
    assert config_utils.get_num_key_value_heads(path) == config.text_config.num_key_value_heads == 32
    assert config_utils.get_layer_num(path) == config.text_config.num_hidden_layers == 32
    assert config_utils.get_vocab_size(path) == config.text_config.vocab_size == 32064


def test_public_getters_prefer_text_over_root(checkpoint):
    path = checkpoint(
        {
            "model_type": "qwen3_vl",
            "hidden_size": 128,
            "vocab_size": 100,
            "text_config": {"hidden_size": 2048, "num_attention_heads": 16, "head_dim": 128, "vocab_size": 151936},
        }
    )
    assert config_utils.get_hidden_size(path) == 2048
    assert config_utils.get_vocab_size(path) == 151936
    assert config_utils.get_head_dim(path) == 128


def test_hf_attribute_map_and_runtime_aliases(checkpoint):
    path = checkpoint({"model_type": "gpt_bigcode", "n_embd": 2048, "n_head": 16, "n_layer": 24})
    config = load_model_config(path)
    assert config.hidden_size == config_utils.get_hidden_size(path) == 2048
    assert config.num_attention_heads == config_utils.get_num_attention_heads(path) == 16
    assert config.num_hidden_layers == config_utils.get_layer_num(path) == 24
    before = copy.deepcopy(config.to_dict())
    runtime = to_model_config_dict(config)
    assert runtime["hidden_size"] == runtime["n_embed"] == runtime["n_embd"] == 2048
    assert runtime["num_hidden_layers"] == runtime["n_layer"] == 24
    runtime["n_embed"] = 1
    assert config.to_dict() == before


@pytest.mark.parametrize(
    "source",
    [
        {"model_type": "llama", "hidden_size": 2048},
        {"model_type": "internvl_chat", "llm_config": {"model_type": "llama", "hidden_size": 2048}},
        {"model_type": "qwen3_vl", "text_config": {"hidden_size": 2048}},
        {"model_type": "qwen3_omni_moe", "thinker_config": {"text_config": {"hidden_size": 2048}}},
    ],
)
def test_each_load_owns_its_tree_and_adapter_owns_its_dictionary(checkpoint, source):
    path = checkpoint(source)
    first, second = load_model_config(path), load_model_config(path)
    text = get_text_config(first)
    assert isinstance(text, PretrainedConfig)
    assert text is get_text_config(first)
    runtime = to_model_config_dict(text)
    runtime["hidden_size"] = 7
    assert text.hidden_size == 2048
    text.hidden_size = 1
    assert get_text_config(second).hidden_size == 2048


def test_modern_rope_is_adapted_without_mutating_hf_config(checkpoint):
    path = checkpoint(
        {
            "model_type": "qwen3",
            "rope_theta": 500000,
            "rope_scaling": {"rope_type": "yarn", "factor": 4.0, "original_max_position_embeddings": 32768},
        }
    )
    config = load_model_config(path)
    runtime = to_model_config_dict(config)
    assert runtime["rope_theta"] == 500000
    assert runtime["rope_scaling"]["rope_type"] == "yarn"
    runtime["rope_scaling"]["factor"] = 99
    assert config.rope_parameters["factor"] == 4.0
    assert config_utils._derive_max_req_total_len_from_model_config(path) == config.max_position_embeddings


def test_custom_auto_config_requires_trust_and_uses_class_semantics(checkpoint, tmp_path):
    path = checkpoint(
        {
            "model_type": "custom_llava",
            "auto_map": {"AutoConfig": "configuration_custom.CustomConfig"},
            "text_config": {"model_type": "llama"},
        }
    )
    (tmp_path / "configuration_custom.py").write_text(
        "from transformers import LlavaConfig\nclass CustomConfig(LlavaConfig):\n"
        "    model_type = 'custom_llava'\n    def __init__(self, **kwargs):\n"
        "        super().__init__(**kwargs)\n        self.text_config.num_key_value_heads = 4\n"
    )
    with pytest.raises(ValueError, match="trust_remote_code"):
        load_model_config(path)
    config = load_model_config(path, trust_remote_code=True)
    assert type(config).__name__ == "CustomConfig"
    assert config.model_type == "custom_llava"
    assert config.text_config.num_key_value_heads == 4


def test_versioned_local_config_uses_hf_resolution(checkpoint, tmp_path):
    path = checkpoint({"model_type": "llama", "hidden_size": 1, "configuration_files": ["config.4.0.0.json"]})
    (tmp_path / "config.4.0.0.json").write_text(json.dumps({"model_type": "llama", "hidden_size": 2048}))
    assert load_model_config(path).hidden_size == 2048


@pytest.mark.parametrize(
    "family,name,attribute,expected",
    [
        ("deepseek_v32", "DeepseekV32Config", "hidden_size", 7168),
        ("internlm", "InternLMConfig", "vocab_size", 103168),
        ("internlm2", "InternLM2Config", "num_key_value_heads", 32),
        ("minicpm", "MiniCPMConfig", "tie_word_embeddings", True),
        ("qwen", "QWenConfig", "max_position_embeddings", 8192),
    ],
)
def test_named_legacy_config_defaults(checkpoint, family, name, attribute, expected):
    config = load_model_config(checkpoint({"model_type": family}))
    assert type(config).__name__ == name
    assert config.model_type == family
    assert getattr(config, attribute) == expected


def test_minicpm_identity_can_come_from_config_class(checkpoint):
    config = load_model_config(checkpoint({"auto_map": {"AutoConfig": "configuration_minicpm.MiniCPMConfig"}}))
    assert config.model_type == "minicpm"
    assert config.scale_emb == config.scale_depth == config.dim_model_base == 1


def test_tarsier_has_its_own_component_contract(checkpoint):
    config = load_model_config(
        checkpoint(
            {
                "model_type": "llava",
                "architectures": ["TarsierForConditionalGeneration"],
                "text_config": {"model_type": "qwen2", "hidden_size": 3584},
            }
        )
    )
    assert type(config).__name__ == "TarsierConfig"
    assert config.text_config.hidden_size == 3584
    assert config.vision_config is None
    assert config.image_newline_idx == 32002
    assert config.image_token_index == 32000
    assert config.vision_feature_layer == -2


def test_original_llava_keeps_flat_backbone(checkpoint):
    config = load_model_config(
        checkpoint({"model_type": "llava", "hidden_size": 512, "mm_vision_tower": "openai/clip-vit-large-patch14-336"})
    )
    assert get_text_config(config) is config
    assert config.hidden_size == 512
    assert not hasattr(config, "text_config")


@pytest.mark.parametrize("family", ["gemma4", "deepseek_v32", "minicpm", "internvl_chat"])
def test_config_loading_does_not_import_lightllm_models_or_initialize_cuda(checkpoint, family):
    path = checkpoint({"model_type": family})
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            """
import importlib.abc
import sys
class RejectModelDependencies(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.startswith('lightllm.models.'):
            raise ImportError('Config reader imported model dependency: ' + fullname)
sys.meta_path.insert(0, RejectModelDependencies())
from lightllm.utils.model_config import load_model_config
config = load_model_config(sys.argv[1])
import torch
assert not torch.cuda.is_initialized()
assert "flash_attn" not in sys.modules
""",
            path,
        ],
        cwd=ROOT,
        env={**os.environ, "CUDA_VISIBLE_DEVICES": ""},
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stdout + result.stderr


def test_runtime_registry_keeps_explicit_source_identity(checkpoint, tmp_path):
    from lightllm.utils.model_config import load_model_config_dict

    path = checkpoint(
        {
            "model_type": "llava",
            "architectures": ["LlavaForConditionalGeneration"],
            "auto_map": {"AutoConfig": "configuration_custom.CustomConfig"},
            "text_config": {"model_type": "llama"},
        }
    )
    (tmp_path / "configuration_custom.py").write_text(
        "from transformers import LlavaConfig\nclass CustomConfig(LlavaConfig):\n"
        "    model_type = 'custom_llava'\n    def __init__(self, **kwargs):\n"
        "        super().__init__(**kwargs)\n        self.vision_feature_layer = -3\n"
        "        self.architectures = ['TarsierForConditionalGeneration']\n"
    )
    config = load_model_config_dict(path, trust_remote_code=True)
    assert config["model_type"] == "llava"
    assert config["architectures"] == ["LlavaForConditionalGeneration"]
    assert config["vision_feature_layer"] == -3


@pytest.mark.parametrize("audio_type", ["whisper", "clap_audio_model"])
def test_extra_audio_dictionary_still_detected(checkpoint, audio_type):
    path = checkpoint({"model_type": "qwen", "audio_config": {"model_type": audio_type}})
    assert config_utils.has_audio_module(path)


def test_periodic_attention_adapter_rejects_unrepresentable_layout(checkpoint):
    from lightllm.utils.model_config import get_full_attention_interval

    config = load_model_config(
        checkpoint(
            {
                "model_type": "qwen3_5_text",
                "num_hidden_layers": 4,
                "layer_types": ["linear_attention", "full_attention", "full_attention", "linear_attention"],
            }
        )
    )
    with pytest.raises(ValueError, match="periodic"):
        get_full_attention_interval(config)


def test_explicit_startup_trust_works_before_environment_is_set(checkpoint, tmp_path, monkeypatch):
    from types import SimpleNamespace

    monkeypatch.delenv("LIGHTLLM_START_ARGS", raising=False)
    path = checkpoint({"model_type": "custom_llama", "auto_map": {"AutoConfig": "configuration_startup.CustomConfig"}})
    (tmp_path / "configuration_startup.py").write_text(
        "from transformers import LlamaConfig\nclass CustomConfig(LlamaConfig):\n"
        "    model_type = 'custom_llama'\n    def __init__(self, **kwargs):\n"
        "        super().__init__(**kwargs)\n        self.max_position_embeddings = 8192\n"
        "        self.dtype = 'bfloat16'\n        self.eos_token_id = 9\n"
    )
    args = SimpleNamespace(model_dir=path, max_req_total_len=None, trust_remote_code=True)
    config_utils.auto_set_max_req_total_len(args)
    assert args.max_req_total_len == 8192
    assert config_utils.get_dtype(path, trust_remote_code=True) == "bfloat16"
    assert config_utils.get_eos_token_ids(path, trust_remote_code=True) == [9]


@pytest.mark.parametrize("component", ["llm_config", "text_config"])
@pytest.mark.parametrize("family", ["internlm2", "qwen2", "qwen3"])
def test_internvl_component_alias_keeps_the_selected_backbone(checkpoint, component, family):
    from lightllm.utils.model_config import load_model_config_dict

    path = checkpoint({"model_type": "internvl_chat", component: {"model_type": family}})
    config = load_model_config(path)
    assert config.llm_config.model_type == family
    assert get_text_config(config) is config.llm_config
    runtime = load_model_config_dict(path)
    assert runtime["llm_config"]["model_type"] == family
    assert runtime.get("text_config", {}).get("model_type", family) == family


@pytest.mark.parametrize("trust", [False, True])
def test_early_shm_estimation_preserves_cli_trust(monkeypatch, trust):
    from types import SimpleNamespace
    from lightllm.utils import shm_size_check

    monkeypatch.delenv("LIGHTLLM_START_ARGS", raising=False)
    observed = []

    def tokenizer(path, *, trust_remote_code):
        observed.append(("tokenizer", trust_remote_code))
        return SimpleNamespace(get_image_token_length=lambda image: 576)

    def hidden_size(path, *, trust_remote_code):
        observed.append(("config", trust_remote_code))
        return 4096

    monkeypatch.setattr(shm_size_check, "get_tokenizer", tokenizer)
    monkeypatch.setattr(shm_size_check, "get_hidden_size", hidden_size)
    args = SimpleNamespace(
        model_dir="test",
        trust_remote_code=trust,
        running_max_req_size=1,
        max_req_total_len=1024,
        enable_multimodal=True,
        cache_capacity=1,
    )
    assert shm_size_check._get_recommended_shm_size_gb(args) > 2
    assert observed == [("tokenizer", trust), ("config", trust)]


def test_tarsier_legacy_qwen2_vl_components_keep_distinct_semantics(checkpoint):
    # Constructed regression case, based on the official Tarsier2-7b-0115 layout.
    # Both old components say qwen2_vl. This is not a pinned complete checkpoint.
    from lightllm.utils.model_config import load_model_config_dict

    path = checkpoint(
        {
            "model_type": "llava",
            "architectures": ["TarsierForConditionalGeneration"],
            "text_config": {"model_type": "qwen2_vl", "hidden_size": 3584, "num_attention_heads": 28},
            "vision_config": {
                "model_type": "qwen2_vl",
                "hidden_size": 3584,
                "embed_dim": 1280,
                "hidden_act": "quick_gelu",
                "depth": 32,
                "auto_map": {"AutoConfig": "models--modeling_qwen2_vl_fast.Qwen2VLVisionConfig"},
            },
        }
    )
    config = load_model_config(path)
    assert type(config.vision_config).__name__ == "Qwen2VLVisionConfig"
    assert get_text_config(config).hidden_size == config_utils.get_hidden_size(path) == 3584
    runtime = load_model_config_dict(path)
    assert runtime["vision_config"]["hidden_size"] == 3584
    assert runtime["vision_config"]["embed_dim"] == 1280
    assert runtime["vision_config"]["hidden_act"] == "quick_gelu"
    assert runtime["text_config"]["hidden_size"] == 3584
    assert runtime["text_config"]["model_type"] == "qwen2_vl"


def test_explicit_custom_config_is_not_shadowed_by_local_registration(checkpoint, tmp_path):
    path = checkpoint(
        {
            "model_type": "qwen",
            "_name_or_path": "old-checkpoint",
            "auto_map": {"AutoConfig": "configuration_local.CustomQwenConfig"},
        }
    )
    (tmp_path / "configuration_local.py").write_text(
        "from transformers import PretrainedConfig\n"
        "class CustomQwenConfig(PretrainedConfig):\n"
        "    model_type = 'qwen'\n"
        "    def __init__(self, **kwargs):\n"
        "        super().__init__(**kwargs)\n"
        "        self.hidden_size = 777\n"
    )
    # An earlier no-trust load registers the local Qwen compatibility class.
    local = load_model_config(path, trust_remote_code=False)
    assert type(local).__name__ == "QWenConfig"
    custom = load_model_config(path, trust_remote_code=True)
    assert type(custom).__name__ == "CustomQwenConfig"
    assert custom.hidden_size == 777
    assert custom.name_or_path == path
    assert type(load_model_config(path, trust_remote_code=False)).__name__ == "QWenConfig"


def test_explicit_custom_config_failure_does_not_fall_back(checkpoint, tmp_path):
    path = checkpoint({"model_type": "qwen", "auto_map": {"AutoConfig": "configuration_broken.BrokenConfig"}})
    (tmp_path / "configuration_broken.py").write_text(
        "from transformers import PretrainedConfig\n"
        "class BrokenConfig(PretrainedConfig):\n"
        "    model_type = 'qwen'\n"
        "    def __init__(self, **kwargs):\n"
        "        raise RuntimeError('checkpoint config initialization failed')\n"
    )
    load_model_config(path, trust_remote_code=False)
    with pytest.raises(RuntimeError, match="checkpoint config initialization failed"):
        load_model_config(path, trust_remote_code=True)


@pytest.mark.parametrize(
    "architecture,text_type,class_name",
    [
        ("LlavaForConditionalGeneration", None, "LlavaTpPartModel"),
        ("TarsierForConditionalGeneration", "llama", "Tarsier2LlamaTpPartModel"),
        ("TarsierForConditionalGeneration", "qwen2", "Tarsier2Qwen2TpPartModel"),
        ("TarsierForConditionalGeneration", "qwen2_vl", "Tarsier2Qwen2VLTpPartModel"),
    ],
)
def test_config_normalization_preserves_multimodal_family(checkpoint, architecture, text_type, class_name):
    from lightllm.models import get_model_class
    from lightllm.utils.model_config import load_model_config_dict

    source = {"model_type": "llava", "architectures": [architecture], "text_config": {}}
    if text_type is not None:
        source["text_config"]["model_type"] = text_type
    path = checkpoint(source)
    before = get_model_class(source)
    after = get_model_class(load_model_config_dict(path))
    assert before is after
    assert after.__name__ == class_name
