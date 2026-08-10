import json
from importlib import import_module
from types import SimpleNamespace

import pytest

import lightllm.common.basemodel.attention.base_att as base_att_module
import lightllm.server.router.model_infer.mode_backend.base_backend as base_backend_module
from lightllm.common.basemodel.attention.base_att import BaseAttBackend
from lightllm.models import get_draft_model_class
from lightllm.models.qwen3_eagle.layer_weights.transformer_layer_weight import Qwen3EagleTransformerLayerWeight
from lightllm.server.router.model_infer.mode_backend.base_backend import ModeBackend
from lightllm.utils import envs_utils


@pytest.mark.parametrize(
    "module_name, class_name",
    [
        ("lightllm.models.deepseek_mtp.model", "Deepseek3MTPModel"),
        ("lightllm.models.glm4_moe_lite_mtp.model", "Glm4MoeLiteMTPModel"),
        ("lightllm.models.mistral_mtp.model", "MistralMTPModel"),
        ("lightllm.models.qwen3_moe_mtp.model", "Qwen3MOEMTPModel"),
        ("lightllm.models.qwen3_5_mtp.model", "Qwen3_5MTPModel"),
        ("lightllm.models.qwen3_5_moe_mtp.model", "Qwen3_5MoeMTPModel"),
        ("lightllm.models.qwen3_eagle.model", "Qwen3EagleModel"),
        ("lightllm.models.qwen3_dflash.model", "Qwen3DFlashModel"),
        ("lightllm.models.qwen3_5_dflash.model", "Qwen3_5DFlashModel"),
        ("lightllm.models.qwen3_dspark.model", "Qwen3DSparkModel"),
        ("lightllm.models.qwen3_5_dspark.model", "Qwen3_5DSparkModel"),
    ],
)
def test_spec_draft_model_class_is_marked(module_name, class_name):
    model_class = getattr(import_module(module_name), class_name)
    assert model_class.is_mtp_draft_model is True


def test_qwen3_eagle_uses_layers_checkpoint_prefix():
    layer_weight = Qwen3EagleTransformerLayerWeight.__new__(Qwen3EagleTransformerLayerWeight)
    layer_weight.layer_num_ = 2

    layer_weight._init_weight_names()

    assert layer_weight._q_weight_name == "layers.2.self_attn.q_proj.weight"
    assert layer_weight._hidden_norm_weight_name == "layers.2.hidden_norm.weight"


@pytest.mark.parametrize(
    "mtp_mode, is_draft_model, draft_step, dynamic_spec, expected",
    [
        ("dspark", False, 7, True, True),
        ("dspark", True, 7, True, False),
        ("vanilla_with_att", True, 7, True, True),
        ("eagle3", True, 0, True, False),
        ("eagle3", False, 7, False, False),
    ],
)
def test_attention_backend_selects_dynamic_spec_layout(
    monkeypatch,
    mtp_mode,
    is_draft_model,
    draft_step,
    dynamic_spec,
    expected,
):
    monkeypatch.setattr(base_att_module, "enable_dynamic_spec", lambda: dynamic_spec)
    monkeypatch.setattr(base_att_module, "get_env_start_args", lambda: SimpleNamespace(mtp_mode=mtp_mode))
    backend = SimpleNamespace(model=SimpleNamespace(is_mtp_draft_model=is_draft_model))
    infer_state = SimpleNamespace(draft_step=draft_step)

    assert BaseAttBackend.uses_dynamic_spec_verify_layout(backend, infer_state) is expected


@pytest.mark.parametrize(
    "spec_mode, architecture, is_linear_att_mixed_model, expected_class_name",
    [
        ("dflash", "Qwen3DFlashModel", False, "Qwen3DFlashModel"),
        ("dflash", "Qwen3DSparkModel", False, "Qwen3DFlashModel"),
        ("dflash", "Qwen3_5DFlashModel", True, "Qwen3_5DFlashModel"),
        ("dspark", "Qwen3DSparkModel", False, "Qwen3DSparkModel"),
        ("dspark", "Qwen3DSparkModel", True, "Qwen3_5DSparkModel"),
        ("eagle3", "Qwen3Eagle3Model", False, "Qwen3EagleModel"),
    ],
)
def test_draft_model_registry(
    spec_mode,
    architecture,
    is_linear_att_mixed_model,
    expected_class_name,
):
    model_class = get_draft_model_class(
        model_cfg={"model_type": "qwen3", "architectures": [architecture]},
        spec_mode=spec_mode,
        is_linear_att_mixed_model=is_linear_att_mixed_model,
    )

    assert model_class.__name__ == expected_class_name


def test_dspark_env_helper_enables_dynamic_spec_without_legacy_flag(monkeypatch):
    monkeypatch.setenv(
        "LIGHTLLM_START_ARGS",
        json.dumps(
            {
                "mtp_mode": "dspark",
                "mtp_dynamic_verify": False,
                "mtp_step": 4,
            }
        ),
    )
    envs_utils.get_env_start_args.cache_clear()
    envs_utils.enable_dynamic_spec.cache_clear()

    assert envs_utils.enable_dynamic_spec()


@pytest.mark.parametrize(
    "mtp_mode, mtp_dynamic_verify, expected",
    [
        ("dspark", False, True),
        ("dflash", True, False),
        ("eagle3", True, True),
        ("eagle3", False, False),
    ],
)
def test_env_helper_uses_normalized_dynamic_spec(monkeypatch, mtp_mode, mtp_dynamic_verify, expected):
    monkeypatch.setenv(
        "LIGHTLLM_START_ARGS",
        json.dumps(
            {
                "mtp_mode": mtp_mode,
                "mtp_step": 7,
                "mtp_dynamic_verify": mtp_dynamic_verify,
            }
        ),
    )
    envs_utils.get_env_start_args.cache_clear()
    envs_utils.enable_dynamic_spec.cache_clear()

    assert envs_utils.enable_dynamic_spec() is expected


def test_draft_model_registry_rejects_unsupported_architecture():
    with pytest.raises(ValueError, match="Unsupported speculative draft model"):
        get_draft_model_class(
            model_cfg={"model_type": "gemma4", "architectures": ["Gemma4DSparkModel"]},
            spec_mode="dspark",
        )


def test_draft_model_registry_validates_target_family():
    with pytest.raises(ValueError, match="linear-attention mixed targets"):
        get_draft_model_class(
            model_cfg={"model_type": "qwen3", "architectures": ["Qwen3DFlashModel"]},
            spec_mode="dflash",
            is_linear_att_mixed_model=True,
        )


def test_target_hidden_layer_ids_reads_text_config(monkeypatch):
    monkeypatch.setattr(
        base_backend_module.PretrainedConfig,
        "get_config_dict",
        lambda _: ({"target_layer_ids": [1, 20, 36]}, {}),
    )
    backend = ModeBackend.__new__(ModeBackend)
    backend.args = SimpleNamespace(mtp_mode="dspark", mtp_draft_model_dir=["/models/dspark"])

    layer_ids = backend._target_hidden_layer_ids({"model_type": "qwen3_5", "text_config": {"num_hidden_layers": 40}})

    assert layer_ids == (1, 20, 36)


def test_dflash_added_kv_layers_come_from_draft_config(tmp_path):
    config_path = tmp_path / "config.json"
    config_path.write_text(json.dumps({"num_hidden_layers": 5}))

    envs_utils.get_env_start_args.cache_clear()
    envs_utils.get_added_mtp_kv_layer_num.cache_clear()
    envs_utils.set_env_start_args(
        {
            "mtp_mode": "dflash",
            "mtp_step": 7,
            "mtp_dynamic_verify": False,
            "mtp_draft_model_dir": [str(tmp_path)],
        }
    )

    assert envs_utils.get_added_mtp_kv_layer_num() == 5


def test_dspark_added_kv_layers_come_from_draft_config(tmp_path):
    config_path = tmp_path / "config.json"
    config_path.write_text(json.dumps({"num_hidden_layers": 6}))

    envs_utils.get_env_start_args.cache_clear()
    envs_utils.get_added_mtp_kv_layer_num.cache_clear()
    envs_utils.set_env_start_args(
        {
            "mtp_mode": "dspark",
            "mtp_step": 7,
            "mtp_dynamic_verify": True,
            "mtp_draft_model_dir": [str(tmp_path)],
        }
    )

    assert envs_utils.get_added_mtp_kv_layer_num() == 6


def test_eagle3_added_kv_layers_come_from_draft_config(tmp_path):
    config_path = tmp_path / "config.json"
    config_path.write_text(json.dumps({"num_hidden_layers": 2}))

    envs_utils.get_env_start_args.cache_clear()
    envs_utils.get_added_mtp_kv_layer_num.cache_clear()
    envs_utils.set_env_start_args(
        {
            "mtp_mode": "eagle3",
            "mtp_step": 7,
            "mtp_dynamic_verify": False,
            "mtp_draft_model_dir": [str(tmp_path)],
        }
    )

    assert envs_utils.get_added_mtp_kv_layer_num() == 2
