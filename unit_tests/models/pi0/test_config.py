from pathlib import Path

import pytest

from lightllm.models.pi0.model import PI_VOCAB_SIZE
from lightllm.utils.config_utils import (
    get_pi_config,
    get_vocab_size,
    with_pi_config_overrides,
)


PI0_DIR = "/mtc/baishihao/vla/lerobot_models_for_vla/pi0_base"
PI05_DIR = "/mtc/baishihao/vla/lerobot_models_for_vla/pi05_base"


@pytest.mark.skipif(not Path(PI0_DIR).is_dir(), reason="pi0 checkpoint is not mounted")
def test_loads_pi0_lerobot_config():
    config = get_pi_config(PI0_DIR)
    assert config["model_type"] == "pi0"
    assert config["action_dim"] == 32
    assert config["action_horizon"] == 50
    assert config["num_denoise_steps"] == 10
    assert config["tokenizer_max_length"] == 48
    assert config["action_expert_uses_state"] and not config["state_in_prefix"]
    assert get_vocab_size(PI0_DIR) == PI_VOCAB_SIZE
    assert config["image_keys"] == (
        "observation.images.base_0_rgb",
        "observation.images.left_wrist_0_rgb",
        "observation.images.right_wrist_0_rgb",
    )


@pytest.mark.skipif(not Path(PI05_DIR).is_dir(), reason="pi0.5 checkpoint is not mounted")
def test_loads_pi05_lerobot_config():
    config = get_pi_config(PI05_DIR)
    assert config["model_type"] == "pi05"
    assert config["tokenizer_max_length"] == 200
    assert config["state_in_prefix"] and not config["action_expert_uses_state"]


@pytest.mark.skipif(not Path(PI0_DIR).is_dir(), reason="pi0 checkpoint is not mounted")
def test_rejects_model_type_mismatch():
    with pytest.raises(ValueError, match="does not match checkpoint"):
        get_pi_config(PI0_DIR, model_type="pi05")


@pytest.mark.skipif(not Path(PI05_DIR).is_dir(), reason="pi0.5 checkpoint is not mounted")
def test_validates_request_overrides():
    config = with_pi_config_overrides(
        get_pi_config(PI05_DIR),
        action_dim=7,
        action_horizon=10,
        num_denoise_steps=5,
    )
    assert (
        config["action_dim"],
        config["action_horizon"],
        config["num_denoise_steps"],
    ) == (
        7,
        10,
        5,
    )
    with pytest.raises(ValueError, match="action_dim"):
        with_pi_config_overrides(config, action_dim=33)


@pytest.mark.skipif(not Path(PI0_DIR).is_dir(), reason="pi0 checkpoint is not mounted")
def test_dtype_override_and_zero_values_are_validated():
    config = get_pi_config(PI0_DIR, dtype="bf16")
    assert config["dtype"] == "bfloat16"
    with pytest.raises(ValueError, match="action_horizon"):
        get_pi_config(PI0_DIR, action_horizon=0)


def test_pi_vocab_size_does_not_depend_on_checkpoint_shape(tmp_path):
    import json

    (tmp_path / "config.json").write_text(json.dumps({"type": "pi0"}), encoding="utf-8")
    (tmp_path / "model.safetensors").touch()

    config = get_pi_config(str(tmp_path))
    assert "vocab_size" not in config
    assert config["image_keys"] == ()
    assert get_vocab_size(str(tmp_path)) == PI_VOCAB_SIZE
