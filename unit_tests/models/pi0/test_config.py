from pathlib import Path

import pytest

from lightllm.models.pi0.config import (
    Pi0VLAConfig,
    StateMode,
    VLAModelType,
)
from lightllm.utils.config_utils import get_vocab_size


PI0_DIR = "/mtc/baishihao/vla/lerobot_models_for_vla/pi0_base"
PI05_DIR = "/mtc/baishihao/vla/lerobot_models_for_vla/pi05_base"


@pytest.mark.skipif(not Path(PI0_DIR).is_dir(), reason="pi0 checkpoint is not mounted")
def test_loads_pi0_lerobot_config():
    config = Pi0VLAConfig.from_model_dir(PI0_DIR)
    assert config.model_type is VLAModelType.PI0
    assert config.state_mode is StateMode.SUFFIX_CONTINUOUS
    assert config.action_dim == 32
    assert config.action_horizon == 50
    assert config.num_denoise_steps == 10
    assert config.tokenizer_max_length == 48
    assert config.vocab_size == 257152
    assert config.image_keys == (
        "observation.images.base_0_rgb",
        "observation.images.left_wrist_0_rgb",
        "observation.images.right_wrist_0_rgb",
    )


@pytest.mark.skipif(not Path(PI05_DIR).is_dir(), reason="pi0.5 checkpoint is not mounted")
def test_loads_pi05_lerobot_config():
    config = Pi0VLAConfig.from_model_dir(PI05_DIR)
    assert config.model_type is VLAModelType.PI05
    assert config.state_mode is StateMode.PREFIX_DISCRETE
    assert config.tokenizer_max_length == 200


@pytest.mark.skipif(not Path(PI0_DIR).is_dir(), reason="pi0 checkpoint is not mounted")
def test_rejects_model_state_mode_mismatch():
    with pytest.raises(ValueError, match="suffix_continuous"):
        Pi0VLAConfig.from_model_dir(PI0_DIR, state_mode="prefix_discrete")
    with pytest.raises(ValueError, match="does not match checkpoint"):
        Pi0VLAConfig.from_model_dir(PI0_DIR, model_type="pi05")


@pytest.mark.skipif(not Path(PI05_DIR).is_dir(), reason="pi0.5 checkpoint is not mounted")
def test_validates_request_overrides():
    config = Pi0VLAConfig.from_model_dir(PI05_DIR).with_overrides(
        action_dim=7,
        action_horizon=10,
        num_denoise_steps=5,
    )
    assert (config.action_dim, config.action_horizon, config.num_denoise_steps) == (
        7,
        10,
        5,
    )
    with pytest.raises(ValueError, match="action_dim"):
        config.with_overrides(action_dim=33)


@pytest.mark.skipif(not Path(PI0_DIR).is_dir(), reason="pi0 checkpoint is not mounted")
def test_dtype_override_and_zero_values_are_validated():
    config = Pi0VLAConfig.from_model_dir(PI0_DIR, dtype="bf16")
    assert str(config.torch_dtype) == "torch.bfloat16"
    with pytest.raises(ValueError, match="action_horizon"):
        Pi0VLAConfig.from_model_dir(PI0_DIR, action_horizon=0)


def test_vocab_size_comes_from_checkpoint_lm_head(tmp_path):
    import json

    import torch
    from safetensors.torch import save_file

    (tmp_path / "config.json").write_text(json.dumps({"type": "pi0"}), encoding="utf-8")
    save_file(
        {
            "paligemma_with_expert.paligemma.lm_head.weight": torch.zeros(17, 2),
        },
        tmp_path / "model.safetensors",
    )

    assert Pi0VLAConfig.from_model_dir(str(tmp_path)).vocab_size == 17
    assert get_vocab_size(str(tmp_path)) == 17
