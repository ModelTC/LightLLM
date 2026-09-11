import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from lightllm.models.pi0.preprocessing import (
    Pi0PrePostProcessor,
    build_pi_domain_profiles,
    load_pi_action_config,
)
from lightllm.utils.config_utils import get_pi_config


PI0_DIR = "/mtc/baishihao/vla/lerobot_models_for_vla/pi0_base"


def test_pi_domain_profiles_resolve_benchmark_roles_at_startup():
    image_keys = (
        "observation.images.base_0_rgb",
        "observation.images.left_wrist_0_rgb",
        "observation.images.right_wrist_0_rgb",
    )

    profiles = build_pi_domain_profiles(image_keys)

    assert profiles["libero"]["image_map"]["agentview"] == image_keys[0]
    assert profiles["libero"]["image_map"]["wrist"] == image_keys[1]
    assert profiles["robotwin"]["required_images"] == list(image_keys)


def test_pi_runtime_config_owns_custom_domain_aliases(tmp_path):
    image_keys = (
        "observation.images.base_0_rgb",
        "observation.images.left_wrist_0_rgb",
        "observation.images.right_wrist_0_rgb",
    )
    (tmp_path / "config.json").write_text(
        json.dumps(
            {
                "type": "pi0",
                "input_features": {
                    image_key: {"type": "VISUAL"} for image_key in image_keys
                },
            }
        ),
        encoding="utf-8",
    )
    (tmp_path / "model.safetensors").touch()
    domain_path = tmp_path / "domains.json"
    domain_path.write_text(
        json.dumps(
            {
                "aliases": {"sim_eval": "my_robot"},
                "domains": {
                    "my_robot": {
                        "state_dim": 3,
                        "action_dim": 2,
                        "image_map": {"front": "base"},
                        "required_images": ["base"],
                    }
                },
            }
        ),
        encoding="utf-8",
    )
    args = SimpleNamespace(
        model_dir=str(tmp_path),
        vla_action_dim=None,
        vla_action_horizon=None,
        vla_num_denoise_steps=None,
        vla_robot_adapter=None,
        vla_norm_config=None,
        vla_action_postprocess_config=None,
        vla_domain_config=str(domain_path),
        data_type="bf16",
    )

    config = load_pi_action_config(args)

    assert config["domain_aliases"]["sim_eval"] == "my_robot"
    assert config["domain_profiles"]["my_robot"]["image_map"] == {
        "front": image_keys[0]
    }


@pytest.mark.skipif(not Path(PI0_DIR).is_dir(), reason="pi0 checkpoint config not mounted")
def test_mean_std_state_and_action_roundtrip_matches_openpi_formula():
    config = get_pi_config(PI0_DIR)
    config.update(
        norm_config={
            "state": {"mode": "mean_std", "mean": [1.0, 2.0], "std": [2.0, 4.0]},
            "action": {"mode": "mean_std", "mean": [1.0, 2.0], "std": [2.0, 4.0]},
        },
    )
    processor = Pi0PrePostProcessor(config)
    normalized = processor.normalize_state(torch.tensor([[3.0, 6.0]]))
    torch.testing.assert_close(normalized, torch.tensor([[1.0, 1.0]]), atol=1e-6, rtol=1e-6)
    restored = processor.postprocess_actions(normalized)
    torch.testing.assert_close(restored, torch.tensor([[3.0, 6.0]]), atol=2e-6, rtol=2e-6)


@pytest.mark.skipif(not Path(PI0_DIR).is_dir(), reason="pi0 checkpoint config not mounted")
def test_quantile_normalization_and_relative_action_adapter():
    config = get_pi_config(PI0_DIR)
    config.update(
        norm_config={
            "state": {"mode": "quantiles", "q01": [0.0, 10.0], "q99": [2.0, 14.0]},
            "action": {"mode": "quantiles", "q01": [0.0, 10.0], "q99": [2.0, 14.0]},
        },
        robot_adapter={"relative_action_mask": [True, False]},
    )
    processor = Pi0PrePostProcessor(config)
    normalized = processor.normalize_state(torch.tensor([[1.0, 12.0]]))
    torch.testing.assert_close(normalized, torch.zeros_like(normalized), atol=1e-6, rtol=0)
    actions = processor.postprocess_actions(torch.zeros(1, 2), raw_state=torch.tensor([[5.0, 7.0]]))
    torch.testing.assert_close(actions, torch.tensor([[6.0, 12.0]]), atol=2e-6, rtol=0)


@pytest.mark.skipif(not Path(PI0_DIR).is_dir(), reason="pi0 checkpoint config not mounted")
def test_relative_action_adapter_preserves_batch_state():
    config = get_pi_config(PI0_DIR)
    config.update(
        robot_adapter={"relative_action_mask": [True, False]},
    )
    processor = Pi0PrePostProcessor(config)
    actual = processor.postprocess_actions(
        torch.zeros(2, 1, 2),
        raw_state=torch.tensor([[1.0, 2.0], [3.0, 4.0]]),
    )
    expected = torch.tensor([[[1.0, 0.0]], [[3.0, 0.0]]])
    torch.testing.assert_close(actual, expected)


def test_domain_selects_its_own_normalization_and_postprocess():
    config = {
        "norm_config": None,
        "robot_adapter": None,
        "action_postprocess_config": None,
        "domain_profiles": {
            "libero": {
                "state_dim": 2,
                "action_dim": 2,
                "norm_config": {
                    "state": {
                        "mode": "mean_std",
                        "mean": [1.0, 2.0],
                        "std": [2.0, 4.0],
                    },
                    "action": {
                        "mode": "mean_std",
                        "mean": [1.0, 2.0],
                        "std": [2.0, 4.0],
                    },
                },
                "action_postprocess_config": {"scale": [1.0, -1.0]},
            }
        },
    }
    processor = Pi0PrePostProcessor(config)

    normalized = processor.normalize_state(torch.tensor([3.0, 6.0]), domain="libero")
    torch.testing.assert_close(normalized, torch.ones(2), atol=1e-6, rtol=1e-6)
    actions = processor.postprocess_actions(torch.ones(1, 2), domain="libero")
    torch.testing.assert_close(
        actions, torch.tensor([[3.000002, -6.000004]]), atol=2e-6, rtol=1e-6
    )


def test_openpi_aloha_robot_coordinates():
    config = {
        "domain_profiles": {
            "aloha": {
                "robot_adapter": {"type": "openpi_aloha"},
            }
        }
    }
    processor = Pi0PrePostProcessor(config)
    state = torch.zeros(14)
    state[1] = 2.0
    state[[6, 13]] = torch.tensor([0.0, 1.0])

    normalized = processor.normalize_state(state, domain="aloha")
    assert normalized[1] == -2.0
    torch.testing.assert_close(
        normalized[[6, 13]],
        torch.tensor([-1.0798653, 0.9456529]),
        atol=1e-6,
        rtol=1e-6,
    )

    actions = torch.zeros(1, 14)
    actions[0, 1] = 2.0
    restored = processor.postprocess_actions(actions, domain="aloha")
    assert restored[0, 1] == -2.0
    torch.testing.assert_close(
        restored[0, [6, 13]],
        torch.tensor([0.5533779, 0.5533779]),
        atol=1e-6,
        rtol=1e-6,
    )
