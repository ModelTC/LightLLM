from dataclasses import replace
from pathlib import Path

import pytest
import torch

from lightllm.models.pi0.config import Pi0VLAConfig
from lightllm.models.pi0.preprocessing import Pi0PrePostProcessor


PI0_DIR = "/mtc/baishihao/vla/lerobot_models_for_vla/pi0_base"


@pytest.mark.skipif(not Path(PI0_DIR).is_dir(), reason="pi0 checkpoint config not mounted")
def test_mean_std_state_and_action_roundtrip_matches_openpi_formula():
    config = replace(
        Pi0VLAConfig.from_model_dir(PI0_DIR),
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
    config = replace(
        Pi0VLAConfig.from_model_dir(PI0_DIR),
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
    config = replace(
        Pi0VLAConfig.from_model_dir(PI0_DIR),
        robot_adapter={"relative_action_mask": [True, False]},
    )
    processor = Pi0PrePostProcessor(config)
    actual = processor.postprocess_actions(
        torch.zeros(2, 1, 2),
        raw_state=torch.tensor([[1.0, 2.0], [3.0, 4.0]]),
    )
    expected = torch.tensor([[[1.0, 0.0]], [[3.0, 0.0]]])
    torch.testing.assert_close(actual, expected)
