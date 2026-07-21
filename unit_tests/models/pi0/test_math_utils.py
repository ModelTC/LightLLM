import math

import pytest
import torch

from lightllm.models.pi0.math_utils import (
    create_sinusoidal_pos_embedding,
    denoise_schedule,
)


def test_timestep_embedding_endpoints_match_formula():
    time = torch.tensor([0.0, 1.0], dtype=torch.float32)
    result = create_sinusoidal_pos_embedding(time, 4, min_period=0.004, max_period=4.0)
    expected_at_one = torch.tensor(
        [
            math.sin(2 * math.pi / 0.004),
            math.sin(2 * math.pi / 4.0),
            math.cos(2 * math.pi / 0.004),
            math.cos(2 * math.pi / 4.0),
        ],
        dtype=torch.float64,
    )
    torch.testing.assert_close(result[0], torch.tensor([0.0, 0.0, 1.0, 1.0], dtype=torch.float64))
    torch.testing.assert_close(result[1], expected_at_one, atol=1e-12, rtol=1e-12)


@pytest.mark.parametrize("steps", [1, 4, 10])
def test_denoise_schedule_has_exact_requested_step_count(steps):
    times, dt = denoise_schedule(steps)
    assert len(times) == steps
    assert times[0] == 1.0
    torch.testing.assert_close(times[-1], torch.tensor(1.0 / steps, dtype=torch.float32))
    torch.testing.assert_close(dt, torch.tensor(-1.0 / steps, dtype=torch.float32))
