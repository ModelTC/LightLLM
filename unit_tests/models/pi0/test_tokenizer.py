import numpy as np
import torch

from lightllm.models.pi0.tokenizer import discretize_state_256, format_pi05_prompt


def test_discrete_state_matches_numpy_digitize_boundaries():
    state = np.array([-1.1, -1.0, -0.5, 0.0, 0.5, 0.9921875, 1.0, 1.1])
    boundaries = np.linspace(-1, 1, 257)[:-1]
    expected = np.digitize(state, bins=boundaries) - 1
    np.testing.assert_array_equal(discretize_state_256(state), expected)
    torch.testing.assert_close(
        discretize_state_256(torch.from_numpy(state)),
        torch.from_numpy(expected),
    )


def test_pi05_prompt_format_is_openpi_compatible():
    prompt = format_pi05_prompt("pick_up\nblock", np.array([-1.0, 0.0, 1.0]))
    assert prompt == "Task: pick up block, State: 0 128 255;\nAction: "
