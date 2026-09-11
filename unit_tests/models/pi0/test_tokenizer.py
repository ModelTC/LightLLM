from types import SimpleNamespace

import numpy as np
import pytest
import torch

from lightllm.models.pi0.tokenizer import Pi0TokenizerAdapter
from lightllm.models.pi05.tokenizer import (
    Pi05TokenizerAdapter,
    discretize_state_256,
    format_pi05_prompt,
)


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


def test_default_camera_order_belongs_to_multimodal_input_adapter():
    assert Pi0TokenizerAdapter.DEFAULT_IMAGE_KEYS == (
        "observation.images.base_0_rgb",
        "observation.images.left_wrist_0_rgb",
        "observation.images.right_wrist_0_rgb",
    )


def test_named_pi_images_follow_checkpoint_camera_order():
    adapter = Pi0TokenizerAdapter.__new__(Pi0TokenizerAdapter)
    adapter.image_keys = ("base", "left", "right")
    images = [
        SimpleNamespace(input_key="right"),
        SimpleNamespace(input_key="base"),
    ]

    assert [image.input_key for image in adapter._ordered_images(images)] == [
        "base",
        "right",
    ]

    with pytest.raises(ValueError, match="all named or all positional"):
        adapter._ordered_images([images[0], SimpleNamespace(input_key=None)])


def test_pi05_state_normalization_uses_action_domain():
    adapter = Pi05TokenizerAdapter.__new__(Pi05TokenizerAdapter)
    calls = []
    adapter.preprocessor = SimpleNamespace(
        normalize_state=lambda state, *, domain: calls.append((state, domain))
        or torch.tensor(state)
    )
    adapter.policy_tokenizer = SimpleNamespace(
        tokenize=lambda prompts, *, states: (prompts, states)
    )
    multimodal = SimpleNamespace(
        state=[0.25, -0.5],
        action=SimpleNamespace(domain="libero"),
    )

    prompts, states = adapter._tokenize_prompt("move", multimodal)

    assert calls == [([0.25, -0.5], "libero")]
    assert prompts == ["move"]
    torch.testing.assert_close(states[0], torch.tensor([0.25, -0.5]))
