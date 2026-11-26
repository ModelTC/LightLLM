"""Construct the packed sliding-window State selected by its encoding config."""

import torch

from .packed_swa_state import PackedSlidingWindowConfig, PackedSlidingWindowState
from .fp8_e8m0_swa_state import FP8E8M0SlidingWindowState
from .nvfp4_swa_state import NVFP4SlidingWindowState


def create_sliding_window_state(
    config: PackedSlidingWindowConfig,
    request_capacity: int,
    mtp_step: int = 0,
    device: str | torch.device = "cuda",
) -> PackedSlidingWindowState:
    state_class = {
        "fp8_group64_e8m0": FP8E8M0SlidingWindowState,
        "nvfp4_block16": NVFP4SlidingWindowState,
    }[config.kv_format]
    return state_class(config=config, request_capacity=request_capacity, mtp_step=mtp_step, device=device)
