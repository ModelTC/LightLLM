"""FP8E8M0SlidingWindowState 的独立创建入口。"""

import torch

from .fp8_e8m0_swa_state import FP8E8M0SlidingWindowConfig, FP8E8M0SlidingWindowState


def create_sliding_window_state(
    config: FP8E8M0SlidingWindowConfig,
    request_capacity: int,
    mtp_step: int = 0,
    device: str | torch.device = "cuda",
) -> FP8E8M0SlidingWindowState:
    return FP8E8M0SlidingWindowState(
        config=config,
        request_capacity=request_capacity,
        mtp_step=mtp_step,
        device=device,
    )
