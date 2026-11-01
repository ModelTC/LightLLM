"""WindowCompressorState 的独立创建入口。"""

import torch

from .window_compressor_state import WindowCompressorStateConfig, WindowCompressorState


def create_window_compressor_state(
    config: WindowCompressorStateConfig,
    request_capacity: int,
    mtp_step: int = 0,
    device: str | torch.device = "cuda",
) -> WindowCompressorState:
    return WindowCompressorState(
        config=config,
        request_capacity=request_capacity,
        mtp_step=mtp_step,
        device=device,
    )
