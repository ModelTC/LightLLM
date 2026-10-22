"""KDA State 的统一创建入口。"""

from __future__ import annotations

import torch

from ..conv_ssm_types import ConvSSMStateConfig
from .kda_state import KDAState


def create_kda_state(
    config: ConvSSMStateConfig,
    request_capacity: int,
    mtp_step: int = 0,
    device: str | torch.device = "cuda",
    *,
    ssm_state_mode: str = "native",
    ssm_replay_config: dict | None = None,
    gate_lower_bound: float = -5.0,
) -> KDAState:
    """创建 KDA 请求运行态，大小页池由后续容量分配流程初始化。"""
    return KDAState(
        config=config,
        request_capacity=request_capacity,
        mtp_step=mtp_step,
        device=device,
        ssm_state_mode=ssm_state_mode,
        ssm_replay_config=ssm_replay_config,
        gate_lower_bound=gate_lower_bound,
    )
