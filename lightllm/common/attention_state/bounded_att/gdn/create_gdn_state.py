"""GDN State 的统一创建入口。"""

from __future__ import annotations

import torch

from ..conv_ssm_types import ConvSSMStateConfig
from .gdn_state import GDNNativeState, GDNReplayState, GDNState


def create_gdn_state(
    config: ConvSSMStateConfig,
    request_capacity: int,
    mtp_step: int = 0,
    device: str | torch.device = "cuda",
    *,
    ssm_state_mode: str = "native",
    ssm_replay_config: dict[str, object] | None = None,
    gate_lower_bound: float = -5.0,
) -> GDNState:
    """创建 GDN 请求运行态，大小页池由后续容量分配流程初始化。"""
    state_cls = {
        "native": GDNNativeState,
        "replay": GDNReplayState,
    }.get(ssm_state_mode)
    if state_cls is None:
        raise ValueError(f"unknown GDN state mode: {ssm_state_mode}")
    return state_cls(
        config=config,
        request_capacity=request_capacity,
        mtp_step=mtp_step,
        device=device,
        ssm_state_mode=ssm_state_mode,
        ssm_replay_config=ssm_replay_config,
        gate_lower_bound=gate_lower_bound,
    )
