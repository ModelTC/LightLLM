"""K-pool attention tail State 的统一创建入口。"""

from __future__ import annotations

import torch

from .kpool_state import KPoolAttentionTailState, KPoolStateConfig


def create_kpool_state(
    config: KPoolStateConfig,
    request_capacity: int,
    mtp_step: int = 0,
    *,
    device: str | torch.device = "cuda",
) -> KPoolAttentionTailState:
    """按池化配置和请求容量创建 K-pool 尾部运行态，包含 MTP 额外槽位。"""
    return KPoolAttentionTailState(config=config, request_capacity=request_capacity, mtp_step=mtp_step, device=device)
