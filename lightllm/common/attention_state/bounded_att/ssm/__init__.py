from __future__ import annotations

from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from ..conv_ssm_types import ConvSSMStateConfig
    from .base import SSMExecutor


def create_ssm_executor(
    config: ConvSSMStateConfig,
    request_capacity: int,
    mtp_step: int,
    device: str | torch.device,
    mode: str,
    replay_config: dict[str, object] | None,
    *,
    kda: bool,
    gate_lower_bound: float,
) -> SSMExecutor:
    """根据状态模式创建 Native 或 Replay SSM executor。"""
    """按统一 state layout 创建 native 或 replay executor。

    native 为每个 MTP 位置分配物理槽位，replay 为每个请求分配一个槽位，
    并在 executor 内保存短 history。
    """

    if mode not in ("native", "replay"):
        raise ValueError(f"unknown SSM state mode: {mode}")
    slots = mtp_step + 1 if mode == "native" else 1
    state = torch.zeros(
        (len(config.logical_layer_ids), request_capacity * slots, config.num_v_heads, config.key_dim, config.value_dim),
        dtype=config.ssm_dtype,
        device=device,
    )
    kwargs = dict(kda=kda, gate_lower_bound=gate_lower_bound)
    if mode == "native":
        from .native import NativeSSMExecutor

        return NativeSSMExecutor(state, mtp_step + 1, **kwargs)
    kwargs.update(num_key_heads=config.num_k_heads)
    from .replay import ReplaySSMExecutor

    replay_config = replay_config or {}
    return ReplaySSMExecutor(
        state,
        replay_config.get("cache_len", 16),
        mtp_step + 1,
        config.conv_dtype,
        compute_mode=replay_config.get("compute_mode", "precompute" if mtp_step > 0 else "inline"),
        **kwargs,
    )
