"""GQA（含 MHA/MQA）的统一创建入口，选择普通精度或量化 State。"""

from __future__ import annotations

from typing import Literal

import torch

from .base_state import FullAttentionConfig
from .gqa_state import GQAFullAttentionState
from .group_quantized_state import GroupQuantizedAttentionConfig, GroupQuantizedFullAttentionState
from .static_fp8_state import StaticFP8AttentionConfig, StaticFP8FullAttentionState


def create_gqa_state(
    config: FullAttentionConfig,
    *,
    kv_type: Literal["None", "int4kv", "int8kv", "fp8kv_spt", "fp8kv_sph"] = "None",
    page_size: int = 1,
    device: str | torch.device = "cuda",
    group_size: int = 8,
    scales: torch.Tensor | None = None,
    q_scales: torch.Tensor | None = None,
    calibration_id: str = "",
) -> GQAFullAttentionState | GroupQuantizedFullAttentionState | StaticFP8FullAttentionState:
    """按编码参数创建 GQA（含 MHA/MQA）State，普通 token 容量由后续流程分配。

    config 描述逻辑 KV 维度和 TP 信息；kv_type 沿用启动参数中的编码名称。
    group_size 仅用于 INT4/INT8；scales、q_scales 和 calibration_id 仅用于静态 FP8。
    静态标定数据由调用方加载，本入口不读取启动参数、配置文件或绑定模型角色。
    """
    if kv_type == "None":
        return GQAFullAttentionState(layout_config=config, page_size=page_size, device=device)
    if kv_type in ("int4kv", "int8kv"):
        quant_config = GroupQuantizedAttentionConfig(
            layer_ids=config.layer_ids,
            global_head_num=config.global_head_num,
            head_dim=config.head_dim,
            dtype=config.dtype,
            tp_rank=config.tp_rank,
            tp_world_size=config.tp_world_size,
            replicated=config.replicated,
            bits=4 if kv_type == "int4kv" else 8,
            group_size=group_size,
        )
        return GroupQuantizedFullAttentionState(layout_config=quant_config, page_size=page_size, device=device)
    assert kv_type in ("fp8kv_spt", "fp8kv_sph"), f"unsupported GQA encoding: {kv_type}"
    assert scales is not None, "static FP8 requires calibration scales"
    assert kv_type != "fp8kv_sph" or q_scales is not None, "per-head FP8 requires Q calibration scales"
    quant_config = StaticFP8AttentionConfig(
        layer_ids=config.layer_ids,
        global_head_num=config.global_head_num,
        head_dim=config.head_dim,
        dtype=torch.uint8,
        tp_rank=config.tp_rank,
        tp_world_size=config.tp_world_size,
        replicated=config.replicated,
        scale_mode="per_head" if kv_type == "fp8kv_sph" else "per_tensor",
        calibration_id=calibration_id,
    )
    return StaticFP8FullAttentionState(
        layout_config=quant_config,
        page_size=page_size,
        device=device,
        scales=scales,
        q_scales=q_scales,
    )
