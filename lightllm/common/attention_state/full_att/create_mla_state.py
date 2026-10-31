"""MLA 的统一创建入口，选择普通、packed 或 packed FP8 State。"""

from __future__ import annotations

from typing import Literal

import torch

from .base_state import FullAttentionState
from .mla_state import MLAAttentionConfig, MLAFullAttentionState
from .packed_mla_state import PackedMLAAttentionConfig, PackedMLAFullAttentionState
from .fp8_e8m0_packed_mla_state import FP8E8M0PackedMLAConfig, FP8E8M0PackedMLAState


def create_mla_state(
    config: MLAAttentionConfig,
    kv_type: Literal["None", "fp8kv_dsa"] = "None",
    packed: bool = False,
    indexer_dim: int = 128,
    page_size: int = 1,
    device: str | torch.device = "cuda",
    fp8_format: Literal["group128_fp32", "group64_e8m0"] = "group128_fp32",
) -> FullAttentionState:
    """创建普通或 packed MLA State，统一处理量化编码、TP 副本和物理行宽。

    config.latent_dim 始终是逻辑 KV latent 维度，尚未包含 indexer、scale 和 padding。
    packed=True 时把 indexer 字段存入同一行；fp8kv_dsa 仅适用于 packed MLA。
    fp8_format 由模型显式选择，仅在 packed FP8 时生效；不根据 latent 维度推断编码。
    group128_fp32 与 group64_e8m0 使用独立 State。
    """
    if not packed:
        assert kv_type == "None", f"unsupported MLA encoding: {kv_type}"
        return MLAFullAttentionState(
            layout_config=config,
            page_size=page_size,
            device=device,
        )
    assert kv_type in ("None", "fp8kv_dsa"), f"unsupported packed MLA encoding: {kv_type}"
    fp8 = kv_type == "fp8kv_dsa"
    assert not fp8 or fp8_format in ("group128_fp32", "group64_e8m0"), f"unknown packed MLA FP8 format: {fp8_format}"
    if fp8 and fp8_format == "group64_e8m0":
        assert config.latent_dim == FP8E8M0PackedMLAConfig.LATENT_DIM, "FP8/E8M0 MLA requires a 512-dimensional latent"
        return FP8E8M0PackedMLAState(
            layout_config=FP8E8M0PackedMLAConfig(
                logical_layer_groups=config.logical_layer_groups,
                tp_rank=config.tp_rank,
                tp_world_size=config.tp_world_size,
                indexer_dim=indexer_dim,
            ),
            page_size=page_size,
            device=device,
        )
    packed_config = PackedMLAAttentionConfig(
        logical_layer_groups=config.logical_layer_groups,
        latent_dim=config.latent_dim,
        dtype=torch.float8_e4m3fn if fp8 else config.dtype,
        tp_rank=config.tp_rank,
        tp_world_size=config.tp_world_size,
        indexer_dim=indexer_dim,
        fp8_nope_dim=512 if fp8 else 0,
    )
    return PackedMLAFullAttentionState(layout_config=packed_config, page_size=page_size, device=device)
