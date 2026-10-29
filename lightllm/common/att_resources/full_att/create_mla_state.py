"""MLA 的统一创建入口，选择普通、packed 或 packed FP8 State。"""

from __future__ import annotations

from dataclasses import replace
from typing import Literal

import torch

from .base_state import FullAttentionConfig
from .mla_state import MLAFullAttentionState
from .packed_mla_state import PackedMLAAttentionConfig, PackedMLAFullAttentionState


def create_mla_state(
    config: FullAttentionConfig,
    *,
    kv_type: Literal["None", "fp8kv_dsa"] = "None",
    packed: bool = False,
    indexer_dim: int = 128,
    page_size: int = 1,
    device: str | torch.device = "cuda",
) -> MLAFullAttentionState | PackedMLAFullAttentionState:
    """创建普通或 packed MLA State，统一处理量化编码、TP 副本和物理行宽。

    config.head_dim 始终是逻辑 KV latent 维度，尚未包含 indexer、scale 和 padding。
    packed=True 时把 indexer 字段存入同一行；fp8kv_dsa 仅适用于 packed MLA。
    """
    if not packed:
        assert kv_type == "None", f"unsupported MLA encoding: {kv_type}"
        return MLAFullAttentionState(
            layout_config=replace(config, replicated=True),
            page_size=page_size,
            device=device,
        )
    assert kv_type in ("None", "fp8kv_dsa"), f"unsupported packed MLA encoding: {kv_type}"
    fp8 = kv_type == "fp8kv_dsa"
    storage_dtype = torch.uint8 if fp8 else config.dtype
    # FP8 latent 包含 512 字节编码、16 字节 scale 和 128 字节 BF16 RoPE。
    # indexer key 使用 INT8 并附带一个 FP32 scale；整行按 16 字节对齐。
    kv_bytes = 656 if fp8 else config.head_dim * config.dtype.itemsize
    row_bytes = (kv_bytes + indexer_dim + 4 + 15) // 16 * 16
    packed_config = PackedMLAAttentionConfig(
        layer_ids=config.layer_ids,
        global_head_num=config.global_head_num,
        head_dim=row_bytes // storage_dtype.itemsize,
        dtype=storage_dtype,
        tp_rank=config.tp_rank,
        tp_world_size=config.tp_world_size,
        replicated=True,
        kv_dim=config.head_dim,
        indexer_dim=indexer_dim,
        fp8_nope_dim=512 if fp8 else 0,
    )
    return PackedMLAFullAttentionState(layout_config=packed_config, page_size=page_size, device=device)
