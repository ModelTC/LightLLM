"""Creation entry point for packed MLA States."""

from __future__ import annotations

import torch

from ..base_state import FullAttentionState
from .packed_mla_state import PackedMLAAttentionConfig, PackedMLALogicalConfig, PackedMLAFullAttentionState
from .fp8_e8m0_packed_mla_state import FP8E8M0PackedMLAConfig, FP8E8M0PackedMLAState


def create_packed_mla_state(
    config: PackedMLALogicalConfig,
    page_size: int = 1,
    device: str | torch.device = "cuda",
) -> FullAttentionState:
    """Create a packed MLA State; packed formats always include an indexer field."""
    assert type(config) is PackedMLALogicalConfig, "packed MLA creation requires PackedMLALogicalConfig"
    if config.indexer_pool_size > 1:
        from .pooled_mla_state import PooledMLAConfig, PooledMLAState

        assert config.rope_dim == 0, "pooled MLA supports NoPE latent only"
        return PooledMLAState(
            layout_config=PooledMLAConfig(
                logical_layer_groups=config.logical_layer_groups,
                format=config.format,
                latent_dim=config.latent_dim,
                indexer_dim=config.indexer_dim,
                tp_rank=config.tp_rank,
                tp_world_size=config.tp_world_size,
            ),
            page_size=page_size,
            device=device,
        )
    if config.format == "packed_nvfp4":
        from ..mla.nvfp4_state import NVFP4MLAConfig, NVFP4MLAState

        return NVFP4MLAState(
            layout_config=NVFP4MLAConfig(
                logical_layer_groups=config.logical_layer_groups,
                latent_dim=config.latent_dim - config.rope_dim,
                rope_dim=config.rope_dim,
                indexer_dim=config.indexer_dim,
                tp_rank=config.tp_rank,
                tp_world_size=config.tp_world_size,
            ),
            page_size=page_size,
            device=device,
        )
    if config.format == "packed_group64_e8m0":
        assert config.latent_dim == FP8E8M0PackedMLAConfig.LATENT_DIM, "FP8/E8M0 MLA requires a 512-dimensional latent"
        return FP8E8M0PackedMLAState(
            layout_config=FP8E8M0PackedMLAConfig(
                logical_layer_groups=config.logical_layer_groups,
                tp_rank=config.tp_rank,
                tp_world_size=config.tp_world_size,
                indexer_dim=config.indexer_dim,
            ),
            page_size=page_size,
            device=device,
        )

    return PackedMLAFullAttentionState(
        layout_config=PackedMLAAttentionConfig(logical_config=config),
        page_size=page_size,
        device=device,
    )
