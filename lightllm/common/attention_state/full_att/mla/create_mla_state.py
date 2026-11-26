"""Independent creation entry points for normal and packed MLA States."""

from __future__ import annotations

import torch

from ..base_state import FullAttentionState
from .mla_state import MLAConfig, MLAFullAttentionState


def create_mla_state(
    config: MLAConfig,
    page_size: int = 1,
    device: str | torch.device = "cuda",
    kv_type: str = "None",
    rope_dim: int = 0,
) -> FullAttentionState:
    """Create a normal latent MLA State without indexer fields."""
    assert type(config) is MLAConfig, "normal MLA creation requires MLAConfig"
    if kv_type == "nvfp4kv":
        from .nvfp4_state import NVFP4MLAConfig, NVFP4MLAState

        return NVFP4MLAState(
            NVFP4MLAConfig(
                logical_layer_groups=config.logical_layer_groups,
                latent_dim=config.latent_dim - rope_dim,
                rope_dim=rope_dim,
                dtype=config.latent_dtype,
                tp_rank=config.tp_rank,
                tp_world_size=config.tp_world_size,
            ),
            page_size=page_size,
            device=device,
        )
    assert kv_type == "None", f"unsupported dense MLA KV type: {kv_type}"
    return MLAFullAttentionState(layout_config=config, page_size=page_size, device=device)
