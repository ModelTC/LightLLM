"""Independent creation entry points for normal and packed MLA States."""

from __future__ import annotations

import torch

from ..base_state import FullAttentionState
from .mla_state import MLAConfig, MLAFullAttentionState


def create_mla_state(
    config: MLAConfig,
    page_size: int = 1,
    device: str | torch.device = "cuda",
) -> FullAttentionState:
    """Create a normal latent MLA State without indexer fields."""
    assert type(config) is MLAConfig, "normal MLA creation requires MLAConfig"
    return MLAFullAttentionState(layout_config=config, page_size=page_size, device=device)
