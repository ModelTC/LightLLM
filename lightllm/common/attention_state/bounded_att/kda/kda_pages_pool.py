"""KDA pages: Conv history and square recurrent matrices; allocators own the page slots."""

from __future__ import annotations


import torch

from ..conv_ssm_types import ConvSSMStateConfig
from ..conv_ssm_pages_pool import ConvSSMPagesPool


class KDAPagesPool(ConvSSMPagesPool):
    def __init__(
        self,
        size: int,
        config: ConvSSMStateConfig,
        device: str | torch.device | int = "cpu",
        pin_memory: bool = True,
    ) -> None:
        assert size >= 0, "pool size must be nonnegative"
        assert (
            config.num_k_heads == config.num_v_heads and config.key_dim == config.value_dim
        ), "KDA pages require equal Q/K/V heads and dimensions"
        super().__init__(size=size, config=config, device=device, pin_memory=pin_memory)
