"""GDN pages with separate Q/K and V dimensions; allocators own the page slots."""

from __future__ import annotations


import torch

from .conv_ssm_types import ConvSSMStateConfig
from .utils import create_page_pool_tensor, register_page_pool_tensor


class GDNPagesPool:
    def __init__(
        self,
        size: int,
        config: ConvSSMStateConfig,
        *,
        device: str | torch.device | int = "cpu",
        pin_memory: bool = True,
    ) -> None:
        assert size >= 0, "pool size must be nonnegative"
        self.size = size
        device = torch.device(device)
        pin_memory = pin_memory and device.type == "cpu"
        self.conv_state = create_page_pool_tensor(
            size=(size, len(config.layer_ids), config.conv_channels, config.conv_width),
            dtype=config.conv_dtype,
            device=device,
            pin_memory=pin_memory,
        )
        self.ssm_state = create_page_pool_tensor(
            size=(size, len(config.layer_ids), config.num_v_heads, config.key_dim, config.value_dim),
            dtype=config.ssm_dtype,
            device=device,
            pin_memory=pin_memory,
        )

    # ========================================================================
    # 跨进程映射：在接收进程的当前 GPU 上注册共享页池
    # ========================================================================

    def register_pin_memory(self) -> None:
        """注册 Conv/SSM 共享存储，供本进程 GPU 使用；重复调用不会重复注册。"""
        register_page_pool_tensor(tensor=self.conv_state)
        register_page_pool_tensor(tensor=self.ssm_state)
