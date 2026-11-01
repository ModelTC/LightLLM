"""KDA pages: Conv history and square recurrent matrices; allocators own the page slots."""

from __future__ import annotations


import torch

from ..conv_ssm_types import ConvSSMStateConfig
from ..utils import create_page_pool_tensor, register_page_pool_tensor


class KDAPagesPool:
    def __init__(
        self,
        size: int,
        config: ConvSSMStateConfig,
        device: str | torch.device | int = "cpu",
        pin_memory: bool = True,
        snapshot_fields=None,
    ) -> None:
        assert size >= 0, "pool size must be nonnegative"
        assert (
            config.num_k_heads == config.num_v_heads and config.key_dim == config.value_dim
        ), "KDA pages require equal Q/K/V heads and dimensions"
        self.size = size
        device = torch.device(device)
        pin_memory = pin_memory and device.type == "cpu"
        if snapshot_fields is None:
            from ..formats import conv_ssm_spec

            snapshot_fields = conv_ssm_spec(config, request_capacity=0, mtp_step=0, algorithm="kda").snapshot_fields
        for field in snapshot_fields:
            setattr(
                self,
                field.attribute,
                create_page_pool_tensor(
                    size=(size, *field.shape),
                    dtype=field.dtype,
                    device=device,
                    pin_memory=pin_memory,
                ),
            )

    # ========================================================================
    # 跨进程映射：在接收进程的当前 GPU 上注册共享页池
    # ========================================================================

    def register_pin_memory(self) -> None:
        """注册 Conv/SSM 共享存储，供本进程 GPU 使用；重复调用不会重复注册。"""
        register_page_pool_tensor(tensor=self.conv_state)
        register_page_pool_tensor(tensor=self.ssm_state)
