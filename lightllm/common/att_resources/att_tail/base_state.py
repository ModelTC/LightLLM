from __future__ import annotations

from typing import TYPE_CHECKING, Any
from abc import ABC, abstractmethod

from lightllm.common.att_resources.types import (
    StateTransferLayout,
    TransferRegion,
)

if TYPE_CHECKING:
    import torch


class AttentionTailState(ABC):
    """注意力相关模块的尾部状态接口，保存增量计算所需的尾部数据及控制信息。"""

    # ========================================================================
    # 一、资源定义与初始化
    # ========================================================================

    @abstractmethod
    def get_state_type(self) -> str:
        """返回具体算法与存储方案标记，用于生成资源成员名；绑定后须保持稳定。"""
        ...

    # ========================================================================
    # 二、Attention tail：计算视图与无快照重置
    # ========================================================================

    @abstractmethod
    def get_layer_view(self, logical_layer_id: int) -> Any:
        """按逻辑层号返回尾部状态计算所需的存储视图。"""
        ...

    @abstractmethod
    def reset_attention_tail_state(
        self,
        req_indices: list[int],
    ) -> None:
        """清零指定请求的 tail；调用方保证用于空前缀或可重置的缓存边界，不用于 MTP 回滚。"""
        ...

    # ========================================================================
    # 三、PD 分离传输：prefill 逻辑窗口
    # ========================================================================

    @abstractmethod
    def get_pd_transfer_regions(self) -> tuple[TransferRegion, ...]:
        """返回独立于本地 MTP 容量的 PD 字段、全局形状和 TP 切片，区域偏移相对本 State 起点。"""
        ...

    @abstractmethod
    def write_to_pd_buffer(
        self, req_idx: int, pd_page_buffers: torch.Tensor,
        *, pd_page_index: int, state_token_len: int, layout: StateTransferLayout,
    ) -> None:
        """按 state_token_len 对应的 prefill 终点导出逻辑窗口，不导出推测候选历史。"""
        ...

    @abstractmethod
    def read_from_pd_buffer(
        self, pd_page_buffers: torch.Tensor, req_idx: int,
        *, pd_page_index: int, state_token_len: int, layout: StateTransferLayout,
    ) -> None:
        """按 state_token_len 将逻辑窗口恢复到本地运行态，适配本地 MTP 容量。"""
        ...
