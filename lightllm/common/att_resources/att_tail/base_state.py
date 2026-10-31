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
        """返回算法与存储方案标记，用于生成资源成员名。

        返回小写 snake_case 的“算法＋存储方案”，不带 full_att/bounded_att/att_tail 类别前缀，
        也不带 main/draftN 角色或最终的 _state；类别由 Python State 类型表达。
        管理器统一生成 resource_id = f"{model_name}_{get_state_type()}_state"，与成员名完全一致。
        同角色下不同算法、精度或量化方案必须使用不同标记，同方案的全部逻辑层放在同一实例。
        层号、容量、设备、TP rank 和请求信息不进入名称；完整布局仍由 config 描述。
        注册后返回值保持稳定，实际分配与 CPU/DP/PD 传输均使用相同标记。

        K-pool 格式为 kpool_c{压缩比例}_{存储精度}，例如 kpool_c4_bf16，
        对应 main_kpool_c4_bf16_state；同压缩比例的不同存储精度使用不同名称。
        新实现使用具体算法及必要方案参数，不仅返回 indexer 或 tail 这种通用名称。
        """
        ...

    # ========================================================================
    # 二、Attention tail：计算视图与无快照重置
    # ========================================================================

    @abstractmethod
    def contains_logical_layer(self, logical_layer_id: int) -> bool:
        """判断指定逻辑层是否属于当前 State 的层映射，不依赖存储分配。"""
        ...

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
        self,
        req_idx: int,
        pd_page_buffers: torch.Tensor,
        pd_page_index: int,
        state_token_len: int,
        layout: StateTransferLayout,
    ) -> None:
        """按 state_token_len 对应的 prefill 终点导出逻辑窗口，不导出推测候选历史。"""
        ...

    @abstractmethod
    def read_from_pd_buffer(
        self,
        pd_page_buffers: torch.Tensor,
        req_idx: int,
        pd_page_index: int,
        state_token_len: int,
        layout: StateTransferLayout,
    ) -> None:
        """按 state_token_len 将逻辑窗口恢复到本地运行态，适配本地 MTP 容量。"""
        ...
