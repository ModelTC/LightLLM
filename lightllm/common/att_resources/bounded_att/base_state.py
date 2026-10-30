"""有界 attention 状态接口；具体算法实现运行态与大小页存储。"""

from __future__ import annotations

from typing import Any
from abc import ABC, abstractmethod
from typing import List

import torch

from lightllm.common.att_resources.types import (
    StateTransferLayout,
    PageKind,
    TransferRegion,
)


class BoundedAttentionState(ABC):
    """单种 bounded-attention 方案的数据操作，索引和槽位由调用方分配。"""

    # ========================================================================
    # 一、资源定义与初始化
    # ========================================================================

    @abstractmethod
    def get_state_type(self) -> str:
        """返回具体算法与存储方案标记，用于生成资源成员名；绑定后须保持稳定。"""
        ...

    @abstractmethod
    def initialize_page_pools(
        self,
        big_page_capacity: int,
        small_page_capacity: int = 0,
        *,
        device: str | torch.device | int = "cpu",
        pin_memory: bool = True,
    ) -> None:
        """容量规划完成后创建大小页池；容量为零表示不启用该页池。"""
        ...

    # ========================================================================
    # 二、Bounded attention：计算视图、运行态重置、大小页与 MTP 提交
    # ========================================================================

    @abstractmethod
    def get_layer_view(self, logical_layer_id: int) -> Any:
        """按逻辑层号返回运行态和算法控制信息的计算视图。"""
        ...

    @abstractmethod
    def reset_bounded_att_state(self, req_indices: List[int]) -> None:
        """将指定槽位的运行态及控制信息恢复到长度 0，保留大小页快照。"""
        ...

    @abstractmethod
    def save_bounded_att_pages(self, req_indices: List[int], *, page_kind: PageKind, slot_ids: List[int]) -> None:
        """将 prefill 完成后的基准运行态保存到指定页槽位，前缀关联由调用方维护。"""
        ...

    @abstractmethod
    def restore_bounded_att_pages(self, req_indices: List[int], *, page_kind: PageKind, slot_ids: List[int]) -> None:
        """从调用方选定的页槽位恢复运行态和控制信息，保留源快照数据。"""
        ...

    @abstractmethod
    def commit_bounded_att_state(
        self,
        req_indices: torch.Tensor,
        accepted_steps: torch.Tensor,
    ) -> None:
        """两个输入均为 CUDA Tensor；int32 接受步数包含基准输入步，候选区与游标由实现管理。"""
        ...

    # ========================================================================
    # 三、CPU Cache 数据交换：读写大小页快照
    # ========================================================================

    @abstractmethod
    def get_cpu_transfer_regions(self) -> tuple[TransferRegion, ...]:
        """返回一份 CPU 快照的字段区间、全局形状与 TP 切片。

        byte_start / byte_end 相对本 State 起点，字段间的必要 padding 由区间间隔体现；
        参与 CPU 页时，管理器按字段最大终点计算占用，不乘页内 token 数。
        """
        ...

    @abstractmethod
    def write_page_state_to_cpu(
        self,
        page_kind: PageKind,
        slot_ids: List[int],
        cpu_page_buffers: torch.Tensor,
        cpu_page_indices: List[int],
        *,
        layout: StateTransferLayout,
        cpu_page_readies: list[bool] | None = None,
    ) -> None:
        """将指定页池槽位的快照写入 CPU 页；跳过 cpu_page_readies 中为 True 的页面。"""
        ...

    @abstractmethod
    def read_page_state_from_cpu(
        self,
        cpu_page_buffers: torch.Tensor,
        cpu_page_indices: List[int],
        *,
        big_page_slot_ids: List[int],
        layout: StateTransferLayout,
    ) -> None:
        """读入指定大页槽位；请求运行态另行通过 restore_bounded_att_pages 恢复。"""
        ...

    # ========================================================================
    # 四、DP 迁移：大页快照与基准运行态
    # ========================================================================

    @abstractmethod
    def copy_big_page_state_from_other_dp(
        self,
        src_page_ids: List[int],
        dst_page_ids: List[int],
        remote_dp_bounded_state: BoundedAttentionState,
    ) -> None:
        """将其他 DP 同一 TP 分片的大页快照复制到本地大页，不修改运行态或分配页槽位。"""
        ...

    @abstractmethod
    def copy_run_time_state_from_other_dp(
        self,
        src_req_indices: List[int],
        dst_req_indices: List[int],
        remote_dp_bounded_state: BoundedAttentionState,
    ) -> None:
        """复制其他 DP 同一 TP 分片的 prefill 基准运行态并重置目标游标，不修改页池。

        调用方保证源状态尚未执行 MTP 提交；KDA/GDN 的源游标必须为 0。
        """
        ...

    # ========================================================================
    # 五、PD 分离传输：读写基准运行态
    # ========================================================================

    @abstractmethod
    def get_pd_transfer_regions(self) -> tuple[TransferRegion, ...]:
        """返回 PD 需要的字段、全局形状和 TP 切片，区域偏移相对本 State 起点。"""
        ...

    @abstractmethod
    def write_to_pd_buffer(
        self, req_idx: int, pd_page_buffers: torch.Tensor,
        *, pd_page_index: int, layout: StateTransferLayout,
    ) -> None:
        """将指定槽位的 prefill 基准运行态写入 PD 页。

        调用方保证源状态尚未执行 MTP 提交；KDA/GDN 的源游标必须为 0。
        """
        ...

    @abstractmethod
    def read_from_pd_buffer(
        self, pd_page_buffers: torch.Tensor, req_idx: int,
        *, pd_page_index: int, layout: StateTransferLayout,
    ) -> None:
        """将 PD 页恢复到指定请求槽位并规范化控制信息，不写入大小页池。"""
        ...
