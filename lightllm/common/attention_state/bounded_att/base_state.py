"""有界 attention 状态接口；具体算法实现运行态与大小页存储。"""

from __future__ import annotations

from functools import lru_cache


from typing import Any
from abc import ABC, abstractmethod
from typing import List

import torch

from ..utils import deepcopy_result
from lightllm.common.attention_state.types import (
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
        """返回算法与存储方案标记，用于生成资源成员名。

        返回小写 snake_case 的“算法＋存储方案”，不带 full_att/bounded_att/att_tail 类别前缀，
        也不带 main/draftN 角色或最终的 _state；类别由 Python State 类型表达。
        管理器统一生成 state_name = f"{model_role}_{get_state_type()}_state"，与成员名完全一致。
        同角色下不同算法、精度或量化方案必须使用不同标记，同方案的全部逻辑层放在同一实例。
        层号、容量、设备、TP rank 和请求信息不进入名称；完整布局仍由 config 描述。
        注册后返回值保持稳定，实际分配与 CPU/DP/PD 传输均使用相同标记。

        KDA/GDN 格式为 {kda|gdn}_conv_{Conv精度}_ssm_{SSM精度}。
        两份状态精度分别标明，不能只写一个 dtype，例如 kda_conv_bf16_ssm_fp32，
        对应 draft0_kda_conv_bf16_ssm_fp32_state。不同精度方案可以在同一角色下共存。
        其他 bounded 算法使用自己的算法名及方案参数，不复用 kda/gdn 标记。
        """
        ...

    @abstractmethod
    def initialize_page_pools(
        self,
        big_page_capacity: int,
        small_page_capacity: int = 0,
        device: str | torch.device | int = "cpu",
        pin_memory: bool = True,
    ) -> None:
        """容量规划完成后创建大小页池；容量为零表示不启用该页池。"""
        ...

    # ========================================================================
    # 二、Bounded attention：计算视图、运行态重置、大小页与 MTP 提交
    # ========================================================================

    @abstractmethod
    def contains_logical_layer(self, logical_layer_id: int) -> bool:
        """判断指定逻辑层是否属于当前 State 的层映射，不依赖存储分配。"""
        ...

    @abstractmethod
    def get_layer_view(self, logical_layer_id: int) -> Any:
        """按逻辑层号返回运行态和算法控制信息的计算视图。"""
        ...

    @abstractmethod
    def reset_bounded_att_state(self, req_indices: List[int]) -> None:
        """将指定槽位的运行态及控制信息恢复到长度 0，保留大小页快照。"""
        ...

    @abstractmethod
    def save_bounded_att_runtime_to_pages(
        self,
        req_indices: List[int],
        state_token_lens: List[int],
        page_kind: PageKind,
        page_slot_ids: List[int],
    ) -> None:
        """按各请求已执行的绝对 token 长度保存基准运行态，三个列表逐项对应。

        长度不包含尚未执行的采样 token；ring 状态据此确定窗口和槽位位置。
        前缀与页槽位的关联仍由调用方维护，不要求页池保存长度元数据。
        """
        ...

    @abstractmethod
    def restore_bounded_att_runtime_from_pages(
        self,
        req_indices: List[int],
        state_token_lens: List[int],
        page_kind: PageKind,
        page_slot_ids: List[int],
    ) -> None:
        """将页快照恢复到对应请求的绝对 token 长度，保留源快照数据。

        req_indices、state_token_lens、page_slot_ids 逐项对应；长度用于恢复 ring 位置等控制信息，
        不表示本批搬运的 token 数，调用方保证与所选快照的前缀一致。
        """
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
    @deepcopy_result
    @lru_cache(maxsize=None)
    def get_cpu_transfer_regions(self) -> tuple[TransferRegion, ...]:
        """返回一份 CPU 快照的字段区间、全局形状与 TP 切片。

        byte_start / byte_end 相对本 State 起点，字段间的必要 padding 由区间间隔体现；
        参与 CPU 页时，管理器按字段最大终点计算占用，不乘页内 token 数。
        在合法缓存边界无需保留历史的 State 返回空元组，不占用 CPU 页面。
        具体实现使用 lru_cache 缓存，外层 deepcopy_result 每次返回独立副本。
        管理器分配后统一清理，子类覆盖时须保留缓存和副本语义。
        """
        ...

    @abstractmethod
    def write_state_pages_to_cpu(
        self,
        page_kind: PageKind,
        page_slot_ids: List[int],
        cpu_page_buffers: torch.Tensor,
        cpu_page_indices: List[int],
        layout: StateTransferLayout,
        cpu_page_ready_flags: list[bool] | None = None,
    ) -> None:
        """将指定页池槽位的快照写入 CPU 页；跳过 cpu_page_ready_flags 中为 True 的页面。"""
        ...

    @abstractmethod
    def read_state_pages_from_cpu(
        self,
        cpu_page_buffers: torch.Tensor,
        cpu_page_indices: List[int],
        big_page_slot_ids: List[int],
        layout: StateTransferLayout,
    ) -> None:
        """读入指定大页槽位；请求运行态另行通过 restore_bounded_att_runtime_from_pages 恢复。"""
        ...

    # ========================================================================
    # 四、DP 迁移：大页快照与基准运行态
    # ========================================================================

    @abstractmethod
    def copy_big_pages_from_dp(
        self,
        src_page_ids: List[int],
        dst_page_ids: List[int],
        remote_dp_bounded_state: BoundedAttentionState,
    ) -> None:
        """将其他 DP 同一 TP 分片的大页快照复制到本地大页，不修改运行态或分配页槽位。"""
        ...

    @abstractmethod
    def copy_runtime_state_from_dp(
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
    def write_to_pd_page(
        self,
        req_idx: int,
        pd_page_buffers: torch.Tensor,
        pd_page_index: int,
        state_token_len: int,
        layout: StateTransferLayout,
    ) -> None:
        """将指定槽位在 state_token_len 绝对长度处的 prefill 基准运行态写入 PD 页。

        调用方保证源状态尚未执行 MTP 提交；KDA/GDN 的源游标必须为 0。
        """
        ...

    @abstractmethod
    def read_from_pd_page(
        self,
        pd_page_buffers: torch.Tensor,
        req_idx: int,
        pd_page_index: int,
        state_token_len: int,
        layout: StateTransferLayout,
    ) -> None:
        """按 state_token_len 恢复请求运行态及 ring 位置等控制信息，不写入大小页池。"""
        ...
