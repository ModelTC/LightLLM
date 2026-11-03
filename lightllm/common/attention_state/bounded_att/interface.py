"""有界 attention 状态的功能接口；具体算法实现运行态与大小页存储。"""

from __future__ import annotations

from abc import ABC, abstractmethod
from lightllm.common.attention_state.utils import weak_cached_method
from typing import Any


from ..utils import deepcopy_result
from lightllm.common.attention_state.types import TransferRegion


class CoreInterface(ABC):
    """有界 attention State 的资源、层视图和运行态核心能力。"""

    # ========================================================================
    # 一、资源定义与初始化
    # ========================================================================

    @abstractmethod
    def get_state_type(self) -> str:
        """返回算法与存储方案标记，用于生成资源成员名。

        返回小写 snake_case 的“算法＋存储方案”，不带 full_att/bounded_att 类别前缀，
        也不带 main/draftN 角色或最终的 _state；类别由 Python State 类型表达。
        管理器统一生成 state_name = f"{model_role}_{get_state_type()}_state"，与成员名完全一致。
        同角色下不同算法、精度或量化方案使用不同标记；同方案可以按不同逻辑层域注册多个实例。
        层号、容量、设备、TP rank 和请求信息不进入名称；完整布局仍由 config 描述。
        注册后返回值保持稳定，实际分配与 CPU/DP/PD 传输均使用相同标记。

        KDA/GDN 格式为 {kda|gdn}_conv_{Conv精度}_ssm_{SSM精度}。
        两份状态精度分别标明，不能只写一个 dtype，例如 kda_conv_bf16_ssm_fp32，
        对应 draft0_kda_conv_bf16_ssm_fp32_state。不同精度方案可以在同一角色下共存。
        其他 bounded 算法使用自己的算法名及方案参数，不复用 kda/gdn 标记。
        """
        ...

    @abstractmethod
    def get_snapshot_storage_specs(self):
        raise NotImplementedError()

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


class CpuCacheInterface(ABC):
    # ========================================================================
    # 三、CPU Cache 数据交换：读写大小页快照
    # ========================================================================

    @abstractmethod
    @deepcopy_result
    @weak_cached_method
    def get_cpu_transfer_regions(self) -> tuple[TransferRegion, ...]:
        """返回一份 CPU 快照的字段区间、全局形状与 TP 切片。

        byte_start / byte_end 相对本 State 起点，字段间的必要 padding 由区间间隔体现；
        参与 CPU 页时，管理器按字段最大终点计算占用，不乘页内 token 数。
        在合法缓存边界无需保留历史的 State 返回空元组，不占用 CPU 页面。
        具体实现使用 lru_cache 缓存，外层 deepcopy_result 每次返回独立副本。
        管理器分配后统一清理，子类覆盖时须保留缓存和副本语义。
        """
        ...


class PdInterface(ABC):
    # ========================================================================
    # 五、PD 分离传输：读写基准运行态
    # ========================================================================

    @abstractmethod
    @deepcopy_result
    @weak_cached_method
    def get_pd_transfer_regions(self) -> tuple[TransferRegion, ...]:
        """返回 PD 需要的字段、全局形状和 TP 切片，区域偏移相对本 State 起点。

        内层 lru_cache 缓存区域定义，外层 deepcopy_result 每次返回独立副本。
        管理器分配存储后统一清理，子类覆盖时须保留缓存和副本语义。
        """
        ...


__all__ = [
    "CoreInterface",
    "CpuCacheInterface",
    "PdInterface",
]
