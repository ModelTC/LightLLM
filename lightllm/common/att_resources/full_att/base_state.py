from __future__ import annotations

from typing import List
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any

import torch

from lightllm.common.att_resources.types import (
    StateTransferLayout,
    TransferRegion,
)


@dataclass(frozen=True)
class FullAttentionConfig:
    # 当前 State 管理的逻辑层号；元组顺序对应 storage 的 layer 轴顺序，不要求层号连续。
    layer_ids: tuple[int, ...]
    # TP 切分前的 KV head 数：GQA 的 K/V 各有这么多个，不把二者相加，也不计 TP 副本。
    # 不是 query head 数；例如全局 8 个 KV head，实际有 8 个 K head 和 8 个 V head。
    # MLA 的压缩 latent 作为一个 KV 字段存储，通常设为 1，并在各 TP rank 复制。
    global_head_num: int
    # GQA 中每个 K/V head 的原始向量维度；INT4 的实际编码宽度由具体 State 换算。
    # 普通 MLA 中为 latent 宽度；packed MLA 中为整行元素数，包含 indexer、scale 和 padding。
    head_dim: int
    # 具体方案使用的数据类型：普通 GQA/MLA 为 KV 精度，分组 INT4/INT8 为 scale 精度。
    # 静态 FP8 和 FP8 packed MLA 使用 uint8 承载编码字节，实际字段类型由 State 解释。
    dtype: torch.dtype
    # 当前 State 所属模型的 TP 组内 rank，不是全局进程 rank 或 DP rank。
    tp_rank: int = 0
    # 当前模型 TP 组大小，用于计算本地 head 数以及传输时的全局 head 范围。
    tp_world_size: int = 1
    # True：每个 rank 都保存全部全局 heads，例如 MLA latent。
    # False：heads 足够时均匀切分；heads 少于 TP rank 数时，每组相邻 ranks 复制一个 head。
    # 例如 global_head_num=2、TP=8 时，rank 0..3 保存 head 0，rank 4..7 保存 head 1。
    replicated: bool = False

    @property
    def head_num(self) -> int:
        """当前 TP rank 实际存储的 KV head 数。"""
        # 仍是单份 K 或 V 的 head 数；GQA 的 K/V 独立分配，各自的 head 轴大小均为 head_num。
        return self.global_head_num if self.replicated else max(1, self.global_head_num // self.tp_world_size)

    def __post_init__(self) -> None:
        assert self.global_head_num > 0 and 0 <= self.tp_rank < self.tp_world_size
        assert (
            self.replicated
            or self.global_head_num % self.tp_world_size == 0
            or self.tp_world_size % self.global_head_num == 0
        ), "KV heads require complete TP shards or replicas"


class FullAttentionState(ABC):
    """单种 full-attention 存储方案的数据接口，不分配 token 索引或维护请求映射。

    所有表示 token 索引的参数都须为调用方准备的 CUDA Tensor，位于对应存储设备上；
    State 不转换设备、dtype 或连续布局，int32/int64 由具体算子支持，DP 目标索引须为 int64。
    CPU 页位于主机内存不改变 token 索引的 CUDA 契约。
    """

    # ========================================================================
    # 一、资源定义与初始化
    # ========================================================================

    # 具体存储和量化方案标识，用于 main/draft State 成员命名。
    @abstractmethod
    def get_state_type(self) -> str:
        """返回具体算法与存储方案标记，用于生成资源成员名；绑定后须保持稳定。"""
        ...

    @abstractmethod
    def __init__(
        self,
        layout_config: FullAttentionConfig,
        page_size: int = 1,
        *,
        device: str | torch.device = "cuda",
    ) -> None:
        """只准备 State 配置，不分配 KV 或 HOLD 页；存储由后续统一容量初始化。"""
        raise NotImplementedError()

    @abstractmethod
    def initialize_storage(self, token_capacity: int) -> None:
        """按 token 容量创建独立存储，沿用已设置的页大小、设备及配套参数。"""
        raise NotImplementedError()

    @abstractmethod
    def get_token_bytes(self) -> int:
        """返回当前 TP rank 的单 token 存储字节数，包含动态量化参数和行 padding。"""
        raise NotImplementedError()

    # ========================================================================
    # 二、Full attention：计算视图、KV 写入与完整 token 复制
    # ========================================================================

    @abstractmethod
    def get_layer_view(self, layer_id: int) -> Any:
        """按逻辑层号返回计算所需的存储视图，具体字段由该布局定义。"""
        raise NotImplementedError()

    @abstractmethod
    def write_full_att_kv(self, layer_id: int, token_indices: torch.Tensor, kv: Any) -> None:
        """向已分配位置写入一层 KV，并完成该布局要求的编码、量化和关联参数更新。"""
        raise NotImplementedError()

    @abstractmethod
    def copy_full_att_tokens(self, src_indices: torch.Tensor, dst_indices: torch.Tensor) -> None:
        """复制全部所辖层的完整 token 数据，包含动态 scale、indexer 字段，保持原存储表示。

        源和目标索引为存储设备上的等长一维 CUDA 整数 Tensor，位置不重叠，直接使用原 dtype 和 stride。"""
        raise NotImplementedError()

    # ========================================================================
    # 三、CPU Cache 数据交换
    # ========================================================================

    @abstractmethod
    def get_cpu_transfer_regions(self) -> tuple[TransferRegion, ...]:
        """按全局 cpu_cache_token_page_size 返回 CPU 字段、全局形状和 TP 切片。

        byte_start / byte_end 相对本 State 起点，字段间的必要 padding 由区间间隔体现；
        管理器按字段最大终点计算该 State 的 CPU 页占用。
        本次有效 token 数由读写流程处理，不改变布局。
        """
        ...

    @abstractmethod
    def write_to_cpu(
        self,
        token_indices: torch.Tensor,
        cpu_page_buffers: torch.Tensor,
        cpu_page_indices: List[int],
        *,
        layout: StateTransferLayout,
        cpu_page_readies: list[bool] | None = None,
    ) -> None:
        """按布局写出 token 数据及配套字段；cpu_page_readies 中为 True 的页跳过写入。"""
        raise NotImplementedError()

    @abstractmethod
    def read_from_cpu(
        self,
        cpu_page_buffers: torch.Tensor,
        cpu_page_indices: List[int],
        token_indices: torch.Tensor,
        *,
        layout: StateTransferLayout,
    ) -> None:
        """从指定 CPU 页读回 token 数据，保留目标中由其他资源或 TP 分片持有的区域。"""
        raise NotImplementedError()

    # ========================================================================
    # 四、DP 迁移
    # ========================================================================

    @abstractmethod
    def copy_tokens_from_other_dp(
        self,
        src_token_indices: torch.Tensor,
        dst_token_indices: torch.Tensor,
        remote_dp_full_att_state: FullAttentionState,
    ) -> None:
        """复制其他 DP 同一 TP 分片的完整 token 数据，包含量化参数及 indexer 字段。

        源、目标索引须已位于各自存储的 CUDA 设备；目标索引为 int64，接口不转换设备或 dtype。
        """
        raise NotImplementedError()

    # ========================================================================
    # 五、PD 分离传输
    # ========================================================================

    @abstractmethod
    def get_pd_transfer_regions(self) -> tuple[TransferRegion, ...]:
        """按全局 pd_kv_page_size 返回 PD 字段、全局形状和 TP 切片，区域偏移相对本 State 起点。"""
        ...

    @abstractmethod
    def write_to_pd_buffer(
        self,
        token_indices: torch.Tensor,
        pd_page_buffers: torch.Tensor,
        *,
        pd_page_index: int,
        layout: StateTransferLayout,
    ) -> None:
        """将 token 数据写入PD 页集合中的指定发送页，网络发送与同步由调用方负责。"""
        raise NotImplementedError()

    @abstractmethod
    def read_from_pd_buffer(
        self,
        pd_page_buffers: torch.Tensor,
        token_indices: torch.Tensor,
        *,
        pd_page_index: int,
        layout: StateTransferLayout,
    ) -> None:
        """从PD 页集合中的指定接收页还原 token 数据，不分配索引或修改请求映射。"""
        raise NotImplementedError()
