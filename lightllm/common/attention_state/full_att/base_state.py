from __future__ import annotations

from functools import lru_cache
from typing import List
from abc import ABC, abstractmethod
from typing import Any

import torch

from ..utils import deepcopy_result
from lightllm.common.attention_state.types import (
    StateTransferLayout,
    TransferRegion,
)


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
        """返回算法与存储方案标记，用于生成资源成员名。

        返回小写 snake_case 的“算法＋存储方案”，不带 full_att/bounded_att/att_tail 类别前缀，
        也不带 main/draftN 角色或最终的 _state；类别由 Python State 类型表达。
        管理器统一生成 state_name = f"{model_role}_{get_state_type()}_state"，与成员名完全一致。
        同角色下不同算法、精度或量化方案必须使用不同标记，同方案的全部逻辑层放在同一实例。
        层号、容量、设备、TP rank 和请求信息不进入名称；完整布局仍由 config 描述。
        注册后返回值保持稳定，实际分配与 CPU/DP/PD 传输均使用相同标记。

        - GQA 浮点 KV：gqa_{bf16|fp16|fp32}。
        - GQA 分组量化：gqa_int{4|8}_group{分组大小}_{scale精度}_scale。
        - GQA 静态 FP8：gqa_fp8_static_{per_tensor|per_head}。
        - MLA 浮点：mla_{精度}；KV/indexer 联合存储：mla_packed_{精度}。
        - MLA packed FP8：mla_packed_fp8_group128，或 mla_packed_fp8_group64_e8m0。
        - 序列压缩历史：compressed_c{压缩比例}_fp8_group64_e8m0。
        例如 gqa_int8_group128_bf16_scale 对应 main_gqa_int8_group128_bf16_scale_state。
        新量化实现必须区分影响方案选择的位宽、分组、scale 方式及精度，不编码标定数值。
        """
        ...

    @abstractmethod
    def allocate_storage(self, token_capacity: int) -> None:
        """按 token 容量和 logical_layer_groups 的物理层数创建存储，沿用已设置的页大小、设备及配套参数。"""
        raise NotImplementedError()

    @abstractmethod
    def get_token_bytes(self) -> int:
        """累计当前 TP rank 各物理层的单 token 字节数，包含动态量化参数和行 padding；共享逻辑层不重复计数。"""
        raise NotImplementedError()

    # ========================================================================
    # 二、Full attention：计算视图、KV 写入与完整 token 复制
    # ========================================================================

    @abstractmethod
    def contains_logical_layer(self, logical_layer_id: int) -> bool:
        """判断指定逻辑层是否属于当前 State 的层映射，不依赖存储分配。"""
        ...

    @abstractmethod
    def get_storage_layer_index(self, logical_layer_id: int) -> int:
        """按逻辑层号查询存储及配套字段的物理行；共享逻辑层返回同一下标。"""
        ...

    @abstractmethod
    def get_layer_view(self, logical_layer_id: int) -> Any:
        """按逻辑层号返回存储视图；同组逻辑层返回同一物理层的数据，具体字段由该布局定义。"""
        raise NotImplementedError()

    @abstractmethod
    def write_full_att_kv(self, logical_layer_id: int, token_indices: torch.Tensor, kv: Any) -> None:
        """向已分配位置写入一层 KV，并完成该布局要求的编码、量化和关联参数更新。

        融合压缩等专用写入路径可直接使用计算视图，不调用本接口。
        """
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
    @deepcopy_result
    @lru_cache(maxsize=None)
    def get_cpu_transfer_regions(self) -> tuple[TransferRegion, ...]:
        """按全局 cpu_cache_token_page_size 返回 CPU 字段、全局形状和 TP 切片。

        byte_start / byte_end 相对本 State 起点，字段间的必要 padding 由区间间隔体现；
        管理器按字段最大终点计算该 State 的 CPU 页占用。
        本次有效 token 数由读写流程处理，不改变布局。
        具体实现使用 lru_cache 缓存，外层 deepcopy_result 每次返回独立副本。
        首次查询后配置与全局 CPU 页规格保持不变。
        管理器实际分配存储后统一清除缓存，以兼容物理页规格变化；子类覆盖时须保留缓存能力。
        """
        ...

    @abstractmethod
    def write_to_cpu(
        self,
        token_indices: torch.Tensor,
        cpu_page_buffers: torch.Tensor,
        cpu_page_indices: List[int],
        layout: StateTransferLayout,
        cpu_page_ready_flags: list[bool] | None = None,
    ) -> None:
        """按布局写出 token 数据及配套字段；cpu_page_ready_flags 中为 True 的页跳过写入。"""
        raise NotImplementedError()

    @abstractmethod
    def read_from_cpu(
        self,
        cpu_page_buffers: torch.Tensor,
        cpu_page_indices: List[int],
        token_indices: torch.Tensor,
        layout: StateTransferLayout,
    ) -> None:
        """从指定 CPU 页读回 token 数据，保留目标中由其他资源或 TP 分片持有的区域。"""
        raise NotImplementedError()

    # ========================================================================
    # 四、DP 迁移
    # ========================================================================

    @abstractmethod
    def copy_tokens_from_dp(
        self,
        src_token_indices: torch.Tensor,
        dst_token_indices: torch.Tensor,
        remote_dp_full_att_state: FullAttentionState,
    ) -> None:
        """复制其他 DP 同一 TP 分片的完整 token 数据，包含量化参数及 indexer 字段。

        源、目标索引须已位于各自存储的 CUDA 设备，可为 int32/int64；接口不转换设备或 dtype。
        调用方保证 peer/IPC 可访问、源数据就绪、目标索引唯一，且源目标数据区域不重叠。
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
    def write_to_pd_page(
        self,
        token_indices: torch.Tensor,
        pd_page_buffers: torch.Tensor,
        pd_page_index: int,
        layout: StateTransferLayout,
    ) -> None:
        """将 token 数据写入PD 页集合中的指定发送页，网络发送与同步由调用方负责。"""
        raise NotImplementedError()

    @abstractmethod
    def read_from_pd_page(
        self,
        pd_page_buffers: torch.Tensor,
        token_indices: torch.Tensor,
        pd_page_index: int,
        layout: StateTransferLayout,
    ) -> None:
        """从PD 页集合中的指定接收页还原 token 数据，不分配索引或修改请求映射。"""
        raise NotImplementedError()
