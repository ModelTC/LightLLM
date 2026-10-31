from __future__ import annotations

from typing import List
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any

import torch

from lightllm.common.attention_state.types import (
    StateTransferLayout,
    TransferRegion,
)


@dataclass(frozen=True)
class FullAttentionConfig:
    # 物理存储层与逻辑推理层的对应关系，外层顺序就是所有 Tensor 的 layer 轴顺序。
    # 每个非空内层元组列出共享同一份存储的逻辑层号；层号可以不连续，但不能重复出现。
    # 例如 ((0, 4), (8,)) 只分配两层：逻辑层 0/4 使用 storage[0]，逻辑层 8 使用 storage[1]。
    # 不共享时写成 ((0,), (4,), (8,))，此时分配三层；容量和传输按 len(logical_layer_groups) 计算。
    # 组内共享该方案的全部字段（KV、scale、indexer 等），因此布局及配套参数必须兼容。
    # 分组不决定执行顺序或写入者：模型负责让共享消费者跳过 KV 写入，避免覆盖生产层数据。
    logical_layer_groups: tuple[tuple[int, ...], ...]
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
        # 只保留这一份分组配置；层号映射在 State 构造时展开，不另存逻辑/物理层数。
        assert isinstance(self.logical_layer_groups, tuple) and all(
            isinstance(logical_layer_group, tuple) and logical_layer_group
            for logical_layer_group in self.logical_layer_groups
        ), "layer groups must be a tuple of nonempty tuples"
        logical_layer_ids = [
            logical_layer_id
            for logical_layer_group in self.logical_layer_groups
            for logical_layer_id in logical_layer_group
        ]
        assert all(isinstance(logical_layer_id, int) and logical_layer_id >= 0 for logical_layer_id in logical_layer_ids), (
            "logical layer IDs must be nonnegative integers"
        )
        assert len(logical_layer_ids) == len(set(logical_layer_ids)), "a logical layer must belong to exactly one storage group"
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
        - MLA packed FP8：mla_packed_fp8_group{分组大小}，当前内核固定为 group128。
        例如 gqa_int8_group128_bf16_scale 对应 main_gqa_int8_group128_bf16_scale_state。
        新量化实现必须区分影响方案选择的位宽、分组、scale 方式及精度，不编码标定数值。
        """
        ...

    @abstractmethod
    def __init__(
        self,
        layout_config: FullAttentionConfig,
        page_size: int = 1,
        device: str | torch.device = "cuda",
    ) -> None:
        """只准备 State 配置，不分配 KV 或 HOLD 页；存储由后续统一容量初始化。"""
        raise NotImplementedError()

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
