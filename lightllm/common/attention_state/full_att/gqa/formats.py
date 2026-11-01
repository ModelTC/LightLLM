from __future__ import annotations

from dataclasses import dataclass
import torch
from ...types import FieldName
from ...utils import dtype_name
from ..token_storage import TokenField, TokenStorageSpec


@dataclass(frozen=True)
class GQAAttentionConfig:
    """普通 GQA（含 MHA/MQA）的逻辑 KV 布局；不描述量化编码或物理行 padding。"""

    # 外层顺序对应物理存储层，组内逻辑层共享该方案的全部字段；各组非空，逻辑层不能重复。
    # 例如 ((0, 4), (8,)) 分配两层，逻辑层 0/4 共用第一层的全部字段。
    # 分组不决定执行顺序或写入者；模型须让共享消费者跳过写入，避免覆盖生产层数据。
    logical_layer_groups: tuple[tuple[int, ...], ...]
    # TP 切分前单份 K 或 V 的 head 数，不将 K/V 相加，也不是 query head 数。
    global_head_num: int
    # 每个 K/V head 编码前的特征维度，量化位宽不改变此逻辑维度。
    head_dim: int
    # K/V 浮点存储精度。
    dtype: torch.dtype
    # 所属模型 TP 组内的 rank，不是全局 rank 或 DP rank。
    tp_rank: int = 0
    # 所属模型 TP 组大小，用于确定分片或副本的访问关系。
    tp_world_size: int = 1
    # True 时各 rank 保存全部 heads；False 时切分，heads 少于 ranks 时相邻 ranks 复制同一 head。
    replicated: bool = False

    @property
    def head_num(self) -> int:
        """当前 TP rank 保存的单份 K 或 V 的 head 数。"""
        return self.global_head_num if self.replicated else max(1, self.global_head_num // self.tp_world_size)

    def __post_init__(self) -> None:
        assert isinstance(self.logical_layer_groups, tuple) and all(
            isinstance(group, tuple) and group for group in self.logical_layer_groups
        ), "layer groups must be a tuple of nonempty tuples"
        layers = [layer for group in self.logical_layer_groups for layer in group]
        assert all(
            isinstance(layer, int) and layer >= 0 for layer in layers
        ), "logical layer IDs must be nonnegative integers"
        assert len(layers) == len(set(layers)), "a logical layer must belong to exactly one storage group"
        assert self.global_head_num > 0 and self.head_dim > 0 and 0 <= self.tp_rank < self.tp_world_size
        assert (
            self.replicated
            or self.global_head_num % self.tp_world_size == 0
            or self.tp_world_size % self.global_head_num == 0
        ), "KV heads require complete TP shards or replicas"


@dataclass(frozen=True)
class GroupQuantizedAttentionConfig:
    """INT4/INT8 GQA 的独立布局；KV 编码为整数，动态 scale 使用单独精度。"""

    # 外层顺序对应物理存储层，组内逻辑层共享该方案的全部字段；各组非空，逻辑层不能重复。
    # 例如 ((0, 4), (8,)) 分配两层，逻辑层 0/4 共用第一层的全部字段。
    # 分组不决定执行顺序或写入者；模型须让共享消费者跳过写入，避免覆盖生产层数据。
    logical_layer_groups: tuple[tuple[int, ...], ...]
    # TP 切分前单份 K 或 V 的 head 数，不将 K/V 相加，也不是 query head 数。
    global_head_num: int
    # 每个 K/V head 编码前的特征维度，量化位宽不改变此逻辑维度。
    head_dim: int
    # 每个 token、每组 K/V 量化 scale 的浮点精度，不是 KV 编码类型。
    scale_dtype: torch.dtype
    # 所属模型 TP 组内的 rank，不是全局 rank 或 DP rank。
    tp_rank: int = 0
    # 所属模型 TP 组大小，用于确定分片或副本的访问关系。
    tp_world_size: int = 1
    # True 时各 rank 保存全部 heads；False 时切分，heads 少于 ranks 时相邻 ranks 复制同一 head。
    replicated: bool = False
    # KV 量化位数；INT4 将两个编码值打包进一个字节。
    bits: int = 8
    # 沿 head_dim 分组，每组共用一个动态 scale；必须为二次幂且整除 head_dim。
    group_size: int = 8

    @property
    def head_num(self) -> int:
        """当前 TP rank 保存的单份 K 或 V 的 head 数。"""
        return self.global_head_num if self.replicated else max(1, self.global_head_num // self.tp_world_size)

    def __post_init__(self) -> None:
        assert isinstance(self.logical_layer_groups, tuple) and all(
            isinstance(group, tuple) and group for group in self.logical_layer_groups
        ), "layer groups must be a tuple of nonempty tuples"
        layers = [layer for group in self.logical_layer_groups for layer in group]
        assert all(
            isinstance(layer, int) and layer >= 0 for layer in layers
        ), "logical layer IDs must be nonnegative integers"
        assert len(layers) == len(set(layers)), "a logical layer must belong to exactly one storage group"
        assert self.global_head_num > 0 and self.head_dim > 0 and 0 <= self.tp_rank < self.tp_world_size
        assert (
            self.replicated
            or self.global_head_num % self.tp_world_size == 0
            or self.tp_world_size % self.global_head_num == 0
        ), "KV heads require complete TP shards or replicas"
        assert (
            self.bits in (4, 8) and self.group_size >= 2 and not (self.group_size & (self.group_size - 1))
        ), "INT4/INT8 quantization requires a power-of-two group size"
        assert self.head_dim % self.group_size == 0 and self.scale_dtype in (
            torch.float16,
            torch.bfloat16,
            torch.float32,
        ), "group scales require a floating dtype and complete groups"


@dataclass(frozen=True)
class StaticFP8AttentionConfig:
    """静态 FP8 GQA 的逻辑布局；KV 固定为 FP8 编码，标定 Tensor 由后续流程绑定。"""

    # 外层顺序对应物理存储层，组内逻辑层共享该方案的全部字段；各组非空，逻辑层不能重复。
    # 例如 ((0, 4), (8,)) 分配两层，逻辑层 0/4 共用第一层的全部字段。
    # 分组不决定执行顺序或写入者；模型须让共享消费者跳过写入，避免覆盖生产层数据。
    logical_layer_groups: tuple[tuple[int, ...], ...]
    # TP 切分前单份 K 或 V 的 head 数，不将 K/V 相加，也不是 query head 数。
    global_head_num: int
    # 每个 K/V head 编码前的特征维度，量化位宽不改变此逻辑维度。
    head_dim: int
    # 所属模型 TP 组内的 rank，不是全局 rank 或 DP rank。
    tp_rank: int = 0
    # 所属模型 TP 组大小，用于确定分片或副本的访问关系。
    tp_world_size: int = 1
    # True 时各 rank 保存全部 heads；False 时切分，heads 少于 ranks 时相邻 ranks 复制同一 head。
    replicated: bool = False
    # 每层 K/V 各一个 scale，或每个本地 K/V head 各一个；不占用逐 token 页空间。
    scale_mode: str = "per_tensor"

    @property
    def head_num(self) -> int:
        """当前 TP rank 保存的单份 K 或 V 的 head 数。"""
        return self.global_head_num if self.replicated else max(1, self.global_head_num // self.tp_world_size)

    def __post_init__(self) -> None:
        assert isinstance(self.logical_layer_groups, tuple) and all(
            isinstance(group, tuple) and group for group in self.logical_layer_groups
        ), "layer groups must be a tuple of nonempty tuples"
        layers = [layer for group in self.logical_layer_groups for layer in group]
        assert all(
            isinstance(layer, int) and layer >= 0 for layer in layers
        ), "logical layer IDs must be nonnegative integers"
        assert len(layers) == len(set(layers)), "a logical layer must belong to exactly one storage group"
        assert self.global_head_num > 0 and self.head_dim > 0 and 0 <= self.tp_rank < self.tp_world_size
        assert (
            self.replicated
            or self.global_head_num % self.tp_world_size == 0
            or self.tp_world_size % self.global_head_num == 0
        ), "KV heads require complete TP shards or replicas"
        assert self.scale_mode in ("per_tensor", "per_head"), "invalid static FP8 scale mode"


def storage_spec(config, encoding: str, fields: tuple[TokenField, ...]) -> TokenStorageSpec:
    return TokenStorageSpec(
        encoding=encoding,
        logical_layer_groups=config.logical_layer_groups,
        global_head_num=config.global_head_num,
        head_num=config.head_num,
        tp_rank=config.tp_rank,
        tp_world_size=config.tp_world_size,
        fields=fields,
    )


def plain_spec(config: GQAAttentionConfig) -> TokenStorageSpec:
    return storage_spec(
        config,
        "gqa_" + dtype_name(config.dtype),
        (
            TokenField(FieldName.K, config.dtype, config.head_dim),
            TokenField(FieldName.V, config.dtype, config.head_dim),
        ),
    )


def group_quantized_spec(config: GroupQuantizedAttentionConfig) -> TokenStorageSpec:
    encoding = f"gqa_int{config.bits}_group{config.group_size}_{dtype_name(config.scale_dtype)}_scale"
    return storage_spec(
        config,
        encoding,
        (
            TokenField(FieldName.K, torch.int8, config.head_dim * config.bits // 8),
            TokenField(FieldName.V, torch.int8, config.head_dim * config.bits // 8),
            TokenField(FieldName.K_SCALE, config.scale_dtype, config.head_dim // config.group_size, "group"),
            TokenField(FieldName.V_SCALE, config.scale_dtype, config.head_dim // config.group_size, "group"),
        ),
    )


def static_fp8_spec(config: StaticFP8AttentionConfig) -> TokenStorageSpec:
    return storage_spec(
        config,
        "gqa_fp8_static_" + config.scale_mode,
        (
            TokenField(FieldName.K, torch.uint8, config.head_dim),
            TokenField(FieldName.V, torch.uint8, config.head_dim),
        ),
    )
