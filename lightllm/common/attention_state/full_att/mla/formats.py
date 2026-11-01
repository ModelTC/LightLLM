from __future__ import annotations
from dataclasses import dataclass
from typing import Literal, ClassVar
from math import lcm
import torch
from ...types import FieldName
from ..token_storage import TokenField, TokenStorageSpec, PackedField, PackedTokenStorageSpec
from ...utils import dtype_name


@dataclass(frozen=True)
class MLAConfig:
    """普通 latent MLA 的逻辑配置，不包含 indexer 或 packed 物理字段。"""

    # 外层顺序对应物理存储层，组内逻辑层共享该方案的全部字段；各组非空，逻辑层不能重复。
    # 例如 ((0, 4), (8,)) 分配两层，逻辑层 0/4 共用第一层的全部字段。
    # 分组不决定执行顺序或写入者；模型须让共享消费者跳过写入，避免覆盖生产层数据。
    logical_layer_groups: tuple[tuple[int, ...], ...]
    # 编码前的完整 latent 宽度，包含模型需要的 RoPE 部分，不含 indexer 或 padding。
    latent_dim: int
    # 未压缩 latent 在运行态和存储中的浮点精度。
    latent_dtype: torch.dtype
    # 所属模型 TP 组内的 rank；各 rank 保存相同 latent，仅 rank 0 写出共享外部页。
    tp_rank: int = 0
    # 保存完整 latent 副本的 TP rank 数，不改变 latent 的维度或物理行宽。
    tp_world_size: int = 1

    def __post_init__(self) -> None:
        assert isinstance(self.logical_layer_groups, tuple) and all(
            isinstance(group, tuple) and group for group in self.logical_layer_groups
        ), "layer groups must be a tuple of nonempty tuples"
        layers = [layer for group in self.logical_layer_groups for layer in group]
        assert all(
            isinstance(layer, int) and layer >= 0 for layer in layers
        ), "logical layer IDs must be nonnegative integers"
        assert len(layers) == len(set(layers)), "a logical layer must belong to exactly one storage group"
        assert self.latent_dim > 0 and 0 <= self.tp_rank < self.tp_world_size


def mla_spec(config):
    return TokenStorageSpec(
        encoding="mla_" + dtype_name(config.latent_dtype),
        logical_layer_groups=config.logical_layer_groups,
        global_head_num=1,
        head_num=1,
        tp_rank=config.tp_rank,
        tp_world_size=config.tp_world_size,
        fields=(TokenField(FieldName.KV, config.latent_dtype, config.latent_dim),),
    )
