"""GDN/KDA 使用的 Conv/SSM 布局配置。"""

from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class ConvSSMStateConfig:
    """KDA/GDN 的层布局、维度、dtype 和 TP 配置；不包含模型角色或资源身份。"""

    # 本 State 管理的逻辑层编号；当前每个逻辑层独占一份运行态，列表顺序对应物理层轴，编号不能重复。
    logical_layer_ids: tuple[int, ...]
    # TP 切分前的全局 K head 数，不计 TP 副本；Q 使用相同的 head 数和维度。
    global_k_heads: int
    # TP 切分前的全局 V head 数；GDN 可与 K 不同，KDA 要求两者相等。
    global_v_heads: int
    # 每个 Q/K head 的特征维度，也是 SSM 矩阵的 key 轴长度，不随 TP 切分。
    key_dim: int
    # 每个 V head 的特征维度，也是 SSM 矩阵的 value 轴长度；KDA 要求与 key_dim 相等。
    value_dim: int
    # Conv 需要保留的历史 token 数，等于卷积核宽度减 1；不含 MTP 额外候选位置。
    conv_width: int
    # Conv 运行态及大小页快照的存储 dtype，不决定计算算子的中间精度。
    conv_dtype: torch.dtype
    # SSM 运行态及大小页快照的存储 dtype，可与 Conv 不同，不决定算子的累加精度。
    ssm_dtype: torch.dtype
    # 当前进程在本模型 TP 组中的编号，范围为 [0, tp_world_size)，用于确定本地 head 分片。
    tp_rank: int = 0
    # 本模型的 TP 并行度；head 足够时均分，少于 TP 并行度时按组复制；1 表示不切分。
    tp_world_size: int = 1

    @property
    def num_k_heads(self) -> int:
        """当前 TP rank 的 K head 数；head 少于 TP 时按组复制。"""
        return max(1, self.global_k_heads // self.tp_world_size)

    @property
    def num_v_heads(self) -> int:
        """当前 TP rank 的 V head 数，独立于 K head 的切分计算。"""
        return max(1, self.global_v_heads // self.tp_world_size)

    @property
    def conv_channels(self) -> int:
        """当前 TP rank 的 Conv 通道总数，按 Q、K、V 三段拼接。"""
        return 2 * self.num_k_heads * self.key_dim + self.num_v_heads * self.value_dim

    def __post_init__(self) -> None:
        assert 0 <= self.tp_rank < self.tp_world_size
        for heads in (self.global_k_heads, self.global_v_heads):
            assert heads > 0 and (
                heads % self.tp_world_size == 0 or self.tp_world_size % heads == 0
            ), "Conv/SSM heads require complete TP shards or replicas"
