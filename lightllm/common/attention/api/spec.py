from dataclasses import dataclass
from enum import Enum
from typing import Optional

import torch


class AttentionFamily(str, Enum):
    """计算类型；Triton、FA3 等实现库由 selector 另行选择。"""

    DENSE = "dense"
    MLA = "mla"
    SPARSE_MLA = "sparse_mla"
    GDN = "gdn"
    KDA = "kda"


class AttentionPhase(str, Enum):
    PREFILL = "prefill"
    DECODE = "decode"


@dataclass(frozen=True)
class CacheLayout:
    """存储布局的静态描述，不持有或分配存储 tensor。

    prefill / decode 必须接受同一布局；dtype、量化粒度和 scale 布局也属于兼容性条件。
    storage_schema 记录 State 显式声明的布局版本及配置，排除逻辑层号，使同布局的层可以归为一组。
    """

    format: str
    page_size: int
    dtype: torch.dtype
    key_heads: int
    key_dim: int
    value_dim: int
    state_dtype: Optional[torch.dtype] = None
    storage_schema: tuple = ()
    quant_group_size: int = 0
    scale_mode: str = "none"

    def __post_init__(self):
        if self.page_size <= 0:
            raise ValueError("cache page_size must be positive")


@dataclass(frozen=True)
class LayerSpec:
    """一个层在一个 phase 下的静态计算需求，用于能力检查和分组。

    初始化按完整的 (prefill_spec, decode_spec) 分组，只有相同需求的层才共享 backend / plan。
    层权重和存储视图放在 LayerBinding.params，批次长度与请求索引放在 AttentionBatch。
    """

    family: AttentionFamily
    phase: AttentionPhase
    dtype: torch.dtype
    cache: CacheLayout
    num_q_heads: int
    head_size: int
    value_head_size: int
    sliding_window: Optional[tuple[int, int]] = None
    use_alibi: bool = False
    use_sink: bool = False
    causal: bool = True
    use_mtp: bool = False
    dynamic_verify: bool = False
    variant: str = ""

    def __post_init__(self):
        if min(self.num_q_heads, self.head_size, self.value_head_size) <= 0:
            raise ValueError("attention head counts and dimensions must be positive")
        if self.family in (AttentionFamily.GDN, AttentionFamily.KDA) and self.sliding_window is not None:
            raise ValueError("recurrent attention cannot use a paged sliding window")
