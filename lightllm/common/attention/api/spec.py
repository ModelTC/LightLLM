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

    @property
    def is_recurrent(self):
        return self in (AttentionFamily.GDN, AttentionFamily.KDA)


class AttentionFeature(str, Enum):
    ALIBI = "alibi"
    SINK = "sink"
    SLIDING_WINDOW = "sliding_window"
    NONCAUSAL = "noncausal"
    MTP = "mtp"
    DYNAMIC_VERIFY = "dynamic_verify"
    MTP_WINDOW = "mtp_window"
    ALIBI_WINDOW = "alibi_window"


class CacheEncoding(str, Enum):
    PLAIN = "plain"
    INT4 = "int4kv"
    INT8 = "int8kv"
    FP8_HEAD = "fp8kv_sph"
    FP8_TENSOR = "fp8kv_spt"
    FP8_SPARSE_MLA = "fp8kv_dsa"

    @classmethod
    def from_option(cls, value):
        return cls.PLAIN if value in (None, "None", "gdn", "kda") else cls(value)


class AttentionPhase(str, Enum):
    PREFILL = "prefill"
    DECODE = "decode"


@dataclass(frozen=True)
class CacheLayout:
    """存储布局的静态描述，不持有或分配存储 tensor。

    prefill / decode 必须接受同一布局；dtype、量化粒度和 scale 布局也属于兼容性条件。
    storage_schema 记录 State 显式声明的布局版本及配置，排除逻辑层号，使同布局的层可以归为一组。
    """

    encoding: CacheEncoding
    page_size: int
    dtype: torch.dtype
    key_heads: int
    key_dim: int
    value_dim: int
    state_mode: str = "native"
    state_dtype: Optional[torch.dtype] = None
    storage_schema: tuple = ()
    quant_group_size: int = 0
    scale_mode: str = "none"

    def __post_init__(self):
        object.__setattr__(self, "encoding", CacheEncoding.from_option(self.encoding))
        if self.page_size <= 0:
            raise ValueError("cache page_size must be positive")


@dataclass(frozen=True)
class LayerSpec:
    """一个层在一个 phase 下的静态计算需求，用于能力检查和分组。

    初始化按完整的 (prefill_spec, decode_spec) 分组，只有相同需求的层才共享 backend / context。
    层权重和存储视图放在 LayerAttention.resources，批次长度与请求索引放在 AttentionBatch。
    """

    family: AttentionFamily
    phase: AttentionPhase
    dtype: torch.dtype
    cache: CacheLayout
    num_q_heads: int
    head_size: int
    value_head_size: int
    sliding_window: Optional[tuple[int, int]] = None
    features: frozenset[AttentionFeature] = frozenset()
    variant: str = ""
    softmax_scale: float | None = None

    def __post_init__(self):
        features = set(self.features)
        if self.sliding_window is not None:
            features.add(AttentionFeature.SLIDING_WINDOW)
            if AttentionFeature.ALIBI in features:
                features.add(AttentionFeature.ALIBI_WINDOW)
            if AttentionFeature.MTP in features:
                features.add(AttentionFeature.MTP_WINDOW)
        object.__setattr__(self, "features", frozenset(features))
        if min(self.num_q_heads, self.head_size, self.value_head_size) <= 0:
            raise ValueError("attention head counts and dimensions must be positive")
        if self.family.is_recurrent and self.sliding_window is not None:
            raise ValueError("recurrent attention cannot use a paged sliding window")

    @property
    def use_mtp(self):
        return AttentionFeature.MTP in self.features

    @property
    def dynamic_verify(self):
        return AttentionFeature.DYNAMIC_VERIFY in self.features

    @property
    def causal(self):
        return AttentionFeature.NONCAUSAL not in self.features
