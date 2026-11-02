from dataclasses import dataclass

import torch

from lightllm.common.attention_state.types import ResourceId


@dataclass(frozen=True)
class ResourceBinding:
    resource: ResourceId
    implementation: object
    view: object


@dataclass(frozen=True)
class KVFormat:
    encoding: str
    dtype: torch.dtype
    global_heads: int
    local_heads: int
    head_dim: int
    page_size: int
    tp_rank: int
    tp_size: int
    replicated: bool = False
    layout: str = "separate_kv_nhd"
    group_size: int | None = None
    scale_dtype: torch.dtype | None = None
    packing: str | None = None
    version: int = 1

    def __post_init__(self):
        if min(self.global_heads, self.local_heads, self.head_dim, self.page_size, self.tp_size) <= 0:
            raise ValueError("KV dimensions and page size must be positive")
        if not 0 <= self.tp_rank < self.tp_size:
            raise ValueError("invalid KV shard rank")


@dataclass(frozen=True)
class AttentionSpec:
    model_role: str
    logical_layer: int
    resource: ResourceId
    q_heads: int
    kv_heads: int
    head_dim: int
    dtype: torch.dtype
    family: str = "gqa"
    input_stage: str = "post_rope"
    cache_writer: str = "upstream"
    window: tuple[int, int] = (-1, -1)
    alibi: bool = False
    attention_sink: bool = False
    causal: bool = True

    def __post_init__(self):
        if min(self.q_heads, self.kv_heads, self.head_dim) <= 0 or self.q_heads % self.kv_heads:
            raise ValueError("attention requires positive dimensions and complete Q/KV head groups")
        if self.model_role != self.resource.model_role:
            raise ValueError("attention cannot bind another model role's state")
        if self.cache_writer not in ("upstream", "shared"):
            raise ValueError("unknown cache writer")

    @property
    def shape_key(self):
        return (
            self.family,
            self.q_heads,
            self.kv_heads,
            self.head_dim,
            self.dtype,
            self.window,
            self.alibi,
            self.attention_sink,
            self.causal,
        )


@dataclass(frozen=True)
class MlaFormat:
    dtype: torch.dtype
    latent_dim: int
    page_size: int
    encoding: str = "mla_plain"
    version: int = 1


@dataclass(frozen=True)
class MlaSpec:
    model_role: str
    logical_layer: int
    resource: ResourceId
    q_heads: int
    nope_dim: int
    rope_dim: int
    latent_dim: int
    value_dim: int
    dtype: torch.dtype
    softmax_scale: float
    family: str = "mla"

    def __post_init__(self):
        if min(self.q_heads, self.nope_dim, self.rope_dim, self.latent_dim, self.value_dim) <= 0:
            raise ValueError("MLA requires positive projection and latent dimensions")
        if self.model_role != self.resource.model_role or self.softmax_scale <= 0:
            raise ValueError("invalid MLA resource or softmax scale")

    @property
    def shape_key(self):
        return (
            self.q_heads,
            self.nope_dim,
            self.rope_dim,
            self.latent_dim,
            self.value_dim,
            self.dtype,
            self.softmax_scale,
        )
