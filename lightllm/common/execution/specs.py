"""Startup contracts. Tensor storage is owned by the caller, never by a spec."""

from dataclasses import dataclass
from math import isfinite
from typing import Any, Literal

Phase = Literal["prefill", "decode", "verify"]


@dataclass(frozen=True)
class DenseAttentionSpec:
    q_heads: int
    kv_heads: int
    head_dim: int
    causal: bool = True
    window: tuple[int, int] = (-1, -1)
    score_scale: float | None = None
    softcap: float = 0.0
    dtype: Any = None
    attention_sink: bool = False

    def __post_init__(self):
        if min(self.q_heads, self.kv_heads, self.head_dim) <= 0 or self.q_heads % self.kv_heads:
            raise ValueError("attention needs positive dimensions and whole query groups per KV head")
        if len(self.window) != 2 or any(x < -1 for x in self.window):
            raise ValueError("window must contain two bounds >= -1")
        if self.score_scale is not None and (not isfinite(self.score_scale) or self.score_scale <= 0):
            raise ValueError("score_scale must be finite and positive")
        if not isfinite(self.softcap) or self.softcap < 0:
            raise ValueError("softcap must be finite and nonnegative")

    @property
    def scale(self):
        return self.head_dim**-0.5 if self.score_scale is None else self.score_scale


@dataclass(frozen=True)
class MLAAttentionSpec:
    q_heads: int
    latent_dim: int
    nope_dim: int
    rope_dim: int
    value_dim: int
    dtype: Any
    score_scale: float | None = None

    def __post_init__(self):
        if min(self.q_heads, self.latent_dim, self.nope_dim, self.rope_dim, self.value_dim) <= 0:
            raise ValueError("MLA requires positive projection dimensions")
        if self.score_scale is not None and (not isfinite(self.score_scale) or self.score_scale <= 0):
            raise ValueError("MLA score scale must be finite and positive")

    @property
    def scale(self):
        return (self.nope_dim + self.rope_dim) ** -0.5 if self.score_scale is None else self.score_scale


@dataclass(frozen=True)
class SparseMLAAttentionSpec(MLAAttentionSpec):
    index_heads: int = 64
    index_dim: int = 128
    index_topk: int = 2048
    index_quant_block: int = 128
    index_scale_fmt: str | None = None

    def __post_init__(self):
        super().__post_init__()
        if min(self.index_heads, self.index_dim, self.index_topk, self.index_quant_block) <= 0:
            raise ValueError("sparse MLA requires positive indexer dimensions and top-k")


@dataclass(frozen=True)
class GatedDeltaNetSpec:
    k_heads: int
    v_heads: int
    key_dim: int
    value_dim: int
    conv_size: int
    dtype: Any
    activation: str = "silu"

    def __post_init__(self):
        if min(self.k_heads, self.v_heads, self.key_dim, self.value_dim) <= 0 or self.v_heads % self.k_heads:
            raise ValueError("GDN requires positive dimensions and whole value groups per key head")
        if self.conv_size < 2:
            raise ValueError("GDN convolution requires at least two positions")

    @property
    def conv_channels(self):
        return 2 * self.k_heads * self.key_dim + self.v_heads * self.value_dim

    @property
    def input_width(self):
        return self.conv_channels + self.v_heads * self.value_dim + 2 * self.v_heads


@dataclass(frozen=True)
class KimiDeltaAttentionSpec:
    heads: int
    head_dim: int
    conv_size: int
    dtype: Any
    gate_lower_bound: float = -5.0

    def __post_init__(self):
        if min(self.heads, self.head_dim) <= 0 or self.conv_size < 2:
            raise ValueError("KDA requires positive heads, dimensions and a convolution history")
        if not isfinite(self.gate_lower_bound) or self.gate_lower_bound >= 0:
            raise ValueError("KDA gate lower bound must be finite and negative")


@dataclass(frozen=True)
class PooledNoPEAttentionSpec:
    q_heads: int
    latent_dim: int
    score_scale: float
    index_heads: int
    index_dim: int
    index_topk: int
    pool_size: int
    dtype: Any

    def __post_init__(self):
        if min(self.q_heads, self.latent_dim, self.index_heads, self.index_dim, self.index_topk, self.pool_size) <= 0:
            raise ValueError("pooled NoPE requires positive attention and indexer dimensions")
        if self.index_topk % self.pool_size:
            raise ValueError("pooled top-k must select whole pools")
        if not isfinite(self.score_scale) or self.score_scale <= 0:
            raise ValueError("NoPE score scale must be finite and positive")


@dataclass(frozen=True)
class StateSpec:
    role: str
    resource: str
    producer: int
    consumers: tuple[int, ...]
    shape: tuple[int, ...]
    retention: str = "history"

    def __post_init__(self):
        if not self.role or not self.resource or not self.consumers:
            raise ValueError("state identity and consumers are required")
        if self.producer not in self.consumers or len(set(self.consumers)) != len(self.consumers):
            raise ValueError("state consumers must be unique and include the producer")
        if min(self.consumers) < 0 or not self.shape or min(self.shape) <= 0:
            raise ValueError("state needs nonnegative layer IDs and positive logical dimensions")

    @property
    def key(self):
        return self.role, self.resource


@dataclass(frozen=True)
class FormatSpec:
    encoding: str
    dtype: Any
    layout: str
    fields: tuple[str, ...]
    page_size: int
    shard: tuple[int, int] = (0, 1)
    version: int = 1
    calibration: str | None = None
    group_size: int | None = None
    scale_dtype: Any = None
    field_dtypes: tuple[Any, ...] = ()
    candidate_slots: int = 1

    def __post_init__(self):
        rank, size = self.shard
        if not self.encoding or not self.layout or not self.fields or len(set(self.fields)) != len(self.fields):
            raise ValueError("format needs an encoding, layout and unique fields")
        if self.page_size <= 0 or self.version <= 0 or size <= 0 or not 0 <= rank < size:
            raise ValueError("invalid page size, format version or shard")
        if self.group_size is not None and (self.group_size < 2 or self.group_size & (self.group_size - 1)):
            raise ValueError("quantization group size must be a power of two >= 2")
        if self.field_dtypes and len(self.field_dtypes) != len(self.fields):
            raise ValueError("field dtypes must describe every field")
        if self.candidate_slots <= 0:
            raise ValueError("state candidate slots must be positive")


@dataclass(frozen=True)
class StateView:
    state: StateSpec
    format: FormatSpec
    # A tuple prevents field replacement; native tensor shape/stride/offset remain authoritative.
    fields: tuple[Any, ...]

    def __post_init__(self):
        if len(self.fields) != len(self.format.fields):
            raise ValueError("view fields must match its format")


@dataclass(frozen=True)
class DeviceCaps:
    device_type: str
    index: int | None = None
    capability: tuple[int, int] | None = None


@dataclass(frozen=True)
class ExecutionLimits:
    workspace_bytes: int
    max_batch_size: int = 1
    max_seq_len: int = 1
    microbatch_count: int = 1
    max_verify_tokens: int = 1

    def __post_init__(self):
        if self.workspace_bytes < 0:
            raise ValueError("workspace budget must be nonnegative")
        if min(self.max_batch_size, self.max_seq_len, self.microbatch_count, self.max_verify_tokens) <= 0:
            raise ValueError("execution capacities must be positive")


@dataclass(frozen=True)
class AttentionOffer:
    backend: str
    phase: Phase
    format: FormatSpec
    workspace_bytes: int = 0
    # Only implementations whose layers execute serially may share this scratch.
    # Concurrent microbatches occupy separate slices within the reservation.
    workspace_group: str | None = None
    execution_limits: ExecutionLimits | None = None
    transfer_requires_compute_sync: bool = False

    def __post_init__(self):
        if not self.backend or self.phase not in ("prefill", "decode", "verify") or self.workspace_bytes < 0:
            raise ValueError("invalid attention offer")


@dataclass(frozen=True)
class LinearSpec:
    in_features: int
    out_features: int
    dtype: Any
    batches: int | None = None

    def __post_init__(self):
        if min(self.in_features, self.out_features) <= 0 or (self.batches is not None and self.batches <= 0):
            raise ValueError("linear dimensions must be positive")


@dataclass(frozen=True)
class LinearOffer:
    backend: str
    spec: LinearSpec
    format: FormatSpec


@dataclass(frozen=True)
class MoeSpec:
    hidden_size: int
    intermediate_size: int
    experts: int
    top_k: int
    dtype: Any
    renormalize: bool = False
    scoring: str = "softmax"
    groups: int = 1
    selected_groups: int = 1
    group_score_topk: int = 1
    routed_scale: float = 1.0
    shared_experts: int = 0
    gated_shared: bool = False
    alpha: float = 1.0
    limit: float | None = None
    clamp_up_add_one: bool = True
    # Preserve input-dtype rounding in routing, projections, activations and routed/shared subtotals.
    round_intermediates: bool = False

    def __post_init__(self):
        if min(self.hidden_size, self.intermediate_size, self.experts, self.top_k, self.groups) <= 0:
            raise ValueError("MoE dimensions and routing counts must be positive")
        if self.experts % self.groups or not 0 < self.selected_groups <= self.groups:
            raise ValueError("MoE groups must partition the experts")
        if self.top_k > self.selected_groups * (self.experts // self.groups):
            raise ValueError("MoE top-k exceeds selected group capacity")
        if not 0 < self.group_score_topk <= self.experts // self.groups:
            raise ValueError("MoE group score exceeds group capacity")
        if self.scoring not in ("softmax", "sigmoid", "topk_softmax"):
            raise ValueError("unsupported MoE scoring function")
        if self.scoring == "topk_softmax" and (self.groups != 1 or not self.renormalize):
            raise ValueError("top-k softmax requires ungrouped, normalized routing")
        if self.shared_experts < 0 or (self.gated_shared and not self.shared_experts):
            raise ValueError("gated MoE needs shared experts")
        if not isfinite(self.routed_scale) or not isfinite(self.alpha) or self.alpha <= 0:
            raise ValueError("MoE scales must be finite and activation alpha positive")
        if self.limit is not None and (not isfinite(self.limit) or self.limit <= 0):
            raise ValueError("MoE activation limit must be finite and positive")
        if self.limit is None and self.alpha != 1:
            raise ValueError("MoE activation alpha requires an explicit limit")


@dataclass(frozen=True)
class MoeOffer:
    backend: str
    spec: MoeSpec
    format: FormatSpec
    fused_shared_experts: int = 0


@dataclass(frozen=True)
class ProbeResult:
    offers: tuple[AttentionOffer | LinearOffer | MoeOffer, ...] = ()
    reason: str = ""

    def __post_init__(self):
        if bool(self.offers) == bool(self.reason):
            raise ValueError("probe must return either offers or a rejection reason")
