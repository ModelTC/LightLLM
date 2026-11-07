from lightllm.core.formats.kv.spec import MLAKVStateSpec
from dataclasses import dataclass
from math import isfinite
from typing import Any, Literal


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

    def validate_state(self, request):
        if request.state.params != MLAKVStateSpec(latent_dim=self.latent_dim, rope_dim=self.rope_dim):
            raise ValueError("MLA latent state dimensions disagree with attention semantics")
        if request.params.spec != self:
            raise ValueError("static MLA parameters differ from the probed semantics")
