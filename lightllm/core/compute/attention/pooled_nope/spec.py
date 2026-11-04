from dataclasses import dataclass
from math import isfinite
from typing import Any, Literal


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

    def validate_state(self, request):
        if request.state.shape != (request.spec.latent_dim, request.spec.index_dim):
            raise ValueError("pooled latent state shape disagrees with attention semantics")

        if request.params.spec != request.spec:
            raise ValueError("static pooled parameters differ from the probed semantics")

        if len(request.auxiliary_states) != 1:
            raise ValueError("pooled attention requires its separately declared indexer tail")

        name, tail, _ = request.auxiliary_states[0]

        if (
            name != "indexer_tail"
            or tail.retention != "window"
            or tail.shape != (request.spec.pool_size, request.spec.pool_size, request.spec.index_dim)
        ):
            raise ValueError("indexer tail window disagrees with pooled semantics")
