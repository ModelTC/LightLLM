from dataclasses import dataclass
from math import isfinite
from typing import Any, Literal


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
        return self.head_dim ** -0.5 if self.score_scale is None else self.score_scale

    def validate_state(self, request):
        shape = request.state.shape

        local_heads = max(1, shape[0] // request.format.shard[1])

        if len(shape) != 2 or (local_heads, shape[-1]) != (request.spec.kv_heads, request.spec.head_dim):
            raise ValueError(f"layer {request.layer_id}: state shape disagrees with attention semantics")

        if getattr(request.params, "spec", request.spec) != request.spec:
            raise ValueError("static attention parameters differ from the probed semantics")
