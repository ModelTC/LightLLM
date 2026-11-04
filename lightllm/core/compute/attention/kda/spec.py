from lightllm.core.formats.recurrent.spec import ConvSSMStateSpec
from dataclasses import dataclass
from math import isfinite
from typing import Any, Literal


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

    def validate_state(self, request):
        tp = request.format.shard[1]
        expected = ConvSSMStateSpec(
            key_heads=self.heads * tp,
            value_heads=self.heads * tp,
            key_dim=self.head_dim,
            value_dim=self.head_dim,
            conv_history_length=self.conv_size - 1,
        )
        if request.state.params != expected or request.params.spec != self:
            raise ValueError("KDA state or static parameters disagree with attention semantics")
