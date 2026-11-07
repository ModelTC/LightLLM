from lightllm.core.formats.recurrent.spec import ConvSSMStateSpec
from dataclasses import dataclass
from math import isfinite
from typing import Any, Literal


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

    def validate_state(self, request):
        tp = request.format.shard[1]
        expected = ConvSSMStateSpec(
            key_heads=self.k_heads * tp,
            value_heads=self.v_heads * tp,
            key_dim=self.key_dim,
            value_dim=self.value_dim,
            conv_history_length=self.conv_size - 1,
        )
        if request.state.params != expected or request.params.spec != self:
            raise ValueError("GDN state or static parameters disagree with attention semantics")
