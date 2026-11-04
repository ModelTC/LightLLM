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
        shape, spec = request.state.shape, request.spec

        tp = request.format.shard[1]

        expected = (spec.k_heads, spec.v_heads, spec.key_dim, spec.value_dim, spec.conv_size - 1)

        if len(shape) != 5 or (shape[0] // tp, shape[1] // tp, *shape[2:]) != expected:
            raise ValueError(f"layer {request.layer_id}: recurrent state shape disagrees with semantics")

        if getattr(request.params, "spec", request.spec) != request.spec:
            raise ValueError("static recurrent parameters differ from the probed semantics")
