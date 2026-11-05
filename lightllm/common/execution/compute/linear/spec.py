from dataclasses import dataclass
from math import isfinite
from typing import Any, Literal


@dataclass(frozen=True)
class LinearSpec:
    in_features: int
    out_features: int
    dtype: Any
    batches: int | None = None

    def __post_init__(self):
        if min(self.in_features, self.out_features) <= 0 or (self.batches is not None and self.batches <= 0):
            raise ValueError("linear dimensions must be positive")
