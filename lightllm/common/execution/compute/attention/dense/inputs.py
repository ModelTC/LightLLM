from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class DenseInputs:
    q: Any
    current_k: Any = None
    current_v: Any = None

    def __post_init__(self):
        if (self.current_k is None) != (self.current_v is None):
            raise ValueError("current K and V must be provided together")


def query_tensor(inputs):
    return inputs.q if isinstance(inputs, DenseInputs) else inputs
