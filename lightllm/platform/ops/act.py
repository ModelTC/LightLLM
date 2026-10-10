from dataclasses import dataclass
from typing import Any, Callable

ActOp = Callable[..., Any]


@dataclass(frozen=True)
class ActOps:
    silu_and_mul: ActOp
    gelu_and_mul: ActOp
