from dataclasses import dataclass
from typing import Protocol

import torch


class RotaryEmb(Protocol):
    def __call__(
        self,
        q: torch.Tensor,
        k: torch.Tensor | None,
        cos: torch.Tensor,
        sin: torch.Tensor,
        partial_rotary_factor: float = 1.0,
    ) -> None: ...


@dataclass(frozen=True)
class RotaryOps:
    rotary_emb: RotaryEmb
