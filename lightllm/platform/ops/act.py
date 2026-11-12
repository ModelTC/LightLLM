from dataclasses import dataclass
from typing import Protocol

import torch


class SiluAndMul(Protocol):
    def __call__(self, input: torch.Tensor, output: torch.Tensor) -> None: ...


class GeluAndMul(Protocol):
    def __call__(self, input: torch.Tensor, output: torch.Tensor) -> torch.Tensor: ...


@dataclass(frozen=True)
class ActOps:
    silu_and_mul: SiluAndMul
    gelu_and_mul: GeluAndMul
