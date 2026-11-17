from abc import ABC, abstractmethod
from typing import Generic, TypeVar

import torch


Batch = TypeVar("Batch")
Inputs = TypeVar("Inputs")


class AttentionBackend(ABC, Generic[Batch, Inputs]):
    @abstractmethod
    def forward(self, batch: Batch, inputs: Inputs, *, out: torch.Tensor, alloc_func=torch.empty) -> torch.Tensor:
        pass
