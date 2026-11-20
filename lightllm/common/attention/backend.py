from abc import ABC, abstractmethod
from typing import Generic, Protocol, TypeVar

import torch


Batch = TypeVar("Batch")
Inputs = TypeVar("Inputs")


class TensorAllocator(Protocol):
    """Allocate contiguous scratch with the requested shape, dtype and device.

    Calls pass a tuple/list shape positionally and dtype/device by keyword,
    as supported by torch.empty and the serving allocator. Returned storage
    must not overlap live operands, outputs or other scratch. The caller's
    allocator manages reuse and CUDA graph lifetimes; the backend retains none.
    """

    def __call__(
        self, shape: tuple[int, ...] | list[int], /, *, dtype: torch.dtype, device: torch.device
    ) -> torch.Tensor:
        ...


class AttentionBackend(ABC, Generic[Batch, Inputs]):
    """Execute attention with caller-owned operands and prepared batch metadata.

    Batch describes one invocation using the provider's execution metadata;
    it need not have the same representation for every provider. Preparation,
    persistent storage and CUDA graph lifecycle belong to the caller.
    """

    @abstractmethod
    def forward(
        self, batch: Batch, inputs: Inputs, *, out: torch.Tensor, alloc_func: TensorAllocator = torch.empty
    ) -> torch.Tensor:
        """Fill and return out itself; concrete backends define tensor layouts."""
        pass
