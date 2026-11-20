from abc import ABC, abstractmethod
from typing import Any, Generic, Protocol, TypeVar

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
    """Execute attention with caller-owned operands and planned batch metadata.

    One forward pass follows a fixed lifecycle:

    1. ``batch = backend.plan(...)`` runs once, before the first layer and
       outside CUDA graph capture. The batch holds everything shared by all
       layers of the pass: request metadata and any schedule, workspace or
       library plan derived from it.
    2. ``backend.forward(batch, inputs, out=...)`` runs once per layer.
       Inputs holds only that layer's data, such as Q, the layer's cache view
       and per-layer indices.
    3. Before replaying a captured CUDA graph, the caller rewrites the
       metadata tensors the captured batch references, then calls
       ``backend.refresh(batch, ...)`` so derived state follows the new values.

    plan and refresh arguments and the Batch representation are
    provider-specific. The caller owns operands, outputs, workspace and the
    batch's lifetime.
    """

    @abstractmethod
    def plan(self, *args: Any, **kwargs: Any) -> Batch:
        """Build the batch shared by every layer of one forward pass; providers define the arguments."""

    def refresh(self, batch: Batch, **kwargs: Any) -> None:
        """Update derived state in place after the caller rewrites batch metadata for graph replay.

        Providers that re-plan from host-side values, such as the current maximum KV length,
        take them as keyword arguments. Providers whose kernels read the metadata tensors
        directly have nothing to refresh.
        """

    @abstractmethod
    def forward(
        self, batch: Batch, inputs: Inputs, *, out: torch.Tensor, alloc_func: TensorAllocator = torch.empty
    ) -> torch.Tensor:
        """Fill and return out itself; concrete backends define tensor layouts."""
