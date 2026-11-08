"""Explicit process-group ownership for context-parallel execution.

The serving path currently owns one world-wide tensor-parallel process group.
This module provides a separate, opt-in owner for a DP x context x TP layout.
It deliberately does not change launcher defaults or attention behavior; callers
must construct it with an explicit layout before using context collectives.
"""

from __future__ import annotations

import operator
from dataclasses import dataclass
from typing import Any, Iterable

import torch
import torch.distributed as dist


def _index(value: Any, name: str, *, minimum: int) -> int:
    if isinstance(value, bool):
        raise ValueError(f"{name} must be an integer, got bool")
    try:
        value = operator.index(value)
    except TypeError as exc:
        raise ValueError(f"{name} must be an integer, got {type(value).__name__}") from exc
    if value < minimum:
        raise ValueError(f"{name} must be >= {minimum}, got {value}")
    return value


@dataclass(frozen=True)
class ContextParallelLayout:
    """Validated DP x context x TP rank coordinates.

    ``context_size`` is the PCP or DCP degree.  The layout is shared because
    both modes require a group whose context rank varies while DP and TP ranks
    stay fixed.  The attention and allocator semantics remain mode-specific.
    """

    global_rank: int
    dp_size: int
    context_size: int
    tp_size: int
    order: str = "context_major"

    def __post_init__(self) -> None:
        for name, minimum in (
            ("global_rank", 0),
            ("dp_size", 1),
            ("context_size", 1),
            ("tp_size", 1),
        ):
            object.__setattr__(self, name, _index(getattr(self, name), name, minimum=minimum))
        if self.order not in ("context_major", "tp_major"):
            raise ValueError(f"order must be 'context_major' or 'tp_major', got {self.order!r}")
        if self.global_rank >= self.world_size:
            raise ValueError(f"global_rank must be in [0, {self.world_size}), got {self.global_rank}")

    @property
    def world_size(self) -> int:
        return self.dp_size * self.context_size * self.tp_size

    @property
    def dp_rank(self) -> int:
        return self.global_rank // (self.context_size * self.tp_size)

    @property
    def context_rank(self) -> int:
        if self.order == "tp_major":
            return self.global_rank % self.context_size
        return (self.global_rank // self.tp_size) % self.context_size

    @property
    def tp_rank(self) -> int:
        if self.order == "tp_major":
            return (self.global_rank // self.context_size) % self.tp_size
        return self.global_rank % self.tp_size

    def rank_for(self, dp_rank: int, context_rank: int, tp_rank: int) -> int:
        """Map logical DP/context/TP coordinates to a global rank."""

        if self.order == "tp_major":
            return (dp_rank * self.context_size * self.tp_size) + (tp_rank * self.context_size) + context_rank
        return (dp_rank * self.context_size + context_rank) * self.tp_size + tp_rank

    def group_ranks(self, kind: str) -> tuple[int, ...]:
        """Return global ranks for one TP, context, or DP subgroup."""

        if kind == "tp":
            return tuple(self.rank_for(self.dp_rank, self.context_rank, rank) for rank in range(self.tp_size))
        if kind == "context":
            return tuple(self.rank_for(self.dp_rank, rank, self.tp_rank) for rank in range(self.context_size))
        if kind == "dp":
            return tuple(self.rank_for(rank, self.context_rank, self.tp_rank) for rank in range(self.dp_size))
        raise ValueError(f"unsupported process-group kind: {kind}")


@dataclass
class ContextParallelGroups:
    """The three subgroup handles owned by one worker."""

    layout: ContextParallelLayout
    tp: dist.ProcessGroup
    context: dist.ProcessGroup
    dp: dist.ProcessGroup
    backend: str | None = None
    _closed: bool = False

    def group(self, kind: str) -> dist.ProcessGroup:
        if kind not in ("tp", "context", "dp"):
            raise ValueError(f"unsupported process-group kind: {kind}")
        return getattr(self, kind)

    def rank(self, kind: str) -> int:
        return dist.get_rank(self.group(kind))

    def size(self, kind: str) -> int:
        return dist.get_world_size(self.group(kind))

    def all_reduce(self, tensor: torch.Tensor, *, kind: str = "context") -> torch.Tensor:
        """Reduce a tensor in an explicitly selected subgroup."""

        dist.all_reduce(tensor, group=self.group(kind))
        return tensor

    def close(self) -> None:
        """Destroy owned groups once; callers still own the default WORLD group."""

        if self._closed:
            return
        for group in (self.tp, self.context, self.dp):
            dist.destroy_process_group(group)
        self._closed = True


def _group_specs(layout: ContextParallelLayout) -> Iterable[tuple[str, tuple[int, ...]]]:
    """Yield every subgroup in the same order on every rank."""

    for dp_rank in range(layout.dp_size):
        for context_rank in range(layout.context_size):
            yield "tp", tuple(layout.rank_for(dp_rank, context_rank, rank) for rank in range(layout.tp_size))
    for dp_rank in range(layout.dp_size):
        for tp_rank in range(layout.tp_size):
            yield "context", tuple(layout.rank_for(dp_rank, rank, tp_rank) for rank in range(layout.context_size))
    for context_rank in range(layout.context_size):
        for tp_rank in range(layout.tp_size):
            yield "dp", tuple(layout.rank_for(rank, context_rank, tp_rank) for rank in range(layout.dp_size))


def create_context_parallel_groups(
    layout: ContextParallelLayout, *, backend: str | None = None
) -> ContextParallelGroups:
    """Create deterministic TP/context/DP groups for an initialized world.

    Every rank must call this function with the same layout and in the same
    order.  It does not alter the default process group, and it rejects a
    mismatched world/rank before creating any subgroup.
    """

    if not dist.is_available() or not dist.is_initialized():
        raise RuntimeError("context-parallel groups require an initialized torch.distributed world")
    if dist.get_world_size() != layout.world_size:
        raise ValueError(f"world size {dist.get_world_size()} does not match layout world size {layout.world_size}")
    if dist.get_rank() != layout.global_rank:
        raise ValueError(f"distributed rank {dist.get_rank()} does not match layout rank {layout.global_rank}")

    selected: dict[str, dist.ProcessGroup] = {}
    for kind, ranks in _group_specs(layout):
        group = dist.new_group(ranks=list(ranks), backend=backend)
        if layout.global_rank in ranks and kind not in selected:
            selected[kind] = group
    return ContextParallelGroups(
        layout=layout,
        tp=selected["tp"],
        context=selected["context"],
        dp=selected["dp"],
        backend=backend,
    )
