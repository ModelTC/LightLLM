"""Canonical expert-parallel runtime state shared by fused-MoE components."""

from dataclasses import dataclass
from typing import Optional

import torch


@dataclass
class EPLBState:
    """Mutable EPLB runtime tensors; tensors retain their owner-provided references."""

    redundant_experts_per_rank: int
    logical_to_physical_map: torch.Tensor
    logical_replica_count: torch.Tensor
    route_counter: torch.Tensor
    recording: bool = False
    recorded_sample_count: int = 0

    def __post_init__(self) -> None:
        if self.redundant_experts_per_rank <= 0:
            raise ValueError("EPLB redundant experts per rank must be positive")
        if self.recorded_sample_count < 0:
            raise ValueError("EPLB recorded sample count cannot be negative")
        mapping, replicas, counter = (
            self.logical_to_physical_map,
            self.logical_replica_count,
            self.route_counter,
        )
        if mapping.dtype != torch.int32 or replicas.dtype != torch.int32:
            raise ValueError("EPLB mapping and replica-count tensors must be int32")
        if counter.dtype != torch.int64:
            raise ValueError("EPLB route counter must be int64")
        if not all(tensor.is_contiguous() for tensor in (mapping, replicas, counter)):
            raise ValueError("EPLB tensors must be contiguous")
        if mapping.ndim != 2 or mapping.shape[1] <= 0 or replicas.ndim != 1:
            raise ValueError("EPLB mapping must be 2-D with replicas as a 1-D tensor")
        if counter.ndim != 2 or counter.shape[0] == 0:
            raise ValueError("EPLB route counter must be a non-empty 2-D tensor")
        if not (mapping.device == replicas.device == counter.device):
            raise ValueError("EPLB tensors must be on one device")


@dataclass(frozen=True)
class ExpertParallelState:
    logical_experts: int
    world_size: int
    eplb: Optional[EPLBState] = None

    def __post_init__(self) -> None:
        if self.logical_experts <= 0 or self.world_size <= 0:
            raise ValueError("expert-parallel logical experts and world size must be positive")
        if self.logical_experts % self.world_size:
            raise ValueError("logical experts must divide evenly across expert-parallel ranks")
        if self.eplb is not None:
            if (
                self.eplb.logical_to_physical_map.shape[0] != self.logical_experts
                or self.eplb.logical_replica_count.numel() != self.logical_experts
                or self.eplb.route_counter.shape[1] != self.logical_experts
            ):
                raise ValueError("EPLB tensors must agree with logical expert count")

    @property
    def primary_experts_per_rank(self) -> int:
        return self.logical_experts // self.world_size

    @property
    def total_physical_experts(self) -> int:
        redundant_experts_per_rank = 0 if self.eplb is None else self.eplb.redundant_experts_per_rank
        return self.logical_experts + self.world_size * redundant_experts_per_rank
