from dataclasses import dataclass
from typing import Optional

import torch


@dataclass
class EPLBState:
    redundant_experts_per_rank: int
    logical_to_physical_map: torch.Tensor
    logical_replica_count: torch.Tensor
    route_counter: torch.Tensor
    recording: bool = False
    recorded_sample_count: int = 0


@dataclass(frozen=True)
class ExpertParallelState:
    logical_experts: int
    world_size: int
    eplb: Optional[EPLBState] = None

    @property
    def primary_experts_per_rank(self) -> int:
        return self.logical_experts // self.world_size

    @property
    def total_physical_experts(self) -> int:
        redundant_experts_per_rank = 0 if self.eplb is None else self.eplb.redundant_experts_per_rank
        return self.logical_experts + self.world_size * redundant_experts_per_rank
