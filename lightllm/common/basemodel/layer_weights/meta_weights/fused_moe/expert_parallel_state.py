from dataclasses import dataclass
from typing import Optional

import torch


@dataclass
class EPLBState:
    num_redundant_experts_per_rank: int
    initial_redundant_expert_ids_by_rank: torch.Tensor
    logical_to_physical_map: torch.Tensor
    logical_replica_count: torch.Tensor
    route_counter: torch.Tensor
    recording: bool = False
    recorded_sample_count: int = 0

    def next_sample_index(self) -> int:
        if not self.recording:
            return 0
        sample_index = self.recorded_sample_count % self.route_counter.shape[0]
        self.recorded_sample_count += 1
        return sample_index


@dataclass(frozen=True)
class ExpertParallelState:
    num_logical_experts: int
    world_size: int
    eplb: Optional[EPLBState] = None

    @property
    def num_primary_experts_per_rank(self) -> int:
        return self.num_logical_experts // self.world_size

    @property
    def num_total_physical_experts(self) -> int:
        num_redundant_experts_per_rank = 0 if self.eplb is None else self.eplb.num_redundant_experts_per_rank
        return self.num_logical_experts + self.world_size * num_redundant_experts_per_rank
