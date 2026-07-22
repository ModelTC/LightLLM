from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass
from typing import Iterator, Optional

import torch


_eplb_model_init_disabled: ContextVar[bool] = ContextVar("eplb_model_init_disabled", default=False)


def is_eplb_model_init_disabled() -> bool:
    return _eplb_model_init_disabled.get()


@contextmanager
def disable_eplb_model_init() -> Iterator[None]:
    token = _eplb_model_init_disabled.set(True)
    try:
        yield
    finally:
        _eplb_model_init_disabled.reset(token)


@dataclass
class EPLBState:
    num_redundant_experts_per_rank: int
    logical_to_physical_map: torch.Tensor
    logical_replica_count: torch.Tensor
    route_counter: torch.Tensor
    recording: bool = False
    recorded_sample_count: int = 0
    physical_to_logical: Optional[torch.Tensor] = None
    placement_generation: int = 0

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


def create_expert_parallel_state(num_logical_experts: int, world_size: int, rank: int) -> ExpertParallelState:
    from lightllm.common.eplb_utils import EPLB_STEADY_SAMPLE_STEPS
    from lightllm.utils.dist_utils import get_node_world_size
    from lightllm.utils.envs_utils import (
        get_env_start_args,
        get_prefill_eplb_step_interval,
        get_prefill_eplb_steady_sample_steps,
    )
    from .eplb_placement import (
        build_initial_redundant_expert_ids,
        expand_redundant_placement,
        build_logical_to_physical_map,
    )

    args = get_env_start_args()
    eplb = None
    if args.enable_prefill_eplb and not is_eplb_model_init_disabled():
        if args.run_mode != "prefill":
            raise ValueError("EPLB requires --run_mode prefill")
        redundant = args.eplb_num_redundant_experts_per_rank
        physical_to_logical = expand_redundant_placement(
            build_initial_redundant_expert_ids(num_logical_experts, world_size, redundant), num_logical_experts
        )
        logical_to_physical, replica_count = build_logical_to_physical_map(
            physical_to_logical, num_logical_experts, source_rank=rank, node_world_size=get_node_world_size()
        )
        # Cover both initial/steady windows and two microbatch dispatches per step.
        sample_steps = max(
            get_prefill_eplb_step_interval(), min(get_prefill_eplb_steady_sample_steps(), EPLB_STEADY_SAMPLE_STEPS)
        )
        eplb = EPLBState(
            num_redundant_experts_per_rank=redundant,
            physical_to_logical=physical_to_logical,
            logical_to_physical_map=logical_to_physical.cuda(),
            logical_replica_count=replica_count.cuda(),
            route_counter=torch.zeros((2 * sample_steps, num_logical_experts), dtype=torch.int64, device="cuda"),
        )
    return ExpertParallelState(num_logical_experts, world_size, eplb)
