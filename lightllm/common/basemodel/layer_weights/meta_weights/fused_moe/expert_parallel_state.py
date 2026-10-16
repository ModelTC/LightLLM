from dataclasses import dataclass
from typing import Callable, Optional

import torch


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
    migration_poll: Optional[Callable[[], None]] = None
    replica_rotation_rank: int = 0
    tp_output_sync_src: Optional[int] = None

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

    def sync_output(self, output: torch.Tensor, is_prefill: Optional[bool], dist_group=None) -> torch.Tensor:
        """Apply the output-consistency contract of this expert-parallel layout."""
        eplb = self.eplb
        if is_prefill is True and eplb is not None and eplb.tp_output_sync_src is not None:
            from lightllm.distributed import broadcast

            broadcast(output, src=eplb.tp_output_sync_src, group=dist_group)
        return output


def create_expert_parallel_state(
    num_logical_experts: int,
    world_size: int,
    rank: int,
    *,
    tp_rank: int,
    tp_world_size: int,
    is_mtp_draft_model: bool = False,
) -> ExpertParallelState:
    from lightllm.common.eplb_utils import EPLB_STEADY_SAMPLE_STEPS, is_eplb_enabled
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
        get_max_reachable_replicas,
    )

    args = get_env_start_args()
    eplb = None
    if is_eplb_enabled(args):
        node_world_size = get_node_world_size()
        replica_rotation_rank = rank
        tp_output_sync_src = None
        if tp_world_size > 1 and not args.enable_tpsp_mix_mode:
            # TP ranks hold the same tokens. Identical physical routes keep
            # DeepEP's BF16 partial-sum groups consistent without a collective.
            leader = rank - tp_rank
            if leader // node_world_size == (leader + tp_world_size - 1) // node_world_size:
                replica_rotation_rank = leader
            else:
                # Preserve node-local replica selection for cross-node TP and
                # synchronize the complete routed output after MoE execution.
                tp_output_sync_src = leader
        redundant = args.eplb_num_redundant_experts_per_rank
        if is_mtp_draft_model:
            from lightllm.distributed import dist_group_manager

            physical_experts = num_logical_experts + world_size * redundant
            if getattr(dist_group_manager, "ll_num_experts", physical_experts) != physical_experts:
                raise ValueError("EPLB target and draft must share the same DeepEP expert space")
        physical_to_logical = expand_redundant_placement(
            build_initial_redundant_expert_ids(num_logical_experts, world_size, redundant), num_logical_experts
        )
        logical_to_physical, replica_count = build_logical_to_physical_map(
            physical_to_logical,
            num_logical_experts,
            source_rank=rank,
            node_world_size=node_world_size,
            replica_rotation_rank=replica_rotation_rank,
        )
        max_hash_replicas = get_max_reachable_replicas(physical_to_logical, num_logical_experts, node_world_size)
        # Cover both initial/steady windows and two microbatch dispatches per step.
        # Draft routing stays static and unrecorded, but must use the same
        # physical expert count as the target: DeepEP shares its signal buffer.
        sample_steps = (
            1
            if is_mtp_draft_model
            else max(
                get_prefill_eplb_step_interval(), min(get_prefill_eplb_steady_sample_steps(), EPLB_STEADY_SAMPLE_STEPS)
            )
        )
        eplb = EPLBState(
            num_redundant_experts_per_rank=redundant,
            physical_to_logical=physical_to_logical,
            logical_to_physical_map=logical_to_physical.cuda(),
            logical_replica_count=replica_count.cuda(),
            # Logical routes plus exact hash buckets for every replica count.
            route_counter=torch.zeros(
                (2 * sample_steps, num_logical_experts, 1 + max_hash_replicas * (max_hash_replicas + 1) // 2),
                dtype=torch.int64,
                device="cuda",
            ),
            replica_rotation_rank=replica_rotation_rank,
            tp_output_sync_src=tp_output_sync_src,
        )
    return ExpertParallelState(num_logical_experts, world_size, eplb)
