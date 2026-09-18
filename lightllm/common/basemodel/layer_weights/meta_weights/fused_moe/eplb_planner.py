"""Pure-Python redundant-expert placement planning for EPLB.

The planner deliberately uses nested lists instead of tensors.  Tensor
conversion belongs to the manager's distributed-communication and migration
boundaries; keeping it out of this module makes placement algorithms easy to
read, test, and replace.
"""

from abc import ABC, abstractmethod
from collections import Counter
from math import ceil
from typing import List


# [layer][logical expert]
LogicalExpertLoad = List[List[float]]
# [layer][rank][local physical expert] -> logical expert
ExpertPlacement = List[List[List[int]]]
# [layer][rank]
RankLoad = List[List[float]]


class EPLBPlanner(ABC):
    """Interface for planning redundant expert placement."""

    @abstractmethod
    def plan(
        self,
        logical_expert_load: LogicalExpertLoad,
        current_placement: ExpertPlacement,
    ) -> ExpertPlacement:
        """Return a concrete ``[layer][rank][local physical expert]`` placement."""


class GreedyEPLBPlanner(EPLBPlanner):
    """Greedy planner based on global logical-expert loads."""

    def __init__(
        self,
        world_size: int,
        num_redundant_experts_per_rank: int,
        *,
        expert_alignment: int = 1,
        rebalance_gain_threshold: float = 0.0,
    ):
        if world_size <= 1:
            raise ValueError("world_size must be greater than one")
        if num_redundant_experts_per_rank <= 0:
            raise ValueError("num_redundant_experts_per_rank must be positive")
        if expert_alignment <= 0:
            raise ValueError("expert_alignment must be positive")
        if not 0.0 <= rebalance_gain_threshold <= 1.0:
            raise ValueError("rebalance_gain_threshold must be between 0.0 and 1.0")
        self.world_size = world_size
        self.num_redundant_experts_per_rank = num_redundant_experts_per_rank
        self.expert_alignment = expert_alignment
        self.rebalance_gain_threshold = rebalance_gain_threshold

    def plan(
        self,
        logical_expert_load: LogicalExpertLoad,
        current_placement: ExpertPlacement,
    ) -> ExpertPlacement:
        """Return a new concrete placement.

        ``logical_expert_load`` is ``[layer][logical_expert]`` and contains
        the load summed across all ranks.
        ``current_placement`` is ``[layer][rank][local_physical_expert]``.
        Each rank row contains its fixed primary experts followed by its
        movable redundant experts.  The returned placement has the same shape.
        """
        load = [[float(value) for value in layer] for layer in logical_expert_load]
        current = [[[int(expert) for expert in rank] for rank in layer] for layer in current_placement]
        self._validate_inputs(load, current)
        before_rank_load = self.estimate_rank_load(load, current)

        candidates = [self._plan_layer(layer_load, current_layer) for layer_load, current_layer in zip(load, current)]
        candidate_rank_load = self.estimate_rank_load(load, candidates)
        placement = []
        for layer, candidate in enumerate(candidates):
            before = max(before_rank_load[layer])
            after = max(candidate_rank_load[layer])
            gain = (before - after) / max(before, 1.0)
            changed = candidate != current[layer] and gain > self.rebalance_gain_threshold
            placement.append(candidate if changed else current[layer])
        return placement

    def estimate_rank_load(
        self,
        logical_expert_load: LogicalExpertLoad,
        placement: ExpertPlacement,
    ) -> RankLoad:
        """Estimate aligned physical-expert work for each layer and rank."""
        load = [[float(value) for value in layer] for layer in logical_expert_load]
        normalized_placement = [[[int(expert) for expert in rank] for rank in layer] for layer in placement]
        num_logical_experts = self._validate_inputs(load, normalized_placement)
        estimated = []
        for layer_load, layer_placement in zip(load, normalized_placement):
            locations = self._expert_locations(layer_placement, num_logical_experts)
            rank_load = [0.0] * self.world_size
            for expert, expert_locations in enumerate(locations):
                for rank, value in enumerate(self._physical_load(layer_load[expert], expert_locations)):
                    rank_load[rank] += value
            estimated.append(rank_load)
        return estimated

    def _plan_layer(
        self,
        logical_load: List[float],
        current_placement: List[List[int]],
    ) -> List[List[int]]:
        num_logical_experts = len(logical_load)
        experts_per_rank = num_logical_experts // self.world_size
        owner = [expert // experts_per_rank for expert in range(num_logical_experts)]
        copy_count = self._allocate_copy_count(logical_load, owner)
        current_redundant_placement = [row[experts_per_rank:] for row in current_placement]

        redundant_placement = [[-1] * self.num_redundant_experts_per_rank for _ in range(self.world_size)]
        locations = [{owner_rank} for owner_rank in owner]
        expert_rank_load = [
            self._physical_load(
                logical_load[expert],
                locations[expert],
            )
            for expert in range(num_logical_experts)
        ]
        rank_load = [
            sum(expert_rank_load[expert][rank] for expert in range(num_logical_experts))
            for rank in range(self.world_size)
        ]
        instances = sorted(
            [expert for expert, copies in enumerate(copy_count) for _ in range(copies - 1)],
            key=lambda expert: (
                -copy_count[expert],
                -logical_load[expert] / copy_count[expert],
                expert,
            ),
        )

        for index, expert in enumerate(instances):
            best = None
            for rank in range(self.world_size):
                if rank in locations[expert]:
                    continue
                empty_slots = [slot for slot, value in enumerate(redundant_placement[rank]) if value < 0]
                if not empty_slots:
                    continue
                slot = min(
                    empty_slots,
                    key=lambda candidate: (
                        current_redundant_placement[rank][candidate] != expert,
                        candidate,
                    ),
                )
                trial_redundant_placement = [row[:] for row in redundant_placement]
                trial_redundant_placement[rank][slot] = expert
                trial_locations = [set(ranks) for ranks in locations]
                trial_locations[expert].add(rank)
                if not self._can_complete(
                    instances[index + 1 :],
                    trial_redundant_placement,
                    trial_locations,
                    owner,
                ):
                    continue

                next_expert_load = self._physical_load(
                    logical_load[expert],
                    trial_locations[expert],
                )
                trial_rank_load = [
                    value - expert_rank_load[expert][target_rank] + next_expert_load[target_rank]
                    for target_rank, value in enumerate(rank_load)
                ]
                candidate = (
                    max(trial_rank_load),
                    current_redundant_placement[rank][slot] != expert,
                    sum(trial_rank_load),
                    rank,
                    slot,
                    trial_rank_load,
                    next_expert_load,
                )
                if best is None or candidate[:5] < best[:5]:
                    best = candidate

            if best is None:
                raise RuntimeError("EPLB planner found no valid redundant expert placement")
            _, _, _, rank, slot, rank_load, next_expert_load = best
            redundant_placement[rank][slot] = expert
            locations[expert].add(rank)
            expert_rank_load[expert] = next_expert_load

        return [
            list(range(rank * experts_per_rank, (rank + 1) * experts_per_rank)) + redundant_experts
            for rank, redundant_experts in enumerate(redundant_placement)
        ]

    def _allocate_copy_count(self, logical_load: List[float], owner: List[int]) -> List[int]:
        copy_count = [1] * len(logical_load)
        replicas_by_owner = [0] * self.world_size
        owner_capacity = self.num_redundant_experts_per_rank * (self.world_size - 1)
        total_replicas = self.num_redundant_experts_per_rank * self.world_size
        for _ in range(total_replicas):
            candidates = [
                expert
                for expert in range(len(logical_load))
                if copy_count[expert] < self.world_size and replicas_by_owner[owner[expert]] < owner_capacity
            ]
            if not candidates:
                raise RuntimeError("EPLB planner cannot allocate all redundant copies")
            expert = max(
                candidates,
                key=lambda candidate: (
                    logical_load[candidate] / copy_count[candidate],
                    -candidate,
                ),
            )
            copy_count[expert] += 1
            replicas_by_owner[owner[expert]] += 1
        return copy_count

    def _can_complete(
        self,
        remaining_instances: List[int],
        redundant_placement: List[List[int]],
        locations: List[set],
        owner: List[int],
    ) -> bool:
        """Check that a greedy choice leaves a legal assignment for all slots."""
        remaining = Counter(remaining_instances)
        capacity = [sum(expert < 0 for expert in row) for row in redundant_placement]
        memo = set()

        def search() -> bool:
            if not remaining:
                return True
            state = (
                tuple(sorted(remaining.items())),
                tuple(capacity),
                tuple(tuple(sorted(ranks)) for ranks in locations),
            )
            if state in memo:
                return False
            memo.add(state)

            expert = min(
                remaining,
                key=lambda item: (
                    sum(capacity[rank] > 0 and rank not in locations[item] for rank in range(self.world_size))
                    - remaining[item],
                    -remaining[item],
                    item,
                ),
            )
            candidate_ranks = [
                rank
                for rank in range(self.world_size)
                if capacity[rank] > 0 and rank not in locations[expert] and rank != owner[expert]
            ]
            if len(candidate_ranks) < remaining[expert]:
                return False
            count = remaining.pop(expert)
            if count > 1:
                remaining[expert] = count - 1
            for rank in sorted(candidate_ranks, key=lambda item: (-capacity[item], item)):
                capacity[rank] -= 1
                locations[expert].add(rank)
                if search():
                    locations[expert].remove(rank)
                    capacity[rank] += 1
                    remaining[expert] = count
                    return True
                locations[expert].remove(rank)
                capacity[rank] += 1
            remaining[expert] = count
            return False

        return search()

    def _physical_load(self, logical_expert_load: float, locations: set) -> List[float]:
        physical_expert_load = logical_expert_load / len(locations)
        aligned_load = ceil(physical_expert_load / self.expert_alignment) * self.expert_alignment
        result = [0.0] * self.world_size
        for rank in locations:
            result[rank] = aligned_load
        return result

    def _expert_locations(
        self,
        placement: List[List[int]],
        num_logical_experts: int,
    ) -> List[set]:
        locations = [set() for _ in range(num_logical_experts)]
        for rank, row in enumerate(placement):
            for expert in row:
                if rank in locations[expert]:
                    raise ValueError(f"logical expert {expert} appears twice on rank {rank}")
                locations[expert].add(rank)
        return locations

    def _validate_inputs(
        self,
        logical_expert_load: LogicalExpertLoad,
        placement: ExpertPlacement,
    ) -> int:
        if not logical_expert_load:
            raise ValueError("logical_expert_load must contain at least one layer")
        if len(placement) != len(logical_expert_load):
            raise ValueError("load and placement must have the same number of layers")
        num_logical_experts = len(logical_expert_load[0])
        if num_logical_experts == 0 or num_logical_experts % self.world_size:
            raise ValueError("logical expert count must be positive and divisible by world_size")
        if self.num_redundant_experts_per_rank > num_logical_experts - num_logical_experts // self.world_size:
            raise ValueError("too many redundant slots to avoid local or duplicate replicas")
        num_primary_experts_per_rank = num_logical_experts // self.world_size
        num_local_experts_per_rank = num_primary_experts_per_rank + self.num_redundant_experts_per_rank

        for layer_load, layer_placement in zip(logical_expert_load, placement):
            if len(layer_load) != num_logical_experts:
                raise ValueError("each load layer must contain every logical expert")
            if any(value < 0 for value in layer_load):
                raise ValueError("logical expert load must be non-negative")
            if len(layer_placement) != self.world_size or any(
                len(rank) != num_local_experts_per_rank for rank in layer_placement
            ):
                raise ValueError("each placement layer must be [world_size][local_experts]")
            if any(expert < 0 or expert >= num_logical_experts for rank in layer_placement for expert in rank):
                raise ValueError("placement contains an invalid logical expert")
            for rank, local_expert_ids in enumerate(layer_placement):
                expected_primary_expert_ids = list(
                    range(
                        rank * num_primary_experts_per_rank,
                        (rank + 1) * num_primary_experts_per_rank,
                    )
                )
                if local_expert_ids[:num_primary_experts_per_rank] != expected_primary_expert_ids:
                    raise ValueError("placement primary experts do not match their owning rank")
            self._expert_locations(layer_placement, num_logical_experts)
        return num_logical_experts
