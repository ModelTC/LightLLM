from typing import Tuple

import torch


def build_initial_redundant_expert_ids(
    num_logical_experts: int,
    num_ranks: int,
    redundant_experts_per_rank: int,
) -> torch.Tensor:
    """Build a deterministic initial placement without local duplicates."""
    assert num_logical_experts % num_ranks == 0
    experts_per_rank = num_logical_experts // num_ranks
    assert 0 < redundant_experts_per_rank <= num_logical_experts - experts_per_rank

    placement = torch.empty(
        (num_ranks, redundant_experts_per_rank),
        dtype=torch.int64,
    )
    for rank in range(num_ranks):
        local_begin = rank * experts_per_rank
        local_end = local_begin + experts_per_rank
        candidate = ((rank + 1) % num_ranks) * experts_per_rank
        selected = []
        while len(selected) < redundant_experts_per_rank:
            expert_id = candidate % num_logical_experts
            if not local_begin <= expert_id < local_end:
                selected.append(expert_id)
            candidate += 1
        placement[rank] = torch.tensor(selected, dtype=torch.int64)
    return placement


def build_logical_to_physical_map(
    redundant_expert_ids: torch.Tensor,
    num_logical_experts: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Build logical-to-physical maps for fixed primary and redundant slots."""
    assert redundant_expert_ids.ndim == 2
    num_ranks, redundant_experts_per_rank = redundant_expert_ids.shape
    assert num_logical_experts % num_ranks == 0
    experts_per_rank = num_logical_experts // num_ranks
    physical_experts_per_rank = experts_per_rank + redundant_experts_per_rank
    max_replicas = 1 + num_ranks * redundant_experts_per_rank

    logical_to_physical = torch.full(
        (num_logical_experts, max_replicas),
        -1,
        dtype=torch.int64,
    )
    replica_count = torch.ones((num_logical_experts,), dtype=torch.int64)

    for expert_id in range(num_logical_experts):
        owner_rank = expert_id // experts_per_rank
        local_slot = expert_id % experts_per_rank
        logical_to_physical[expert_id, 0] = owner_rank * physical_experts_per_rank + local_slot

    for rank in range(num_ranks):
        for slot in range(redundant_experts_per_rank):
            expert_id = int(redundant_expert_ids[rank, slot].item())
            assert 0 <= expert_id < num_logical_experts
            replica_index = int(replica_count[expert_id].item())
            physical_id = rank * physical_experts_per_rank + experts_per_rank + slot
            logical_to_physical[expert_id, replica_index] = physical_id
            replica_count[expert_id] += 1

    return logical_to_physical, replica_count


def estimate_rank_load(
    expert_load: torch.Tensor,
    redundant_expert_ids: torch.Tensor,
) -> torch.Tensor:
    """Estimate per-rank load assuming replicas evenly split logical load."""
    assert expert_load.ndim == 2
    assert redundant_expert_ids.ndim == 3
    num_layers, num_logical_experts = expert_load.shape
    assert redundant_expert_ids.shape[0] == num_layers
    num_ranks, redundant_experts_per_rank = redundant_expert_ids.shape[1:]
    assert num_logical_experts % num_ranks == 0
    experts_per_rank = num_logical_experts // num_ranks

    replica_count = torch.ones_like(expert_load, dtype=torch.float64)
    replica_count.scatter_add_(
        1,
        redundant_expert_ids.reshape(num_layers, -1),
        torch.ones(
            (num_layers, num_ranks * redundant_experts_per_rank),
            dtype=torch.float64,
        ),
    )
    load_per_replica = expert_load.to(torch.float64) / replica_count
    rank_load = load_per_replica.reshape(
        num_layers,
        num_ranks,
        experts_per_rank,
    ).sum(dim=-1)
    rank_load.scatter_add_(
        1,
        torch.arange(num_ranks, dtype=torch.int64).repeat_interleave(redundant_experts_per_rank).expand(num_layers, -1),
        load_per_replica.gather(
            1,
            redundant_expert_ids.reshape(num_layers, -1),
        ),
    )
    return rank_load


def plan_redundant_experts(
    expert_load: torch.Tensor,
    num_ranks: int,
    redundant_experts_per_rank: int,
) -> torch.Tensor:
    """Plan hot-expert replicas and greedily place them on the lightest ranks.

    The replication step follows the DeepSeek/vLLM EPLB policy: repeatedly
    replicate the logical expert with the largest load per current replica.
    LightLLM keeps primary experts fixed, so only the redundant replicas are
    packed into rank-local spare slots.
    """
    assert expert_load.ndim == 2
    num_layers, num_logical_experts = expert_load.shape
    assert num_logical_experts % num_ranks == 0
    assert redundant_experts_per_rank > 0
    experts_per_rank = num_logical_experts // num_ranks
    num_redundant = num_ranks * redundant_experts_per_rank
    assert num_redundant <= num_logical_experts * (num_ranks - 1)

    load = expert_load.to(dtype=torch.float64, device="cpu")
    placement = torch.full(
        (num_layers, num_ranks, redundant_experts_per_rank),
        -1,
        dtype=torch.int64,
    )

    for layer_idx in range(num_layers):
        layer_load = load[layer_idx]
        replica_count = torch.ones(
            (num_logical_experts,),
            dtype=torch.int64,
        )
        duplicate_ids = []
        for _ in range(num_redundant):
            score = layer_load / replica_count.to(torch.float64)
            score[replica_count >= num_ranks] = -1
            expert_id = int(torch.argmax(score).item())
            duplicate_ids.append(expert_id)
            replica_count[expert_id] += 1

        load_per_replica = layer_load / replica_count.to(torch.float64)
        rank_load = load_per_replica.reshape(
            num_ranks,
            experts_per_rank,
        ).sum(dim=-1)
        remaining_slots = torch.full(
            (num_ranks,),
            redundant_experts_per_rank,
            dtype=torch.int64,
        )
        assigned = [set() for _ in range(num_ranks)]
        duplicate_ids.sort(key=lambda expert_id: (-float(load_per_replica[expert_id]), expert_id))

        for expert_id in duplicate_ids:
            owner_rank = expert_id // experts_per_rank
            candidates = [
                rank
                for rank in range(num_ranks)
                if remaining_slots[rank] > 0 and rank != owner_rank and expert_id not in assigned[rank]
            ]
            if not candidates:
                candidates = [
                    rank for rank in range(num_ranks) if remaining_slots[rank] > 0 and expert_id not in assigned[rank]
                ]
            if not candidates:
                candidates = [rank for rank in range(num_ranks) if remaining_slots[rank] > 0]
            rank = min(
                candidates,
                key=lambda candidate: (float(rank_load[candidate]), candidate),
            )
            slot = redundant_experts_per_rank - int(remaining_slots[rank].item())
            placement[layer_idx, rank, slot] = expert_id
            assigned[rank].add(expert_id)
            remaining_slots[rank] -= 1
            rank_load[rank] += load_per_replica[expert_id]

    assert torch.all(placement >= 0)
    return placement
