from typing import Dict, Tuple

import torch


EPLB_MIN_IMBALANCE_RATIO = 1.05
EPLB_MIN_RELATIVE_IMPROVEMENT = 0.05


def build_initial_redundant_expert_ids(
    num_logical_experts: int,
    num_ranks: int,
    redundant_experts_per_rank: int,
) -> torch.Tensor:
    """Build a deterministic initial placement without local duplicates."""
    assert num_logical_experts % num_ranks == 0
    experts_per_rank = num_logical_experts // num_ranks
    assert 0 < redundant_experts_per_rank <= num_logical_experts - experts_per_rank

    rank_offsets = torch.arange(1, num_ranks + 1, dtype=torch.int64)[:, None] * experts_per_rank
    expert_offsets = torch.arange(redundant_experts_per_rank, dtype=torch.int64)
    return (rank_offsets + expert_offsets) % num_logical_experts


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
    max_replicas = num_ranks

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
        assert torch.unique(redundant_expert_ids[rank]).numel() == redundant_experts_per_rank
        for slot in range(redundant_experts_per_rank):
            expert_id = int(redundant_expert_ids[rank, slot].item())
            assert 0 <= expert_id < num_logical_experts
            replica_index = int(replica_count[expert_id].item())
            assert replica_index < max_replicas, "an expert can have at most one replica per rank"
            physical_id = rank * physical_experts_per_rank + experts_per_rank + slot
            logical_to_physical[expert_id, replica_index] = physical_id
            replica_count[expert_id] += 1

    return logical_to_physical, replica_count


def estimate_rank_load(
    expert_load: torch.Tensor,
    redundant_expert_ids: torch.Tensor,
    expert_alignment: int | None = None,
) -> torch.Tensor:
    """Estimate per-rank load, aligning every sample and physical replica separately."""
    assert expert_load.ndim in (2, 3)
    assert redundant_expert_ids.ndim == 3
    has_samples = expert_load.ndim == 3
    if not has_samples:
        expert_load = expert_load.unsqueeze(0)
    num_samples, num_layers, num_logical_experts = expert_load.shape
    assert redundant_expert_ids.shape[0] == num_layers
    num_ranks, redundant_experts_per_rank = redundant_expert_ids.shape[1:]
    assert num_logical_experts % num_ranks == 0
    if expert_alignment is not None:
        assert expert_alignment > 0
    experts_per_rank = num_logical_experts // num_ranks

    replica_count = torch.ones((num_layers, num_logical_experts), dtype=torch.float64, device=expert_load.device)
    replica_count.scatter_add_(
        1,
        redundant_expert_ids.reshape(num_layers, -1),
        torch.ones(
            (num_layers, num_ranks * redundant_experts_per_rank),
            dtype=torch.float64,
            device=expert_load.device,
        ),
    )
    load_per_replica = expert_load.to(torch.float64) / replica_count.unsqueeze(0)
    primary_load = load_per_replica.reshape(
        num_samples,
        num_layers,
        num_ranks,
        experts_per_rank,
    )
    replica_load = load_per_replica.gather(
        2,
        redundant_expert_ids.reshape(1, num_layers, -1).expand(num_samples, -1, -1),
    )
    if expert_alignment is not None:
        primary_load = torch.ceil(primary_load / expert_alignment) * expert_alignment
        replica_load = torch.ceil(replica_load / expert_alignment) * expert_alignment
    rank_load = primary_load.sum(dim=-1)
    rank_load.scatter_add_(
        2,
        torch.arange(num_ranks, dtype=torch.int64, device=expert_load.device)
        .repeat_interleave(redundant_experts_per_rank)
        .reshape(1, 1, -1)
        .expand(num_samples, num_layers, -1),
        replica_load,
    )
    return rank_load if has_samples else rank_load.squeeze(0)


def select_improving_placements(
    expert_load: torch.Tensor,
    current_placement: torch.Tensor,
    candidate_placement: torch.Tensor,
    expert_alignment: int | None = None,
) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, float | int]]:
    """Select strictly better layers, gated by full-model imbalance and gain."""
    assert current_placement.shape == candidate_placement.shape
    current_rank_load = estimate_rank_load(expert_load, current_placement, expert_alignment)
    candidate_rank_load = estimate_rank_load(expert_load, candidate_placement, expert_alignment)
    if expert_load.ndim == 2:
        current_critical = current_rank_load.max(dim=1).values
        candidate_critical = candidate_rank_load.max(dim=1).values
    else:
        current_critical = current_rank_load.max(dim=2).values.sum(dim=0)
        candidate_critical = candidate_rank_load.max(dim=2).values.sum(dim=0)
    # A changed layer must reduce its own aggregate critical load.  The
    # 1.05/5% gates are intentionally evaluated only after all such changes
    # are combined, so a hot individual layer cannot churn the whole model.
    improved = candidate_critical < current_critical
    selected = current_placement.clone()
    selected[improved] = candidate_placement[improved]
    selected_rank_load = estimate_rank_load(expert_load, selected, expert_alignment)
    model_current_critical = current_critical.sum()
    if expert_load.ndim == 2:
        model_current_mean = current_rank_load.mean(dim=1).sum()
        model_selected_critical = selected_rank_load.max(dim=1).values.sum()
        model_selected_mean = selected_rank_load.mean(dim=1).sum()
    else:
        model_current_mean = current_rank_load.mean(dim=2).sum()
        model_selected_critical = selected_rank_load.max(dim=2).values.sum()
        model_selected_mean = selected_rank_load.mean(dim=2).sum()
    model_ratio = model_current_critical / model_current_mean.clamp_min(1.0)
    candidate_model_ratio = model_selected_critical / model_selected_mean.clamp_min(1.0)
    model_relative_improvement = (model_current_critical - model_selected_critical) / model_current_critical.clamp_min(
        1.0
    )
    metrics = {
        "model_imbalance_ratio": float(model_ratio.item()),
        "candidate_model_imbalance_ratio": float(candidate_model_ratio.item()),
        "candidate_relative_improvement": float(model_relative_improvement.item()),
        "candidate_changed_layer_count": int(improved.sum().item()),
    }
    if model_ratio >= EPLB_MIN_IMBALANCE_RATIO and model_relative_improvement >= EPLB_MIN_RELATIVE_IMPROVEMENT:
        return selected, improved, metrics
    return current_placement.clone(), torch.zeros_like(improved), metrics


def plan_redundant_experts(
    expert_load: torch.Tensor,
    num_ranks: int,
    redundant_experts_per_rank: int,
    expert_alignment: int | None = None,
) -> torch.Tensor:
    """Greedily reduce sample-wise aligned critical rank load with replicas."""
    assert expert_load.ndim in (2, 3)
    if expert_alignment is not None:
        assert expert_alignment > 0
    if expert_load.ndim == 2:
        expert_load = expert_load.unsqueeze(0)
    num_samples, num_layers, num_logical_experts = expert_load.shape
    assert num_logical_experts % num_ranks == 0
    assert redundant_experts_per_rank > 0
    experts_per_rank = num_logical_experts // num_ranks
    num_redundant = num_ranks * redundant_experts_per_rank
    assert num_redundant <= num_logical_experts * (num_ranks - 1)

    load = expert_load.to(dtype=torch.float64, device="cpu")
    placement = torch.full((num_layers, num_ranks, redundant_experts_per_rank), -1, dtype=torch.int64)
    layer_indices = torch.arange(num_layers, dtype=torch.int64)
    expert_ids = torch.arange(num_logical_experts, dtype=torch.int64)
    owner_rank = expert_ids // experts_per_rank

    # [sample, layer, expert, replica_count - 1].  This is the same even-share
    # approximation as estimate_rank_load, but applies alignment before rank
    # aggregation.  All layers are planned together so online work has only
    # one Python iteration per redundant slot, not one per layer.
    divisors = torch.arange(1, num_ranks + 1, dtype=torch.float64)
    share = load.unsqueeze(-1) / divisors
    if expert_alignment is not None:
        share = torch.ceil(share / expert_alignment) * expert_alignment
    replica_count = torch.ones((num_layers, num_logical_experts), dtype=torch.int64)
    rank_load = share[..., 0].reshape(num_samples, num_layers, num_ranks, experts_per_rank).sum(dim=-1)
    remaining_slots = torch.full((num_layers, num_ranks), redundant_experts_per_rank, dtype=torch.int64)
    locations = torch.zeros((num_layers, num_logical_experts, num_ranks), dtype=torch.bool)
    locations[:, expert_ids, owner_rank] = True

    for _ in range(num_redundant):
        # Fully replicated experts have no valid target.  Clamp their unused
        # prospective share index so gathering remains in bounds before the
        # validity mask discards those candidates.
        safe_counts = replica_count.clamp_max(num_ranks - 1)
        count_index = safe_counts.unsqueeze(0).unsqueeze(-1).expand(num_samples, -1, -1, -1)
        old_share = share.gather(3, count_index - 1).squeeze(-1)
        new_share = share.gather(3, count_index).squeeze(-1)
        base = rank_load[:, :, None, :] + (new_share - old_share).unsqueeze(-1) * locations.unsqueeze(0)
        top_values, top_ranks = base.topk(k=2, dim=3)
        target_ranks = torch.arange(num_ranks, dtype=torch.int64).reshape(1, 1, 1, -1)
        other_max = torch.where(target_ranks == top_ranks[..., :1], top_values[..., 1:2], top_values[..., :1])
        candidate_critical = torch.maximum(base + new_share.unsqueeze(-1), other_max).sum(dim=0)
        valid = (
            (remaining_slots[:, None, :] > 0)
            & (target_ranks.squeeze(0).squeeze(0) != owner_rank[:, None]).unsqueeze(0)
            & ~locations
        )
        candidate_critical.masked_fill_(~valid, torch.inf)
        best = candidate_critical.reshape(num_layers, -1).argmin(dim=1)
        selected_experts = best // num_ranks
        selected_ranks = best % num_ranks
        if torch.isinf(candidate_critical[layer_indices, selected_experts, selected_ranks]).any():
            raise RuntimeError("EPLB planner found no valid redundant expert placement")

        selected_locations = locations[layer_indices, selected_experts]
        selected_old = old_share[:, layer_indices, selected_experts]
        selected_new = new_share[:, layer_indices, selected_experts]
        rank_load += (selected_new - selected_old).unsqueeze(-1) * selected_locations.unsqueeze(0)
        rank_load += selected_new.unsqueeze(-1) * torch.nn.functional.one_hot(selected_ranks, num_ranks).unsqueeze(0)
        slot = redundant_experts_per_rank - remaining_slots[layer_indices, selected_ranks]
        placement[layer_indices, selected_ranks, slot] = selected_experts
        locations[layer_indices, selected_experts, selected_ranks] = True
        replica_count[layer_indices, selected_experts] += 1
        remaining_slots[layer_indices, selected_ranks] -= 1

    assert torch.all(placement >= 0)
    return placement
