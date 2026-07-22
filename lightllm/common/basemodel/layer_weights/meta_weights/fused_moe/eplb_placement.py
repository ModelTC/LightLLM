from typing import Dict, Tuple

import torch


EPLB_IMBALANCE_THRESHOLD = 1.05
EPLB_REBALANCE_GAIN_THRESHOLD = 0.05


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
    source_rank: int | None = None,
    node_world_size: int | None = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Build logical-to-physical maps for fixed primary and redundant slots.

    With a source rank, prefer copies on that source node.  If an expert has
    no local copy, retain every global copy as the fallback.
    """
    assert redundant_expert_ids.ndim == 2
    num_ranks, redundant_experts_per_rank = redundant_expert_ids.shape
    assert num_logical_experts % num_ranks == 0
    if source_rank is not None:
        assert 0 <= source_rank < num_ranks
        assert node_world_size is not None and 0 < node_world_size <= num_ranks
        assert num_ranks % node_world_size == 0
    experts_per_rank = num_logical_experts // num_ranks
    physical_experts_per_rank = experts_per_rank + redundant_experts_per_rank
    max_replicas = num_ranks

    global_logical_to_physical = torch.full(
        (num_logical_experts, max_replicas),
        -1,
        dtype=torch.int64,
    )
    global_replica_count = torch.ones((num_logical_experts,), dtype=torch.int64)

    for expert_id in range(num_logical_experts):
        owner_rank = expert_id // experts_per_rank
        local_slot = expert_id % experts_per_rank
        global_logical_to_physical[expert_id, 0] = owner_rank * physical_experts_per_rank + local_slot

    for rank in range(num_ranks):
        assert torch.unique(redundant_expert_ids[rank]).numel() == redundant_experts_per_rank
        for slot in range(redundant_experts_per_rank):
            expert_id = int(redundant_expert_ids[rank, slot].item())
            assert 0 <= expert_id < num_logical_experts
            replica_index = int(global_replica_count[expert_id].item())
            assert replica_index < max_replicas, "an expert can have at most one replica per rank"
            physical_id = rank * physical_experts_per_rank + experts_per_rank + slot
            global_logical_to_physical[expert_id, replica_index] = physical_id
            global_replica_count[expert_id] += 1

    if source_rank is None:
        return global_logical_to_physical, global_replica_count

    logical_to_physical = torch.full_like(global_logical_to_physical, -1)
    replica_count = torch.empty_like(global_replica_count)
    source_node = source_rank // node_world_size
    for expert_id in range(num_logical_experts):
        global_count = int(global_replica_count[expert_id].item())
        replicas = global_logical_to_physical[expert_id, :global_count]
        replica_ranks = torch.div(replicas, physical_experts_per_rank, rounding_mode="floor")
        local_replicas = replicas[torch.div(replica_ranks, node_world_size, rounding_mode="floor") == source_node]
        selected = local_replicas if local_replicas.numel() else replicas
        logical_to_physical[expert_id, : selected.numel()] = selected
        replica_count[expert_id] = selected.numel()

    return logical_to_physical, replica_count


def estimate_rank_load(
    expert_load: torch.Tensor,
    redundant_expert_ids: torch.Tensor,
    expert_alignment: int | None = None,
    node_world_size: int | None = None,
) -> torch.Tensor:
    """Estimate runtime source-node-local routing load per physical expert.

    ``expert_load`` accepts the historic ``[layers, experts]`` and
    ``[samples, layers, experts]`` forms, which are both one source node, and
    the distributed ``[samples, layers, source_nodes, experts]`` form.  Source
    loads are kept separate until they are assigned to physical replicas, then
    combined before applying the per-expert alignment used by DeepEP.
    """
    source_load, squeeze_sample, node_world_size = _as_source_node_load(
        expert_load, redundant_expert_ids.shape[1], node_world_size
    )
    num_samples, num_layers, num_nodes, num_logical_experts = source_load.shape
    assert redundant_expert_ids.ndim == 3 and redundant_expert_ids.shape[0] == num_layers
    num_ranks, redundant_experts_per_rank = redundant_expert_ids.shape[1:]
    assert num_logical_experts % num_ranks == 0
    if expert_alignment is not None:
        assert expert_alignment > 0

    locations = _expert_locations(redundant_expert_ids, num_logical_experts)
    route = _source_route(locations, num_nodes, node_world_size)
    physical_load = torch.einsum("slne,lner->sler", source_load.to(torch.float64), route)
    if expert_alignment is not None:
        physical_load = torch.ceil(physical_load / expert_alignment) * expert_alignment
    rank_load = physical_load.sum(dim=2)
    return rank_load.squeeze(0) if squeeze_sample else rank_load


def select_improving_placements(
    expert_load: torch.Tensor,
    current_placement: torch.Tensor,
    candidate_placement: torch.Tensor,
    expert_alignment: int | None = None,
    node_world_size: int | None = None,
    rebalance_gain_threshold: float = EPLB_REBALANCE_GAIN_THRESHOLD,
) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, float | int]]:
    """Select strictly better layers, gated by full-model imbalance and gain."""
    if not 0.0 <= rebalance_gain_threshold <= 1.0:
        raise ValueError("rebalance_gain_threshold must be between 0.0 and 1.0")
    assert current_placement.shape == candidate_placement.shape
    current_rank_load = estimate_rank_load(expert_load, current_placement, expert_alignment, node_world_size)
    candidate_rank_load = estimate_rank_load(expert_load, candidate_placement, expert_alignment, node_world_size)
    if expert_load.ndim == 2:
        current_critical = current_rank_load.max(dim=1).values
        candidate_critical = candidate_rank_load.max(dim=1).values
    else:
        current_critical = current_rank_load.max(dim=2).values.sum(dim=0)
        candidate_critical = candidate_rank_load.max(dim=2).values.sum(dim=0)
    # A changed layer must reduce its own aggregate critical load. The
    # fixed imbalance gate and configurable gain gate are evaluated only
    # after all changes are combined, so a hot individual layer cannot churn the whole model.
    improved = candidate_critical < current_critical
    selected = current_placement.clone()
    selected[improved] = candidate_placement[improved]
    selected_rank_load = estimate_rank_load(expert_load, selected, expert_alignment, node_world_size)
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
    candidate_rebalance_gain = (model_current_critical - model_selected_critical) / model_current_critical.clamp_min(
        1.0
    )
    metrics = {
        "model_imbalance_ratio": float(model_ratio.item()),
        "candidate_model_imbalance_ratio": float(candidate_model_ratio.item()),
        "candidate_rebalance_gain": float(candidate_rebalance_gain.item()),
        "candidate_changed_layer_count": int(improved.sum().item()),
    }
    if model_ratio >= EPLB_IMBALANCE_THRESHOLD and candidate_rebalance_gain >= rebalance_gain_threshold:
        return selected, improved, metrics
    return current_placement.clone(), torch.zeros_like(improved), metrics


def plan_redundant_experts(
    expert_load: torch.Tensor,
    num_ranks: int,
    redundant_experts_per_rank: int,
    expert_alignment: int | None = None,
    node_world_size: int | None = None,
) -> torch.Tensor:
    """Plan replicas using source-node-local copies, with global fallback."""
    assert expert_load.ndim in (2, 3, 4)
    if expert_alignment is not None:
        assert expert_alignment > 0
    use_legacy_topology_preference = expert_load.ndim < 4
    legacy_node_world_size = node_world_size if use_legacy_topology_preference else None
    source_load, _squeeze_sample, node_world_size = _as_source_node_load(
        expert_load, num_ranks, node_world_size
    )
    num_samples, num_layers, num_nodes, num_logical_experts = source_load.shape
    assert num_logical_experts % num_ranks == 0
    assert redundant_experts_per_rank > 0
    experts_per_rank = num_logical_experts // num_ranks
    num_redundant = num_ranks * redundant_experts_per_rank
    assert num_redundant <= num_logical_experts * (num_ranks - 1)

    load = source_load.to(dtype=torch.float64, device="cpu")
    placement = torch.full((num_layers, num_ranks, redundant_experts_per_rank), -1, dtype=torch.int64)
    owner_rank = torch.arange(num_logical_experts, dtype=torch.int64) // experts_per_rank

    locations = _expert_locations(placement, num_logical_experts)
    expert_rank = _expert_rank_load_all(load, locations, num_nodes, node_world_size, expert_alignment)
    rank_load = expert_rank.sum(dim=2)
    remaining_slots = torch.full((num_layers, num_ranks), redundant_experts_per_rank, dtype=torch.int64)
    layer_indices = torch.arange(num_layers, dtype=torch.int64)
    rank_nodes = (
        torch.arange(num_ranks, dtype=torch.int64) // legacy_node_world_size
        if legacy_node_world_size is not None and legacy_node_world_size < num_ranks
        else None
    )

    # Every iteration fills one slot per layer.  Candidate expert evaluation
    # is vectorized across all layers and logical experts, which keeps large
    # GLM/Qwen planning comfortably on the CPU fast path.
    for _ in range(num_redundant):
        rank_order = torch.argsort(rank_load.sum(dim=0), dim=1, stable=True)
        target_ranks = torch.full((num_layers,), -1, dtype=torch.int64)
        legal = torch.zeros((num_layers, num_logical_experts), dtype=torch.bool)
        for layer in range(num_layers):
            for target_rank in rank_order[layer].tolist():
                if remaining_slots[layer, target_rank] == 0:
                    continue
                candidate_legal = (owner_rank != target_rank) & ~locations[layer, :, target_rank]
                # Legacy 2D/3D callers have no source-node axis.  Retain the
                # previous topology preference for that compatibility path;
                # node-aware [S,L,N,E] planning uses only the exact load
                # objective below.
                if rank_nodes is not None:
                    existing_on_target_node = locations[layer, :, rank_nodes == rank_nodes[target_rank]].any(dim=1)
                    new_node_legal = candidate_legal & ~existing_on_target_node
                    if torch.any(new_node_legal):
                        candidate_legal = new_node_legal
                if torch.any(candidate_legal):
                    target_ranks[layer] = target_rank
                    legal[layer] = candidate_legal
                    break
        if torch.any(target_ranks < 0):
            raise RuntimeError("EPLB planner found no valid redundant expert placement")

        candidate_locations = locations.clone()
        candidate_locations[layer_indices[:, None], torch.arange(num_logical_experts)[None, :], target_ranks[:, None]] = True
        candidate_expert_rank = _expert_rank_load_all(
            load, candidate_locations, num_nodes, node_world_size, expert_alignment
        )
        candidate_rank_load = rank_load[:, :, None, :] - expert_rank + candidate_expert_rank
        critical = candidate_rank_load.max(dim=3).values.sum(dim=0)
        critical.masked_fill_(~legal, torch.inf)
        selected_experts = critical.argmin(dim=1)
        if torch.isinf(critical[layer_indices, selected_experts]).any():
            raise RuntimeError("EPLB planner found no valid redundant expert placement")

        slots = redundant_experts_per_rank - remaining_slots[layer_indices, target_ranks]
        placement[layer_indices, target_ranks, slots] = selected_experts
        selected_next = candidate_expert_rank[:, layer_indices, selected_experts]
        selected_old = expert_rank[:, layer_indices, selected_experts]
        rank_load += selected_next - selected_old
        expert_rank[:, layer_indices, selected_experts] = selected_next
        locations[layer_indices, selected_experts, target_ranks] = True
        remaining_slots[layer_indices, target_ranks] -= 1

    assert torch.all(placement >= 0)
    return placement


def _as_source_node_load(
    expert_load: torch.Tensor, num_ranks: int, node_world_size: int | None
) -> Tuple[torch.Tensor, bool, int]:
    """Normalize load to ``[samples, layers, source_nodes, experts]``."""
    assert expert_load.ndim in (2, 3, 4)
    squeeze_sample = expert_load.ndim == 2
    if expert_load.ndim == 2:
        source_load = expert_load.unsqueeze(0).unsqueeze(2)
    elif expert_load.ndim == 3:
        source_load = expert_load.unsqueeze(2)
    else:
        source_load = expert_load
    num_nodes = source_load.shape[2]
    # Historic 2D/3D loads represent one source node containing every rank.
    if expert_load.ndim < 4:
        return source_load, squeeze_sample, num_ranks
    if node_world_size is None:
        assert num_ranks % num_nodes == 0
        node_world_size = num_ranks // num_nodes
    assert 0 < node_world_size <= num_ranks and num_ranks % node_world_size == 0
    assert num_nodes == num_ranks // node_world_size
    return source_load, squeeze_sample, node_world_size


def _expert_locations(redundant_expert_ids: torch.Tensor, num_logical_experts: int) -> torch.Tensor:
    """Return ``[layer, logical expert, rank]`` physical-copy occupancy."""
    num_layers, num_ranks, redundant_experts_per_rank = redundant_expert_ids.shape
    assert num_logical_experts % num_ranks == 0
    experts_per_rank = num_logical_experts // num_ranks
    locations = torch.zeros(
        (num_layers, num_logical_experts, num_ranks),
        dtype=torch.bool,
        device=redundant_expert_ids.device,
    )
    expert_ids = torch.arange(num_logical_experts, device=locations.device)
    owners = expert_ids // experts_per_rank
    locations[:, expert_ids, owners] = True
    layers = torch.arange(num_layers, device=locations.device)[:, None]
    ranks = torch.arange(num_ranks, device=locations.device).repeat_interleave(redundant_experts_per_rank)[None, :]
    redundant_ids = redundant_expert_ids.reshape(num_layers, -1)
    valid = redundant_ids >= 0
    if torch.any(valid):
        expanded_layers = layers.expand_as(redundant_ids)
        expanded_ranks = ranks.expand_as(redundant_ids)
        locations[
            expanded_layers[valid],
            redundant_ids[valid],
            expanded_ranks[valid],
        ] = True
    return locations


def _source_route(slots: torch.Tensor, num_nodes: int, node_world_size: int) -> torch.Tensor:
    """Route each source node to its local copies, or all copies as fallback."""
    num_ranks = slots.shape[-1]
    assert num_ranks % node_world_size == 0 and num_nodes == num_ranks // node_world_size
    rank_nodes = torch.arange(num_ranks, device=slots.device) // node_world_size
    source_nodes = torch.arange(num_nodes, device=slots.device)
    copies = slots.unsqueeze(-3).expand(*slots.shape[:-2], num_nodes, *slots.shape[-2:])
    rank_node_shape = (1,) * slots.ndim + (num_ranks,)
    source_node_shape = (1,) * (slots.ndim - 2) + (num_nodes, 1, 1)
    local = copies & (rank_nodes.reshape(rank_node_shape) == source_nodes.reshape(source_node_shape))
    selected = torch.where(local.any(dim=-1, keepdim=True), local, copies)
    return selected.to(torch.float64) / selected.sum(dim=-1, keepdim=True)


def _expert_rank_load(
    source_load: torch.Tensor,
    slots: torch.Tensor,
    num_nodes: int,
    node_world_size: int,
    expert_alignment: int | None,
) -> torch.Tensor:
    """Return aligned contribution ``[samples, expert, rank]`` for one layer."""
    route = _source_route(slots, num_nodes, node_world_size)
    physical_load = torch.einsum("sne,ner->ser", source_load.to(torch.float64), route)
    if expert_alignment is not None:
        physical_load = torch.ceil(physical_load / expert_alignment) * expert_alignment
    return physical_load


def _expert_rank_load_all(
    source_load: torch.Tensor,
    locations: torch.Tensor,
    num_nodes: int,
    node_world_size: int,
    expert_alignment: int | None,
) -> torch.Tensor:
    """Return aligned ``[samples, layers, expert, rank]`` contributions."""
    route = _source_route(locations, num_nodes, node_world_size)
    physical_load = torch.einsum("slne,lner->sler", source_load.to(torch.float64), route)
    if expert_alignment is not None:
        physical_load = torch.ceil(physical_load / expert_alignment) * expert_alignment
    return physical_load
