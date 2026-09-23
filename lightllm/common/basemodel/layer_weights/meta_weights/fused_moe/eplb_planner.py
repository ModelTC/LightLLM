"""Bounded CPU candidate search, scored with the runtime's node-local routing.

The score is samplewise aligned critical token load, not a prediction of latency.
Full layouts preserve each node's expert set; cross-node placement is deliberately
left to a future planner with a calibrated communication cost model.
"""

import torch

from .eplb_placement import (
    _estimate_rank_load,
    _expert_locations,
    _expert_rank_load_all,
    _resolve_node_world_size,
    validate_physical_placement,
)


def _critical_load(load, placement, alignment, node_world_size, full_layout):
    return _estimate_rank_load(load, placement, alignment, node_world_size, full_layout).max(dim=-1).values.sum(dim=0)


def refine_placement_candidates(
    expert_load: torch.Tensor,
    current: torch.Tensor,
    candidates: list[torch.Tensor],
    *,
    expert_alignment: int | None = None,
    node_world_size: int | None = None,
    full_layout: bool = False,
    stickiness: float = 0.0,
    search_steps: int = 4,
    candidate_budget: int = 256,
) -> torch.Tensor:
    """Keep the best layer of each seed and its bounded local search result.

    Current placement is always included and wins ties. Each search step scores
    at most ``candidate_budget`` replacements/swaps per layer. Only the two
    affected experts are re-evaluated for each move, bounding temporary memory.
    Stickiness is a minimum cumulative improvement over a search seed, in
    aligned-token units; it must not block a series of individually small moves.
    """
    if search_steps < 0 or candidate_budget <= 0 or not 0 <= stickiness <= 1:
        raise ValueError("invalid EPLB search budget or stickiness")
    node_world_size = _resolve_node_world_size(expert_load, current.shape[1], node_world_size)
    load = expert_load.to(device="cpu", dtype=torch.float64)
    best = current.cpu().clone()
    score = _critical_load(load, best, expert_alignment, node_world_size, full_layout)
    for seed in [current, *candidates]:
        if seed.shape != current.shape:
            raise ValueError("candidate capacity differs from the current placement")
        seed = seed.cpu().clone()
        if full_layout:
            validate_physical_placement(seed, load.shape[-1])
            # Preserve node holdings, including when a caller supplies a seed.
            for node_start in range(0, current.shape[1], node_world_size):
                for layer in range(current.shape[0]):
                    if set(seed[layer, node_start : node_start + node_world_size].flatten().tolist()) != set(
                        current[layer, node_start : node_start + node_world_size].flatten().tolist()
                    ):
                        raise ValueError("full EPLB candidates must preserve each node's expert set")
        for proposal in (
            seed,
            _local_search(
                load, seed, expert_alignment, node_world_size, full_layout, stickiness, search_steps, candidate_budget
            ),
        ):
            proposal_score = _critical_load(load, proposal, expert_alignment, node_world_size, full_layout)
            improved = proposal_score < score
            best[improved] = proposal[improved]
            score[improved] = proposal_score[improved]
    return best


def _local_search(load, placement, alignment, node_world_size, full_layout, stickiness, steps, budget):
    result = placement.clone()
    experts = load.shape[-1]
    for layer in range(placement.shape[0]):
        source = load[:, layer : layer + 1]
        margin = stickiness * float(source.sum()) / experts
        initial_score = _critical_load(source, placement[layer : layer + 1], alignment, node_world_size, full_layout)
        for _ in range(steps):
            rows = result[layer]
            locations = _expert_locations(rows[None], experts, full_layout)
            contributions = _expert_rank_load_all(source, locations, source.shape[2], node_world_size, alignment)[:, 0]
            rank_load = contributions.sum(dim=1)
            proposals = _propose_moves(rows, locations[0], contributions, node_world_size, full_layout, budget)
            if not proposals:
                break
            # Each move changes two distinct experts. Build only their routing
            # contributions, never a [candidates, all experts] load tensor.
            old_ids = torch.tensor([move[2] for move in proposals])
            new_ids = torch.tensor([move[3] for move in proposals])
            ids = torch.stack((old_ids, new_ids), dim=1)
            new_locations = locations[0, ids].clone()
            for index, (rank, slot, old, new, other_rank, other_slot) in enumerate(proposals):
                new_locations[index, 0, rank] = False
                new_locations[index, 1, rank] = True
                if other_rank >= 0:
                    new_locations[index, 1, other_rank] = False
                    new_locations[index, 0, other_rank] = True
            move_load = source[:, 0, :, ids].permute(0, 2, 1, 3)
            next_contributions = _expert_rank_load_all(
                move_load, new_locations, source.shape[2], node_world_size, alignment
            ).sum(dim=2)
            previous = contributions[:, ids].sum(dim=2)
            scores = (rank_load[:, None] - previous + next_contributions).max(dim=-1).values.sum(dim=0)
            winner = int(scores.argmin())
            if float(scores[winner]) >= float(rank_load.max(dim=-1).values.sum()):
                break
            rank, slot, old, new, other_rank, other_slot = proposals[winner]
            rows[rank, slot] = new
            if other_rank >= 0:
                rows[other_rank, other_slot] = old
        final_score = _critical_load(source, result[layer : layer + 1], alignment, node_world_size, full_layout)
        if final_score + margin >= initial_score:
            result[layer].copy_(placement[layer])
    return result


def _propose_moves(rows, locations, contributions, node_world_size, full_layout, budget):
    """Deterministic interleaving avoids spending the whole budget on one rank."""
    ranks, capacity = rows.shape
    rank_order = contributions.sum(dim=(0, 1)).argsort(descending=True, stable=True).tolist()
    expert_order = contributions.sum(dim=(0, 2)).argsort(descending=True, stable=True).tolist()
    occupancy = locations.tolist()
    per_copy_load = contributions.sum(dim=0).tolist()
    row_ids = rows.tolist()
    per_rank = []
    for rank in rank_order:
        moves = []
        node_start = rank // node_world_size * node_world_size
        node_end = node_start + node_world_size
        # Replacement candidates include cold redundant copies, freeing space
        # for hot experts. Last copies on a node may only move through swaps.
        for new in expert_order:
            if occupancy[new][rank]:
                continue
            if full_layout and not any(occupancy[new][node_start:node_end]):
                continue
            for slot, old in enumerate(row_ids[rank]):
                if full_layout and sum(occupancy[old][node_start:node_end]) <= 1:
                    continue
                moves.append((rank, slot, old, new, -1, -1))
                if len(moves) >= budget:
                    break
            if len(moves) >= budget:
                break
        swaps = []
        # Swap hot rows of heavy ranks with cold rows of lighter ranks. In
        # fixed-primary mode only the redundant rows are interchangeable.
        hot_slots = sorted(range(capacity), key=lambda s: -per_copy_load[row_ids[rank][s]][rank])
        for other in reversed(rank_order):
            if other == rank or (full_layout and other // node_world_size != rank // node_world_size):
                continue
            cold_slots = sorted(range(capacity), key=lambda s: per_copy_load[row_ids[other][s]][other])
            for slot in hot_slots:
                old = row_ids[rank][slot]
                if occupancy[old][other]:
                    continue
                for other_slot in cold_slots:
                    new = row_ids[other][other_slot]
                    if occupancy[new][rank]:
                        continue
                    swaps.append((rank, slot, old, new, other, other_slot))
                    if len(swaps) >= budget:
                        break
                if len(swaps) >= budget:
                    break
            if len(swaps) >= budget:
                break
        per_rank.append(
            [move for pair in zip(moves, swaps) for move in pair] + moves[len(swaps) :] + swaps[len(moves) :]
        )
    proposals = []
    for index in range(budget):
        for moves in per_rank:
            if index < len(moves):
                proposals.append(moves[index])
                if len(proposals) == budget:
                    return proposals
    return proposals


def plan_full_experts(
    expert_load: torch.Tensor,
    current: torch.Tensor,
    *,
    expert_alignment: int | None = None,
    node_world_size: int | None = None,
    stickiness: float = 0.0,
) -> torch.Tensor:
    """Allocate load-driven replica counts, pack within nodes, then refine.

    The capacity and expert set of each node stay fixed. A load-aware pack is
    attempted first; a largest-remaining-capacity pack guarantees feasibility
    for equal per-rank capacities and replica counts bounded by node size.
    """
    validate_physical_placement(current, expert_load.shape[-1])
    node_world_size = _resolve_node_world_size(expert_load, current.shape[1], node_world_size)
    load = expert_load.to(device="cpu", dtype=torch.float64)
    locations = _expert_locations(current, load.shape[-1], full_layout=True)
    # Sum unpadded traffic reaching a node under the current node-local policy.
    traffic = _expert_rank_load_all(load, locations, load.shape[2], node_world_size, None).sum(dim=0)
    candidate = current.clone()
    for layer in range(current.shape[0]):
        for start in range(0, current.shape[1], node_world_size):
            end = start + node_world_size
            ids = sorted(set(current[layer, start:end].flatten().tolist()))
            weights = traffic[layer, :, start:end].sum(dim=-1).tolist()
            counts = {expert: 1 for expert in ids}
            for _ in range(node_world_size * current.shape[2] - len(ids)):
                legal = [expert for expert in ids if counts[expert] < node_world_size]
                expert = max(legal, key=lambda e: (weights[e] / counts[e], -e))
                counts[expert] += 1
            packed = _pack_node(counts, weights, node_world_size, current.shape[2], prefer_load=True)
            if packed is None:
                packed = _pack_node(counts, weights, node_world_size, current.shape[2], prefer_load=False)
            assert packed is not None
            candidate[layer, start:end] = torch.tensor(packed, dtype=current.dtype)
    return refine_placement_candidates(
        load,
        current,
        [candidate],
        expert_alignment=expert_alignment,
        node_world_size=node_world_size,
        full_layout=True,
        stickiness=stickiness,
    )


def _pack_node(counts, weights, ranks, capacity, *, prefer_load):
    rows = [[] for _ in range(ranks)]
    loads = [0.0] * ranks
    order = sorted(counts, key=lambda e: (-weights[e] / counts[e], -counts[e], e))
    for expert in order:
        available = [rank for rank in range(ranks) if len(rows[rank]) < capacity]
        if len(available) < counts[expert]:
            return None
        available.sort(key=lambda r: (loads[r], len(rows[r]), r) if prefer_load else (len(rows[r]), loads[r], r))
        for rank in available[: counts[expert]]:
            rows[rank].append(expert)
            loads[rank] += weights[expert] / counts[expert]
    return rows
