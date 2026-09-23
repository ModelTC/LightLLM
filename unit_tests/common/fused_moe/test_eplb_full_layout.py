"""Offline EPLB planning and publication regressions; no model/service startup."""
import itertools
import math
from types import SimpleNamespace

import pytest
import torch

from lightllm.common.basemodel.layer_weights.meta_weights.fused_moe.eplb_placement import (
    _estimate_rank_load,
    build_initial_redundant_expert_ids,
    build_logical_to_physical_maps_for_layers,
    expand_redundant_placement,
    plan_redundant_experts,
    select_improving_placements,
    validate_physical_placement,
)
from lightllm.common.basemodel.layer_weights.meta_weights.fused_moe.eplb_planner import (
    plan_full_experts,
    refine_placement_candidates,
)
from lightllm.common.eplb_utils import get_eplb_staging_shape
from lightllm.server.router.model_infer.mode_backend.eplb_transfer import (
    _commit_staging_rows,
    align_target_placement,
    build_transfer_plan,
)


@pytest.fixture(scope="module", autouse=True)
def single_cpu_thread():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


def _initial(redundant=1, layers=1):
    extras = build_initial_redundant_expert_ids(8, 4, redundant)[None].expand(layers, -1, -1).clone()
    return expand_redundant_placement(extras, 8)


def _independent_rank_load(source_load, physical, alignment, node_size):
    # Slow scalar oracle: merge sources at each physical expert BEFORE padding.
    rows = physical.tolist()
    result = [0.0] * len(rows)
    for expert in range(len(source_load[0])):
        owners = [rank for rank, row in enumerate(rows) if expert in row]
        received = [0.0] * len(rows)
        for node, loads in enumerate(source_load):
            local = [rank for rank in owners if rank // node_size == node]
            selected = local or owners
            for rank in selected:
                received[rank] += loads[expert] / len(selected)
        for rank, count in enumerate(received):
            result[rank] += math.ceil(count / alignment) * alignment
    return result


@pytest.mark.parametrize("redundant", [0, 1, 2])
@pytest.mark.parametrize("node_size", [1, 2, 4])
def test_full_maps_and_estimator_match_independent_physical_layout(redundant, node_size):
    layout = _initial(redundant, layers=2)
    # Move original experts: the map must use absolute rows, never E/W owners.
    layout[0] = layout[0].roll(1, dims=0)
    layout[1] = layout[1].flip(dims=(0, 1))
    capacity = layout.shape[-1]
    for source_rank in range(4):
        maps, counts = build_logical_to_physical_maps_for_layers(
            layout, 8, source_rank=source_rank, node_world_size=node_size, full_layout=True
        )
        assert maps.shape == (2, 8, 4)
        for layer, expert in itertools.product(range(2), range(8)):
            candidates = [index for index, value in enumerate(layout[layer].flatten().tolist()) if value == expert]
            local = [index for index in candidates if index // capacity // node_size == source_rank // node_size]
            expected = local or candidates
            chosen = maps[layer, expert, : counts[layer, expert]].tolist()
            assert set(chosen) == set(expected)
            assert chosen[0] == expected[source_rank % len(expected)]
            assert maps[layer, expert, counts[layer, expert] :].eq(-1).all()
    generator = torch.Generator().manual_seed(42)
    loads = torch.randint(0, 1024, (3, 2, 4 // node_size, 8), generator=generator)
    actual = _estimate_rank_load(loads, layout, 128, node_size, full_layout=True)
    for sample, layer in itertools.product(range(3), range(2)):
        expected = _independent_rank_load(loads[sample, layer].tolist(), layout[layer], 128, node_size)
        torch.testing.assert_close(actual[sample, layer], torch.tensor(expected, dtype=torch.float64))


@pytest.mark.parametrize(
    "layout,message",
    [
        ([[[0, 0], [1, 2]]], "duplicate"),
        ([[[0, 1], [1, 2]]], "cover"),
        ([[[0, 1], [2, 4]]], "invalid"),
        ([[[0, 1], [2, -1]]], "invalid"),
    ],
)
def test_full_layout_rejects_unrepresentable_routes(layout, message):
    with pytest.raises(ValueError, match=message):
        build_logical_to_physical_maps_for_layers(torch.tensor(layout), 4, full_layout=True)


def test_refinement_closes_known_fixed_primary_greedy_gap():
    loads = torch.tensor([640, 512, 1536, 3584, 3840, 8320, 9984, 512]).reshape(1, 1, 1, 8)
    current = build_initial_redundant_expert_ids(8, 4, 1)[None]
    greedy = plan_redundant_experts(loads, 4, 1, 128, current_placement=current, stickiness=0.1)
    unbiased = plan_redundant_experts(loads, 4, 1, 128)
    refined = refine_placement_candidates(loads, current, [greedy, unbiased], expert_alignment=128, stickiness=0.1)
    # Exhaust the 6^4 feasible layouts with an independent scalar oracle.
    legal = [[expert for expert in range(8) if expert // 2 != rank] for rank in range(4)]
    optimum = min(
        max(
            _independent_rank_load(
                loads[0, 0].tolist(),
                torch.tensor([[2 * rank, 2 * rank + 1, choice[rank]] for rank in range(4)]),
                128,
                4,
            )
        )
        for choice in itertools.product(*legal)
    )
    assert optimum == 7936
    assert _estimate_rank_load(loads, refined, 128).max() == optimum
    assert _estimate_rank_load(loads, greedy, 128).max() > optimum
    assert torch.equal(
        refined, refine_placement_candidates(loads, current, [greedy, unbiased], expert_alignment=128, stickiness=0.1)
    )


@pytest.mark.parametrize("redundant", [0, 1, 2])
@pytest.mark.parametrize("node_size", [1, 2, 4])
def test_full_planner_preserves_topology_and_never_worsens_samplewise_score(redundant, node_size):
    current = _initial(redundant, layers=2)
    generator = torch.Generator().manual_seed(1337 + node_size + redundant)
    load = torch.randint(0, 10000, (4, 2, 4 // node_size, 8), generator=generator)
    planned = plan_full_experts(load, current, expert_alignment=128, node_world_size=node_size)
    validate_physical_placement(planned, 8)
    assert planned.shape == current.shape
    for layer, node in itertools.product(range(2), range(4 // node_size)):
        ranks = slice(node * node_size, (node + 1) * node_size)
        assert set(current[layer, ranks].flatten().tolist()) == set(planned[layer, ranks].flatten().tolist())
    old = _estimate_rank_load(load, current, 128, node_size, True).max(dim=-1).values.sum(dim=0)
    new = _estimate_rank_load(load, planned, 128, node_size, True).max(dim=-1).values.sum(dim=0)
    assert torch.all(new <= old)
    assert torch.equal(current, _initial(redundant, layers=2))


def test_uniform_workload_keeps_current_layout_and_zero_replicas_can_rebalance():
    current = _initial(0)
    uniform = torch.full((2, 1, 1, 8), 12800)
    assert torch.equal(plan_full_experts(uniform, current, expert_alignment=128), current)
    skewed = torch.tensor([12800, 12800, 128, 128, 12800, 12800, 128, 128]).reshape(1, 1, 1, 8)
    candidate = plan_full_experts(skewed, current, expert_alignment=128)
    placement, improved, metrics, before, after = select_improving_placements(
        skewed, current, candidate, rebalance_gain_threshold=0.05, expert_alignment=128, full_layout=True
    )
    assert improved.all() and metrics["candidate_rebalance_gain"] > 0.4
    assert after.max() < before.max()
    assert placement.numel() == 8


def test_refinement_keeps_each_best_seed_layer_and_respects_zero_budget():
    current = build_initial_redundant_expert_ids(8, 4, 1)[None].expand(2, -1, -1).clone()
    load = torch.randint(1, 10000, (4, 2, 1, 8), generator=torch.Generator().manual_seed(12))
    greedy = plan_redundant_experts(load, 4, 1, 128)
    result = refine_placement_candidates(load, current, [greedy], expert_alignment=128, search_steps=0)
    cost = lambda placement: _estimate_rank_load(load, placement, 128).max(dim=-1).values.sum(dim=0)
    torch.testing.assert_close(cost(result), torch.minimum(cost(current), cost(greedy)))
    with pytest.raises(ValueError, match="budget"):
        refine_placement_candidates(load, current, [greedy], candidate_budget=0)


def test_full_candidate_cannot_change_node_holdings():
    current = _initial(0)
    with pytest.raises(ValueError, match="node's expert set"):
        refine_placement_candidates(
            torch.ones((1, 1, 2, 8)), current, [current.roll(1, dims=1)], node_world_size=2, full_layout=True
        )


def test_absolute_row_transfer_handles_overwrite_cycle_and_second_generation():
    current = torch.arange(6).reshape(3, 2)
    live = [current[rank, :, None].to(torch.float32).repeat(1, 5) for rank in range(3)]
    scales = [row[:, :1].clone() + 0.5 for row in live]
    for shift in (1, -1):
        target = align_target_placement(current, current.roll(shift, dims=0))
        plan = build_transfer_plan(current, target, 6, 3, 3, full_layout=True)
        assert len(plan) == 6  # all primary rows move, even with R=0
        staged_weights = [torch.empty_like(row) for row in live]
        staged_scales = [torch.empty_like(row) for row in scales]
        for step in plan:
            staged_weights[step.dst_rank][step.dst_slot].copy_(live[step.src_rank][step.src_local_row])
            staged_scales[step.dst_rank][step.dst_slot].copy_(scales[step.src_rank][step.src_local_row])
        for rank in range(3):
            slots = [step.dst_slot for step in plan if step.dst_rank == rank]
            _commit_staging_rows(live[rank], staged_weights[rank], 0, slots)
            _commit_staging_rows(scales[rank], staged_scales[rank], 0, slots)
            torch.testing.assert_close(live[rank][:, 0], target[rank].float())
            torch.testing.assert_close(scales[rank][:, 0], target[rank].float() + 0.5)
        current = target


def test_full_transfer_retains_slots_and_uses_sources_from_current_layout():
    current = torch.tensor([[2, 1, 4, 5], [3, 4, 0, 5]])
    target = torch.tensor([[4, 1, 0, 5], [2, 3, 4, 5]])
    aligned = align_target_placement(current, target)
    plan = build_transfer_plan(current, target, 6, 2, 2, full_layout=True)
    assert len(plan) == 2
    for step in plan:
        assert current[step.src_rank, step.src_local_row] == aligned[step.dst_rank, step.dst_slot]


@pytest.mark.parametrize("full,expected", [(True, (1, 34)), (False, (8, 2))])
def test_staging_capacity_is_shared_with_memory_profiling(full, expected):
    state = SimpleNamespace(full_layout=full, num_redundant_experts_per_rank=2)
    weight = SimpleNamespace(expert_parallel_state=SimpleNamespace(eplb=state, num_primary_experts_per_rank=32))
    assert get_eplb_staging_shape([weight] * 40) == expected


def test_manager_full_plan_is_canonical_and_publication_tracks_partial_generations(monkeypatch):
    from lightllm.server.router.model_infer.mode_backend import eplb_manager as module
    from lightllm.common.basemodel.layer_weights.meta_weights.fused_moe.expert_parallel_state import EPLBState

    manager = module.EPLBManager.__new__(module.EPLBManager)
    manager.global_rank, manager.world_size, manager.node_world_size = 0, 4, 4
    manager.num_logical_experts, manager.num_redundant_experts_per_rank = 8, 0
    manager.full_layout = True
    manager.current_placement = _initial(0, layers=2)
    original = manager.current_placement.clone()
    manager.placement_stickiness, manager.rebalance_gain_threshold = 0.1, 0.05
    manager.evaluation_group = object()
    broadcasts = []
    monkeypatch.setattr(module.dist, "broadcast_object_list", lambda result, **kwargs: broadcasts.append(result[0]))
    loads = torch.tensor([12800, 12800, 128, 128, 12800, 12800, 128, 128]).reshape(1, 1, 1, 8).expand(1, 2, 1, 8)
    result = manager._plan_and_broadcast(loads)
    assert result["kind"] == "planned" and broadcasts[0] is result
    assert torch.equal(manager.current_placement, original)
    for current, target in zip(original, result["placement"]):
        assert torch.equal(target, align_target_placement(current, target))
    maps, counts = build_logical_to_physical_maps_for_layers(original, 8, full_layout=True)
    manager._eplb_states = [
        EPLBState(
            0,
            torch.empty((4, 0), dtype=torch.int64),
            maps[layer].clone(),
            counts[layer].clone(),
            torch.zeros((1, 8)),
            full_layout=True,
            physical_to_logical=original[layer].clone(),
        )
        for layer in range(2)
    ]
    targets, target_counts = build_logical_to_physical_maps_for_layers(result["placement"], 8, full_layout=True)
    manager.target_placement = result["placement"]
    manager.target_metadata = list(zip(targets, target_counts))
    pointers = [state.logical_to_physical_map.data_ptr() for state in manager._eplb_states]
    manager._commit_layer_metadata(1)
    assert [state.placement_generation for state in manager._eplb_states] == [0, 1]
    assert torch.equal(manager.current_placement[0], original[0])
    assert torch.equal(manager.current_placement[1], manager.target_placement[1])
    manager._commit_layer_metadata(0)
    for layer, state in enumerate(manager._eplb_states):
        assert state.placement_generation == 1
        assert state.logical_to_physical_map.data_ptr() == pointers[layer]
        torch.testing.assert_close(state.logical_to_physical_map, targets[layer])
        torch.testing.assert_close(state.physical_to_logical, manager.target_placement[layer])


def test_full_checkpoint_partition_uses_absolute_layout_rows(monkeypatch):
    from lightllm.common.basemodel.layer_weights.meta_weights.fused_moe import fused_moe_weight as module
    from lightllm.common.basemodel.layer_weights.meta_weights.fused_moe.expert_parallel_state import (
        EPLBState,
        ExpertParallelState,
    )

    weight = module.FusedMoeWeight.__new__(module.FusedMoeWeight)
    weight.enable_ep_moe = True
    weight.global_rank_, weight.layer_num_ = 1, 0
    weight.n_routed_experts, weight.num_fused_shared_experts, weight.moe_intermediate_size = 8, 0, 32
    weight.quant_method = SimpleNamespace(method_name="test")
    weight._initial_redundant_expert_ids = [4]
    physical = _initial(1)[0].roll(1, dims=0).flip(dims=(1,))
    state = EPLBState(
        1, build_initial_redundant_expert_ids(8, 4, 1), None, None, None, full_layout=True, physical_to_logical=physical
    )
    weight.expert_parallel_state = ExpertParallelState(8, 4, state)
    monkeypatch.setattr(module, "get_row_slice_mixin", lambda *a, **kw: None)
    monkeypatch.setattr(module, "get_col_slice_mixin", lambda *a, **kw: None)
    weight._init_weight_partition()
    assert weight.local_n_routed_experts == 3
    assert weight.expert_idx_to_local_idx == {expert: slot for slot, expert in enumerate(physical[1].tolist())}
    assert weight._initial_redundant_expert_idx_to_local_idx == {}


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
@pytest.mark.parametrize("redundant", [0, 1])
def test_full_state_initialization_including_zero_replicas(monkeypatch, redundant):
    from lightllm.common.basemodel.layer_weights.meta_weights.fused_moe import fused_moe_weight as module

    args = SimpleNamespace(
        enable_prefill_eplb=True,
        eplb_placement_mode="full",
        run_mode="prefill",
        eplb_num_redundant_experts_per_rank=redundant,
    )
    weight = module.FusedMoeWeight.__new__(module.FusedMoeWeight)
    weight.n_routed_experts, weight.global_world_size, weight.global_rank_ = 8, 4, 1
    weight.enable_ep_moe = True
    monkeypatch.setattr(module, "get_env_start_args", lambda: args)
    monkeypatch.setattr(module, "get_node_world_size", lambda: 4)
    monkeypatch.setattr(module, "get_prefill_eplb_step_interval", lambda: 4)
    weight._init_expert_parallel_state()
    state = weight.expert_parallel_state.eplb
    assert state.full_layout and state.placement_generation == 0
    assert state.route_counter.shape == (8, 8)
    assert state.physical_to_logical.shape == (4, 2 + redundant)
    assert weight.expert_parallel_state.num_total_physical_experts == 8 + 4 * redundant
    for expert, count in enumerate(state.logical_replica_count.tolist()):
        for index in state.logical_to_physical_map[expert, :count].tolist():
            assert state.physical_to_logical.flatten()[index] == expert
