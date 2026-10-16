"""Replica routing must save aligned critical work without changing logical experts."""
import threading
from types import SimpleNamespace

import pytest
import torch

from lightllm.common.basemodel.layer_weights.meta_weights.fused_moe.eplb_placement import (
    _estimate_rank_load,
    build_initial_redundant_expert_ids,
    build_logical_to_physical_maps_for_layers,
    expand_redundant_placement,
    select_improving_placements,
    select_replica_routing,
)
from lightllm.common.basemodel.layer_weights.meta_weights.fused_moe.expert_parallel_state import EPLBState
from lightllm.server.router.model_infer.mode_backend.eplb_manager import EPLBManager


def _placement(layers=1):
    return expand_redundant_placement(build_initial_redundant_expert_ids(288, 4, 2), 288)[None].repeat(layers, 1, 1)


def test_multinode_routing_keeps_existing_source_local_policy():
    placement = _placement()
    loads = torch.ones((2, 1, 2, 288))
    expected = _estimate_rank_load(loads, placement, 128, 2)
    policy, actual = select_replica_routing(loads, placement, expected, 128, 2)
    assert policy is None
    torch.testing.assert_close(actual, expected)


def test_uniform_padding_and_hot_experts_choose_different_policies():
    placement = _placement(2)
    load = torch.zeros((4, 2, 1, 288), dtype=torch.int64)
    load[:, 0] = 32
    load[:, 1, :, 0] = 8192
    policy, actual = select_replica_routing(load, placement, expert_alignment=128, node_world_size=4)
    assert policy.tolist() == [False, True]
    torch.testing.assert_close(actual[:, 0], torch.full((4, 4), 9216.0, dtype=torch.float64))
    assert actual[:, 1].max().item() == 4096
    # Policy changes need no weight migration or lowered rebalance threshold.
    selected, improved, metrics, before, after = select_improving_placements(
        load,
        placement,
        placement,
        expert_alignment=128,
        node_world_size=4,
        rebalance_gain_threshold=0.05,
        adaptive_replica_routing=True,
    )
    assert not improved.any()
    assert metrics["replica_routing"].tolist() == [False, True]
    torch.testing.assert_close(selected, placement)
    torch.testing.assert_close(before, actual)
    torch.testing.assert_close(after, actual)


@pytest.mark.parametrize("migrated", [False, True])
def test_collapsed_maps_use_live_rows_and_preserve_one_expert_per_route(migrated):
    placement = _placement(2)
    if migrated:
        placement = placement.flip(1).roll(11, dims=2)
    for rank in range(4):
        maps, counts = build_logical_to_physical_maps_for_layers(
            placement,
            288,
            source_rank=rank,
            node_world_size=4,
            replica_routing=torch.tensor([False, True]),
        )
        assert counts[0].eq(0).all()
        original_maps, _ = build_logical_to_physical_maps_for_layers(
            placement, 288, source_rank=rank, node_world_size=4
        )
        torch.testing.assert_close(maps[:, :, 0], original_maps[:, :, 0], rtol=0, atol=0)
        for layer in range(2):
            for expert in range(288):
                ids = (maps[layer, expert, -1:] if layer == 0 else maps[layer, expert, : counts[layer, expert]]).long()
                assert placement[layer].flatten()[ids].eq(expert).all()
        if not migrated:
            torch.testing.assert_close(maps[0, :, -1] // 74, torch.arange(288, dtype=torch.int32) // 72)


def test_policy_only_publication_orders_copies_without_migrating(monkeypatch):
    from lightllm.server.router.model_infer import infer_batch

    manager = EPLBManager.__new__(EPLBManager)
    placement = _placement()
    old_maps, old_counts = build_logical_to_physical_maps_for_layers(placement, 288)
    maps, counts = build_logical_to_physical_maps_for_layers(
        placement,
        288,
        node_world_size=4,
        replica_routing=torch.tensor([False]),
    )
    state = EPLBState(2, old_maps[0], old_counts[0], torch.zeros((1, 288), dtype=torch.int64))
    manager._eplb_states = [state]
    ordered = []
    monkeypatch.setattr(
        torch.cuda, "current_stream", lambda: SimpleNamespace(wait_stream=lambda stream: ordered.append(stream))
    )
    monkeypatch.setattr(infer_batch.g_infer_context, "get_overlap_stream", lambda: "previous_forward")
    manager._publish_unchanged_routing({"improved": torch.tensor([False]), "metadata": [(maps[0], counts[0], False)]})
    assert ordered == ["previous_forward"]
    assert state.replica_routing is False
    assert state.logical_replica_count.eq(0).all()
    assert state.placement_generation == 0
    # A changed-weight layer must keep its old policy until the migration commits.
    manager._publish_unchanged_routing(
        {"improved": torch.tensor([True]), "metadata": [(old_maps[0], old_counts[0], True)]}
    )
    assert state.replica_routing is False
    assert len(ordered) == 1


def test_small_hot_window_updates_policy_without_planning_or_migration(monkeypatch):
    from lightllm.server.router.model_infer import infer_batch
    from lightllm.server.router.model_infer.mode_backend import eplb_manager as module

    placement = _placement(2)
    maps, counts = build_logical_to_physical_maps_for_layers(
        placement, 288, node_world_size=4, replica_routing=torch.tensor([False, True])
    )
    states = [
        EPLBState(2, maps[i], counts[i], torch.zeros((40, 288), dtype=torch.int64), replica_routing=bool(i))
        for i in range(2)
    ]
    manager = EPLBManager.__new__(EPLBManager)
    manager.__dict__.update(
        weights=[None, None],
        _eplb_states=states,
        global_rank=0,
        world_size=4,
        node_world_size=4,
        num_logical_experts=288,
        rebalance_gain_threshold=0.05,
        current_placement=placement.clone(),
        evaluation_group=object(),
        _evaluation_lock=threading.Lock(),
        _evaluation_error=None,
        _evaluation_snapshot={"placement": placement, "steps": 20},
        _initial_plan_complete=False,
        _evaluation_thread=SimpleNamespace(join=lambda: None),
        evaluation_in_flight=True,
    )
    # Complete hash buckets as recorded independently of the active route policy.
    load = torch.zeros((20, 2, 288, 11), dtype=torch.int64)
    load[:, 0, :2, 0] = 256
    load[:, 1, :, 0] = 1
    for copies in range(1, 5):
        offset = 1 + copies * (copies - 1) // 2
        for bucket in range(copies):
            load[..., offset + bucket] = (load[..., 0] + copies - 1 - bucket) // copies
    manager._collect_local_samples = lambda: load
    manager._prepare_next_sampling_window = lambda: None
    manager.transfer = SimpleNamespace(prepare_transfer=lambda _: pytest.fail("policy-only needs no weight transfer"))
    monkeypatch.setattr(module, "plan_eplb_candidate", lambda *_a, **_k: pytest.fail("insufficient for migration"))
    monkeypatch.setattr(module.dist, "all_reduce", lambda *_a, **_k: None)
    monkeypatch.setattr(module.dist, "broadcast_object_list", lambda *_a, **_k: None)
    monkeypatch.setattr(torch.cuda, "set_device", lambda _: None)
    ordered = []
    monkeypatch.setattr(
        torch.cuda, "current_stream", lambda: SimpleNamespace(wait_stream=lambda stream: ordered.append(stream))
    )
    monkeypatch.setattr(infer_batch.g_infer_context, "get_overlap_stream", lambda: "previous_forward")

    manager._evaluate_after_event(SimpleNamespace(synchronize=lambda: None))
    assert manager._evaluation_error is None
    result = manager._evaluation_result
    assert result["kind"] == "insufficient"
    assert result["minimum_layer_samples"] < result["minimum"]
    assert result["replica_routing"].tolist() == [True, False]
    assert not result["improved"].any()
    assert result["layer_plans"] == result["prepared_batches"] == []
    assert not manager._poll_evaluation()
    assert ordered == ["previous_forward"]
    assert [state.replica_routing for state in states] == [True, False]
    assert states[0].logical_replica_count[:2].eq(2).all()
    assert states[1].logical_replica_count.eq(0).all()
    assert all(state.placement_generation == 0 for state in states)
    torch.testing.assert_close(manager.current_placement, placement)

    # Repeated decisions do not rebuild/publish unchanged route metadata.
    manager._evaluate_after_event(SimpleNamespace(synchronize=lambda: None))
    assert manager._evaluation_error is None
    assert manager._evaluation_result["metadata"] == [None, None]


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.parametrize("tokens", [1, 64])
@pytest.mark.parametrize("phase", [0, 1, 2, 3])
@pytest.mark.parametrize("migrated", [False, True])
def test_sampling_and_decode_routes_are_independent_of_prefill_policy(tokens, phase, migrated):
    from lightllm.common.basemodel.triton_kernel.fused_moe.grouped_topk import triton_grouped_topk_eplb

    generator = torch.Generator(device="cuda").manual_seed(671)
    logits = torch.randn((tokens, 288), device="cuda", generator=generator)
    logits[:, 0] += 10
    placement = _placement()
    if migrated:
        # Also exercise a full row of replicas: canonical mode overwrites its tail.
        placement[0, 1:, -2:] = torch.tensor([0, 1])
        placement = placement.flip(1).roll(11, dims=2)
    outputs = []
    for policy in (True, False):
        maps, counts = build_logical_to_physical_maps_for_layers(
            placement,
            288,
            source_rank=0,
            node_world_size=4,
            replica_rotation_rank=phase,
            replica_routing=torch.tensor([policy]),
        )
        counter = torch.zeros((1, 288, 11), dtype=torch.int64, device="cuda")
        weights, physical, logical = triton_grouped_topk_eplb(
            hidden_states=torch.empty((tokens, 32), device="cuda"),
            gating_output=logits,
            correction_bias=None,
            topk=8,
            renormalize=True,
            num_expert_group=1,
            topk_group=1,
            scoring_func="sigmoid",
            logical_to_physical_map=maps[0].cuda(),
            logical_replica_count=counts[0].cuda(),
            expert_counter=counter,
            sample_index=0,
            record_load=True,
            use_grouped_topk=True,
            return_logical_ids=True,
            group_score_used_topk_num=1,
            replica_rotation_rank=phase,
        )
        torch.testing.assert_close(placement.flatten().cuda()[physical.long()], logical.long(), rtol=0, atol=0)
        _, decode_physical, _ = triton_grouped_topk_eplb(
            hidden_states=torch.empty((tokens, 32), device="cuda"),
            gating_output=logits,
            correction_bias=None,
            topk=8,
            renormalize=True,
            num_expert_group=1,
            topk_group=1,
            scoring_func="sigmoid",
            logical_to_physical_map=maps[0].cuda(),
            logical_replica_count=counts[0].cuda(),
            expert_counter=None,
            sample_index=0,
            record_load=False,
            use_grouped_topk=True,
            use_replicas=False,
            return_logical_ids=True,
            group_score_used_topk_num=1,
            replica_rotation_rank=phase,
        )
        outputs.append((weights, logical, counter, decode_physical))
    for full, canonical in zip(*outputs):
        torch.testing.assert_close(full, canonical, rtol=0, atol=0)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.parametrize("draft", [False, True])
def test_factory_preserves_initial_routing_without_online_samples(monkeypatch, draft):
    from lightllm.common import eplb_utils
    from lightllm.distributed import dist_group_manager
    from lightllm.utils import dist_utils, envs_utils
    from lightllm.common.basemodel.layer_weights.meta_weights.fused_moe.expert_parallel_state import (
        create_expert_parallel_state,
    )

    args = SimpleNamespace(enable_tpsp_mix_mode=False, eplb_num_redundant_experts_per_rank=2)
    monkeypatch.setattr(eplb_utils, "is_eplb_enabled", lambda _: True)
    monkeypatch.setattr(envs_utils, "get_env_start_args", lambda: args)
    monkeypatch.setattr(dist_utils, "get_node_world_size", lambda: 4)
    monkeypatch.setattr(dist_group_manager, "ll_num_experts", 296, raising=False)
    state = create_expert_parallel_state(288, 4, 0, tp_rank=0, tp_world_size=4, is_mtp_draft_model=draft).eplb
    maps, counts = build_logical_to_physical_maps_for_layers(
        _placement(), 288, source_rank=0, node_world_size=4, replica_rotation_rank=0
    )
    assert state.replica_routing is True
    assert state.recording == (not draft)
    torch.testing.assert_close(state.logical_to_physical_map.cpu(), maps[0], rtol=0, atol=0)
    torch.testing.assert_close(state.logical_replica_count.cpu(), counts[0], rtol=0, atol=0)
