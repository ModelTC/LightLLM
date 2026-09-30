import threading
import time
from contextlib import nullcontext
from types import SimpleNamespace

import pytest
import torch

from lightllm.server.router.model_infer.mode_backend.eplb.placement import (
    EPLBPlanner,
    GlobalBalanceEPLBPlanner,
    TopologyAwareEPLBPlanner,
    build_initial_local_expert_ids,
    build_logical_to_physical_map,
    create_eplb_planner,
)
from lightllm.server.api_cli import make_argument_parser
from lightllm.server.core.objs.start_args_type import StartArgs
from lightllm.server.router.model_infer.infer_batch import g_infer_context
from lightllm.server.router.model_infer.mode_backend.eplb import (
    runtime_manager as manager_module,
)
from lightllm.server.router.model_infer.mode_backend.eplb import async_task as async_task_module
from lightllm.server.router.model_infer.mode_backend.eplb import (
    async_load_gather_task as load_gather_module,
)
from lightllm.server.router.model_infer.mode_backend.eplb import metrics as eplb_metrics
from lightllm.server.router.model_infer.mode_backend.eplb import eplb_utils as eplb_utils_module
from lightllm.server.router.model_infer.mode_backend.eplb import (
    async_placement_plan_task as plan_module,
)
from lightllm.server.router.model_infer.mode_backend.eplb import (
    async_expert_transfer as transfer_module,
)
from lightllm.server.router.model_infer.mode_backend.eplb import (
    async_transfer_planner as transfer_planner_module,
)
from lightllm.common.basemodel.layer_weights.meta_weights.fused_moe.impl import (
    deepgemm_impl as deepgemm_module,
)
from lightllm.common.basemodel.layer_weights.meta_weights.fused_moe.impl import (
    create_fuse_moe_impl,
    FuseMoeMarlin,
    FuseMoeTriton,
)
from lightllm.common.basemodel.layer_weights.meta_weights.fused_moe.impl.base_impl import (
    FuseMoeBaseImpl,
)
from lightllm.common.basemodel.layer_weights.meta_weights.fused_moe import (
    fused_moe_weight as fused_weight_module,
)
from lightllm.server.router.model_infer.mode_backend.eplb.eplb_utils import extract_eplb_expert_tensors
from lightllm.server.router.model_infer.mode_backend.eplb.async_expert_transfer import (
    EPLBTransferInfo,
    ExpertTensorBuffer,
    PinnedMemoryEPLBTransfer,
    build_transfer_plan,
)


def _test_moe_impl(
    eplb=False,
    num_logical_experts=128,
    world_size=16,
    num_redundant_experts_per_rank=1,
    prefill_route_counter=None,
    prefill_route_sample_index=None,
    decode_route_counter=None,
    run_mode="prefill",
):
    logical_to_physical_map = None
    if eplb:
        if run_mode == "prefill":
            if prefill_route_counter is None:
                prefill_route_counter = torch.zeros((24, num_logical_experts), dtype=torch.int64)
            if prefill_route_sample_index is None:
                prefill_route_sample_index = torch.zeros((2,), dtype=torch.int64)
            decode_route_counter = None
        else:
            prefill_route_counter = None
            prefill_route_sample_index = None
            if decode_route_counter is None:
                decode_route_counter = torch.zeros((num_logical_experts, num_logical_experts), dtype=torch.int64)
        logical_to_physical_map = torch.zeros((num_logical_experts, world_size + 3), dtype=torch.int32)
        logical_to_physical_map[:, :3] = 1
    else:
        num_redundant_experts_per_rank = 0
    return SimpleNamespace(
        n_routed_experts=num_logical_experts,
        num_total_physical_experts=(num_logical_experts + world_size * num_redundant_experts_per_rank),
        num_redundant_experts_per_rank=num_redundant_experts_per_rank,
        local_logics_expert_ids_list=list(range(num_logical_experts // world_size + num_redundant_experts_per_rank)),
        logical_to_physical_map=logical_to_physical_map,
        prefill_route_counter=prefill_route_counter,
        prefill_route_sample_index=prefill_route_sample_index,
        decode_route_counter=decode_route_counter,
    )


def _set_deepgemm_runtime(impl, runtime):
    for name in (
        "num_total_physical_experts",
        "num_redundant_experts_per_rank",
        "logical_to_physical_map",
        "prefill_route_counter",
        "prefill_route_sample_index",
        "decode_route_counter",
    ):
        setattr(impl, name, getattr(runtime, name))


def _initial_expert_placement(num_logical_experts, world_size, num_redundant_experts_per_rank):
    return torch.tensor(
        build_initial_local_expert_ids(
            num_logical_experts,
            world_size,
            num_redundant_experts_per_rank,
        ),
        dtype=torch.int64,
    )


def _planner_load(load_by_layer, world_size):
    """构造 [rank, layer, sample, logical expert] CPU planner 输入。"""
    aggregated_load = torch.as_tensor(load_by_layer, dtype=torch.float64)
    assert aggregated_load.ndim == 2
    load = torch.zeros(
        (world_size, aggregated_load.shape[0], 1, aggregated_load.shape[1]),
        dtype=torch.float64,
    )
    load[0, :, 0] = aggregated_load
    return load


def _rank_to_logic_expert_ids(redundant_placement, num_logical_experts):
    num_ranks = len(redundant_placement)
    num_primary_experts_per_rank = num_logical_experts // num_ranks
    return [
        list(
            range(
                rank * num_primary_experts_per_rank,
                (rank + 1) * num_primary_experts_per_rank,
            )
        )
        + list(rank_redundant_expert_ids)
        for rank, rank_redundant_expert_ids in enumerate(redundant_placement)
    ]


def test_base_call_template_forwards_selection_and_capture_callback():
    class Impl(FuseMoeBaseImpl):
        def _select_experts(
            self,
            input_tensor,
            router_logits,
            correction_bias,
            top_k,
            renormalize,
            use_grouped_topk,
            topk_group,
            num_expert_group,
            scoring_func,
            per_expert_scale=None,
        ):
            return "weights", "logical_ids"

        def _prepare_expert_execution(self, topk_weights, topk_ids, is_prefill, shared_expert_gate=None):
            seen["prepare"] = {"topk_ids": topk_ids, "is_prefill": is_prefill}
            return topk_weights, "physical_ids"

        def _fused_experts(
            self,
            input_tensor,
            w13,
            w2,
            topk_weights,
            topk_ids,
            is_prefill,
            router_logits=None,
        ):
            seen["fused"] = {"topk_ids": topk_ids}
            return "output"

    seen, captured = {}, []
    impl = Impl(4, 0, 1.0, SimpleNamespace())
    result = impl(
        "input",
        "logits",
        "w13",
        "w2",
        None,
        "softmax",
        2,
        False,
        False,
        0,
        0,
        moe_capture_callback=captured.append,
        is_prefill=True,
    )
    assert result == "output"
    assert captured == ["logical_ids"]
    assert seen["prepare"]["topk_ids"] == "logical_ids"
    assert seen["prepare"]["is_prefill"] is True
    assert seen["fused"]["topk_ids"] == "physical_ids"

    with pytest.raises(AssertionError, match="is_prefill must be explicitly specified"):
        impl(
            "input",
            "logits",
            "w13",
            "w2",
            None,
            "softmax",
            2,
            False,
            False,
            0,
            0,
        )


def test_factory_selects_all_paths_without_ep_constructor_state(monkeypatch):
    plain_quant = SimpleNamespace(method_name="none")
    marlin_quant = SimpleNamespace(method_name="awq_marlin")
    monkeypatch.setattr(FuseMoeMarlin, "create_workspace", lambda self: None)
    monkeypatch.setattr(
        deepgemm_module,
        "get_env_start_args",
        lambda: SimpleNamespace(eplb_num_redundant_experts_per_rank=0),
    )
    monkeypatch.setattr(deepgemm_module, "get_global_world_size", lambda: 2)
    monkeypatch.setattr(deepgemm_module, "get_global_rank", lambda: 0)
    ep_impl = create_fuse_moe_impl(
        n_routed_experts=4,
        num_fused_shared_experts=0,
        routed_scaling_factor=1.0,
        quant_method=plain_quant,
        enable_ep_moe=True,
    )
    assert isinstance(ep_impl, deepgemm_module.FuseMoeDeepGEMM)
    assert ep_impl.num_total_physical_experts == 4
    assert not hasattr(ep_impl, "num_primary_experts_per_rank")
    assert not hasattr(ep_impl, "prefill_route_counter")
    assert not hasattr(ep_impl, "expert_parallel_state")
    assert isinstance(
        create_fuse_moe_impl(
            n_routed_experts=4,
            num_fused_shared_experts=0,
            routed_scaling_factor=1.0,
            quant_method=plain_quant,
        ),
        FuseMoeTriton,
    )
    assert isinstance(
        create_fuse_moe_impl(
            n_routed_experts=4,
            num_fused_shared_experts=0,
            routed_scaling_factor=1.0,
            quant_method=marlin_quant,
        ),
        FuseMoeMarlin,
    )


def test_find_fused_moe_weights_uses_layer_experts_in_model_order(monkeypatch):
    class FakeFusedMoeWeight:
        def __init__(self, layer_num, enable_ep_moe=True):
            self.layer_num_ = layer_num
            self.enable_ep_moe = enable_ep_moe

    monkeypatch.setattr(manager_module, "FusedMoeWeight", FakeFusedMoeWeight)
    first = FakeFusedMoeWeight(1)
    second = FakeFusedMoeWeight(3)
    disabled = FakeFusedMoeWeight(2, enable_ep_moe=False)
    model = SimpleNamespace(
        trans_layers_weight=[
            SimpleNamespace(experts=first),
            SimpleNamespace(),
            SimpleNamespace(experts=disabled),
            SimpleNamespace(experts=second),
        ]
    )

    assert manager_module._find_fused_moe_weights(model) == [first, second]


def test_eplb_redundant_experts_default_to_disabled():
    parser = make_argument_parser()

    assert parser.parse_args([]).eplb_num_redundant_experts_per_rank == 0
    assert parser.parse_args(["--eplb_num_redundant_experts_per_rank", "3"]).eplb_num_redundant_experts_per_rank == 3
    assert StartArgs().eplb_num_redundant_experts_per_rank == 0
    assert parser.parse_args([]).eplb_run_mode == "prefill"
    assert parser.parse_args(["--eplb_run_mode", "prefill"]).eplb_run_mode == "prefill"
    assert parser.parse_args(["--eplb_run_mode", "decode"]).eplb_run_mode == "decode"
    assert StartArgs().eplb_run_mode == "prefill"
    assert parser.parse_args([]).eplb_plan_mode == "topology_aware"
    assert parser.parse_args(["--eplb_plan_mode", "global_balance"]).eplb_plan_mode == "global_balance"
    assert parser.parse_args(["--eplb_plan_mode", "topology_aware"]).eplb_plan_mode == "topology_aware"
    assert StartArgs().eplb_plan_mode == "topology_aware"
    assert parser.parse_args([]).eplb_rebalance_count == 1
    assert parser.parse_args(["--eplb_rebalance_count", "-1"]).eplb_rebalance_count == -1
    assert parser.parse_args(["--eplb_rebalance_count", "0"]).eplb_rebalance_count == 0
    assert StartArgs().eplb_rebalance_count == 1
    assert parser.parse_args([]).eplb_config_path is None
    assert parser.parse_args(["--eplb_config_path", "/tmp/eplb.json"]).eplb_config_path == "/tmp/eplb.json"
    assert StartArgs().eplb_config_path is None


@pytest.mark.parametrize(
    ("num_logical_experts", "num_ranks", "num_redundant_experts_per_rank", "expected"),
    [
        (8, 4, 2, [[0, 1, 2, 3], [2, 3, 4, 5], [4, 5, 6, 7], [6, 7, 0, 1]]),
        (6, 3, 4, [[0, 1, 2, 3, 4, 5], [2, 3, 4, 5, 0, 1], [4, 5, 0, 1, 2, 3]]),
    ],
)
def test_build_initial_local_expert_ids(
    num_logical_experts,
    num_ranks,
    num_redundant_experts_per_rank,
    expected,
):
    actual = build_initial_local_expert_ids(
        num_logical_experts,
        num_ranks,
        num_redundant_experts_per_rank,
    )

    assert actual == expected


def test_build_initial_local_expert_ids_rejects_local_or_duplicate_replicas():
    with pytest.raises(AssertionError):
        build_initial_local_expert_ids(8, 4, 7)


def test_eplb_planner_defines_an_abstract_planning_interface():
    with pytest.raises(TypeError):
        EPLBPlanner()

    assert isinstance(GlobalBalanceEPLBPlanner(2, 1), EPLBPlanner)
    assert isinstance(
        TopologyAwareEPLBPlanner(2, 1, node_world_size=2),
        EPLBPlanner,
    )


def test_create_eplb_planner_selects_requested_algorithm():
    planner = create_eplb_planner(
        plan_mode="global_balance",
        run_mode="prefill",
        num_ranks=2,
        num_redundant_experts_per_rank=1,
        expert_alignment=1,
        node_world_size=2,
    )

    assert isinstance(planner, GlobalBalanceEPLBPlanner)

    planner = create_eplb_planner(
        plan_mode="topology_aware",
        run_mode="prefill",
        num_ranks=4,
        num_redundant_experts_per_rank=1,
        expert_alignment=128,
        node_world_size=2,
    )

    assert isinstance(planner, TopologyAwareEPLBPlanner)
    assert planner.node_world_size == 2

    with pytest.raises(AssertionError):
        create_eplb_planner(
            plan_mode="unknown",
            run_mode="prefill",
            num_ranks=2,
            num_redundant_experts_per_rank=1,
            expert_alignment=1,
            node_world_size=2,
        )

    for prefill_plan_mode in ("global_balance", "topology_aware"):
        with pytest.raises(AssertionError, match="is for 'prefill'"):
            create_eplb_planner(
                plan_mode=prefill_plan_mode,
                run_mode="decode",
                num_ranks=2,
                num_redundant_experts_per_rank=1,
                expert_alignment=1,
                node_world_size=2,
            )


def test_topology_aware_planner_minimizes_samplewise_aligned_critical_load():
    planner = TopologyAwareEPLBPlanner(
        2,
        1,
        node_world_size=2,
        expert_alignment=128,
        rebalance_gain_threshold=0.0,
    )
    load = torch.zeros((2, 1, 2, 4), dtype=torch.int64)
    load[0, 0] = torch.tensor(
        [
            [300, 20, 20, 200],
            [100, 300, 40, 160],
        ]
    )
    current = [[[0, 1, 2], [2, 3, 0]]]

    result = planner.plan(route_statistics=load, current_placement=current)

    assert result == [[[0, 1, 3], [2, 3, 0]]]


def test_topology_aware_planner_keeps_primary_slots_and_builds_legal_redundant_rows():
    planner = TopologyAwareEPLBPlanner(
        4,
        2,
        node_world_size=4,
        expert_alignment=128,
        rebalance_gain_threshold=0.0,
    )
    current = _initial_expert_placement(8, 4, 2).unsqueeze(0).tolist()
    load = torch.ones((4, 1, 2, 8), dtype=torch.int64)
    load[:, :, :, 0] = 1000
    load[:, :, :, 4] = 500

    result = planner.plan(route_statistics=load, current_placement=current)

    for rank, rank_placement in enumerate(result[0]):
        primary_experts = list(range(rank * 2, (rank + 1) * 2))
        redundant_experts = rank_placement[2:]
        assert rank_placement[:2] == primary_experts
        assert len(rank_placement) == len(set(rank_placement))
        assert all(expert_id // 2 != rank for expert_id in redundant_experts)
    placed_experts = {expert_id for rank_placement in result[0] for expert_id in rank_placement}
    assert placed_experts == set(range(8))


def test_topology_aware_planner_prefers_replica_on_the_request_source_node():
    planner = TopologyAwareEPLBPlanner(
        4,
        1,
        expert_alignment=128,
        node_world_size=2,
        rebalance_gain_threshold=0.0,
    )
    load = torch.zeros((4, 1, 1, 8), dtype=torch.int64)
    load[0, 0, 0, 0] = 1024
    current = _initial_expert_placement(8, 4, 1).unsqueeze(0).tolist()

    result = planner.plan(route_statistics=load, current_placement=current)

    # Expert 0 is owned by rank 0. Rank 1 is the only other rank on source
    # node 0, so a local-first plan puts its redundant copy there.
    assert result[0][1][-1] == 0


def test_topology_aware_planner_plans_each_layer_independently():
    planner = TopologyAwareEPLBPlanner(
        4,
        2,
        node_world_size=2,
        rebalance_gain_threshold=0.0,
    )
    generator = torch.Generator().manual_seed(17)
    load = torch.randint(0, 1000, (4, 3, 4, 16), generator=generator)
    current = _initial_expert_placement(16, 4, 2).unsqueeze(0).repeat(3, 1, 1).tolist()

    combined_result = planner.plan(
        route_statistics=load,
        current_placement=current,
    )
    layer_results = [
        planner.plan(
            route_statistics=load[:, layer_index : layer_index + 1],
            current_placement=[current[layer_index]],
        )[0]
        for layer_index in range(load.shape[1])
    ]

    assert combined_result == layer_results


def test_topology_aware_planner_reuses_unchanged_layout_and_physical_slots():
    generator = torch.Generator().manual_seed(7)
    load = torch.randint(1, 1000, (4, 3, 2, 16), generator=generator)
    initial = _initial_expert_placement(16, 4, 2).unsqueeze(0).repeat(3, 1, 1).tolist()
    candidate_planner = TopologyAwareEPLBPlanner(
        4,
        2,
        node_world_size=4,
        rebalance_gain_threshold=0.0,
    )
    current = candidate_planner.plan(
        route_statistics=load,
        current_placement=initial,
    )
    default_planner = TopologyAwareEPLBPlanner(4, 2, node_world_size=4)

    result = default_planner.plan(
        route_statistics=load,
        current_placement=current,
    )

    assert result == current


def test_topology_aware_planner_restores_canonical_primaries_from_full_slot_layout():
    planner = TopologyAwareEPLBPlanner(
        2,
        1,
        node_world_size=2,
        rebalance_gain_threshold=0.0,
    )
    load = torch.tensor([[[[100, 90, 80, 70]]], [[[60, 50, 40, 30]]]])
    full_slot_layout = [[[0, 2, 3], [1, 3, 0]]]

    result = planner.plan(
        route_statistics=load,
        current_placement=full_slot_layout,
    )

    assert result[0][0][:2] == [0, 1]
    assert result[0][1][:2] == [2, 3]


def test_global_balance_planner_builds_legal_concrete_slot_layout():
    planner = GlobalBalanceEPLBPlanner(
        4,
        1,
        expert_alignment=1,
    )
    current = _initial_expert_placement(8, 4, 1).unsqueeze(0).tolist()
    load = torch.ones((4, 1, 1, 8), dtype=torch.int64)
    load[:, :, :, 0] = 1000
    load[:, :, :, 4] = 500

    result = planner.plan(route_statistics=load, current_placement=current)
    placement = result[0]

    for row in placement:
        assert len(row) == 3
        assert len(row) == len(set(row))
        assert 0 in row
    assert set(expert for row in placement for expert in row) == set(range(8))
    assert any(row[:2] != list(range(rank * 2, (rank + 1) * 2)) for rank, row in enumerate(placement))
    assert isinstance(result, list)


def test_global_balance_planner_returns_deterministic_layout_for_zero_load_experts():
    planner = GlobalBalanceEPLBPlanner(2, 1)
    current = [[[0, 1, 3], [2, 3, 1]]]

    result = planner.plan(
        route_statistics=_planner_load([[0, 0, 0, 0]], world_size=2),
        current_placement=current,
    )

    assert result == [[[0, 1, 3], [2, 0, 1]]]


def test_global_balance_planner_aggregates_rank_and_sample_dimensions():
    planner = GlobalBalanceEPLBPlanner(2, 1)
    current = [[[0, 1, 3], [2, 3, 1]]]
    raw_load = torch.tensor(
        [
            [[[500, 1, 2, 3], [300, 4, 5, 6]]],
            [[[100, 7, 8, 9], [200, 10, 11, 12]]],
        ],
        dtype=torch.int64,
    )
    aggregated_load = raw_load.sum(dim=(0, 2))
    equivalent_raw_load = torch.zeros_like(raw_load)
    equivalent_raw_load[0, :, 0] = aggregated_load

    result = planner.plan(route_statistics=raw_load, current_placement=current)

    assert result == planner.plan(
        route_statistics=equivalent_raw_load,
        current_placement=current,
    )


def test_global_balance_planner_plans_each_layer_independently_then_combines_results():
    planner = GlobalBalanceEPLBPlanner(2, 1)
    current_layer = [[0, 1, 3], [2, 3, 1]]
    current = [[row[:] for row in current_layer], [row[:] for row in current_layer]]

    load = _planner_load(
        [
            [1000, 1, 1, 1],
            [0, 0, 0, 0],
        ],
        world_size=2,
    )

    result = planner.plan(route_statistics=load, current_placement=current)

    assert result == [
        [[0, 1, 3], [2, 0, 1]],
        [[0, 1, 3], [2, 0, 1]],
    ]


def test_global_balance_planner_iteratively_places_hot_expert_on_idle_rank():
    planner = GlobalBalanceEPLBPlanner(2, 1)
    current = [[[0, 1, 3], [2, 3, 1]]]

    result = planner.plan(
        route_statistics=_planner_load([[1000, 1, 1, 1]], world_size=2),
        current_placement=current,
    )

    assert result == [[[0, 1, 3], [2, 0, 1]]]


def test_global_balance_planner_repeatedly_splits_the_hottest_remaining_expert():
    planner = GlobalBalanceEPLBPlanner(4, 3)
    current = _initial_expert_placement(8, 4, 3).unsqueeze(0).tolist()

    result = planner.plan(
        route_statistics=_planner_load([[1000, 900, 800, 700, 1, 1, 1, 1]], world_size=4),
        current_placement=current,
    )

    replica_counts = [sum(expert in row for row in result[0]) for expert in range(8)]
    assert replica_counts == [4, 4, 4, 4, 1, 1, 1, 1]


def test_global_balance_planner_balances_expert_groups_with_equal_replica_counts():
    planner = GlobalBalanceEPLBPlanner(4, 1)

    placement = planner._distribute_remaining_experts(
        redundant_experts=[0],
        expert_groups=[
            (1, 1, 8.0),
            (2, 1, 7.0),
            (3, 1, 6.0),
            (4, 1, 5.0),
            (5, 1, 4.0),
            (6, 1, 3.0),
            (7, 1, 2.0),
            (8, 1, 1.0),
        ],
    )

    assert placement == [
        [0, 1, 8],
        [0, 2, 7],
        [0, 3, 6],
        [0, 4, 5],
    ]


def test_global_balance_planner_places_replicas_of_one_expert_on_distinct_ranks():
    planner = GlobalBalanceEPLBPlanner(2, 1)

    placement = planner._distribute_remaining_experts(
        redundant_experts=[0],
        expert_groups=[
            (1, 2, 5.0),
            (2, 1, 8.0),
            (3, 1, 1.0),
        ],
    )

    assert placement == [
        [0, 1, 2],
        [0, 1, 3],
    ]
    assert all(len(row) == len(set(row)) for row in placement)


def test_global_balance_planner_places_single_replicas_by_rank_load_before_free_slots():
    planner = GlobalBalanceEPLBPlanner(4, 1)

    placement = planner._distribute_remaining_experts(
        redundant_experts=[0],
        expert_groups=[
            (1, 3, 10.0),
            (2, 2, 1.0),
            (3, 1, 8.0),
            (4, 1, 7.0),
            (5, 1, 6.0),
            (6, 1, 5.0),
            (7, 1, 4.0),
            (8, 1, 3.0),
            (9, 1, 2.0),
        ],
    )

    # 多副本专家平铺后，rank 3 的剩余槽位比 rank 1、2 少，但负载最低；
    # 因此它仍连续取得最热的两个单副本专家，并率先填满。
    assert placement == [
        [0, 1, 2, 7],
        [0, 1, 5, 9],
        [0, 1, 6, 8],
        [0, 2, 3, 4],
    ]


def test_global_balance_planner_matches_documented_two_stage_distribution_example():
    planner = GlobalBalanceEPLBPlanner(4, 1)
    expert_groups = [
        (1, 2, 6.0),
        (2, 1, 9.0),
        (3, 1, 8.0),
        (4, 1, 7.0),
        (5, 1, 5.0),
        (6, 1, 4.0),
        (7, 1, 3.0),
    ]

    placement = planner._distribute_remaining_experts(
        redundant_experts=[0],
        expert_groups=expert_groups,
    )

    assert placement == [
        [0, 1, 4],
        [0, 1, 5],
        [0, 2, 7],
        [0, 3, 6],
    ]
    load_per_replica = {expert: load for expert, _, load in expert_groups}
    assert [sum(load_per_replica[expert] for expert in row[1:]) for row in placement] == [13.0, 11.0, 12.0, 12.0]


def test_global_balance_planner_matches_candidate_ranks_before_reusing_slots():
    planner = GlobalBalanceEPLBPlanner(3, 1)
    current = [
        [0, 1, 2],
        [3, 4, 5],
        [6, 7, 8],
    ]
    candidate = [
        [3, 4, 9],
        [6, 7, 10],
        [0, 1, 11],
    ]

    placement = planner._reuse_current_slots(candidate, current)

    assert placement == [
        [0, 1, 11],
        [3, 4, 9],
        [6, 7, 10],
    ]


def test_global_balance_planner_keeps_selected_experts_in_their_current_slots():
    planner = GlobalBalanceEPLBPlanner(4, 2)
    current = _initial_expert_placement(8, 4, 2).unsqueeze(0).tolist()

    result = planner.plan(
        route_statistics=_planner_load([[50, 98, 54, 6, 34, 66, 63, 52]], world_size=4),
        current_placement=current,
    )

    # 只要专家仍分配在同一个 rank，就保留其原物理槽位。
    for current_row, target_row in zip(current[0], result[0]):
        for slot, expert in enumerate(current_row):
            if expert in target_row:
                assert target_row[slot] == expert


def test_global_balance_planner_fills_every_rank_with_distinct_nonlocal_experts():
    planner = GlobalBalanceEPLBPlanner(
        4,
        1,
    )
    current = _initial_expert_placement(16, 4, 1).unsqueeze(0).tolist()
    load_by_layer_and_rank = torch.randint(
        0,
        10000,
        (1, 4, 16),
        generator=torch.Generator().manual_seed(2),
    )
    load = load_by_layer_and_rank.permute(1, 0, 2).unsqueeze(dim=2)

    result = planner.plan(route_statistics=load, current_placement=current)

    assert len(result) == len(current)
    assert all(len(actual) == len(expected) for actual, expected in zip(result[0], current[0]))
    for row in result[0]:
        assert len(row) == len(set(row)) == 5
    assert set(expert for row in result[0] for expert in row) == set(range(16))


def test_global_balance_planner_supports_multiple_redundant_experts_per_rank():
    planner = GlobalBalanceEPLBPlanner(4, 3)
    current = _initial_expert_placement(16, 4, 3).unsqueeze(0).tolist()
    load = _planner_load(
        [
            [
                22613,
                26852,
                21852,
                23480,
                13270,
                14695,
                28735,
                22303,
                15324,
                19604,
                21492,
                25458,
                14120,
                12130,
                18620,
                22888,
            ]
        ],
        world_size=4,
    )

    result = planner.plan(route_statistics=load, current_placement=current)

    for row in result[0]:
        assert len(row) == len(set(row)) == 7
    replica_counts = [sum(expert in row for row in result[0]) for expert in range(16)]
    assert replica_counts[1] == replica_counts[6] == replica_counts[11] == 4
    assert sum(replica_counts) == 28


def test_fused_moe_loads_default_replicas_into_their_physical_rows():
    weight = object.__new__(fused_weight_module.FusedMoeWeight)
    weight.lock = threading.Lock()
    loaded = []

    def load_weight(expert, local, _weights):
        loaded.append(("weight", expert, local))

    def load_scale(expert, local, _weights):
        loaded.append(("scale", expert, local))

    def load_zero_point(expert, local, _weights):
        loaded.append(("zero", expert, local))

    weight._load_expert = load_weight
    weight._load_expert_scale = load_scale
    weight._load_expert_zero_point = load_zero_point
    local_logic_expert_ids_list = build_initial_local_expert_ids(8, 4, 2)[0]

    weight._load_weight(local_logic_expert_ids_list, {})

    assert local_logic_expert_ids_list == [0, 1, 2, 3]
    assert [entry for entry in loaded if entry[0] == "weight"] == [
        ("weight", 0, 0),
        ("weight", 1, 1),
        ("weight", 2, 2),
        ("weight", 3, 3),
    ]


def test_logical_to_physical_map_selects_one_physical_expert():
    rank_to_logic_expert_ids = [[0, 1, 2, 3], [2, 3, 0, 1]]
    logical_to_physical = build_logical_to_physical_map(
        rank_to_logic_expert_ids,
        num_logical_experts=4,
        current_rank=0,
        node_world_size=2,
    )

    assert isinstance(logical_to_physical, list)
    assert len(logical_to_physical) == 4
    assert all(len(row) == 11 for row in logical_to_physical)
    assert [row[0] for row in logical_to_physical] == [2, 2, 2, 2]
    assert [row[1] for row in logical_to_physical] == [2, 2, 2, 2]
    assert [row[2] for row in logical_to_physical] == [1, 1, 1, 1]
    assert [row[3] for row in logical_to_physical] == [0, 1, 2, 3]
    assert all(physical_id >= 0 for row in logical_to_physical for physical_id in row[3 : 3 + row[0]])
    assert all(physical_id == -1 for row in logical_to_physical for physical_id in row[3 + row[0] :])


def test_logical_to_physical_map_requires_expert_count_divisible_by_rank_count():
    rank_to_logic_expert_ids = [[0, 1, 0], [2, 3, 1]]

    with pytest.raises(AssertionError):
        build_logical_to_physical_map(
            rank_to_logic_expert_ids,
            num_logical_experts=5,
            current_rank=0,
            node_world_size=2,
        )


def test_logical_to_physical_map_supports_all_redundant_slots_for_one_expert():
    logical_to_physical = build_logical_to_physical_map(
        [[0, 1, 0, 0], [2, 3, 0, 0]],
        num_logical_experts=4,
        current_rank=0,
        node_world_size=2,
    )

    # 1 个主副本加上 2 个 rank 的全部 4 个冗余槽。
    assert logical_to_physical[0][:3] == [5, 5, 3]
    assert len(logical_to_physical[0][3:]) == 8
    assert len(set(logical_to_physical[0][3:8])) == 5
    assert logical_to_physical[0][8:] == [-1, -1, -1]


def test_logical_to_physical_map_prefers_current_rank_replica():
    redundant = [[4], [5], [0], [1]]
    rank_to_logic_expert_ids = _rank_to_logic_expert_ids(redundant, 8)
    rank0_map = build_logical_to_physical_map(
        rank_to_logic_expert_ids,
        num_logical_experts=8,
        current_rank=0,
        node_world_size=2,
    )
    rank1_map = build_logical_to_physical_map(
        rank_to_logic_expert_ids,
        num_logical_experts=8,
        current_rank=1,
        node_world_size=2,
    )
    fallback_redundant = [[4], [5], [0], [1], [2], [3]]
    rank4_map = build_logical_to_physical_map(
        _rank_to_logic_expert_ids(fallback_redundant, 12),
        12,
        current_rank=4,
        node_world_size=2,
    )

    assert rank0_map[0][:3] == [2, 1, 1]
    assert rank1_map[0][:3] == [2, 1, 0]
    assert rank0_map[0][3] == 0
    assert set(rank0_map[0][3:5]) == {0, 8}
    assert set(rank1_map[0][3:5]) == {0, 8}
    assert rank4_map[0][:3] == [2, 0, 0]
    assert set(rank4_map[0][3:5]) == {0, 8}


def test_nonlocal_rank_without_same_node_replica_routes_across_all_replicas():
    redundant = [[4], [5], [0], [1]]
    rank_to_logic_expert_ids = _rank_to_logic_expert_ids(redundant, 8)
    maps = [
        build_logical_to_physical_map(
            rank_to_logic_expert_ids,
            8,
            current_rank=rank,
            node_world_size=1,
        )
        for rank in range(4)
    ]
    assert maps[0][0][:3] == [2, 1, 1]
    assert maps[1][0][:3] == [2, 0, 0]
    assert maps[2][0][:3] == [2, 1, 1]
    assert maps[3][0][:3] == [2, 0, 0]
    assert all(set(logical_map[0][3:5]) == {0, 8} for logical_map in maps)


def test_nonlocal_rank_prefers_same_node_replica():
    rank_to_logic_expert_ids = [
        [1, 2],
        [0, 3],
        [0, 4],
        [5, 6],
        [0, 7],
        [1, 2],
        [3, 4],
        [5, 6],
    ]

    rank0_map = build_logical_to_physical_map(
        rank_to_logic_expert_ids,
        num_logical_experts=8,
        current_rank=0,
        node_world_size=4,
    )

    # Expert 0 is on same-node ranks 1 and 2 (physical IDs 2 and 4), plus
    # remote rank 4 (physical ID 8). Hash routing only uses the first two.
    assert rank0_map[0][:6] == [3, 2, 0, 2, 4, 8]


def test_current_rank_moves_local_replica_to_front_without_changing_copies():
    # Expert 0 is primary on rank 0 and redundant on rank 1。两个 rank 都优先
    # 自己的本地副本，因此路由槽的起点不同，但候选集合和数量保持一致。
    redundant = [[1], [0], [3], [2]]
    rank_to_logic_expert_ids = _rank_to_logic_expert_ids(redundant, 4)
    rank0_map = build_logical_to_physical_map(
        rank_to_logic_expert_ids,
        4,
        current_rank=0,
        node_world_size=1,
    )
    rank1_map = build_logical_to_physical_map(
        rank_to_logic_expert_ids,
        4,
        current_rank=1,
        node_world_size=1,
    )

    assert rank0_map[0] == [2, 1, 1, 0, 3, -1, -1, -1, -1, -1, -1]
    assert rank1_map[0] == [2, 1, 1, 3, 0, -1, -1, -1, -1, -1, -1]


def test_current_rank_stably_moves_all_local_physical_ids_to_front():
    # Expert 0 在 rank 0/1/2 上依次对应 physical IDs [0, 1, 3, 5]。
    # 对 rank 1 构建路由表时，只把本地 ID 3 移到最前面；其余远端 ID
    # 仍保持原来的 [0, 1, 5] 顺序。
    rank_to_logic_expert_ids = [[0, 0], [1, 0], [2, 0]]

    rank1_map = build_logical_to_physical_map(
        rank_to_logic_expert_ids,
        num_logical_experts=3,
        current_rank=1,
        node_world_size=1,
    )

    assert rank1_map[0] == [4, 1, 1, 3, 0, 1, 5, -1, -1]


def test_transfer_plan_respects_explicit_target_slots():
    current = [[0, 1, 4, 5], [2, 3, 6, 7], [4, 5, 0, 1], [6, 7, 2, 3]]
    target = [[0, 1, 5, 4], [2, 3, 7, 6], [4, 5, 1, 0], [6, 7, 3, 2]]

    plan = build_transfer_plan(current, target, 3, num_logical_experts=8, world_size=4)
    transfer_infos = [transfer_info for transfer_batch in plan for transfer_info in transfer_batch]

    assert all(info.layer_index == 3 for info in transfer_infos)
    assert all(
        current[info.source_rank][info.source_local_expert_index] == info.source_logical_expert_id
        for info in transfer_infos
    )
    assert {(info.dest_rank, info.source_logical_expert_id) for info in transfer_infos} == {
        (rank, target[rank][slot]) for rank in range(4) for slot in range(2, 4)
    }


def test_manager_evaluating_starts_raw_load_gather_without_modifying_counters(monkeypatch):
    counters = [
        torch.tensor([[10, 11]], dtype=torch.int64),
        torch.tensor([[40, 41]], dtype=torch.int64),
    ]
    manager = manager_module.EPLBManager.__new__(manager_module.EPLBManager)
    manager.state = manager_module.EPLBManagerState.EVALUATING
    manager._eplb_impls = [
        _test_moe_impl(
            eplb=True,
            prefill_route_counter=counter,
            num_logical_experts=2,
            world_size=1,
        )
        for counter in counters
    ]
    manager.num_logical_experts = 2
    manager.run_mode = "prefill"
    manager.global_rank = 1
    manager.world_size = 1
    manager.load_gather_group = object()
    manager.control_group = object()
    manager.max_rebalance_count = -1
    manager.completed_rebalance_count = 0
    tasks = []

    class LoadGatherTask:
        def __init__(self, **kwargs):
            self.kwargs = kwargs
            self.started = False
            tasks.append(self)

        def start(self):
            self.started = True

    monkeypatch.setattr(manager_module, "EPLBLoadGatherTask", LoadGatherTask)
    monkeypatch.setattr(
        manager_module.dist,
        "broadcast_object_list",
        lambda values, **_kwargs: values.__setitem__(0, True),
    )

    manager._step_evaluating()

    assert manager.state is manager_module.EPLBManagerState.WAIT_LOAD_GATHER_FINISHED
    assert len(tasks) == 1
    assert tasks[0].started
    assert tasks[0].kwargs["load_gather_group"] is manager.load_gather_group
    assert torch.equal(
        tasks[0].kwargs["local_route_statistics"],
        torch.tensor([[[10, 11]], [[40, 41]]], dtype=torch.int64),
    )
    assert torch.equal(counters[0], torch.tensor([[10, 11]], dtype=torch.int64))
    assert torch.equal(counters[1], torch.tensor([[40, 41]], dtype=torch.int64))


def test_manager_evaluating_gathers_decode_upper_triangle(monkeypatch):
    decode_counter = torch.tensor([[300, 50, 20], [0, 400, 30], [0, 0, 500]], dtype=torch.int64)
    manager = manager_module.EPLBManager.__new__(manager_module.EPLBManager)
    manager.state = manager_module.EPLBManagerState.EVALUATING
    manager._eplb_impls = [
        _test_moe_impl(
            eplb=True,
            decode_route_counter=decode_counter,
            num_logical_experts=3,
            world_size=1,
            run_mode="decode",
        )
    ]
    manager.num_logical_experts = 3
    manager.run_mode = "decode"
    manager.global_rank = 0
    manager.world_size = 1
    manager.load_gather_group = object()
    manager.control_group = object()
    manager.max_rebalance_count = -1
    manager.completed_rebalance_count = 0
    manager.metric_client = object()
    tasks = []
    published_loads = []

    class LoadGatherTask:
        def __init__(self, **kwargs):
            self.kwargs = kwargs
            self.started = False
            tasks.append(self)

        def start(self):
            self.started = True

    monkeypatch.setattr(manager_module, "EPLBLoadGatherTask", LoadGatherTask)
    monkeypatch.setattr(
        manager_module.dist,
        "broadcast_object_list",
        lambda _values, **_kwargs: None,
    )
    monkeypatch.setattr(
        eplb_metrics,
        "publish_expert_load_metrics",
        lambda **kwargs: published_loads.append(kwargs["expert_load"]),
    )

    manager._step_evaluating()

    assert manager.state is manager_module.EPLBManagerState.WAIT_LOAD_GATHER_FINISHED
    assert tasks[0].started
    assert torch.equal(tasks[0].kwargs["local_route_statistics"], decode_counter.unsqueeze(0))
    assert len(published_loads) == 1
    assert torch.equal(published_loads[0], torch.tensor([[300, 400, 500]], dtype=torch.int64))
    assert torch.equal(decode_counter, tasks[0].kwargs["local_route_statistics"][0])


def test_load_gather_task_preserves_rank_layer_and_sample_dimensions(monkeypatch):
    local_load = torch.tensor(
        [
            [[1, 2], [3, 4]],
            [[5, 6], [7, 8]],
        ],
        dtype=torch.int64,
    )
    load_gather_group = object()
    calls = []

    def all_gather(output, local, group):
        calls.append((local, group))
        output[0].copy_(local)
        output[1].copy_(local + 100)

    monkeypatch.setattr(load_gather_module.dist, "all_gather", all_gather)
    monkeypatch.setattr(load_gather_module.dist, "get_world_size", lambda group: 2)
    task = load_gather_module.EPLBLoadGatherTask(
        local_route_statistics=local_load,
        load_gather_group=load_gather_group,
    )

    task._run()

    assert task.is_finished()
    assert len(calls) == 1
    assert torch.equal(calls[0][0], local_load)
    assert calls[0][1] is load_gather_group
    assert task.result is not None
    assert task.result.shape == (2, 2, 2, 2)
    assert torch.equal(task.result[0], local_load)
    assert torch.equal(task.result[1], local_load + 100)


def test_manager_delegates_distribution_planning_to_planner_class():
    current_placement = [[[0, 1]]]
    logical_load = torch.tensor([[[[10, 20]]]])
    calls = []
    planned_placement = [[[0, 1]]]
    planner = SimpleNamespace(
        plan=lambda **kwargs: (
            calls.append((kwargs["route_statistics"], kwargs["current_placement"])) or planned_placement
        )
    )
    task = plan_module.EPLBPlanTask(
        planner=planner,
        route_statistics=logical_load,
        current_placement=current_placement,
    )

    task._run()

    assert task.status == "succeeded"
    assert task.result == planned_placement
    assert len(calls) == 1
    assert calls[0][0] is logical_load
    assert calls[0][1] == current_placement


def test_plan_task_exits_process_on_failure(monkeypatch):
    def fail(**_kwargs):
        raise RuntimeError("planning boom")

    task = plan_module.EPLBPlanTask(
        planner=SimpleNamespace(plan=fail),
        route_statistics=torch.tensor([[[[10, 20]]]]),
        current_placement=[[[1]]],
    )
    exits = []
    logs = []
    monkeypatch.setattr(async_task_module.os, "_exit", exits.append)
    monkeypatch.setattr(async_task_module.logger, "exception", logs.append)

    task._run()

    assert exits == [1]
    assert logs == ["EPLBPlanTask failed"]


def test_transfer_planner_combines_all_layer_batches(monkeypatch):
    current_placement = [[[0, 1], [2, 3]], [[0, 2], [1, 3]]]
    target_placement = [[[2, 1], [0, 3]], [[0, 3], [1, 2]]]
    transfer_infos = [
        EPLBTransferInfo(0, 2, 1, 0, 0, 0),
        EPLBTransferInfo(1, 3, 1, 1, 0, 1),
    ]
    calls = []

    def build_plan(**kwargs):
        calls.append(kwargs)
        return [[transfer_infos[kwargs["layer_index"]]]]

    monkeypatch.setattr(transfer_planner_module, "build_transfer_plan", build_plan)
    planner = transfer_planner_module.EPLBTransferPlanner(
        current_placement=current_placement,
        target_placement=target_placement,
        num_logical_experts=4,
        world_size=2,
        transfer_layer_parallelism=8,
    )

    planner._run()

    assert planner.status == "succeeded"
    assert planner.result == [[transfer_infos[0], transfer_infos[1]]]
    assert calls == [
        {
            "current_placement": current_placement[0],
            "target_placement": target_placement[0],
            "layer_index": 0,
            "num_logical_experts": 4,
            "world_size": 2,
        },
        {
            "current_placement": current_placement[1],
            "target_placement": target_placement[1],
            "layer_index": 1,
            "num_logical_experts": 4,
            "world_size": 2,
        },
    ]


def test_transfer_planner_limits_parallel_layers_and_preserves_layer_order(monkeypatch):
    current_placement = [[[0], [1]]] * 5
    target_placement = [[[1], [0]]] * 5
    transfer_infos_by_layer = [
        [EPLBTransferInfo(layer_index, batch_index, 0, 0, 1, 0) for batch_index in range(batch_count)]
        for layer_index, batch_count in enumerate((3, 0, 1, 2, 1))
    ]

    def build_plan(**kwargs):
        return [[transfer_info] for transfer_info in transfer_infos_by_layer[kwargs["layer_index"]]]

    monkeypatch.setattr(transfer_planner_module, "build_transfer_plan", build_plan)
    planner = transfer_planner_module.EPLBTransferPlanner(
        current_placement=current_placement,
        target_placement=target_placement,
        num_logical_experts=2,
        world_size=2,
        transfer_layer_parallelism=2,
    )

    planner._run()

    assert planner.status == "succeeded"
    assert planner.result == [
        [transfer_infos_by_layer[0][0], transfer_infos_by_layer[2][0]],
        [transfer_infos_by_layer[0][1], transfer_infos_by_layer[3][0]],
        [transfer_infos_by_layer[0][2], transfer_infos_by_layer[3][1]],
        [transfer_infos_by_layer[4][0]],
    ]

    serial_planner = transfer_planner_module.EPLBTransferPlanner(
        current_placement=current_placement,
        target_placement=target_placement,
        num_logical_experts=2,
        world_size=2,
        transfer_layer_parallelism=1,
    )
    serial_planner._run()
    assert serial_planner.result == [
        [transfer_info] for layer_transfer_infos in transfer_infos_by_layer for transfer_info in layer_transfer_infos
    ]


def test_transfer_planner_exits_process_on_failure(monkeypatch):
    def fail(**_kwargs):
        raise RuntimeError("transfer planning boom")

    planner = transfer_planner_module.EPLBTransferPlanner(
        current_placement=[[[0], [1]]],
        target_placement=[[[1], [0]]],
        num_logical_experts=2,
        world_size=2,
        transfer_layer_parallelism=8,
    )
    exits = []
    logs = []
    monkeypatch.setattr(transfer_planner_module, "build_transfer_plan", fail)
    monkeypatch.setattr(async_task_module.os, "_exit", exits.append)
    monkeypatch.setattr(async_task_module.logger, "exception", logs.append)

    planner._run()

    assert exits == [1]
    assert logs == ["EPLBTransferPlanner failed"]


def test_compute_critical_overhead_ratio_estimates_rank_pressure():
    load = torch.tensor([[384, 128, 128, 128]], dtype=torch.int64)
    placement = [[[0, 1], [2, 3]]]

    ratio = eplb_metrics.compute_critical_overhead_ratio(
        logical_expert_load=load,
        placement=placement,
        expert_alignment=128,
    )

    # rank loads are [512, 256], so excess critical / balanced is 128 / 384.
    assert ratio == pytest.approx(1 / 3)


def test_compute_critical_overhead_ratio_preserves_layer_boundaries():
    load = torch.tensor(
        [
            [1, 0],
            [0, 1],
        ],
        dtype=torch.int64,
    )

    ratio = eplb_metrics.compute_critical_overhead_ratio(
        logical_expert_load=load,
        placement=[
            [[0], [1]],
            [[0], [1]],
        ],
        expert_alignment=128,
    )

    # 两层的热点 rank 相反；逐层取关键路径时仍各有 100% 开销，不能相互抵消。
    assert ratio == pytest.approx(1.0)


def test_compute_critical_overhead_ratio_is_zero_without_load():
    assert (
        eplb_metrics.compute_critical_overhead_ratio(
            logical_expert_load=torch.zeros((1, 2), dtype=torch.int64),
            placement=[[[0], [1]]],
            expert_alignment=128,
        )
        == 0.0
    )


def test_logical_expert_imbalance_percentiles_report_layer_distribution():
    load = torch.tensor(
        [
            [3, 3, 3, 3],
            [6, 2, 2, 2],
            [9, 1, 1, 1],
            [12, 0, 0, 0],
            [0, 0, 0, 0],
        ],
        dtype=torch.int64,
    )

    assert eplb_metrics.logical_expert_imbalance_percentiles(expert_load=load) == {
        25: pytest.approx(1.0),
        50: pytest.approx(2.0),
        100: pytest.approx(4.0),
    }


def test_logical_expert_imbalance_percentiles_are_zero_without_load():
    assert eplb_metrics.logical_expert_imbalance_percentiles(expert_load=torch.zeros((3, 4), dtype=torch.int64)) == {
        25: 0.0,
        50: 0.0,
        100: 0.0,
    }


def test_publish_expert_load_metrics():
    calls = []
    metric_client = SimpleNamespace(gauge_set=lambda name, value: calls.append((name, value)))

    eplb_metrics.publish_expert_load_metrics(
        metric_client=metric_client,
        expert_load=torch.tensor([[384, 128, 128, 128]]),
    )

    assert calls == [
        (eplb_metrics.EXPERT_IMBALANCE_RATIO_METRICS[25], pytest.approx(2.0)),
        (eplb_metrics.EXPERT_IMBALANCE_RATIO_METRICS[50], pytest.approx(2.0)),
        (eplb_metrics.EXPERT_IMBALANCE_RATIO_METRICS[100], pytest.approx(2.0)),
    ]


def test_prefill_rebalance_metric_load_uses_one_sample_across_ranks():
    route_statistics = torch.zeros((2, 1, 2, 4), dtype=torch.int64)
    route_statistics[0, 0, 0].fill_(100)
    route_statistics[1, 0, 0].fill_(50)
    route_statistics[1, 0, 1].fill_(1000)

    metric_load = eplb_metrics.prefill_rebalance_metric_load(
        route_statistics=route_statistics,
    )

    assert torch.equal(metric_load, torch.full((1, 4), 150, dtype=torch.int64))


def test_publish_rebalance_compute_metrics_from_sample_load():
    calls = []
    metric_client = SimpleNamespace(gauge_set=lambda name, value: calls.append((name, value)))

    eplb_metrics.publish_rebalance_compute_metrics(
        metric_client=metric_client,
        sample_load=torch.tensor([[384, 128, 128, 128]]),
        current_placement=[[[0, 1, 2], [1, 2, 3]]],
        target_placement=[[[0, 1, 2], [0, 2, 3]]],
        expert_alignment=128,
    )

    assert calls == [
        (
            eplb_metrics.COMPUTE_CRITICAL_OVERHEAD_RATIO_BEFORE_REBALANCE_METRIC,
            pytest.approx(0.25),
        ),
        (
            eplb_metrics.COMPUTE_CRITICAL_OVERHEAD_RATIO_AFTER_REBALANCE_METRIC,
            pytest.approx(0.0),
        ),
    ]


@pytest.mark.parametrize("run_mode", ["prefill", "decode"])
def test_eplb_allocates_only_the_active_route_counter(monkeypatch, run_mode):
    args = type(
        "Args",
        (),
        {
            "eplb_num_redundant_experts_per_rank": 2,
            "eplb_config_path": None,
            "eplb_run_mode": run_mode,
        },
    )()
    monkeypatch.setattr(deepgemm_module, "get_env_start_args", lambda: args)
    monkeypatch.setattr(deepgemm_module, "get_global_world_size", lambda: 2)
    monkeypatch.setattr(deepgemm_module, "get_global_rank", lambda: 0)
    monkeypatch.setattr(deepgemm_module, "get_node_world_size", lambda: 2)
    monkeypatch.setattr(torch.Tensor, "cuda", lambda tensor: tensor)
    original_zeros = torch.zeros

    def cpu_zeros(*shape, **kwargs):
        kwargs.pop("device", None)
        return original_zeros(*shape, **kwargs)

    monkeypatch.setattr(deepgemm_module.torch, "zeros", cpu_zeros)

    impl = deepgemm_module.FuseMoeDeepGEMM(4, 0, 1.0, SimpleNamespace())

    if run_mode == "prefill":
        assert impl.prefill_route_counter.shape == (24, 4)
        assert impl.prefill_route_sample_index.shape == (2,)
        assert impl.decode_route_counter is None
        assert torch.equal(impl.prefill_route_sample_index, torch.zeros(2, dtype=torch.int64))
    else:
        assert impl.prefill_route_counter is None
        assert impl.prefill_route_sample_index is None
        assert impl.decode_route_counter.shape == (4, 4)
        assert torch.count_nonzero(impl.decode_route_counter) == 0
    assert not hasattr(impl, "eplb_dispatch_mode")


@pytest.mark.parametrize(
    ("plan_mode", "dispatch_mode"),
    [
        ("global_balance", "global_first"),
        ("topology_aware", "current_node_first"),
    ],
)
def test_prefill_dispatch_mode_matches_planner_load_model(monkeypatch, plan_mode, dispatch_mode):
    monkeypatch.setattr(
        eplb_utils_module,
        "get_env_start_args",
        lambda: SimpleNamespace(eplb_plan_mode=plan_mode),
    )
    assert eplb_utils_module.get_eplb_dispatch_mode(is_prefill=True) == dispatch_mode


def test_prefill_dispatch_mode_uses_node_first_for_unknown_planner(monkeypatch):
    monkeypatch.setattr(
        eplb_utils_module,
        "get_env_start_args",
        lambda: SimpleNamespace(eplb_plan_mode="unknown"),
    )
    assert eplb_utils_module.get_eplb_dispatch_mode(is_prefill=True) == "current_node_first"


@pytest.mark.parametrize("plan_mode", ["global_balance", "topology_aware"])
def test_decode_dispatch_mode_prefers_current_gpu(monkeypatch, plan_mode):
    monkeypatch.setattr(
        eplb_utils_module,
        "get_env_start_args",
        lambda: SimpleNamespace(eplb_plan_mode=plan_mode),
    )
    assert eplb_utils_module.get_eplb_dispatch_mode(is_prefill=False) == "current_gpu_first"


def test_decode_dispatch_mode_uses_gpu_first_for_unknown_planner(monkeypatch):
    monkeypatch.setattr(
        eplb_utils_module,
        "get_env_start_args",
        lambda: SimpleNamespace(eplb_plan_mode="unknown"),
    )
    assert eplb_utils_module.get_eplb_dispatch_mode(is_prefill=False) == "current_gpu_first"


def test_ep_without_eplb_creates_layout_without_eplb_runtime_state(monkeypatch):
    args = type(
        "Args",
        (),
        {"eplb_num_redundant_experts_per_rank": 0},
    )()
    monkeypatch.setattr(deepgemm_module, "get_env_start_args", lambda: args)
    monkeypatch.setattr(deepgemm_module, "get_global_world_size", lambda: 2)
    monkeypatch.setattr(deepgemm_module, "get_global_rank", lambda: 0)

    impl = deepgemm_module.FuseMoeDeepGEMM(4, 0, 1.0, SimpleNamespace())

    assert impl.num_redundant_experts_per_rank == 0
    assert impl.num_total_physical_experts == impl.n_routed_experts
    assert impl.local_logics_expert_ids_list == [0, 1]
    assert not hasattr(impl, "num_primary_experts_per_rank")
    assert not hasattr(impl, "initial_local_expert_ids_by_rank")
    assert not hasattr(impl, "logical_to_physical_map")
    assert not hasattr(impl, "prefill_route_counter")
    assert not hasattr(impl, "prefill_route_sample_index")
    assert not hasattr(impl, "decode_route_counter")


def test_manager_wait_load_gather_aggregates_rank_and_sample_dimensions(monkeypatch):
    manager = manager_module.EPLBManager.__new__(manager_module.EPLBManager)
    manager._eplb_impls = [object()]
    manager.global_rank = 0
    manager.world_size = 4
    manager.num_logical_experts = 4
    manager.run_mode = "prefill"
    manager.control_group = object()
    manager.state = manager_module.EPLBManagerState.WAIT_LOAD_GATHER_FINISHED
    gathered_load = torch.zeros((4, 1, 24, 4), dtype=torch.int64)
    gathered_load[0, 0, 0].fill_(100)
    gathered_load[1, 0, 1].fill_(100)
    gathered_load[2, 0, 0].fill_(50)
    manager._load_gather_task = SimpleNamespace(
        is_finished=lambda: True,
        result=gathered_load,
    )
    seen = {}

    def all_gather_object(output, local_finished, **kwargs):
        seen["local_finished"] = local_finished
        seen["group"] = kwargs["group"]
        output[:] = [True] * manager.world_size

    monkeypatch.setattr(manager_module.dist, "all_gather_object", all_gather_object)

    manager._step_wait_load_gather_finished()

    assert seen["group"] is manager.control_group
    assert seen["local_finished"] is True
    assert manager.state is manager_module.EPLBManagerState.PLAN_PLACEMENT
    assert manager._pending_route_statistics is gathered_load
    assert manager._load_gather_task is None


def test_manager_wait_load_gather_prepares_decode_statistics_for_planning(monkeypatch):
    manager = manager_module.EPLBManager.__new__(manager_module.EPLBManager)
    manager._eplb_impls = [object()]
    manager.global_rank = 0
    manager.world_size = 2
    manager.num_logical_experts = 3
    manager.run_mode = "decode"
    manager.control_group = object()
    manager.state = manager_module.EPLBManagerState.WAIT_LOAD_GATHER_FINISHED
    gathered_decode_statistics = torch.tensor(
        [
            [[[10, 3, 1], [0, 20, 4], [0, 0, 30]]],
            [[[40, 2, 5], [0, 50, 6], [0, 0, 60]]],
        ],
        dtype=torch.int64,
    )
    manager._load_gather_task = SimpleNamespace(
        is_finished=lambda: True,
        result=gathered_decode_statistics,
    )
    monkeypatch.setattr(
        manager_module.dist,
        "all_gather_object",
        lambda output, local_finished, **_kwargs: output.__setitem__(slice(None), [local_finished] * 2),
    )
    manager._step_wait_load_gather_finished()

    assert manager.state is manager_module.EPLBManagerState.PLAN_PLACEMENT
    assert manager._pending_route_statistics is gathered_decode_statistics


def test_manager_wait_load_gather_does_not_advance_until_every_rank_finishes(monkeypatch):
    manager = manager_module.EPLBManager.__new__(manager_module.EPLBManager)
    manager.state = manager_module.EPLBManagerState.WAIT_LOAD_GATHER_FINISHED
    manager.world_size = 2
    manager.control_group = object()
    load_gather_task = SimpleNamespace(
        is_finished=lambda: True,
        result=torch.ones((2, 1, 1, 2), dtype=torch.int64),
    )
    manager._load_gather_task = load_gather_task
    monkeypatch.setattr(
        manager_module.dist,
        "all_gather_object",
        lambda output, _local_finished, **_kwargs: output.__setitem__(slice(None), [True, False]),
    )

    manager._step_wait_load_gather_finished()

    assert manager.state is manager_module.EPLBManagerState.WAIT_LOAD_GATHER_FINISHED
    assert manager._load_gather_task is load_gather_task


def test_decode_dispatch_uses_physical_ids_and_total_expert_count(monkeypatch):
    class Event:
        def current_stream_wait(self):
            pass

    class Buffer:
        def dispatch(self, _qinput_tensor, **kwargs):
            calls.append(kwargs)
            handle = SimpleNamespace(psum_num_recv_tokens_per_expert="masked")
            return "recv", None, "recv_topk_weights", handle, Event()

    impl = object.__new__(deepgemm_module.FuseMoeDeepGEMM)
    impl.quant_method = type("Quant", (), {"method_name": "fp8"})()
    impl.n_routed_experts = 128
    _set_deepgemm_runtime(impl, _test_moe_impl(eplb=True, run_mode="decode"))
    logical_ids = torch.tensor([[0, 127]], dtype=torch.int32)
    physical_ids = torch.tensor([[128, 143]], dtype=torch.int32)
    impl._select_experts = lambda **_kwargs: (
        torch.ones((1, 2)),
        logical_ids,
    )
    calls, repairs = [], []

    def repair(**kwargs):
        repairs.append(kwargs)
        return physical_ids

    monkeypatch.setattr(deepgemm_module, "eplb_repair_topk_ids", repair)
    monkeypatch.setattr(
        eplb_utils_module,
        "get_env_start_args",
        lambda: SimpleNamespace(eplb_plan_mode="topology_aware", eplb_run_mode="decode"),
    )
    monkeypatch.setattr(
        deepgemm_module,
        "get_deepep_num_max_dispatch_tokens_per_rank_decode",
        lambda: 16,
    )
    monkeypatch.setattr(
        deepgemm_module.grouped_fused_moe_ep,
        "quantize_fused_experts_input",
        lambda *_args: "qinput",
    )
    monkeypatch.setattr(deepgemm_module.dist_group_manager, "ep_buffer", Buffer())

    recv_x, ep_handle, recv_topk_weights, hook = impl.decode_dispatch(
        torch.empty((1, 4)),
        object(),
        torch.empty((1, 128)),
        None,
        False,
        2,
        False,
        0,
        0,
        "softmax",
    )

    assert recv_x == "recv"
    assert ep_handle.psum_num_recv_tokens_per_expert == "masked"
    assert recv_topk_weights == "recv_topk_weights"
    assert callable(hook)
    assert calls[0]["topk_idx"].tolist() == [[128, 143]]
    assert repairs[0]["logical_topk_ids"] is logical_ids
    assert not repairs[0]["update_prefill_route_counter"]
    assert repairs[0]["update_decode_route_counter"]
    assert repairs[0]["decode_route_counter"] is impl.decode_route_counter
    assert repairs[0]["mode"] == "current_gpu_first"
    assert calls[0]["num_experts"] == 144
    assert calls[0]["allocate_on_comm_stream"] is False


def test_select_returns_logical_ids_and_applies_expert_scale(monkeypatch):
    from lightllm.common.basemodel.triton_kernel.fused_moe import topk_select

    impl = object.__new__(deepgemm_module.FuseMoeDeepGEMM)
    impl.routed_scaling_factor = 2.0
    _set_deepgemm_runtime(impl, _test_moe_impl(eplb=True))
    logical_ids = torch.tensor([[3, 4]], dtype=torch.int32)
    calls = []

    def select(**kwargs):
        calls.append(kwargs)
        return torch.tensor([[0.5, 0.25]]), logical_ids

    monkeypatch.setattr(topk_select, "select_experts", select)
    weights, selected = impl._select_experts(
        torch.empty((1, 4)),
        torch.empty((1, 128)),
        None,
        2,
        False,
        False,
        0,
        0,
        "softmax",
        per_expert_scale=torch.tensor([1.0, 1.0, 1.0, 3.0, 5.0]),
    )

    assert len(calls) == 1
    assert weights.tolist() == [[3.0, 2.5]]
    assert selected is logical_ids


def test_eplb_prefill_repairs_ids_after_selection(monkeypatch):
    impl = object.__new__(deepgemm_module.FuseMoeDeepGEMM)
    impl.routed_scaling_factor = 1.0
    impl.quant_method = object()
    _set_deepgemm_runtime(impl, _test_moe_impl(eplb=True))
    logical_ids = torch.tensor([[3, 4]], dtype=torch.int32)
    physical_ids = torch.tensor([[130, 131]], dtype=torch.long)
    calls = []
    impl._select_experts = lambda **_kwargs: (torch.ones((1, 2)), logical_ids)

    def repair(**kwargs):
        calls.append(kwargs)
        return physical_ids

    monkeypatch.setattr(deepgemm_module, "eplb_repair_topk_ids", repair)
    monkeypatch.setattr(
        eplb_utils_module,
        "get_env_start_args",
        lambda: SimpleNamespace(eplb_plan_mode="global_balance", eplb_run_mode="prefill"),
    )
    monkeypatch.setattr(
        deepgemm_module.grouped_fused_moe_ep,
        "quantize_fused_experts_input",
        lambda *_args: "qinput",
    )

    weights, topk_idx, qinput = impl.select_experts_and_quant_input(
        torch.empty((1, 4)),
        torch.empty((1, 128)),
        None,
        object(),
        False,
        2,
        False,
        0,
        0,
        "softmax",
    )

    assert weights.tolist() == [[1.0, 1.0]]
    assert topk_idx is physical_ids
    assert topk_idx.dtype is torch.long
    assert qinput == "qinput"
    assert calls[0]["logical_topk_ids"] is logical_ids
    assert calls[0]["update_prefill_route_counter"]
    assert not calls[0]["update_decode_route_counter"]
    assert calls[0]["mode"] == "global_first"


def test_eplb_prefill_dispatch_consumes_physical_ids_and_event(monkeypatch):
    class Buffer:
        def dispatch(self, _qinput, **kwargs):
            calls.append(kwargs)
            return (
                (torch.empty((4, 2)),),
                "recv_idx",
                "recv_weight",
                SimpleNamespace(num_recv_tokens_per_expert_list=[4]),
                SimpleNamespace(current_stream_wait=lambda: None),
            )

    impl = object.__new__(deepgemm_module.FuseMoeDeepGEMM)
    impl.routed_scaling_factor = 1.0
    impl.quant_method = object()
    runtime = _test_moe_impl(
        eplb=True,
        prefill_route_counter=torch.zeros((24, 128), dtype=torch.int64),
    )
    _set_deepgemm_runtime(impl, runtime)
    calls, repair_calls = [], []
    logical_ids = torch.tensor([[3, 4]], dtype=torch.int32)
    physical_ids = torch.tensor([[130, 131]], dtype=torch.long)
    impl._select_experts = lambda **_kwargs: (torch.ones((1, 2)), logical_ids)

    def repair(**kwargs):
        repair_calls.append(kwargs)
        return physical_ids

    monkeypatch.setattr(deepgemm_module, "eplb_repair_topk_ids", repair)
    monkeypatch.setattr(
        eplb_utils_module,
        "get_env_start_args",
        lambda: SimpleNamespace(eplb_plan_mode="global_balance", eplb_run_mode="prefill"),
    )
    monkeypatch.setattr(
        deepgemm_module.grouped_fused_moe_ep,
        "quantize_fused_experts_input",
        lambda *_args: "qinput",
    )
    monkeypatch.setattr(deepgemm_module.dist_group_manager, "ep_buffer", Buffer())
    monkeypatch.setattr(
        deepgemm_module,
        "get_deepep_num_max_dispatch_tokens_per_rank_prefill",
        lambda: 16,
    )
    monkeypatch.setattr(
        deepgemm_module.grouped_fused_moe_ep,
        "get_ep_num_sms",
        lambda overlap_with_compute: 8,
    )

    weights, topk_idx, qinput = impl.select_experts_and_quant_input(
        torch.empty((1, 4)),
        torch.empty((1, 128)),
        None,
        object(),
        True,
        2,
        False,
        1,
        8,
        "sigmoid",
    )
    caller_event = object()
    impl.prefill_dispatch(
        qinput,
        topk_idx,
        weights,
        overlap_event=caller_event,
    )

    assert topk_idx is physical_ids
    assert len(repair_calls) == 1
    assert repair_calls[0]["logical_topk_ids"] is logical_ids
    assert repair_calls[0]["update_prefill_route_counter"]
    assert not repair_calls[0]["update_decode_route_counter"]
    assert repair_calls[0]["mode"] == "global_first"
    assert calls[0]["topk_idx"] is physical_ids
    assert calls[0]["topk_idx"].dtype is torch.long
    assert calls[0]["previous_event"] is caller_event


def test_prefill_dispatch_preserves_event(monkeypatch):
    class Buffer:
        def dispatch(self, _qinput, **kwargs):
            calls.append(kwargs)
            return (
                (torch.empty((4, 2)),),
                "recv_idx",
                "recv_weight",
                SimpleNamespace(num_recv_tokens_per_expert_list=[4]),
                SimpleNamespace(current_stream_wait=lambda: None),
            )

    impl = object.__new__(deepgemm_module.FuseMoeDeepGEMM)
    _set_deepgemm_runtime(impl, _test_moe_impl(eplb=True))
    calls = []
    caller_event = object()
    monkeypatch.setattr(deepgemm_module.dist_group_manager, "ep_buffer", Buffer())
    monkeypatch.setattr(
        deepgemm_module,
        "get_deepep_num_max_dispatch_tokens_per_rank_prefill",
        lambda: 16,
    )
    monkeypatch.setattr(
        deepgemm_module.grouped_fused_moe_ep,
        "get_ep_num_sms",
        lambda overlap_with_compute: 8,
    )

    impl.prefill_dispatch(
        "qinput",
        torch.tensor([[1, 2]], dtype=torch.long),
        torch.ones((1, 2)),
        caller_event,
    )

    assert calls[0]["previous_event"] is caller_event
    assert calls[0]["topk_idx"].dtype is torch.long


def test_deepgemm_constructor_owns_eplb_runtime(monkeypatch):
    monkeypatch.setattr(
        deepgemm_module,
        "get_env_start_args",
        lambda: SimpleNamespace(
            eplb_num_redundant_experts_per_rank=1,
            eplb_config_path=None,
            eplb_run_mode="prefill",
        ),
    )
    monkeypatch.setattr(deepgemm_module, "get_global_world_size", lambda: 2)
    monkeypatch.setattr(deepgemm_module, "get_global_rank", lambda: 0)
    monkeypatch.setattr(deepgemm_module, "get_node_world_size", lambda: 2)
    monkeypatch.setattr(torch.Tensor, "cuda", lambda tensor: tensor)
    original_zeros = torch.zeros

    def cpu_zeros(*shape, **kwargs):
        kwargs.pop("device", None)
        return original_zeros(*shape, **kwargs)

    monkeypatch.setattr(deepgemm_module.torch, "zeros", cpu_zeros)
    impl = deepgemm_module.FuseMoeDeepGEMM(4, 0, 1.0, SimpleNamespace())

    assert impl.num_redundant_experts_per_rank == 1
    assert impl.num_total_physical_experts == 6
    assert impl.prefill_route_counter.shape == (24, 4)
    assert impl.prefill_route_sample_index.shape == (2,)
    assert impl.decode_route_counter is None
    assert impl.local_logics_expert_ids_list == [0, 1, 2]
    assert not hasattr(impl, "initial_local_expert_ids_by_rank")
    assert not hasattr(impl, "expert_parallel_state")


def test_deepgemm_constructor_loads_saved_layout_before_weight_initialization(monkeypatch):
    saved_placement = [[1, 0, 3], [2, 3, 1]]
    monkeypatch.setattr(
        deepgemm_module,
        "get_env_start_args",
        lambda: SimpleNamespace(
            eplb_num_redundant_experts_per_rank=1,
            eplb_config_path="/tmp/eplb.json",
            eplb_run_mode="prefill",
        ),
    )
    monkeypatch.setattr(deepgemm_module, "get_global_world_size", lambda: 2)
    monkeypatch.setattr(deepgemm_module, "get_global_rank", lambda: 0)
    monkeypatch.setattr(deepgemm_module, "get_node_world_size", lambda: 2)
    monkeypatch.setattr(torch.Tensor, "cuda", lambda tensor: tensor)
    monkeypatch.setattr(
        deepgemm_module,
        "load_layer_placement",
        lambda path, **kwargs: (
            saved_placement
            if path == "/tmp/eplb.json"
            and kwargs
            == {
                "layer_index": 7,
                "num_logical_experts": 4,
                "world_size": 2,
                "num_redundant_experts_per_rank": 1,
            }
            else None
        ),
    )
    original_zeros = torch.zeros
    monkeypatch.setattr(
        deepgemm_module.torch,
        "zeros",
        lambda *shape, **kwargs: original_zeros(*shape, dtype=kwargs.get("dtype")),
    )

    impl = deepgemm_module.FuseMoeDeepGEMM(4, 0, 1.0, SimpleNamespace(), layer_index=7)

    assert impl.local_logics_expert_ids_list == saved_placement[0]
    expected_map = build_logical_to_physical_map(saved_placement, 4, current_rank=0, node_world_size=2)
    assert impl.logical_to_physical_map.tolist() == expected_map


@pytest.mark.parametrize(
    ("run_mode", "is_prefill", "update_prefill_counter", "update_decode_counter"),
    [
        ("prefill", True, True, False),
        ("prefill", False, False, False),
        ("decode", True, False, False),
        ("decode", False, False, True),
    ],
)
def test_eplb_prepare_records_statistics_for_matching_run_mode_and_phase(
    monkeypatch,
    run_mode,
    is_prefill,
    update_prefill_counter,
    update_decode_counter,
):
    impl = object.__new__(deepgemm_module.FuseMoeDeepGEMM)
    runtime = _test_moe_impl(eplb=True, run_mode=run_mode)
    _set_deepgemm_runtime(impl, runtime)
    logical_ids = torch.tensor([[3, 4]], dtype=torch.int32)
    physical_ids = torch.tensor([[13, 14]], dtype=torch.int32)
    calls = []

    def repair(**kwargs):
        calls.append(kwargs)
        return physical_ids

    monkeypatch.setattr(deepgemm_module, "eplb_repair_topk_ids", repair)
    monkeypatch.setattr(
        eplb_utils_module,
        "get_env_start_args",
        lambda: SimpleNamespace(eplb_plan_mode="global_balance", eplb_run_mode=run_mode),
    )
    weights, selected = impl._prepare_expert_execution(torch.ones((1, 2)), logical_ids, is_prefill=is_prefill)

    assert weights.tolist() == [[1.0, 1.0]]
    assert selected is physical_ids
    assert calls[0]["logical_topk_ids"] is logical_ids
    assert calls[0]["update_prefill_route_counter"] is update_prefill_counter
    assert calls[0]["update_decode_route_counter"] is update_decode_counter
    assert calls[0]["mode"] == ("global_first" if is_prefill else "current_gpu_first")
    if run_mode == "prefill":
        assert calls[0]["prefill_route_counter"] is not None
        assert calls[0]["prefill_route_sample_index"] is not None
        assert calls[0]["decode_route_counter"] is None
    else:
        assert calls[0]["prefill_route_counter"] is None
        assert calls[0]["prefill_route_sample_index"] is None
        assert calls[0]["decode_route_counter"] is not None


def test_topology_aware_prepare_uses_node_local_dispatch(monkeypatch):
    impl = object.__new__(deepgemm_module.FuseMoeDeepGEMM)
    runtime = _test_moe_impl(eplb=True)
    _set_deepgemm_runtime(impl, runtime)
    logical_ids = torch.tensor([[3, 4]], dtype=torch.int32)
    calls = []

    def repair(**kwargs):
        calls.append(kwargs)
        return logical_ids

    monkeypatch.setattr(deepgemm_module, "eplb_repair_topk_ids", repair)
    monkeypatch.setattr(
        eplb_utils_module,
        "get_env_start_args",
        lambda: SimpleNamespace(eplb_plan_mode="topology_aware", eplb_run_mode="prefill"),
    )

    impl._prepare_expert_execution(torch.ones((1, 2)), logical_ids, is_prefill=True)

    assert calls[0]["mode"] == "current_node_first"


def test_decode_masked_group_gemm_uses_all_physical_rows_when_eplb_is_enabled(
    monkeypatch,
):
    impl = object.__new__(deepgemm_module.FuseMoeDeepGEMM)
    _set_deepgemm_runtime(impl, _test_moe_impl(eplb=True, num_logical_experts=8, world_size=1))
    captured = {}

    def grouped_gemm(**kwargs):
        captured.update(kwargs)
        return "out"

    monkeypatch.setattr(deepgemm_module.grouped_fused_moe_ep, "decode_masked_group_gemm", grouped_gemm)
    pack = lambda: type(
        "Pack",
        (),
        {"weight": torch.empty((10, 4)), "weight_scale": torch.empty((10, 1))},
    )()

    ep_handle = SimpleNamespace(
        psum_num_recv_tokens_per_expert=torch.empty(8),
        expert_alignment=128,
    )
    assert impl.decode_masked_group_gemm((torch.empty((1, 4)),), pack(), pack(), ep_handle, torch.float16, 1) == "out"
    assert captured["expert_alignment"] == 128
    assert captured["w1"].shape[0] == captured["w2"].shape[0] == 10
    assert captured["w1_scale"].shape[0] == captured["w2_scale"].shape[0] == 10


def test_decode_fused_experts_uses_full_weight_packs_and_physical_experts(
    monkeypatch,
):
    impl = object.__new__(deepgemm_module.FuseMoeDeepGEMM)
    impl.n_routed_experts = 128
    _set_deepgemm_runtime(
        impl,
        _test_moe_impl(
            eplb=True,
            num_logical_experts=128,
            world_size=16,
            num_redundant_experts_per_rank=2,
        ),
    )
    impl.quant_method = object()
    captured = []

    def fused(**kwargs):
        captured.append(kwargs)
        return "out"

    monkeypatch.setattr(deepgemm_module.grouped_fused_moe_ep, "fused_experts", fused)
    pack = lambda: type(
        "Pack",
        (),
        {
            "weight": torch.empty((10, 4)),
            "weight_scale": torch.empty((10, 1)),
            "weight_zero_point": None,
        },
    )()
    w13, w2 = pack(), pack()

    for _ in range(2):
        assert (
            impl._fused_experts(
                torch.empty((1, 4)),
                w13,
                w2,
                torch.ones((1, 2)),
                torch.zeros((1, 2), dtype=torch.int64),
                is_prefill=False,
            )
            == "out"
        )

    assert [call["num_experts"] for call in captured] == [160, 160]
    assert all(call["w13"] is w13 and call["w2"] is w2 for call in captured)


def test_transfer_plan_uses_stable_current_expert_source():
    current = [[0, 1, 4, 5], [2, 3, 6, 7], [4, 5, 0, 1], [6, 7, 2, 3]]
    target = [[0, 1, 6, 5], [2, 3, 6, 7], [4, 5, 0, 4], [6, 7, 2, 3]]
    plan = build_transfer_plan(current, target, 5, num_logical_experts=8, world_size=4)
    assert plan == [
        [
            EPLBTransferInfo(5, 6, 1, 2, 0, 2),
            EPLBTransferInfo(5, 4, 2, 0, 2, 3),
        ],
    ]


def test_transfer_plan_reuses_stable_source_for_repeated_expert():
    current = [[0, 1, 0, 1], [2, 3, 2, 3], [4, 5, 4, 5], [6, 7, 4, 7]]
    target = [[0, 1, 4, 4], [2, 3, 2, 3], [4, 5, 4, 5], [6, 7, 4, 7]]
    first = build_transfer_plan(current, target, 5, 8, 4)
    second = build_transfer_plan(current, target, 5, 8, 4)
    assert first == second
    assert first == [
        [EPLBTransferInfo(5, 4, 2, 0, 0, 2)],
        [EPLBTransferInfo(5, 4, 2, 2, 0, 3)],
    ]


def test_transfer_plan_keeps_primary_slot_swap_in_one_atomic_batch():
    current = [[0, 1], [2, 3]]
    target = [[2, 1], [0, 3]]

    plan = build_transfer_plan(current, target, 0, num_logical_experts=4, world_size=2)

    assert plan == [
        [
            EPLBTransferInfo(0, 2, 1, 0, 0, 0),
            EPLBTransferInfo(0, 0, 0, 0, 1, 0),
        ]
    ]


def test_transfer_plan_keeps_three_way_cycle_in_one_atomic_batch():
    current = [[0], [1], [2]]
    target = [[1], [2], [0]]

    plan = build_transfer_plan(current, target, 0, num_logical_experts=3, world_size=3)

    assert plan == [
        [
            EPLBTransferInfo(0, 1, 1, 0, 0, 0),
            EPLBTransferInfo(0, 0, 0, 0, 2, 0),
            EPLBTransferInfo(0, 2, 2, 0, 1, 0),
        ]
    ]


def test_p2p_message_tag_is_stable_and_identifies_transfer_tensor():
    transfer = object.__new__(PinnedMemoryEPLBTransfer)
    transfer.transfer_info = EPLBTransferInfo(5, 4, 1, 0, 0, 2)
    weight_tag = transfer._build_p2p_message_tag("w13.weight")

    assert weight_tag == transfer._build_p2p_message_tag("w13.weight")
    assert 0 <= weight_tag <= 0x7FFFFFFF
    assert weight_tag != transfer._build_p2p_message_tag("w13.weight_scale")
    transfer.transfer_info = EPLBTransferInfo(5, 6, 1, 0, 0, 2)
    assert weight_tag != transfer._build_p2p_message_tag("w13.weight")
    transfer.transfer_info = EPLBTransferInfo(5, 4, 1, 1, 0, 2)
    assert weight_tag != transfer._build_p2p_message_tag("w13.weight")
    transfer.transfer_info = EPLBTransferInfo(5, 4, 1, 0, 0, 3)
    assert weight_tag != transfer._build_p2p_message_tag("w13.weight")


def test_extract_expert_tensors_includes_quantization_metadata_in_order():
    class Pack:
        def __init__(self, offset, scale=True, zero_point=True):
            self.weight = torch.full((3, 2), offset)
            self.weight_scale = torch.full((3, 1), offset + 1) if scale else None
            self.weight_zero_point = torch.full((3, 1), offset + 2) if zero_point else None

    weight = type("Weight", (), {"w13": Pack(1), "w2": Pack(10, scale=False, zero_point=False)})()
    tensors = extract_eplb_expert_tensors(weight)
    assert [name for name, _ in tensors] == [
        "w13.weight",
        "w13.weight_scale",
        "w13.weight_zero_point",
        "w2.weight",
    ]


def test_manager_commits_transfer_rows_and_metadata(monkeypatch):
    original_copy = torch.Tensor.copy_
    non_blocking_values = []
    copy_sources = []

    def record_copy(tensor, source, non_blocking=False):
        non_blocking_values.append(non_blocking)
        copy_sources.append(source)
        return original_copy(tensor, source, non_blocking=non_blocking)

    monkeypatch.setattr(torch.Tensor, "copy_", record_copy)
    live = torch.arange(20).reshape(5, 4)
    original_primary = live[:3].clone()
    local_expert_ids = [0, 1, 2, 3, 2]
    target_placement = [[[0, 1, 2, 4, 5], [3, 4, 5, 0, 2]]]
    expected_metadata = torch.tensor(
        build_logical_to_physical_map(
            target_placement[0],
            6,
            current_rank=0,
            node_world_size=2,
        ),
        dtype=torch.int32,
    )
    logical_to_physical_map = torch.zeros_like(expected_metadata)
    manager = manager_module.EPLBManager.__new__(manager_module.EPLBManager)
    manager.global_rank = 0
    manager.world_size = 2
    manager.node_world_size = 2
    manager.num_logical_experts = 6
    manager.target_placement = target_placement
    manager.current_placement = [[[0, 1, 2, 3, 2], [3, 4, 5, 0, 2]]]
    manager._eplb_impls = [
        SimpleNamespace(
            local_logics_expert_ids_list=local_expert_ids,
            logical_to_physical_map=logical_to_physical_map,
        )
    ]
    transfers = [
        SimpleNamespace(
            transfer_info=EPLBTransferInfo(0, 4, 1, 1, 0, 3),
            tensor_buffers=[ExpertTensorBuffer("weight", live, torch.full((4,), -4))],
        ),
        SimpleNamespace(
            transfer_info=EPLBTransferInfo(0, 5, 1, 2, 0, 4),
            tensor_buffers=[ExpertTensorBuffer("weight", live, torch.full((4,), -5))],
        ),
    ]
    manager.active_transfers = [transfers[0]]
    manager._commit_transfer(transfers[0].transfer_info)
    assert manager.current_placement[0][0] == [0, 1, 2, 4, 2]
    manager.active_transfers = [transfers[1]]
    manager._commit_transfer(transfers[1].transfer_info)
    manager._publish_layer_metadata(0)

    assert torch.equal(live[:3], original_primary)
    assert torch.equal(live[3], torch.full((4,), -4))
    assert torch.equal(live[4], torch.full((4,), -5))
    assert local_expert_ids == [0, 1, 2, 4, 5]
    assert torch.equal(logical_to_physical_map, expected_metadata)
    assert non_blocking_values == [True, True, True]
    assert copy_sources[-1].is_pinned()


def test_manager_transfers_only_local_tasks_and_gathers_global_status(monkeypatch):
    class Transfer:
        def __init__(self, transfer_info):
            self.transfer_info = transfer_info
            self.finished = finished_by_info[transfer_info]

        def start(self):
            starts.append(self.transfer_info)

        def is_finished(self):
            return self.finished

    remote_info = EPLBTransferInfo(0, 2, 0, 0, 2, 2)
    local_info0 = EPLBTransferInfo(0, 3, 1, 1, 3, 2)
    local_info1 = EPLBTransferInfo(1, 4, 0, 0, 1, 2)
    finished_by_info = {local_info0: False, local_info1: True}
    starts = []
    manager = manager_module.EPLBManager.__new__(manager_module.EPLBManager)
    manager.control_group = object()
    manager.transfer_group = object()
    manager._weights = [object(), object()]
    manager.world_size = 4
    manager.pending_transfer_batches = [[remote_info, local_info0], [local_info1]]
    manager.target_placement = [
        [[0, 1, 2], [2, 3, 3], [4, 5, 2], [6, 7, 0]],
        [[0, 1, 4], [2, 3, 5], [4, 5, 6], [6, 7, 1]],
    ]
    manager.current_placement = [
        [[0, 1, 4], [2, 3, 5], [4, 5, 6], [6, 7, 0]],
        [[0, 1, 2], [2, 3, 4], [4, 5, 6], [6, 7, 0]],
    ]
    manager.global_rank = 1
    manager.state = manager_module.EPLBManagerState.TRANSFERRING
    manager.max_rebalance_count = -1
    manager.completed_rebalance_count = 0
    manager.rebalance_started_at = None
    manager.active_transfer_batch = None
    manager.active_transfers = None
    committed = []
    active_streams = []
    cleared_route_counters = []

    def commit_transfer(transfer_info):
        assert active_streams == [overlap_stream]
        committed.append(transfer_info)

    def publish_layer_metadata(_layer_index):
        assert active_streams == [overlap_stream]

    manager._commit_transfer = commit_transfer
    manager._publish_layer_metadata = publish_layer_metadata
    manager._clear_route_statistics = lambda: cleared_route_counters.append(True)
    used_streams = []
    overlap_stream = object()

    class StreamContext:
        def __init__(self, stream):
            self.stream = stream

        def __enter__(self):
            used_streams.append(self.stream)
            active_streams.append(self.stream)

        def __exit__(self, *_args):
            active_streams.pop()

    monkeypatch.setattr(
        manager_module.torch.cuda,
        "stream",
        StreamContext,
    )
    monkeypatch.setattr(g_infer_context, "get_overlap_stream", lambda: overlap_stream)
    monkeypatch.setattr(
        manager_module,
        "PinnedMemoryEPLBTransfer",
        lambda **kwargs: Transfer(kwargs["transfer_info"]),
    )

    gathered_states = [
        [True, False, True, False],
        [True, True, True, True],
        [True, True, True, True],
    ]
    local_states = []

    def all_gather_object(output, local_state, **_kwargs):
        local_states.append(local_state)
        output[:] = gathered_states.pop(0)

    monkeypatch.setattr(manager_module.dist, "all_gather_object", all_gather_object)

    manager._step_transferring()
    assert starts == [local_info0]
    assert committed == []
    assert manager.active_transfer_batch == [remote_info, local_info0]
    assert manager.pending_transfer_batches == [[local_info1]]

    manager._step_transferring()
    assert committed == []

    manager.active_transfers[0].finished = True
    manager._step_transferring()
    assert committed == [remote_info, local_info0]
    assert manager.active_transfers is None

    manager._step_transferring()
    assert starts == [local_info0, local_info1]

    manager._step_transferring()
    assert committed == [remote_info, local_info0, local_info1]

    manager._step_transferring()
    assert manager.state is manager_module.EPLBManagerState.COLLECTING
    assert manager.completed_rebalance_count == 1
    assert manager.current_placement == [
        [[0, 1, 2], [2, 3, 3], [4, 5, 2], [6, 7, 0]],
        [[0, 1, 4], [2, 3, 5], [4, 5, 6], [6, 7, 1]],
    ]
    assert manager.pending_transfer_batches is None
    assert manager.target_placement is None
    assert manager.rebalance_started_at is None
    assert cleared_route_counters == [True]
    assert local_states == [False, True, True]
    assert used_streams == [overlap_stream, overlap_stream]


def test_manager_returns_to_collecting_after_reaching_rebalance_limit():
    manager = manager_module.EPLBManager.__new__(manager_module.EPLBManager)
    target_placement = [[[0, 1], [1, 0]]]
    manager.state = manager_module.EPLBManagerState.TRANSFERRING
    manager.global_rank = 0
    manager.current_placement = [[[0, 1], [0, 1]]]
    manager.target_placement = target_placement
    manager.pending_transfer_batches = []
    manager.max_rebalance_count = 1
    manager.completed_rebalance_count = 0
    manager.rebalance_started_at = None
    manager.active_transfer_batch = None
    manager.active_transfers = None
    manager._clear_route_statistics = lambda: None
    persisted_placements = []
    manager._persist_current_placement = lambda: persisted_placements.append(manager.current_placement)

    manager._step_transferring()

    assert manager.current_placement is target_placement
    assert manager.completed_rebalance_count == 1
    assert manager.state is manager_module.EPLBManagerState.COLLECTING
    assert persisted_placements == [target_placement]


def test_wait_plan_finish_broadcasts_pending_status(monkeypatch):
    manager = manager_module.EPLBManager.__new__(manager_module.EPLBManager)
    manager.state = manager_module.EPLBManagerState.WAIT_PLAN_PLACEMENT_FINISHED
    manager.global_rank = 0
    manager.control_group = object()
    manager._plan_task = SimpleNamespace(
        is_finished=lambda: False,
        result=None,
    )
    broadcasts = []

    def broadcast(values, **_kwargs):
        broadcasts.append(values[0])

    monkeypatch.setattr(manager_module.dist, "broadcast_object_list", broadcast)
    manager._step_wait_plan_placement_finished()
    assert broadcasts == [None]
    assert manager.state is manager_module.EPLBManagerState.WAIT_PLAN_PLACEMENT_FINISHED


def test_wait_plan_finish_publishes_before_and_after_metrics(monkeypatch):
    manager = manager_module.EPLBManager.__new__(manager_module.EPLBManager)
    manager.state = manager_module.EPLBManagerState.WAIT_PLAN_PLACEMENT_FINISHED
    manager.global_rank = 0
    manager.run_mode = "prefill"
    manager.control_group = object()
    manager.current_placement = [[[0, 1], [2, 3]]]
    target_placement = [[[0, 2], [1, 3]]]
    route_statistics = torch.tensor([[[[4, 3, 2, 1]]]], dtype=torch.int64)
    manager._plan_task = SimpleNamespace(
        is_finished=lambda: True,
        result=target_placement,
        route_statistics=route_statistics,
    )
    manager.metric_client = object()
    published = []
    monkeypatch.setattr(
        eplb_metrics,
        "publish_rebalance_compute_metrics",
        lambda **kwargs: published.append((kwargs["sample_load"], kwargs["target_placement"])),
    )
    monkeypatch.setattr(manager_module.dist, "broadcast_object_list", lambda _values, **_kwargs: None)

    manager._step_wait_plan_placement_finished()

    assert manager.state is manager_module.EPLBManagerState.PLAN_TRANSFER
    assert manager.target_placement is target_placement
    assert len(published) == 1
    assert torch.equal(published[0][0], torch.tensor([[4, 3, 2, 1]], dtype=torch.int64))
    assert published[0][1] is target_placement
    assert manager._plan_task is None


def test_wait_plan_finish_skips_prefill_compute_metrics_for_decode(monkeypatch):
    manager = manager_module.EPLBManager.__new__(manager_module.EPLBManager)
    manager.state = manager_module.EPLBManagerState.WAIT_PLAN_PLACEMENT_FINISHED
    manager.global_rank = 0
    manager.run_mode = "decode"
    manager.control_group = object()
    manager.current_placement = [[[0, 1], [2, 3]]]
    target_placement = [[[0, 2], [1, 3]]]
    manager._plan_task = SimpleNamespace(
        is_finished=lambda: True,
        result=target_placement,
        route_statistics=torch.zeros((1, 1, 4, 4), dtype=torch.int64),
    )
    manager.metric_client = object()
    monkeypatch.setattr(
        eplb_metrics,
        "publish_rebalance_compute_metrics",
        lambda **_kwargs: pytest.fail("decode must not publish prefill compute metrics"),
    )
    monkeypatch.setattr(manager_module.dist, "broadcast_object_list", lambda _values, **_kwargs: None)

    manager._step_wait_plan_placement_finished()

    assert manager.state is manager_module.EPLBManagerState.PLAN_TRANSFER
    assert manager.target_placement is target_placement
    assert manager._plan_task is None


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_manager_transfer_task_commit_orders_live_weights_between_overlap_forwards(
    monkeypatch,
):
    class Transfer:
        def __init__(self, live, received, transfer_info):
            self.tensor_buffers = [ExpertTensorBuffer("weight", live, received)]
            self.transfer_info = transfer_info
            self.status = "succeeded"

        def is_finished(self):
            return True

    live = torch.tensor([1.0], device="cuda")
    received = torch.tensor(2.0, pin_memory=True)
    previous_read = torch.empty_like(live)
    next_read = torch.empty_like(live)
    source_stream = torch.cuda.Stream(device=live.device)
    destination_stream = torch.cuda.Stream(device=live.device)
    initial_stream = torch.cuda.current_stream(device=live.device)
    original_overlap_stream = g_infer_context.overlap_stream

    manager = manager_module.EPLBManager.__new__(manager_module.EPLBManager)
    transfer_info = EPLBTransferInfo(0, 0, 0, 0, 0, 0)
    transfer = Transfer(live, received, transfer_info)
    manager.active_transfers = [transfer]
    manager.active_transfer_batch = [transfer_info]
    manager.control_group = object()
    manager.world_size = 1
    manager.node_world_size = 1
    manager.pending_transfer_batches = []
    manager.num_logical_experts = 1
    manager.global_rank = 0
    manager.target_placement = [[[0]]]
    manager._eplb_impls = [
        SimpleNamespace(
            local_logics_expert_ids_list=[0],
            logical_to_physical_map=torch.zeros((1, 1), dtype=torch.int32, device="cuda"),
        )
    ]
    manager.current_placement = [[[0]]]
    manager.rebalance_started_at = time.time()
    manager.state = manager_module.EPLBManagerState.TRANSFERRING
    monkeypatch.setattr(
        manager_module.dist,
        "all_gather_object",
        lambda output, local_ready, **_kwargs: output.__setitem__(slice(None), [local_ready]),
    )
    monkeypatch.setattr(manager_module, "build_logical_to_physical_map", lambda *_args, **_kwargs: [[0]])

    try:
        g_infer_context.overlap_stream = source_stream
        with torch.cuda.stream(source_stream):
            source_stream.wait_stream(initial_stream)
            torch.cuda._sleep(20_000_000)
            previous_read.copy_(live, non_blocking=True)
        with torch.cuda.stream(destination_stream):
            manager._step_transferring()
        with torch.cuda.stream(source_stream):
            source_stream.wait_stream(destination_stream)
            next_read.copy_(live, non_blocking=True)
        source_stream.synchronize()

        assert previous_read.item() == 1.0
        assert next_read.item() == 2.0
    finally:
        g_infer_context.overlap_stream = original_overlap_stream


def test_manager_step_advances_inflight_transfer():
    manager = manager_module.EPLBManager.__new__(manager_module.EPLBManager)
    manager.state = manager_module.EPLBManagerState.TRANSFERRING
    calls = []
    manager._step_transferring = lambda: calls.append("transfer")

    manager.step()

    assert calls == ["transfer"]


def test_manager_evaluates_only_after_entering_evaluating_state(monkeypatch):
    manager = manager_module.EPLBManager.__new__(manager_module.EPLBManager)
    prefill_route_counter = torch.tensor([[1, 2]], dtype=torch.int64)
    load_gather_tasks = []
    manager.state = manager_module.EPLBManagerState.COLLECTING
    manager.global_rank = 1
    manager.steps = 0
    manager.step_interval = 3
    manager.next_evaluation_step = 3
    manager.num_logical_experts = 2
    manager.run_mode = "prefill"
    manager._eplb_impls = [SimpleNamespace(prefill_route_counter=prefill_route_counter)]
    manager.world_size = 1
    manager.control_group = object()
    manager.max_rebalance_count = -1
    manager.completed_rebalance_count = 0
    monkeypatch.setattr(
        manager_module,
        "EPLBLoadGatherTask",
        lambda **_kwargs: load_gather_tasks.append(True),
    )
    monkeypatch.setattr(
        manager_module.dist,
        "broadcast_object_list",
        lambda values, **_kwargs: values.__setitem__(0, False),
    )

    manager.step()
    manager.step()
    assert manager.state is manager_module.EPLBManagerState.COLLECTING
    manager.step()

    assert manager.state is manager_module.EPLBManagerState.EVALUATING
    assert load_gather_tasks == []
    assert manager.next_evaluation_step == 6

    manager.step()

    assert manager.state is manager_module.EPLBManagerState.COLLECTING
    assert load_gather_tasks == []
    assert manager.next_evaluation_step == 6
    assert torch.equal(prefill_route_counter, torch.tensor([[1, 2]], dtype=torch.int64))


def test_manager_step_uses_explicit_state_instead_of_pending_work():
    manager = manager_module.EPLBManager.__new__(manager_module.EPLBManager)
    manager.state = manager_module.EPLBManagerState.TRANSFERRING
    manager.pending_transfer_batches = [[object()]]
    manager._plan_task = object()
    calls = []
    manager._step_transferring = lambda: calls.append("transfer")
    manager._step_evaluating = lambda: calls.append("evaluation")

    manager.step()

    assert calls == ["transfer"]


def test_manager_plans_transfers_asynchronously_before_entering_transferring(monkeypatch):
    manager = manager_module.EPLBManager.__new__(manager_module.EPLBManager)
    manager.state = manager_module.EPLBManagerState.WAIT_PLAN_PLACEMENT_FINISHED
    manager.global_rank = 1
    manager.control_group = object()
    manager.transfer_group = object()
    manager.world_size = 2
    manager.num_logical_experts = 4
    manager.run_mode = "prefill"
    manager.transfer_layer_parallelism = 8
    manager.pending_transfer_batches = None
    placement = [
        [[0, 1, 2], [2, 3, 3]],
        [[0, 1, 3], [2, 3, 0]],
    ]
    manager.current_placement = [
        [[0, 1, 3], [2, 3, 2]],
        [[0, 1, 2], [2, 3, 1]],
    ]
    monkeypatch.setattr(
        manager_module.dist,
        "broadcast_object_list",
        lambda values, **_kwargs: values.__setitem__(0, placement),
    )

    transfer_infos = [
        EPLBTransferInfo(0, 2, 0, 0, 1, 2),
        EPLBTransferInfo(1, 0, 0, 0, 1, 2),
    ]
    transfer_planners = []

    class TransferPlanner:
        def __init__(self, **kwargs):
            self.current = kwargs["current_placement"]
            self.target = kwargs["target_placement"]
            self.num_logical_experts = kwargs["num_logical_experts"]
            self.world_size = kwargs["world_size"]
            self.transfer_layer_parallelism = kwargs["transfer_layer_parallelism"]
            self.result = [[transfer_infos[0]], [transfer_infos[1]]]
            self.started = False
            self.finished = False
            transfer_planners.append(self)

        def start(self):
            self.started = True

        def is_finished(self):
            return self.finished

    monkeypatch.setattr(manager_module, "EPLBTransferPlanner", TransferPlanner)
    monkeypatch.setattr(
        manager_module,
        "PinnedMemoryEPLBTransfer",
        lambda **_kwargs: pytest.fail("transfer object must not be built while planning transfers"),
    )

    manager._step_wait_plan_placement_finished()

    assert manager.state is manager_module.EPLBManagerState.PLAN_TRANSFER
    assert manager.target_placement is placement
    assert transfer_planners == []
    assert manager.pending_transfer_batches is None

    manager.step()

    assert manager.state is manager_module.EPLBManagerState.WAIT_PLAN_TRANSFER_FINISHED
    assert len(transfer_planners) == 1
    transfer_planner = transfer_planners[0]
    assert transfer_planner.current is manager.current_placement
    assert transfer_planner.target is placement
    assert transfer_planner.num_logical_experts == manager.num_logical_experts
    assert transfer_planner.world_size == manager.world_size
    assert transfer_planner.transfer_layer_parallelism == manager.transfer_layer_parallelism
    assert transfer_planner.started

    remote_finished = False

    def gather_finished(output, local_finished, **_kwargs):
        output[:] = [local_finished, remote_finished]

    monkeypatch.setattr(manager_module.dist, "all_gather_object", gather_finished)
    transfer_planner.finished = True
    manager.step()

    assert manager.state is manager_module.EPLBManagerState.WAIT_PLAN_TRANSFER_FINISHED
    assert manager.pending_transfer_batches is None

    remote_finished = True
    manager.step()

    assert manager.state is manager_module.EPLBManagerState.TRANSFERRING
    assert manager.pending_transfer_batches == [[transfer_infos[0]], [transfer_infos[1]]]
    assert manager._transfer_planner is None


def test_manager_evaluation_with_insufficient_tokens_returns_to_collecting(monkeypatch):
    manager = manager_module.EPLBManager.__new__(manager_module.EPLBManager)
    manager.state = manager_module.EPLBManagerState.EVALUATING
    manager._eplb_impls = [SimpleNamespace(prefill_route_counter=torch.full((1, 4), 100, dtype=torch.int64))]
    manager.num_logical_experts = 4
    manager.run_mode = "prefill"
    manager.steps = 11
    manager.step_interval = 20
    manager.next_evaluation_step = 31
    manager.global_rank = 0
    manager.world_size = 1
    manager.control_group = object()
    manager.load_gather_group = object()
    manager.max_rebalance_count = -1
    manager.completed_rebalance_count = 0
    manager.metric_client = object()
    broadcast_decisions = []
    monkeypatch.setattr(eplb_metrics, "publish_expert_load_metrics", lambda **_kwargs: None)
    monkeypatch.setattr(
        manager_module,
        "EPLBLoadGatherTask",
        lambda **_kwargs: pytest.fail("insufficient rank-0 load must skip all-gather"),
    )
    monkeypatch.setattr(
        manager_module.dist,
        "broadcast_object_list",
        lambda values, **kwargs: broadcast_decisions.append((values[0], kwargs["src"], kwargs["group"])),
    )

    manager.step()

    assert manager.state is manager_module.EPLBManagerState.COLLECTING
    assert manager.next_evaluation_step == 31
    assert broadcast_decisions == [(False, 0, manager.control_group)]


def test_manager_evaluation_with_enough_tokens_enters_planning(monkeypatch):
    manager = manager_module.EPLBManager.__new__(manager_module.EPLBManager)
    local_load = torch.full((1, 4), 256, dtype=torch.int64)
    plan_tasks = []
    published_loads = []
    manager.state = manager_module.EPLBManagerState.EVALUATING
    manager.global_rank = 0
    manager.num_logical_experts = 4
    manager.run_mode = "prefill"
    manager._eplb_impls = [SimpleNamespace(prefill_route_counter=local_load)]
    manager.world_size = 1
    manager.control_group = object()
    manager.load_gather_group = object()
    manager.max_rebalance_count = -1
    manager.completed_rebalance_count = 0

    class LoadGatherTask:
        def __init__(self, **kwargs):
            self.local_route_statistics = kwargs["local_route_statistics"]
            self.result = self.local_route_statistics.unsqueeze(0)
            self.started = False

        def start(self):
            self.started = True

        def is_finished(self):
            return True

    monkeypatch.setattr(manager_module, "EPLBLoadGatherTask", LoadGatherTask)
    monkeypatch.setattr(manager_module.dist, "broadcast_object_list", lambda _values, **_kwargs: None)
    monkeypatch.setattr(
        manager_module.dist,
        "all_gather_object",
        lambda output, local_finished, **_kwargs: output.__setitem__(slice(None), [local_finished]),
    )

    class PlanTask:
        def __init__(self, **kwargs):
            self.planner = kwargs["planner"]
            self.route_statistics = kwargs["route_statistics"]
            self.current_placement = kwargs["current_placement"]
            self.started = False
            plan_tasks.append(self)

        def start(self):
            self.started = True

    manager.planner = object()
    manager.current_placement = [[[0, 1, 2, 3]]]
    manager.metric_client = object()
    monkeypatch.setattr(
        eplb_metrics,
        "publish_expert_load_metrics",
        lambda **kwargs: published_loads.append(kwargs["expert_load"]),
    )
    monkeypatch.setattr(manager_module, "EPLBPlanTask", PlanTask)

    manager.step()

    assert manager.state is manager_module.EPLBManagerState.WAIT_LOAD_GATHER_FINISHED
    assert manager._load_gather_task.started
    assert torch.equal(manager._load_gather_task.local_route_statistics, local_load.unsqueeze(0))
    assert len(published_loads) == 1
    assert torch.equal(published_loads[0], local_load)
    assert plan_tasks == []
    assert getattr(manager, "_plan_task", None) is None

    manager.step()

    assert manager.state is manager_module.EPLBManagerState.PLAN_PLACEMENT
    planning_route_statistics = manager._pending_route_statistics
    assert torch.equal(planning_route_statistics, local_load.unsqueeze(0).unsqueeze(0))
    assert manager._load_gather_task is None
    assert plan_tasks == []

    manager.step()

    assert manager.state is manager_module.EPLBManagerState.WAIT_PLAN_PLACEMENT_FINISHED
    assert plan_tasks[0].route_statistics is planning_route_statistics
    assert torch.equal(plan_tasks[0].route_statistics, local_load.unsqueeze(0).unsqueeze(0))
    assert plan_tasks[0].planner is manager.planner
    assert plan_tasks[0].current_placement is manager.current_placement
    assert plan_tasks[0].started
    assert manager._plan_task is plan_tasks[0]
    assert manager._pending_route_statistics is None
    assert len(published_loads) == 1


def test_manager_keeps_reporting_after_reaching_rebalance_limit(monkeypatch):
    manager = manager_module.EPLBManager.__new__(manager_module.EPLBManager)
    local_load = torch.tensor([[10, 20, 30, 40]], dtype=torch.int64)
    published_loads = []
    cleared_counters = []
    manager.state = manager_module.EPLBManagerState.EVALUATING
    manager.global_rank = 0
    manager.num_logical_experts = 4
    manager.run_mode = "prefill"
    manager._eplb_impls = [SimpleNamespace(prefill_route_counter=local_load)]
    manager.max_rebalance_count = 1
    manager.completed_rebalance_count = 1
    manager.metric_client = object()
    monkeypatch.setattr(
        eplb_metrics,
        "publish_expert_load_metrics",
        lambda **kwargs: published_loads.append(kwargs["expert_load"]),
    )
    manager._clear_route_statistics = lambda: cleared_counters.append(True)
    monkeypatch.setattr(
        manager_module,
        "EPLBLoadGatherTask",
        lambda **_kwargs: pytest.fail("load gathering must stop after reaching the limit"),
    )

    manager.step()

    assert manager.state is manager_module.EPLBManagerState.COLLECTING
    assert len(published_loads) == 1
    assert torch.equal(published_loads[0], local_load)
    assert cleared_counters == [True]


def test_nonzero_rank_enters_plan_wait_without_starting_planner():
    manager = manager_module.EPLBManager.__new__(manager_module.EPLBManager)
    manager.state = manager_module.EPLBManagerState.PLAN_PLACEMENT
    manager.global_rank = 1

    manager.step()

    assert manager.state is manager_module.EPLBManagerState.WAIT_PLAN_PLACEMENT_FINISHED
    assert getattr(manager, "_plan_task", None) is None
    assert getattr(manager, "_pending_route_statistics", None) is None


def test_manager_planning_without_changes_returns_to_collecting(monkeypatch):
    manager = manager_module.EPLBManager.__new__(manager_module.EPLBManager)
    manager.state = manager_module.EPLBManagerState.WAIT_PLAN_PLACEMENT_FINISHED
    manager.global_rank = 1
    manager.control_group = object()
    manager.steps = 11
    manager.step_interval = 20
    manager.next_evaluation_step = 31
    manager.current_placement = [[[0, 1], [1, 0]]]
    result = None

    def broadcast(values, **_kwargs):
        values[0] = result

    monkeypatch.setattr(manager_module.dist, "broadcast_object_list", broadcast)

    manager.step()
    assert manager.state is manager_module.EPLBManagerState.WAIT_PLAN_PLACEMENT_FINISHED

    result = [[[0, 1], [1, 0]]]
    manager.step()

    assert manager.state is manager_module.EPLBManagerState.COLLECTING
    assert manager.next_evaluation_step == 31
    assert getattr(manager, "_plan_task", None) is None


def test_manager_exposes_one_lifecycle_step_entrypoint():
    assert hasattr(manager_module.EPLBManager, "step")
    assert not hasattr(manager_module.EPLBManager, "poll")


def test_pinned_transfer_copies_source_row_and_sends_to_destination(monkeypatch):
    class Stream:
        def __init__(self):
            self.synchronize_count = 0

        def synchronize(self):
            self.synchronize_count += 1

    transfer = object.__new__(PinnedMemoryEPLBTransfer)
    transfer._device = "cuda:0"
    transfer._is_source_rank = True
    transfer._is_destination_rank = False
    transfer._p2p_group = object()
    transfer.transfer_info = EPLBTransferInfo(0, 5, 0, 1, 1, 2)
    transfer._device_to_host_stream = Stream()
    transfer.tensor_buffers = [
        ExpertTensorBuffer(
            "weight",
            torch.tensor([[1.0, 2.0], [3.0, 4.0]]),
            torch.empty(2),
        )
    ]
    transfer.status = "running"
    sends = []
    monkeypatch.setattr(transfer_module.torch.cuda, "set_device", lambda _device: None)
    monkeypatch.setattr(transfer_module.torch.cuda, "stream", lambda _stream: nullcontext())
    monkeypatch.setattr(
        transfer_module.dist,
        "send",
        lambda tensor, dst, group, tag: sends.append((tensor.clone(), dst, group, tag)),
    )

    transfer._run()

    assert transfer.status == "succeeded"
    assert len(sends) == 1
    assert torch.equal(sends[0][0], torch.tensor([3.0, 4.0]))
    expected_tag = transfer._build_p2p_message_tag("weight")
    assert sends[0][1:] == (1, transfer._p2p_group, expected_tag)
    assert torch.equal(transfer.tensor_buffers[0].pinned_row, torch.tensor([3.0, 4.0]))
    assert transfer._device_to_host_stream.synchronize_count == 1


def test_pinned_transfer_skips_p2p_for_local_destination(monkeypatch):
    class Stream:
        def synchronize(self):
            pass

    transfer = object.__new__(PinnedMemoryEPLBTransfer)
    transfer._device = "cuda:0"
    transfer._is_source_rank = True
    transfer._is_destination_rank = True
    transfer._p2p_group = object()
    transfer.transfer_info = EPLBTransferInfo(0, 5, 0, 0, 0, 1)
    transfer._device_to_host_stream = Stream()
    transfer.tensor_buffers = [
        ExpertTensorBuffer(
            "weight",
            torch.tensor([[3.0, 4.0]]),
            torch.empty(2),
        )
    ]
    transfer.status = "running"
    p2p_calls = []
    monkeypatch.setattr(transfer_module.torch.cuda, "set_device", lambda _device: None)
    monkeypatch.setattr(transfer_module.torch.cuda, "stream", lambda _stream: nullcontext())
    monkeypatch.setattr(transfer_module.dist, "send", lambda *_args, **_kwargs: p2p_calls.append("send"))
    monkeypatch.setattr(transfer_module.dist, "recv", lambda *_args, **_kwargs: p2p_calls.append("recv"))

    transfer._run()

    assert transfer.is_finished()
    assert p2p_calls == []
    assert torch.equal(transfer.tensor_buffers[0].pinned_row, torch.tensor([3.0, 4.0]))


def test_pinned_transfer_exits_process_on_failure(monkeypatch):
    transfer = object.__new__(PinnedMemoryEPLBTransfer)
    transfer._device = "cuda:0"
    transfer._is_source_rank = False
    transfer._is_destination_rank = True
    transfer.transfer_info = EPLBTransferInfo(0, 3, 1, 0, 0, 2)
    transfer._p2p_group = object()
    transfer.tensor_buffers = [
        ExpertTensorBuffer(
            "weight",
            torch.empty((1, 1)),
            torch.empty(1),
        )
    ]
    transfer.status = "running"
    logged_messages = []
    exit_codes = []

    def fail_recv(*_args, **_kwargs):
        raise RuntimeError("recv failed")

    monkeypatch.setattr(transfer_module.torch.cuda, "set_device", lambda _device: None)
    monkeypatch.setattr(transfer_module.dist, "recv", fail_recv)
    monkeypatch.setattr(async_task_module.logger, "exception", logged_messages.append)
    monkeypatch.setattr(async_task_module.os, "_exit", exit_codes.append)

    transfer._run()

    assert logged_messages == ["PinnedMemoryEPLBTransfer failed"]
    assert exit_codes == [1]
    assert not transfer.is_finished()


def test_pinned_transfer_is_single_use_and_exposes_pinned_rows(monkeypatch):
    class Stream:
        def synchronize(self):
            pass

    transfer = object.__new__(PinnedMemoryEPLBTransfer)
    transfer._device = "cuda:0"
    transfer._is_source_rank = False
    transfer._is_destination_rank = True
    transfer.transfer_info = EPLBTransferInfo(0, 3, 1, 0, 0, 2)
    transfer._p2p_group = object()
    transfer._device_to_host_stream = Stream()
    transfer.tensor_buffers = [
        ExpertTensorBuffer(
            "weight",
            torch.empty((1, 1)),
            torch.tensor([3.0]),
        )
    ]
    transfer.status = "idle"
    transfer._thread = threading.Thread(target=transfer._run, daemon=True)
    receives = []
    monkeypatch.setattr(transfer_module.torch.cuda, "set_device", lambda _device: None)
    monkeypatch.setattr(
        transfer_module.dist,
        "recv",
        lambda tensor, src, group, tag: receives.append((tensor, src, group, tag)),
    )

    assert not transfer.is_finished()
    transfer.start()
    deadline = time.monotonic() + 2
    while transfer.status == "running" and time.monotonic() < deadline:
        time.sleep(0.001)
    assert transfer.status == "succeeded"
    assert transfer.is_finished()
    assert torch.equal(transfer.tensor_buffers[0].pinned_row, torch.tensor([3.0]))
    assert len(receives) == 1
    assert receives[0][0] is transfer.tensor_buffers[0].pinned_row
    expected_tag = transfer._build_p2p_message_tag("weight")
    assert receives[0][1:] == (1, transfer._p2p_group, expected_tag)
    with pytest.raises(AssertionError, match="already been started"):
        transfer.start()


def test_manager_requires_more_than_one_rank(monkeypatch):
    monkeypatch.setattr(manager_module, "is_sm100_gpu", lambda: False)
    monkeypatch.setattr(manager_module, "_find_fused_moe_weights", lambda model: [object()])
    monkeypatch.setattr(manager_module, "get_global_rank", lambda: 0)
    monkeypatch.setattr(manager_module, "get_global_world_size", lambda: 1)

    with pytest.raises(AssertionError, match="more than one rank"):
        manager_module.EPLBManager(type("Model", (), {})())


def test_manager_rejects_sm100_before_initialization(monkeypatch):
    monkeypatch.setattr(manager_module, "is_sm100_gpu", lambda: True)

    with pytest.raises(AssertionError, match="EPLB does not support SM100"):
        manager_module.EPLBManager(type("Model", (), {})())


def test_manager_clears_all_route_statistics_on_overlap_stream(monkeypatch):
    manager = manager_module.EPLBManager.__new__(manager_module.EPLBManager)
    counters = [torch.tensor([[1, 2]]), torch.tensor([[3, 4]])]
    sample_indices = [torch.tensor([7, 3]), torch.tensor([9, 4])]
    decode_counters = [
        torch.tensor([[1, 2], [3, 4]]),
        torch.tensor([[5, 6], [7, 8]]),
    ]
    prefill_impls = [
        SimpleNamespace(
            prefill_route_counter=counter,
            prefill_route_sample_index=sample_index,
            # 第一层同时持有两种统计，验证三个状态会独立清零。
            decode_route_counter=decode_counters[0] if index == 0 else None,
        )
        for index, (counter, sample_index) in enumerate(zip(counters, sample_indices))
    ]
    decode_impls = [
        SimpleNamespace(
            prefill_route_counter=None,
            prefill_route_sample_index=None,
            decode_route_counter=decode_counters[1],
        )
    ]
    manager._eplb_impls = prefill_impls + decode_impls
    overlap_stream = object()
    used_streams = []
    monkeypatch.setattr(g_infer_context, "get_overlap_stream", lambda: overlap_stream)
    monkeypatch.setattr(
        manager_module.torch.cuda,
        "stream",
        lambda stream: (used_streams.append(stream) or nullcontext()),
    )

    manager._clear_route_statistics()

    assert used_streams == [overlap_stream]
    assert all(torch.count_nonzero(counter) == 0 for counter in counters)
    assert all(torch.count_nonzero(sample_index) == 0 for sample_index in sample_indices)
    assert all(torch.count_nonzero(counter) == 0 for counter in decode_counters)


def test_manager_initializes_without_transfer_task(monkeypatch):
    weight = type(
        "Weight",
        (),
        {
            "n_routed_experts": 4,
            "layer_num_": 0,
            "fuse_moe_impl": _test_moe_impl(
                eplb=True,
                num_logical_experts=4,
                world_size=2,
                num_redundant_experts_per_rank=2,
                prefill_route_counter=torch.zeros((24, 4), dtype=torch.int64),
            ),
        },
    )()
    groups = [object(), object(), object()]
    new_group_calls = []
    monkeypatch.setattr(manager_module, "is_sm100_gpu", lambda: False)
    monkeypatch.setattr(manager_module, "_find_fused_moe_weights", lambda model: [weight])
    monkeypatch.setattr(manager_module, "get_global_rank", lambda: 0)
    monkeypatch.setattr(manager_module, "get_global_world_size", lambda: 2)
    monkeypatch.setattr(manager_module, "get_node_world_size", lambda: 2)
    monkeypatch.setattr(manager_module, "get_eplb_step_interval", lambda: 20)
    monkeypatch.setattr(manager_module, "get_eplb_transfer_layer_parallelism", lambda: 8)
    clear_calls = []
    monkeypatch.setattr(
        manager_module.EPLBManager,
        "_clear_route_statistics",
        lambda manager: clear_calls.append(manager),
    )
    monkeypatch.setattr(manager_module, "get_shm_port_args", lambda: SimpleNamespace(metric_port=1234))
    metric_client = SimpleNamespace()
    metric_client_ports = []
    monkeypatch.setattr(
        manager_module,
        "MetricClient",
        lambda port: (metric_client_ports.append(port) or metric_client),
    )

    def new_group(*args, **kwargs):
        new_group_calls.append((args, kwargs))
        return groups[len(new_group_calls) - 1]

    monkeypatch.setattr(manager_module.dist, "new_group", new_group)
    all_gather_calls = []

    def all_gather_object(output, local_expert_ids_by_layer, group):
        all_gather_calls.append((local_expert_ids_by_layer, group))
        output[:] = [local_expert_ids_by_layer, [[2, 3, 0, 1]]]

    monkeypatch.setattr(manager_module.dist, "all_gather_object", all_gather_object)
    logs = []
    monkeypatch.setattr(manager_module.logger, "info", lambda message: logs.append(message))
    monkeypatch.setattr(
        manager_module,
        "save_placement_config",
        lambda *_args, **_kwargs: pytest.fail("manager initialization must not save the placement"),
    )
    manager = manager_module.EPLBManager(type("Model", (), {})(), config_path="/tmp/eplb.json")
    assert manager._plan_task is None
    assert manager.pending_transfer_batches is None
    assert manager.state is manager_module.EPLBManagerState.COLLECTING
    assert (manager.load_gather_group, manager.control_group, manager.transfer_group) == tuple(groups)
    assert new_group_calls == [(([0, 1],), {"backend": "gloo"})] * 3
    assert all_gather_calls == [([[0, 1, 2, 3]], groups[1])]
    assert manager.current_placement == [[[0, 1, 2, 3], [2, 3, 0, 1]]]
    assert manager.metric_client is metric_client
    assert metric_client_ports == [1234]
    assert manager.next_evaluation_step == manager.step_interval
    assert manager.transfer_layer_parallelism == 8
    assert manager.max_rebalance_count == 1
    assert manager.completed_rebalance_count == 0
    assert manager.run_mode == "prefill"
    assert manager.plan_mode == "topology_aware"
    assert clear_calls == [manager]
    assert isinstance(manager.planner, TopologyAwareEPLBPlanner)
    assert "run_mode=prefill" in logs[0]
    assert "plan_mode=topology_aware" in logs[0]
    assert "planner=TopologyAwareEPLBPlanner" in logs[0]
    assert not hasattr(weight.fuse_moe_impl, "eplb_dispatch_mode")
    assert manager._eplb_impls[0] is weight.fuse_moe_impl

    # 当前两个 planner 都属于 prefill，不能与 decode 运行模式组合。
    weight.fuse_moe_impl = _test_moe_impl(
        eplb=True,
        num_logical_experts=4,
        world_size=2,
        num_redundant_experts_per_rank=2,
        run_mode="decode",
    )
    with pytest.raises(AssertionError, match="is for 'prefill'"):
        manager_module.EPLBManager(
            type("Model", (), {})(),
            run_mode="decode",
        )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required for the Triton EPLB kernel")
@pytest.mark.parametrize("update_prefill_route_counter", [False, True])
@pytest.mark.parametrize("tokens", [1, 32])
@pytest.mark.parametrize("mode", ["current_gpu_first", "current_node_first", "global_first"])
def test_eplb_repair_topk_ids_maps_and_counts(update_prefill_route_counter, tokens, mode):
    from lightllm.common.basemodel.triton_kernel.fused_moe.eplb_topk_ids import (
        eplb_repair_topk_ids,
    )

    topk = 4
    experts = 64
    logical_ids = (torch.arange(tokens * topk, dtype=torch.int32, device="cuda") % experts).view(tokens, topk)
    original_logical_ids = logical_ids.clone()
    logical_experts = torch.arange(experts, dtype=torch.int32, device="cuda")
    replica_counts = torch.where(
        logical_experts % 3 == 0,
        torch.full_like(logical_experts, 3),
        torch.ones_like(logical_experts),
    )
    num_current_gpu_replicas = torch.where(
        logical_experts % 5 == 0,
        torch.ones_like(logical_experts),
        torch.zeros_like(logical_experts),
    )
    num_node_replicas = torch.where(
        num_current_gpu_replicas > 0,
        torch.where(
            replica_counts >= 2,
            torch.full_like(logical_experts, 2),
            num_current_gpu_replicas,
        ),
        torch.where(
            (logical_experts % 7 == 0) & (replica_counts == 3),
            torch.full_like(logical_experts, 2),
            torch.zeros_like(logical_experts),
        ),
    )
    logical_to_physical = torch.stack(
        (
            replica_counts,
            num_node_replicas,
            num_current_gpu_replicas,
            logical_experts,
            torch.where(replica_counts >= 2, logical_experts + experts, logical_experts),
            torch.where(replica_counts == 3, logical_experts + 2 * experts, logical_experts),
        ),
        dim=1,
    )
    counter = torch.zeros((24, experts), dtype=torch.int64, device="cuda")
    sample_index = torch.zeros((2,), dtype=torch.int64, device="cuda")
    expected_counter = torch.zeros_like(counter)

    logical_ids_long = logical_ids.to(torch.long)
    token_indices = torch.arange(tokens, device="cuda", dtype=torch.int64).unsqueeze(1)
    if mode == "current_gpu_first":
        num_preferred_replicas = torch.where(
            num_current_gpu_replicas > 0,
            num_current_gpu_replicas,
            replica_counts,
        )
    elif mode == "current_node_first":
        num_preferred_replicas = torch.where(num_node_replicas > 0, num_node_replicas, replica_counts)
    else:
        num_preferred_replicas = replica_counts
    hash_values = token_indices ^ ((logical_ids.to(torch.int64) + 1) * 0x9E3779B9)
    hash_values &= 0xFFFFFFFF
    hash_values ^= hash_values >> 16
    hash_values = (hash_values * 0x7FEB352D) & 0xFFFFFFFF
    hash_values ^= hash_values >> 15
    hash_values = (hash_values * 0x846CA68B) & 0xFFFFFFFF
    hash_values ^= hash_values >> 16
    replica_indices = hash_values % num_preferred_replicas[logical_ids_long].to(torch.int64)
    expected_ids = logical_to_physical[logical_ids_long, replica_indices + 3]
    if update_prefill_route_counter:
        expected_counter[0].scatter_add_(
            0,
            logical_ids.reshape(-1).to(torch.long),
            torch.ones(logical_ids.numel(), dtype=torch.int64, device="cuda"),
        )

    physical_ids = eplb_repair_topk_ids(
        logical_topk_ids=logical_ids,
        logical_to_physical_map=logical_to_physical,
        prefill_route_counter=counter,
        prefill_route_sample_index=sample_index,
        update_prefill_route_counter=update_prefill_route_counter,
        decode_route_counter=None,
        update_decode_route_counter=False,
        mode=mode,
    )
    torch.cuda.synchronize()

    assert torch.equal(logical_ids, original_logical_ids)
    assert torch.equal(physical_ids, expected_ids)
    assert torch.equal(counter, expected_counter)
    assert sample_index.tolist() == ([1, 0] if update_prefill_route_counter else [0, 0])


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required for the Triton EPLB kernel")
def test_eplb_decode_route_counter_records_exact_upper_triangle():
    from lightllm.common.basemodel.triton_kernel.fused_moe.eplb_topk_ids import (
        eplb_repair_topk_ids,
    )

    num_tokens = 1025
    topk = 4
    experts = 64
    logical_ids = (torch.arange(num_tokens * topk, device="cuda", dtype=torch.int32) % experts).view(
        num_tokens,
        topk,
    )
    # 显式覆盖最小/最大 expert ID，并用反序 top-k 验证 pair 的上三角
    # 落点只由 expert ID 决定；相等边界仍必须准确进入主对角线。
    logical_ids[0] = torch.tensor([0, 1, 2, experts - 1], device="cuda")
    logical_ids[1] = torch.tensor([experts - 1, 2, 1, 0], device="cuda")
    logical_experts = torch.arange(experts, dtype=torch.int32, device="cuda")
    logical_to_physical = torch.stack(
        (
            torch.ones_like(logical_experts),
            torch.ones_like(logical_experts),
            torch.ones_like(logical_experts),
            logical_experts,
        ),
        dim=1,
    )
    decode_counter = torch.zeros((experts, experts), dtype=torch.int64, device="cuda")
    selected_experts = torch.nn.functional.one_hot(logical_ids.cpu().to(torch.long), num_classes=experts).sum(
        dim=1, dtype=torch.int64
    )
    expected_cooccurrence = torch.triu(selected_experts.T @ selected_experts)

    physical_ids = eplb_repair_topk_ids(
        logical_topk_ids=logical_ids,
        logical_to_physical_map=logical_to_physical,
        prefill_route_counter=None,
        prefill_route_sample_index=None,
        update_prefill_route_counter=False,
        decode_route_counter=decode_counter,
        update_decode_route_counter=True,
        mode="current_gpu_first",
    )
    torch.cuda.synchronize()

    assert torch.equal(physical_ids, logical_ids)
    assert torch.equal(decode_counter.cpu(), expected_cooccurrence)
    assert torch.equal(
        torch.diagonal(decode_counter).cpu(),
        torch.bincount(logical_ids.cpu().flatten().to(torch.long), minlength=experts),
    )
    assert torch.count_nonzero(torch.tril(decode_counter, diagonal=-1)) == 0
    assert int(decode_counter.sum().item()) == num_tokens * topk * (topk + 1) // 2


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required for the Triton EPLB kernel")
def test_eplb_prefill_route_counter_multigrid_ring_wrap_is_exact():
    from lightllm.common.basemodel.triton_kernel.fused_moe.eplb_topk_ids import (
        eplb_repair_topk_ids,
    )

    capacity = 24
    experts = 256
    tokens = 1025
    topk = 4
    num_samples = capacity + 2
    base_ids = torch.arange(tokens * topk, dtype=torch.int32, device="cuda").view(tokens, topk)
    logical_experts = torch.arange(experts, dtype=torch.int32, device="cuda")
    logical_to_physical = torch.stack(
        (
            torch.ones_like(logical_experts),
            torch.ones_like(logical_experts),
            torch.ones_like(logical_experts),
            logical_experts,
        ),
        dim=1,
    )
    counter = torch.zeros((capacity, experts), dtype=torch.int64, device="cuda")
    sample_index = torch.zeros((2,), dtype=torch.int64, device="cuda")
    expected = torch.zeros_like(counter)

    for sample in range(num_samples):
        logical_ids = (base_ids + sample) % experts
        physical_ids = eplb_repair_topk_ids(
            logical_topk_ids=logical_ids,
            logical_to_physical_map=logical_to_physical,
            prefill_route_counter=counter,
            prefill_route_sample_index=sample_index,
            update_prefill_route_counter=True,
            decode_route_counter=None,
            update_decode_route_counter=False,
            mode="global_first",
        )
        expected[sample % capacity] = torch.bincount(
            logical_ids.reshape(-1).to(torch.long),
            minlength=experts,
        )
        assert torch.equal(physical_ids, logical_ids)

    torch.cuda.synchronize()

    assert sample_index.tolist() == [num_samples, 0]
    assert torch.equal(counter, expected)
    assert int(counter.sum().item()) == capacity * tokens * topk


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required for the Triton EPLB kernel")
def test_eplb_repair_topk_ids_spreads_strided_expert_tokens():
    from lightllm.common.basemodel.triton_kernel.fused_moe.eplb_topk_ids import (
        eplb_repair_topk_ids,
    )

    num_tokens = 4096
    token_indices = torch.arange(num_tokens, dtype=torch.int32, device="cuda")
    logical_ids = torch.where(token_indices % 4 == 0, 0, 1).view(-1, 1)
    logical_to_physical = torch.tensor(
        [
            [4, 4, 4, 0, 2, 3, 4],
            [1, 1, 1, 1, -1, -1, -1],
        ],
        dtype=torch.int32,
        device="cuda",
    )
    counter = torch.zeros((24, 2), dtype=torch.int64, device="cuda")
    sample_index = torch.zeros((2,), dtype=torch.int64, device="cuda")

    physical_ids = eplb_repair_topk_ids(
        logical_topk_ids=logical_ids,
        logical_to_physical_map=logical_to_physical,
        prefill_route_counter=counter,
        prefill_route_sample_index=sample_index,
        update_prefill_route_counter=False,
        decode_route_counter=None,
        update_decode_route_counter=False,
        mode="global_first",
    )
    torch.cuda.synchronize()

    strided_token_outputs = physical_ids[token_indices % 4 == 0, 0]
    replica_counts = torch.stack([(strided_token_outputs == physical_id).sum() for physical_id in (0, 2, 3, 4)])
    assert torch.all(replica_counts > strided_token_outputs.numel() * 0.2)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required for the Triton EPLB kernel")
@pytest.mark.parametrize("num_replicas", [2, 3, 4, 5, 101, 127, 128, 251])
def test_eplb_replica_hash_is_uniform_across_tokens_and_experts(num_replicas):
    from lightllm.common.basemodel.triton_kernel.fused_moe.eplb_topk_ids import (
        eplb_repair_topk_ids,
    )

    num_tokens = 4097
    num_experts = 256
    expert_ids = torch.arange(num_experts, dtype=torch.int32, device="cuda")
    logical_ids = expert_ids.expand(num_tokens, -1).contiguous()
    replica_counts = torch.full((num_experts, 3), num_replicas, dtype=torch.int32, device="cuda")
    physical_ids = expert_ids.unsqueeze(1) + num_experts * torch.arange(
        num_replicas,
        dtype=torch.int32,
        device="cuda",
    )
    logical_to_physical = torch.cat((replica_counts, physical_ids), dim=1)
    counter = torch.zeros((24, num_experts), dtype=torch.int64, device="cuda")
    sample_index = torch.zeros((2,), dtype=torch.int64, device="cuda")

    routed_physical_ids = eplb_repair_topk_ids(
        logical_topk_ids=logical_ids,
        logical_to_physical_map=logical_to_physical,
        prefill_route_counter=counter,
        prefill_route_sample_index=sample_index,
        update_prefill_route_counter=False,
        decode_route_counter=None,
        update_decode_route_counter=False,
        mode="global_first",
    )
    torch.cuda.synchronize()

    routed_replica_indices = routed_physical_ids // num_experts
    observed_counts = torch.stack(
        [(routed_replica_indices == replica_index).sum(dim=0) for replica_index in range(num_replicas)],
        dim=1,
    )
    expected_count = num_tokens / num_replicas
    deviations = observed_counts - expected_count
    if num_replicas <= 5:
        max_relative_deviation = (deviations.abs() / expected_count).max().item()
        assert max_relative_deviation < 0.12
    else:
        # 副本很多时单槽期望样本较少，使用每个 expert 的归一化卡方值
        # 检查整体形状，并额外检查跨 expert 汇总后的单槽最大偏差。
        normalized_chi_square = (deviations.square() / expected_count).sum(dim=1) / (num_replicas - 1)
        assert normalized_chi_square.max().item() < 1.75

        aggregate_expected_count = num_tokens * num_experts / num_replicas
        aggregate_max_relative_deviation = (
            (observed_counts.sum(dim=0) - aggregate_expected_count).abs() / aggregate_expected_count
        ).max()
        assert aggregate_max_relative_deviation.item() < 0.06


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required for the Triton EPLB kernel")
def test_eplb_repair_topk_ids_empty_input_skips_kernel():
    from lightllm.common.basemodel.triton_kernel.fused_moe.eplb_topk_ids import (
        eplb_repair_topk_ids,
    )

    experts = 64
    logical_ids = torch.empty((0, 4), dtype=torch.int32, device="cuda")
    counter = torch.zeros((24, experts), dtype=torch.int64, device="cuda")
    sample_index = torch.zeros((2,), dtype=torch.int64, device="cuda")
    logical_to_physical = torch.stack(
        (
            torch.ones((experts,), dtype=torch.int32, device="cuda"),
            torch.ones((experts,), dtype=torch.int32, device="cuda"),
            torch.ones((experts,), dtype=torch.int32, device="cuda"),
            torch.arange(experts, dtype=torch.int32, device="cuda"),
        ),
        dim=1,
    )
    physical_ids = eplb_repair_topk_ids(
        logical_topk_ids=logical_ids,
        logical_to_physical_map=logical_to_physical,
        prefill_route_counter=counter,
        prefill_route_sample_index=sample_index,
        update_prefill_route_counter=True,
        decode_route_counter=None,
        update_decode_route_counter=False,
        mode="current_gpu_first",
    )

    assert physical_ids.shape == (0, 4)
    assert physical_ids.dtype is torch.int32
    assert torch.equal(counter, torch.zeros_like(counter))
    assert torch.equal(sample_index, torch.zeros_like(sample_index))


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required for the Triton EPLB kernel")
def test_eplb_repair_topk_ids_rejects_unknown_dispatch_mode():
    from lightllm.common.basemodel.triton_kernel.fused_moe.eplb_topk_ids import (
        eplb_repair_topk_ids,
    )

    logical_ids = torch.empty((0, 1), dtype=torch.int32, device="cuda")
    logical_to_physical = torch.tensor([[1, 1, 1, 0]], dtype=torch.int32, device="cuda")
    counter = torch.zeros((24, 1), dtype=torch.int64, device="cuda")
    sample_index = torch.zeros((2,), dtype=torch.int64, device="cuda")

    with pytest.raises(AssertionError, match="unsupported EPLB dispatch mode"):
        eplb_repair_topk_ids(
            logical_topk_ids=logical_ids,
            logical_to_physical_map=logical_to_physical,
            prefill_route_counter=counter,
            prefill_route_sample_index=sample_index,
            update_prefill_route_counter=False,
            decode_route_counter=None,
            update_decode_route_counter=False,
            mode="unknown",
        )
