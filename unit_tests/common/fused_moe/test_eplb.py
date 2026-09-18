import threading
import time
from concurrent.futures import Future
from contextlib import nullcontext
from types import SimpleNamespace

import pytest
import torch

from lightllm.common.basemodel.layer_weights.meta_weights.fused_moe.eplb_placement import (
    build_initial_local_expert_ids,
    build_logical_to_physical_map,
    build_logical_to_physical_maps_for_layers,
)
from lightllm.common.basemodel.layer_weights.meta_weights.fused_moe.eplb_planner import (
    EPLBPlanner,
    GreedyEPLBPlanner,
)
from lightllm.server.api_cli import make_argument_parser
from lightllm.server.core.objs.start_args_type import StartArgs
from lightllm.server.router.model_infer.infer_batch import g_infer_context
from lightllm.server.router.model_infer.mode_backend import (
    eplb_manager as manager_module,
)
from lightllm.server.router.model_infer.mode_backend import (
    eplb_transfer as transfer_module,
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
from lightllm.common.eplb_utils import extract_eplb_expert_tensors
from lightllm.server.router.model_infer.mode_backend.eplb_transfer import (
    EPLBTransferInfo,
    ExpertTensorBuffer,
    PinnedMemoryEPLBTransfer,
    TransferStatus,
    build_transfer_plan,
)


def _test_moe_impl(
    *,
    eplb=False,
    num_logical_experts=128,
    world_size=16,
    num_redundant_experts_per_rank=1,
    route_counter=None,
    recording=False,
):
    logical_to_physical_map = None
    if eplb:
        if route_counter is None:
            route_counter = torch.zeros((num_logical_experts,), dtype=torch.int64)
        logical_to_physical_map = torch.zeros((num_logical_experts, world_size + 2), dtype=torch.int32)
        logical_to_physical_map[:, 0] = 1
    else:
        num_redundant_experts_per_rank = 0
    return SimpleNamespace(
        n_routed_experts=num_logical_experts,
        num_primary_experts_per_rank=num_logical_experts // world_size,
        num_total_physical_experts=(num_logical_experts + world_size * num_redundant_experts_per_rank),
        num_redundant_experts_per_rank=num_redundant_experts_per_rank,
        local_logics_expert_ids_list=list(range(num_logical_experts // world_size + num_redundant_experts_per_rank)),
        logical_to_physical_map=logical_to_physical_map,
        route_counter=route_counter,
        recording=recording,
    )


def _set_deepgemm_runtime(impl, runtime):
    for name in (
        "num_primary_experts_per_rank",
        "num_total_physical_experts",
        "num_redundant_experts_per_rank",
        "logical_to_physical_map",
        "route_counter",
        "recording",
    ):
        setattr(impl, name, getattr(runtime, name))


def _initial_extra_expert_placement(num_logical_experts, world_size, num_redundant_experts_per_rank):
    num_primary_experts_per_rank = num_logical_experts // world_size
    initial_local_expert_ids_by_rank = build_initial_local_expert_ids(
        num_logical_experts, world_size, num_redundant_experts_per_rank
    )
    return torch.tensor(
        [expert_ids[num_primary_experts_per_rank:] for expert_ids in initial_local_expert_ids_by_rank],
        dtype=torch.int64,
    )


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

        def _prepare_expert_execution(self, topk_weights, topk_ids, shared_expert_gate=None):
            seen["prepare"] = {"topk_ids": topk_ids}
            return topk_weights, "physical_ids"

        def _fused_experts(
            self,
            input_tensor,
            w13,
            w2,
            topk_weights,
            topk_ids,
            router_logits=None,
            is_prefill=None,
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
    )
    assert result == "output"
    assert captured == ["logical_ids"]
    assert seen["prepare"]["topk_ids"] == "logical_ids"
    assert seen["fused"]["topk_ids"] == "physical_ids"


def test_deepgemm_runtime_derives_expert_layout():
    runtime = _test_moe_impl(
        eplb=True,
        num_logical_experts=4,
        world_size=2,
        num_redundant_experts_per_rank=1,
    )
    assert runtime.num_primary_experts_per_rank == 2
    assert runtime.num_total_physical_experts == 6


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
    assert not hasattr(ep_impl, "route_counter")
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


def test_find_fused_moe_weights_discovers_direct_layer_attributes(monkeypatch):
    class FakeFusedMoeWeight:
        def __init__(self, layer_num, enable_ep_moe=True):
            self.layer_num_ = layer_num
            self.enable_ep_moe = enable_ep_moe

    monkeypatch.setattr(manager_module, "FusedMoeWeight", FakeFusedMoeWeight)
    first = FakeFusedMoeWeight(3)
    alternate = FakeFusedMoeWeight(1)
    aliased = FakeFusedMoeWeight(2)
    disabled = FakeFusedMoeWeight(0, enable_ep_moe=False)
    model = SimpleNamespace(
        trans_layers_weight=[
            SimpleNamespace(moe_weight=first),
            SimpleNamespace(alternate_moe_weight=alternate),
            SimpleNamespace(moe_weight=aliased, alternate_moe_weight=aliased),
            SimpleNamespace(moe_weight=disabled),
        ]
    )

    assert manager_module._find_fused_moe_weights(model) == [alternate, aliased, first]


def test_eplb_redundant_experts_default_to_disabled():
    parser = make_argument_parser()

    assert parser.parse_args([]).eplb_num_redundant_experts_per_rank == 0
    assert parser.parse_args(["--eplb_num_redundant_experts_per_rank", "3"]).eplb_num_redundant_experts_per_rank == 3
    assert StartArgs().eplb_num_redundant_experts_per_rank == 0


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

    assert isinstance(GreedyEPLBPlanner(2, 1), EPLBPlanner)


def test_eplb_planner_builds_legal_concrete_slot_layout():
    planner = GreedyEPLBPlanner(
        4,
        1,
        expert_alignment=1,
        rebalance_gain_threshold=0.0,
    )
    current = _initial_extra_expert_placement(8, 4, 1).unsqueeze(0).tolist()
    load = torch.ones((1, 4, 8), dtype=torch.int64)
    load[:, :, 0] = 1000
    load[:, :, 4] = 500

    result = planner.plan(load.sum(dim=1).tolist(), current)
    placement = result.placement[0]

    for rank, row in enumerate(placement):
        assert len(row) == len(set(row))
        assert all(expert // 2 != rank for expert in row)
    assert max(map(max, result.after_rank_load)) <= max(map(max, result.before_rank_load))
    assert isinstance(result.placement, list)
    assert isinstance(result.changed_layers, list)


def test_eplb_planner_estimator_distributes_global_load_across_copies():
    planner = GreedyEPLBPlanner(
        4,
        1,
        expert_alignment=128,
    )
    placement = [[[2], [4], [6], [0]]]
    load = [[100, 200, 300, 400, 500, 600, 700, 800]]

    predicted = planner.estimate_rank_load(load, placement)

    assert predicted == [[640, 1024, 1280, 1408]]


def test_eplb_planner_does_not_move_zero_load_experts():
    planner = GreedyEPLBPlanner(2, 1)
    current = [[[3], [1]]]

    result = planner.plan([[0, 0, 0, 0]], current)

    assert result.reason == "no_improvement"
    assert result.placement == current


def test_eplb_planner_reserves_rank_capacity_for_remaining_copies():
    planner = GreedyEPLBPlanner(
        4,
        1,
        rebalance_gain_threshold=0.0,
    )
    current = _initial_extra_expert_placement(16, 4, 1).unsqueeze(0).tolist()
    load = torch.randint(
        0,
        10000,
        (1, 4, 16),
        generator=torch.Generator().manual_seed(2),
    )

    result = planner.plan(load.sum(dim=1).tolist(), current)

    assert len(result.placement) == len(current)
    assert all(len(actual) == len(expected) for actual, expected in zip(result.placement[0], current[0]))
    for rank, row in enumerate(result.placement[0]):
        assert all(expert // 4 != rank for expert in row)


def test_eplb_planner_keeps_high_redundancy_search_state_isolated():
    planner = GreedyEPLBPlanner(4, 3)
    current = _initial_extra_expert_placement(16, 4, 3).unsqueeze(0).tolist()
    load = [
        [22613, 26852, 21852, 23480, 13270, 14695, 28735, 22303, 15324, 19604, 21492, 25458, 14120, 12130, 18620, 22888]
    ]

    result = planner.plan(load, current)

    for rank, row in enumerate(result.placement[0]):
        assert len(row) == len(set(row)) == 3
        assert all(expert // 4 != rank for expert in row)


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
    logical_to_physical = build_logical_to_physical_map(rank_to_logic_expert_ids, num_logical_experts=4, current_rank=0)

    assert isinstance(logical_to_physical, list)
    assert len(logical_to_physical) == 4
    assert all(len(row) == 7 for row in logical_to_physical)
    assert [row[0] for row in logical_to_physical] == [2, 2, 2, 2]
    assert [row[1] for row in logical_to_physical] == [1, 1, 1, 1]
    assert [row[2] for row in logical_to_physical] == [0, 1, 2, 3]
    assert all(physical_id >= 0 for row in logical_to_physical for physical_id in row[2 : 2 + row[0]])
    assert all(physical_id == -1 for row in logical_to_physical for physical_id in row[2 + row[0] :])


def test_logical_to_physical_map_requires_expert_count_divisible_by_rank_count():
    rank_to_logic_expert_ids = [[0, 1, 0], [2, 3, 1]]

    with pytest.raises(AssertionError):
        build_logical_to_physical_map(rank_to_logic_expert_ids, num_logical_experts=5, current_rank=0)


def test_logical_to_physical_map_supports_all_redundant_slots_for_one_expert():
    logical_to_physical = build_logical_to_physical_map(
        [[0, 1, 0, 0], [2, 3, 0, 0]],
        num_logical_experts=4,
        current_rank=0,
    )

    # 1 个主副本加上 2 个 rank 的全部 4 个冗余槽。
    assert logical_to_physical[0][0] == 5
    assert len(logical_to_physical[0][2:]) == 5
    assert len(set(logical_to_physical[0][2:])) == 5


def test_logical_to_physical_map_prefers_current_rank_replica():
    redundant = [[4], [5], [0], [1]]
    rank_to_logic_expert_ids = _rank_to_logic_expert_ids(redundant, 8)
    rank0_map = build_logical_to_physical_map(rank_to_logic_expert_ids, num_logical_experts=8, current_rank=0)
    rank1_map = build_logical_to_physical_map(rank_to_logic_expert_ids, num_logical_experts=8, current_rank=1)
    fallback_redundant = [[4], [5], [0], [1], [2], [3]]
    rank4_map = build_logical_to_physical_map(_rank_to_logic_expert_ids(fallback_redundant, 12), 12, current_rank=4)

    assert rank0_map[0][0] == rank1_map[0][0] == 2
    assert rank0_map[0][1] == 1
    assert rank1_map[0][1] == 0
    assert rank0_map[0][2] == 0
    assert set(rank0_map[0][2:4]) == {0, 8}
    assert set(rank1_map[0][2:4]) == {0, 8}
    assert rank4_map[0][0] == 2
    assert rank4_map[0][1] == 0
    assert set(rank4_map[0][2:4]) == {0, 8}


def test_nonlocal_rank_routes_across_all_replicas():
    redundant = [[4], [5], [0], [1]]
    rank_to_logic_expert_ids = _rank_to_logic_expert_ids(redundant, 8)
    maps = [build_logical_to_physical_map(rank_to_logic_expert_ids, 8, current_rank=rank) for rank in range(4)]
    assert maps[0][0][1] == 1
    assert maps[1][0][1] == 0
    assert maps[2][0][1] == 1
    assert maps[3][0][1] == 0
    assert all(set(logical_map[0][2:4]) == {0, 8} for logical_map in maps)


def test_current_rank_moves_local_replica_to_front_without_changing_copies():
    # Expert 0 is primary on rank 0 and redundant on rank 1。两个 rank 都优先
    # 自己的本地副本，因此路由槽的起点不同，但候选集合和数量保持一致。
    redundant = [[1], [0], [3], [2]]
    rank_to_logic_expert_ids = _rank_to_logic_expert_ids(redundant, 4)
    rank0_map = build_logical_to_physical_map(rank_to_logic_expert_ids, 4, current_rank=0)
    rank1_map = build_logical_to_physical_map(rank_to_logic_expert_ids, 4, current_rank=1)

    assert rank0_map[0] == [2, 1, 0, 3, -1, -1, -1]
    assert rank1_map[0] == [2, 1, 3, 0, -1, -1, -1]


def test_current_rank_stably_moves_all_local_physical_ids_to_front():
    # Expert 0 在 rank 0/1/2 上依次对应 physical IDs [0, 1, 3, 5]。
    # 对 rank 1 构建路由表时，只把本地 ID 3 移到最前面；其余远端 ID
    # 仍保持原来的 [0, 1, 5] 顺序。
    rank_to_logic_expert_ids = [[0, 0], [1, 0], [2, 0]]

    rank1_map = build_logical_to_physical_map(rank_to_logic_expert_ids, num_logical_experts=3, current_rank=1)

    assert rank1_map[0] == [4, 1, 3, 0, 1, 5]


@pytest.mark.parametrize(
    "current_rank",
    [0, 1, 2, 3],
)
def test_logical_to_physical_maps_for_layers_match_single_layer_api(current_rank):
    placements_by_layer = torch.tensor(
        [
            [[4, 5], [0, 1], [0, 1], [2, 3]],
            [[6, 7], [0, 1], [0, 1], [2, 3]],
            [[4, 5], [0, 1], [0, 1], [2, 3]],
        ],
        dtype=torch.int64,
    )

    rank_to_logic_expert_ids_by_layer = [
        _rank_to_logic_expert_ids(placement.tolist(), 8) for placement in placements_by_layer
    ]
    maps_by_layer = build_logical_to_physical_maps_for_layers(
        rank_to_logic_expert_ids_by_layer,
        num_logical_experts=8,
        current_rank=current_rank,
    )
    expected_by_layer = [
        build_logical_to_physical_map(
            rank_to_logic_expert_ids,
            num_logical_experts=8,
            current_rank=current_rank,
        )
        for rank_to_logic_expert_ids in rank_to_logic_expert_ids_by_layer
    ]

    assert maps_by_layer == expected_by_layer
    assert all(
        physical_expert_id >= 0
        for logical_map in maps_by_layer
        for row in logical_map
        for physical_expert_id in row[2 : 2 + row[0]]
    )
    assert all(
        physical_expert_id == -1
        for logical_map in maps_by_layer
        for row in logical_map
        for physical_expert_id in row[2 + row[0] :]
    )


def test_transfer_plan_respects_explicit_target_slots():
    current = [[4, 5], [6, 7], [0, 1], [2, 3]]
    target = [[5, 4], [7, 6], [1, 0], [3, 2]]

    plan = build_transfer_plan(current, target, 3, num_logical_experts=8, world_size=4)

    assert all(info.layer_index == 3 for info in plan)
    assert {(info.dest_rank, info.source_logical_expert_id) for info in plan} == {
        (rank, target[rank][slot]) for rank in range(4) for slot in range(2)
    }


def test_manager_evaluating_copies_route_counters_to_cpu_without_modifying_them(monkeypatch):
    counters = [
        torch.tensor([10, 11], dtype=torch.int64),
        torch.tensor([40, 41], dtype=torch.int64),
    ]
    manager = manager_module.EPLBManager.__new__(manager_module.EPLBManager)
    manager.state = manager_module.EPLBManagerState.EVALUATING
    manager._eplb_impls = [
        _test_moe_impl(
            eplb=True,
            route_counter=counter,
            recording=False,
            num_logical_experts=2,
            world_size=1,
        )
        for counter in counters
    ]
    manager.num_logical_experts = 2
    manager.global_rank = 1
    manager.world_size = 1
    manager.control_group = object()
    local_token_counts = []

    def all_gather_object(output, local_token_count, **_kwargs):
        local_token_counts.append(local_token_count)
        output[:] = [0]

    monkeypatch.setattr(manager_module.dist, "all_gather_object", all_gather_object)

    manager._step_evaluating()

    assert local_token_counts == [102]
    assert torch.equal(counters[0], torch.tensor([10, 11], dtype=torch.int64))
    assert torch.equal(counters[1], torch.tensor([40, 41], dtype=torch.int64))


def test_manager_delegates_distribution_planning_to_planner_class(monkeypatch):
    manager = manager_module.EPLBManager.__new__(manager_module.EPLBManager)
    manager.global_rank = 0
    manager.world_size = 2
    manager.control_group = object()
    manager.current_placement = [[[1]]]
    logical_load = torch.tensor([[10, 20]])
    calls = []

    class Result:
        def as_dict(self):
            return {"kind": "no_improvement"}

    manager.planner = SimpleNamespace(plan=lambda load, placement: (calls.append((load, placement)) or Result()))
    broadcasts = []
    monkeypatch.setattr(
        manager_module.dist,
        "broadcast_object_list",
        lambda values, **kwargs: broadcasts.append((values, kwargs)),
    )

    result = manager._plan_and_broadcast(logical_load)

    assert result == {"kind": "no_improvement"}
    assert len(calls) == 1
    assert calls[0][0] == logical_load.tolist()
    assert calls[0][1] == manager.current_placement
    assert broadcasts == [([result], {"src": 0, "group": manager.control_group})]


def test_expert_load_imbalance_ratio_averages_layer_ratios():
    global_load = torch.tensor(
        [
            [2, 4, 6],
            [10, 10, 10],
        ],
        dtype=torch.int64,
    )

    ratio = manager_module._expert_load_imbalance_ratio(global_load)

    assert ratio == pytest.approx(1.25)


def test_manager_publishes_expert_load_metrics_from_rank_zero():
    calls = []
    manager = manager_module.EPLBManager.__new__(manager_module.EPLBManager)
    manager.global_rank = 0
    manager.metric_client = SimpleNamespace(gauge_set=lambda name, value: calls.append((name, value)))

    manager._publish_expert_load_metric(
        {
            "expert_imbalance_ratio": 1.25,
        }
    )

    assert calls == [
        (manager_module.EPLB_EXPERT_IMBALANCE_RATIO_METRIC, 1.25),
    ]


def test_manager_does_not_publish_expert_load_metrics_from_other_ranks():
    manager = manager_module.EPLBManager.__new__(manager_module.EPLBManager)
    manager.global_rank = 1

    manager._publish_expert_load_metric({"expert_imbalance_ratio": 1.25})

    assert not hasattr(manager, "metric_client")


def test_eplb_route_counter_has_one_entry_per_logical_expert(monkeypatch):
    args = type(
        "Args",
        (),
        {"eplb_num_redundant_experts_per_rank": 2},
    )()
    monkeypatch.setattr(deepgemm_module, "get_env_start_args", lambda: args)
    monkeypatch.setattr(deepgemm_module, "get_global_world_size", lambda: 2)
    monkeypatch.setattr(deepgemm_module, "get_global_rank", lambda: 0)
    monkeypatch.setattr(torch.Tensor, "cuda", lambda tensor: tensor)
    original_zeros = torch.zeros

    def cpu_zeros(*shape, **kwargs):
        kwargs.pop("device", None)
        return original_zeros(*shape, **kwargs)

    monkeypatch.setattr(deepgemm_module.torch, "zeros", cpu_zeros)

    impl = deepgemm_module.FuseMoeDeepGEMM(4, 0, 1.0, SimpleNamespace())

    assert impl.route_counter.shape == (4,)


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
    assert not hasattr(impl, "route_counter")
    assert not hasattr(impl, "recording")


def test_manager_evaluation_gathers_token_counts_from_all_ranks(monkeypatch):
    manager = manager_module.EPLBManager.__new__(manager_module.EPLBManager)
    manager.weights = [
        type(
            "Weight",
            (),
            {
                "fuse_moe_impl": _test_moe_impl(
                    eplb=True,
                    route_counter=torch.zeros((4,), dtype=torch.int64),
                    num_logical_experts=4,
                    world_size=1,
                )
            },
        )()
    ]
    manager._eplb_impls = [manager.weights[0].fuse_moe_impl]
    manager.global_rank = 2
    manager.world_size = 4
    manager.step_interval = 20
    manager.num_logical_experts = 4
    manager.num_redundant_experts_per_rank = 1
    manager.current_placement = _initial_extra_expert_placement(4, 4, 1).unsqueeze(0).tolist()
    manager.control_group = object()
    local = torch.full((4,), 100, dtype=torch.int64)
    manager._eplb_impls[0].route_counter = local
    manager.state = manager_module.EPLBManagerState.EVALUATING
    seen = {}

    def all_gather_object(output, local_token_count, **kwargs):
        seen["local_token_count"] = local_token_count
        seen["group"] = kwargs["group"]
        # Simulate one other rank contributing the same logical-expert load.
        output[:] = [local_token_count, local_token_count, 0, 0]

    monkeypatch.setattr(manager_module.dist, "all_gather_object", all_gather_object)

    manager._step_evaluating()

    assert seen["group"] is manager.control_group
    assert seen["local_token_count"] == 400
    assert manager.state is manager_module.EPLBManagerState.COLLECTING


def test_manager_planning_builds_improved_metadata_in_one_multilayer_call(
    monkeypatch,
):
    manager = manager_module.EPLBManager.__new__(manager_module.EPLBManager)
    manager.weights = [
        type(
            "Weight",
            (),
            {
                "fuse_moe_impl": _test_moe_impl(
                    eplb=True,
                    route_counter=torch.zeros((4,), dtype=torch.int64),
                    num_logical_experts=4,
                    world_size=4,
                )
            },
        )()
        for _ in range(3)
    ]
    manager._eplb_impls = [weight.fuse_moe_impl for weight in manager.weights]
    manager._weights = manager.weights
    for layer_num, weight in enumerate(manager._weights):
        weight.layer_num_ = layer_num
    manager.global_rank = 1
    manager.world_size = 4
    manager.step_interval = 20
    manager.num_logical_experts = 4
    manager.num_redundant_experts_per_rank = 1
    manager.current_placement = _initial_extra_expert_placement(4, 4, 1).unsqueeze(0).expand(3, -1, -1).clone().tolist()
    manager.control_group = object()
    local = torch.full((3, 4), 100, dtype=torch.int64)
    planned_placement = torch.tensor(
        [
            [[3], [0], [1], [2]],
            [[2], [3], [0], [1]],
            [[2], [3], [0], [1]],
        ],
        dtype=torch.int64,
    )
    manager._plan_and_broadcast = lambda _global_load: {
        "kind": "planned",
        "placement": planned_placement.tolist(),
        "changed_layers": [True, False, True],
    }
    calls = []
    original_build_maps_for_layers = manager_module.build_logical_to_physical_maps_for_layers

    def build_maps_for_layers(*args, **kwargs):
        placements = args[0]
        calls.append((len(placements), len(placements[0]), len(placements[0][0])))
        return original_build_maps_for_layers(*args, **kwargs)

    monkeypatch.setattr(
        manager_module,
        "build_logical_to_physical_maps_for_layers",
        build_maps_for_layers,
    )
    monkeypatch.setattr(manager_module.dist, "all_reduce", lambda _tensor, **_kwargs: None)

    planning = Future()
    manager._plan(local, planning)
    result = planning.result()

    assert calls == [(2, 4, 2)]
    assert result["expert_imbalance_ratio"] == 1.0
    metadata = result["metadata"]
    assert 1 not in metadata
    assert {info.layer_index for info in result["transfer_infos"]} == {0, 2}
    for layer_index in (0, 2):
        item = metadata[layer_index]
        expected = build_logical_to_physical_map(
            _rank_to_logic_expert_ids(planned_placement[layer_index].tolist(), 4),
            4,
            current_rank=manager.global_rank,
        )
        assert torch.equal(item, torch.tensor(expected, dtype=torch.int32))


def test_decode_dispatch_uses_physical_ids_and_total_expert_count(monkeypatch):
    class Buffer:
        def low_latency_dispatch(self, **kwargs):
            calls.append(kwargs)
            return "recv", "masked", "handle", "event", "hook"

    impl = object.__new__(deepgemm_module.FuseMoeDeepGEMM)
    impl.quant_method = type("Quant", (), {"method_name": "fp8"})()
    impl.n_routed_experts = 128
    _set_deepgemm_runtime(impl, _test_moe_impl(eplb=True))
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
        deepgemm_module,
        "get_deepep_num_max_dispatch_tokens_per_rank_decode",
        lambda: 16,
    )
    monkeypatch.setattr(deepgemm_module.dist_group_manager, "ep_low_latency_buffer", Buffer())

    result = impl.low_latency_dispatch(
        torch.empty((1, 4)),
        torch.empty((1, 128)),
        None,
        False,
        2,
        False,
        0,
        0,
        "softmax",
    )

    assert result[2].tolist() == [[128, 143]]
    assert repairs[0]["logical_topk_ids"] is logical_ids
    assert calls[0]["num_experts"] == 144


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
    monkeypatch.setattr(deepgemm_module, "quantize_fused_experts_input", lambda *_args: "qinput")

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
    assert not calls[0]["update_logical_expert_counter"]


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
        route_counter=torch.zeros((128,), dtype=torch.int64),
        recording=True,
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
    monkeypatch.setattr(deepgemm_module, "quantize_fused_experts_input", lambda *_args: "qinput")
    monkeypatch.setattr(deepgemm_module.dist_group_manager, "ep_buffer", Buffer())
    monkeypatch.setattr(
        deepgemm_module,
        "get_deepep_num_max_dispatch_tokens_per_rank_prefill",
        lambda: 16,
    )
    monkeypatch.setattr(deepgemm_module, "get_ep_num_sms", lambda: 8)

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
    impl.dispatch(
        qinput,
        topk_idx,
        weights,
        overlap_event=caller_event,
    )

    assert topk_idx is physical_ids
    assert len(repair_calls) == 1
    assert repair_calls[0]["logical_topk_ids"] is logical_ids
    assert repair_calls[0]["update_logical_expert_counter"]
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
    monkeypatch.setattr(deepgemm_module, "get_ep_num_sms", lambda: 8)

    impl.dispatch(
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
        lambda: SimpleNamespace(eplb_num_redundant_experts_per_rank=1),
    )
    monkeypatch.setattr(deepgemm_module, "get_global_world_size", lambda: 2)
    monkeypatch.setattr(deepgemm_module, "get_global_rank", lambda: 0)
    monkeypatch.setattr(torch.Tensor, "cuda", lambda tensor: tensor)
    original_zeros = torch.zeros

    def cpu_zeros(*shape, **kwargs):
        kwargs.pop("device", None)
        return original_zeros(*shape, **kwargs)

    monkeypatch.setattr(deepgemm_module.torch, "zeros", cpu_zeros)
    impl = deepgemm_module.FuseMoeDeepGEMM(4, 0, 1.0, SimpleNamespace())

    assert impl.num_primary_experts_per_rank == 2
    assert impl.num_redundant_experts_per_rank == 1
    assert impl.num_total_physical_experts == 6
    assert impl.route_counter.shape == (4,)
    assert impl.recording
    assert impl.local_logics_expert_ids_list == [0, 1, 2]
    assert not hasattr(impl, "initial_local_expert_ids_by_rank")
    assert not hasattr(impl, "expert_parallel_state")


def test_eplb_prepare_repairs_logical_ids(monkeypatch):
    impl = object.__new__(deepgemm_module.FuseMoeDeepGEMM)
    runtime = _test_moe_impl(eplb=True, recording=True)
    _set_deepgemm_runtime(impl, runtime)
    logical_ids = torch.tensor([[3, 4]], dtype=torch.int32)
    physical_ids = torch.tensor([[13, 14]], dtype=torch.int32)
    calls = []

    def repair(**kwargs):
        calls.append(kwargs)
        return physical_ids

    monkeypatch.setattr(deepgemm_module, "eplb_repair_topk_ids", repair)
    weights, selected = impl._prepare_expert_execution(torch.ones((1, 2)), logical_ids)

    assert weights.tolist() == [[1.0, 1.0]]
    assert selected is physical_ids
    assert calls[0]["logical_topk_ids"] is logical_ids
    assert calls[0]["update_logical_expert_counter"]


def test_decode_masked_group_gemm_uses_all_physical_rows_when_eplb_is_enabled(
    monkeypatch,
):
    impl = object.__new__(deepgemm_module.FuseMoeDeepGEMM)
    _set_deepgemm_runtime(impl, _test_moe_impl(eplb=True, num_logical_experts=8, world_size=1))
    captured = {}

    def masked(*args, **kwargs):
        captured["w13"] = args[3]
        captured["w13_scale"] = args[4]
        captured["w2"] = args[5]
        captured["w2_scale"] = args[6]
        return "out"

    monkeypatch.setattr(deepgemm_module, "masked_group_gemm", masked)
    pack = lambda: type(
        "Pack",
        (),
        {"weight": torch.empty((10, 4)), "weight_scale": torch.empty((10, 1))},
    )()

    assert impl.masked_group_gemm((torch.empty((1, 4)),), pack(), pack(), torch.empty(8), torch.float16, 1) == "out"
    assert captured["w13"].shape[0] == captured["w2"].shape[0] == 10
    assert captured["w13_scale"].shape[0] == captured["w2_scale"].shape[0] == 10


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

    monkeypatch.setattr(deepgemm_module, "fused_experts", fused)
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


def test_transfer_plan_always_uses_primary_expert_rank():
    current = [[4, 5], [6, 7], [0, 1], [2, 3]]
    target = [[6, 5], [6, 7], [0, 4], [2, 3]]
    plan = build_transfer_plan(current, target, 5, num_logical_experts=8, world_size=4)
    assert plan == [
        EPLBTransferInfo(3, 5, 6, 0, 2),
        EPLBTransferInfo(2, 5, 4, 2, 3),
    ]


def test_transfer_plan_uses_same_primary_source_for_repeated_expert():
    current = [[0, 1], [2, 3], [4, 5], [4, 7]]
    target = [[4, 4], [2, 3], [4, 5], [4, 7]]
    first = build_transfer_plan(current, target, 5, 8, 4)
    second = build_transfer_plan(current, target, 5, 8, 4)
    assert first == second
    assert first == [
        EPLBTransferInfo(2, 5, 4, 0, 2),
        EPLBTransferInfo(2, 5, 4, 0, 3),
    ]


def test_p2p_message_tag_is_stable_and_identifies_transfer_tensor():
    transfer = object.__new__(PinnedMemoryEPLBTransfer)
    transfer.transfer_info = EPLBTransferInfo(1, 5, 4, 0, 2)
    weight_tag = transfer._build_p2p_message_tag("w13.weight")

    assert weight_tag == transfer._build_p2p_message_tag("w13.weight")
    assert 0 <= weight_tag <= 0x7FFFFFFF
    assert weight_tag != transfer._build_p2p_message_tag("w13.weight_scale")
    transfer.transfer_info = EPLBTransferInfo(1, 5, 6, 0, 2)
    assert weight_tag != transfer._build_p2p_message_tag("w13.weight")
    transfer.transfer_info = EPLBTransferInfo(1, 5, 4, 0, 3)
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


def test_manager_commits_completed_transfer_rows_by_planned_destination_index():
    live = torch.arange(20).reshape(5, 4)
    original_primary = live[:3].clone()
    local_expert_ids = [0, 1, 2, 4, 5]
    manager = manager_module.EPLBManager.__new__(manager_module.EPLBManager)
    manager.global_rank = 0
    manager.num_primary_experts_per_rank = 3
    manager.target_placement = [[[7, 6]]]
    manager._eplb_impls = [SimpleNamespace(local_logics_expert_ids_list=local_expert_ids)]
    manager._commit_layer_metadata = lambda _layer: None
    manager.completed_layer_transfers = [
        SimpleNamespace(
            transfer_info=EPLBTransferInfo(1, 0, 7, 0, 3),
            tensor_buffers=[ExpertTensorBuffer("weight", live, torch.full((4,), -7))],
        ),
        SimpleNamespace(
            transfer_info=EPLBTransferInfo(2, 0, 6, 0, 4),
            tensor_buffers=[ExpertTensorBuffer("weight", live, torch.full((4,), -6))],
        ),
    ]

    manager._commit_transferred_layer(0)

    assert torch.equal(live[:3], original_primary)
    assert torch.equal(live[3], torch.full((4,), -7))
    assert torch.equal(live[4], torch.full((4,), -6))
    assert local_expert_ids == [0, 1, 2, 7, 6]


def test_manager_inflight_ready_gate_commits_one_layer(monkeypatch):
    class Transfer:
        def __init__(self, transfer_info):
            self.transfer_info = transfer_info
            self.status = TransferStatus.SUCCEEDED

        def is_finished(self):
            return True

    info0 = EPLBTransferInfo(0, 0, 2, 1, 2)
    info0b = EPLBTransferInfo(1, 0, 3, 0, 2)
    info1 = EPLBTransferInfo(0, 1, 4, 1, 2)
    manager = manager_module.EPLBManager.__new__(manager_module.EPLBManager)
    manager.active_transfer = Transfer(info0)
    manager.control_group = object()
    manager.world_size = 2
    manager.pending_transfer_infos = [info0, info0b, info1]
    manager.completed_layer_transfers = []
    manager.global_rank = 1
    manager.steps = 0
    manager.step_interval = 20
    committed, finished = [], []
    manager._commit_transferred_layer = committed.append
    manager._complete_rebalance = lambda: (finished.append(True) or 0.0)

    def start_next_transfer():
        manager.active_transfer = Transfer(manager.pending_transfer_infos[0])

    manager._start_next_transfer = start_next_transfer
    operations = []

    class CurrentStream:
        def wait_stream(self, stream):
            operations.append(("wait", stream))

    overlap_stream = object()
    monkeypatch.setattr(manager_module.torch.cuda, "current_stream", lambda: CurrentStream())
    monkeypatch.setattr(g_infer_context, "get_overlap_stream", lambda: overlap_stream)

    def set_global_ready(ready):
        def all_gather_object(output, _local_ready, **_kwargs):
            output[:] = [ready] * manager.world_size

        return all_gather_object

    monkeypatch.setattr(manager_module.dist, "all_gather_object", set_global_ready(False))
    manager._step_transferring()
    assert committed == []

    monkeypatch.setattr(manager_module.dist, "all_gather_object", set_global_ready(True))
    manager._step_transferring()
    assert committed == []
    assert operations == []
    assert not finished

    manager._step_transferring()
    assert committed == [0]
    assert operations == [("wait", overlap_stream)]

    manager._step_transferring()
    assert committed == [0, 1]
    assert finished == [True]

    expected = EPLBTransferInfo(0, 2, 5, 1, 2)
    manager.pending_transfer_infos = [expected]
    manager.active_transfer = Transfer(EPLBTransferInfo(1, 2, 5, 1, 2))
    with pytest.raises(RuntimeError, match="does not match"):
        manager._step_transferring()


def test_background_work_ready_reports_pending_completion_and_errors(monkeypatch):
    manager = manager_module.EPLBManager.__new__(manager_module.EPLBManager)
    manager.control_group = object()
    manager.world_size = 2

    work = Future()
    assert not manager._background_work_ready_on_all_ranks(work, "planning")

    work.set_exception(RuntimeError("planning boom"))
    statuses = []

    def retain_local_error(output, local_failed, **_kwargs):
        statuses.append(local_failed)
        output[:] = [local_failed, False]

    monkeypatch.setattr(manager_module.dist, "all_gather_object", retain_local_error)
    with pytest.raises(RuntimeError, match="EPLB planning failed on this rank") as exc_info:
        manager._background_work_ready_on_all_ranks(work, "planning")
    assert isinstance(exc_info.value.__cause__, RuntimeError)
    assert str(exc_info.value.__cause__) == "planning boom"
    assert statuses == [True]

    work = Future()
    work.set_result({"kind": "no_improvement"})
    statuses.clear()

    def remote_error(output, local_failed, **_kwargs):
        statuses.append(local_failed)
        output[:] = [local_failed, True]

    monkeypatch.setattr(manager_module.dist, "all_gather_object", remote_error)
    with pytest.raises(RuntimeError, match="EPLB planning failed on another rank"):
        manager._background_work_ready_on_all_ranks(work, "planning")
    assert statuses == [False]


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_manager_inflight_commit_orders_live_weights_between_overlap_forwards(
    monkeypatch,
):
    class Transfer:
        def __init__(self, live, received, transfer_info):
            self.tensor_buffers = [ExpertTensorBuffer("weight", live, received)]
            self.transfer_info = transfer_info
            self.status = TransferStatus.SUCCEEDED

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
    transfer_info = EPLBTransferInfo(0, 0, 2, 0, 0)
    manager.active_transfer = Transfer(live, received, transfer_info)
    manager.control_group = object()
    manager.world_size = 1
    manager.pending_transfer_infos = [transfer_info]
    manager.completed_layer_transfers = []
    manager.num_primary_experts_per_rank = 0
    manager.global_rank = 0
    manager.target_placement = [[[2]]]
    manager._eplb_impls = [SimpleNamespace(local_logics_expert_ids_list=[1])]
    manager._commit_layer_metadata = lambda _layer: None
    manager._complete_rebalance = lambda: 0.0
    manager.steps = 0
    manager.step_interval = 20
    monkeypatch.setattr(
        manager_module.dist,
        "all_gather_object",
        lambda output, local_ready, **_kwargs: output.__setitem__(slice(None), [local_ready]),
    )

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
    route_counter = torch.tensor([1, 2], dtype=torch.int64)
    local_token_counts = []
    manager.state = manager_module.EPLBManagerState.COLLECTING
    manager.global_rank = 1
    manager.steps = 0
    manager.step_interval = 3
    manager.next_evaluation_step = 3
    manager.num_logical_experts = 2
    manager._eplb_impls = [SimpleNamespace(route_counter=route_counter)]
    manager.world_size = 1
    manager.control_group = object()

    def all_gather_object(output, local_token_count, **_kwargs):
        local_token_counts.append(local_token_count)
        output[:] = [0]

    monkeypatch.setattr(manager_module.dist, "all_gather_object", all_gather_object)

    manager.step()
    manager.step()
    assert manager.state is manager_module.EPLBManagerState.COLLECTING
    manager.step()

    assert manager.state is manager_module.EPLBManagerState.EVALUATING
    assert local_token_counts == []
    assert manager.next_evaluation_step == 6

    manager.step()

    assert manager.state is manager_module.EPLBManagerState.COLLECTING
    assert local_token_counts == [3]
    assert manager.next_evaluation_step == 6
    assert torch.equal(route_counter, torch.tensor([1, 2], dtype=torch.int64))


def test_manager_step_uses_explicit_state_instead_of_pending_work():
    manager = manager_module.EPLBManager.__new__(manager_module.EPLBManager)
    manager.state = manager_module.EPLBManagerState.TRANSFERRING
    manager.pending_transfer_infos = [EPLBTransferInfo(0, 0, 2, 1, 2)]
    manager._planning = Future()
    calls = []
    manager._step_transferring = lambda: calls.append("transfer")
    manager._step_evaluating = lambda: calls.append("evaluation")

    manager.step()

    assert calls == ["transfer"]


def test_manager_enters_transferring_state_with_planned_work():
    manager = manager_module.EPLBManager.__new__(manager_module.EPLBManager)
    transfer_info = EPLBTransferInfo(0, 0, 2, 1, 2)
    starts = []
    manager.state = manager_module.EPLBManagerState.PLANNING
    manager.global_rank = 1
    manager._planning = Future()
    manager._planning.set_result(
        {
            "kind": "planned",
            "placement": [[[2], [3]]],
            "metadata": {0: torch.tensor([1])},
            "transfer_infos": [transfer_info],
            "expert_imbalance_ratio": 1.0,
        }
    )
    manager._background_work_ready_on_all_ranks = lambda _work, _phase: True
    manager._publish_expert_load_metric = lambda _result: None
    manager._start_next_transfer = lambda: starts.append(manager.pending_transfer_infos[0])

    manager._step_planning()

    assert manager.state is manager_module.EPLBManagerState.TRANSFERRING
    assert manager.pending_transfer_infos == [transfer_info]
    assert manager.completed_layer_transfers == []
    assert starts == [transfer_info]


def test_manager_evaluation_with_insufficient_tokens_returns_to_collecting(monkeypatch):
    manager = manager_module.EPLBManager.__new__(manager_module.EPLBManager)
    manager.state = manager_module.EPLBManagerState.EVALUATING
    manager._eplb_impls = [SimpleNamespace(route_counter=torch.full((4,), 255, dtype=torch.int64))]
    manager.num_logical_experts = 4
    manager.steps = 11
    manager.step_interval = 20
    manager.next_evaluation_step = 31
    manager.global_rank = 1
    manager.world_size = 1
    manager.control_group = object()
    monkeypatch.setattr(
        manager_module.dist,
        "all_gather_object",
        lambda output, local_token_count, **_kwargs: output.__setitem__(slice(None), [local_token_count]),
    )

    manager.step()

    assert manager.state is manager_module.EPLBManagerState.COLLECTING
    assert manager.next_evaluation_step == 31


def test_manager_evaluation_with_enough_tokens_enters_planning(monkeypatch):
    manager = manager_module.EPLBManager.__new__(manager_module.EPLBManager)
    local_load = torch.full((1, 4), 256, dtype=torch.int64)
    worker = SimpleNamespace(start=lambda: None)
    thread_args = []
    manager.state = manager_module.EPLBManagerState.EVALUATING
    manager.global_rank = 1
    manager.num_logical_experts = 4
    manager._eplb_impls = [SimpleNamespace(route_counter=local_load[0])]
    manager.world_size = 1
    manager.control_group = object()
    monkeypatch.setattr(
        manager_module.dist,
        "all_gather_object",
        lambda output, local_token_count, **_kwargs: output.__setitem__(slice(None), [local_token_count]),
    )
    monkeypatch.setattr(
        manager_module.threading,
        "Thread",
        lambda **kwargs: (thread_args.append(kwargs["args"]) or worker),
    )

    manager.step()

    assert manager.state is manager_module.EPLBManagerState.PLANNING
    assert torch.equal(manager._local_load, local_load)
    assert thread_args == []
    assert not hasattr(manager, "_planning")

    manager.step()

    assert manager.state is manager_module.EPLBManagerState.PLANNING
    assert torch.equal(thread_args[0][0], local_load)
    assert thread_args[0][1] is manager._planning
    assert not hasattr(manager, "_local_load")


def test_manager_planning_without_changes_returns_to_collecting():
    manager = manager_module.EPLBManager.__new__(manager_module.EPLBManager)
    manager.state = manager_module.EPLBManagerState.PLANNING
    manager.global_rank = 1
    manager.steps = 11
    manager.step_interval = 20
    manager.next_evaluation_step = 31
    manager._planning = Future()
    manager._planning.set_result({"kind": "no_improvement", "expert_imbalance_ratio": 1.0})
    manager._background_work_ready_on_all_ranks = lambda _work, _phase: True
    manager._publish_expert_load_metric = lambda _result: None
    manager.step()

    assert manager.state is manager_module.EPLBManagerState.COLLECTING
    assert manager.next_evaluation_step == 31
    assert not hasattr(manager, "_planning")


def test_manager_complete_rebalance_releases_transferring_state():
    manager = manager_module.EPLBManager.__new__(manager_module.EPLBManager)
    target_placement = [[[2], [3]]]
    manager.global_rank = 1
    manager.steps = 10
    manager.step_interval = 20
    manager.pending_transfer_infos = []
    manager.completed_layer_transfers = []
    manager.active_transfer = object()
    manager.target_placement = target_placement
    manager.target_metadata = {}
    manager.rebalance_started_at = time.time()

    elapsed = manager._complete_rebalance()

    assert manager.current_placement is target_placement
    assert elapsed >= 0
    for attribute in (
        "pending_transfer_infos",
        "completed_layer_transfers",
        "active_transfer",
        "target_placement",
        "target_metadata",
        "rebalance_started_at",
    ):
        assert not hasattr(manager, attribute)


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
    transfer.transfer_info = EPLBTransferInfo(0, 0, 5, 1, 2)
    transfer._device_to_host_stream = Stream()
    transfer._local_logical_expert_ids = [4, 5]
    transfer.tensor_buffers = [
        ExpertTensorBuffer(
            "weight",
            torch.tensor([[1.0, 2.0], [3.0, 4.0]]),
            torch.empty(2),
        )
    ]
    transfer.status = TransferStatus.RUNNING
    sends = []
    monkeypatch.setattr(transfer_module.torch.cuda, "set_device", lambda _device: None)
    monkeypatch.setattr(transfer_module.torch.cuda, "stream", lambda _stream: nullcontext())
    monkeypatch.setattr(
        transfer_module.dist,
        "send",
        lambda tensor, dst, group, tag: sends.append((tensor.clone(), dst, group, tag)),
    )

    transfer._run_transfer()

    assert transfer.status is TransferStatus.SUCCEEDED
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
    transfer.transfer_info = EPLBTransferInfo(0, 0, 5, 0, 1)
    transfer._device_to_host_stream = Stream()
    transfer._local_logical_expert_ids = [5]
    transfer.tensor_buffers = [
        ExpertTensorBuffer(
            "weight",
            torch.tensor([[3.0, 4.0]]),
            torch.empty(2),
        )
    ]
    transfer.status = TransferStatus.RUNNING
    p2p_calls = []
    monkeypatch.setattr(transfer_module.torch.cuda, "set_device", lambda _device: None)
    monkeypatch.setattr(transfer_module.torch.cuda, "stream", lambda _stream: nullcontext())
    monkeypatch.setattr(transfer_module.dist, "send", lambda *_args, **_kwargs: p2p_calls.append("send"))
    monkeypatch.setattr(transfer_module.dist, "recv", lambda *_args, **_kwargs: p2p_calls.append("recv"))

    transfer._run_transfer()

    assert transfer.is_finished()
    assert p2p_calls == []
    assert torch.equal(transfer.tensor_buffers[0].pinned_row, torch.tensor([3.0, 4.0]))


def test_pinned_transfer_exits_process_on_failure(monkeypatch):
    transfer = object.__new__(PinnedMemoryEPLBTransfer)
    transfer._device = "cuda:0"
    transfer._is_source_rank = False
    transfer._is_destination_rank = True
    transfer.transfer_info = EPLBTransferInfo(1, 0, 3, 0, 2)
    transfer._p2p_group = object()
    transfer.tensor_buffers = [
        ExpertTensorBuffer(
            "weight",
            torch.empty((1, 1)),
            torch.empty(1),
        )
    ]
    transfer.status = TransferStatus.RUNNING
    logged_messages = []
    exit_codes = []

    def fail_recv(*_args, **_kwargs):
        raise RuntimeError("recv failed")

    monkeypatch.setattr(transfer_module.torch.cuda, "set_device", lambda _device: None)
    monkeypatch.setattr(transfer_module.dist, "recv", fail_recv)
    monkeypatch.setattr(transfer_module.logger, "exception", logged_messages.append)
    monkeypatch.setattr(transfer_module.os, "_exit", exit_codes.append)

    transfer._run_transfer()

    assert logged_messages == ["EPLB transfer failed"]
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
    transfer.transfer_info = EPLBTransferInfo(1, 0, 3, 0, 2)
    transfer._p2p_group = object()
    transfer._device_to_host_stream = Stream()
    transfer.tensor_buffers = [
        ExpertTensorBuffer(
            "weight",
            torch.empty((1, 1)),
            torch.tensor([3.0]),
        )
    ]
    transfer.status = TransferStatus.IDLE
    transfer._transfer_thread = threading.Thread(target=transfer._run_transfer, daemon=True)
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
    while transfer.status is TransferStatus.RUNNING and time.monotonic() < deadline:
        time.sleep(0.001)
    assert transfer.status is TransferStatus.SUCCEEDED
    assert transfer.is_finished()
    assert torch.equal(transfer.tensor_buffers[0].pinned_row, torch.tensor([3.0]))
    assert len(receives) == 1
    assert receives[0][0] is transfer.tensor_buffers[0].pinned_row
    expected_tag = transfer._build_p2p_message_tag("weight")
    assert receives[0][1:] == (1, transfer._p2p_group, expected_tag)
    with pytest.raises(AssertionError, match="already been started"):
        transfer.start()


def test_manager_requires_more_than_one_rank(monkeypatch):
    monkeypatch.setattr(manager_module, "_find_fused_moe_weights", lambda model: [object()])
    monkeypatch.setattr(manager_module, "get_global_rank", lambda: 0)
    monkeypatch.setattr(manager_module, "get_global_world_size", lambda: 1)

    with pytest.raises(AssertionError, match="more than one rank"):
        manager_module.EPLBManager(type("Model", (), {})())


def test_manager_constructs_pinned_memory_transfer(monkeypatch):
    weight = type(
        "Weight",
        (),
        {
            "n_routed_experts": 4,
            "layer_num_": 0,
            "fuse_moe_impl": _test_moe_impl(
                eplb=True,
                recording=True,
                num_logical_experts=4,
                world_size=2,
                num_redundant_experts_per_rank=2,
                route_counter=torch.zeros((4,), dtype=torch.int64),
            ),
        },
    )()
    transfer_starts = []
    transfer = SimpleNamespace(start=lambda: transfer_starts.append(True))
    groups = [object(), object()]
    new_group_calls = []
    monkeypatch.setattr(manager_module, "_find_fused_moe_weights", lambda model: [weight])
    monkeypatch.setattr(manager_module, "get_global_rank", lambda: 0)
    monkeypatch.setattr(manager_module, "get_global_world_size", lambda: 2)
    monkeypatch.setattr(manager_module, "get_eplb_step_interval", lambda: 20)
    monkeypatch.setattr(manager_module, "get_eplb_rebalance_gain_threshold", lambda: 0.07)
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

    def all_gather_object(output, local_redundant_expert_ids_by_layer, group):
        all_gather_calls.append((local_redundant_expert_ids_by_layer, group))
        output[:] = [local_redundant_expert_ids_by_layer, [[0, 1]]]

    monkeypatch.setattr(manager_module.dist, "all_gather_object", all_gather_object)
    transfer_calls = []
    monkeypatch.setattr(
        manager_module,
        "PinnedMemoryEPLBTransfer",
        lambda weights, group, rank, info: (transfer_calls.append((weights, group, rank, info)) or transfer),
    )
    logs = []
    monkeypatch.setattr(manager_module.logger, "info", lambda message: logs.append(message))
    manager = manager_module.EPLBManager(type("Model", (), {})())
    assert not hasattr(manager, "_planning")
    assert not hasattr(manager, "active_transfer")
    assert not hasattr(manager, "target_placement")
    assert manager.state is manager_module.EPLBManagerState.COLLECTING
    assert (manager.control_group, manager.transfer_group) == tuple(groups)
    assert new_group_calls == [(([0, 1],), {"backend": "gloo"})] * 2
    assert all_gather_calls == [([[2, 3]], groups[0])]
    transfer_info = EPLBTransferInfo(0, 0, 2, 1, 2)
    manager.pending_transfer_infos = [transfer_info]
    manager._start_next_transfer()
    assert transfer_calls == [([weight], groups[1], 0, transfer_info)]
    assert transfer_starts == [True]
    assert manager.planner.rebalance_gain_threshold == 0.07
    assert manager.current_placement == [[[2, 3], [0, 1]]]
    assert manager.metric_client is metric_client
    assert metric_client_ports == [1234]
    assert manager.next_evaluation_step == manager.step_interval
    assert "planner=GreedyEPLBPlanner" in logs[0]
    assert weight.fuse_moe_impl.recording
    assert manager._eplb_impls[0] is weight.fuse_moe_impl
    assert not hasattr(weight.fuse_moe_impl, "update_logical_expert_counter")


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required for the Triton EPLB kernel")
@pytest.mark.parametrize("update_logical_expert_counter", [False, True])
@pytest.mark.parametrize("tokens", [1, 32])
def test_eplb_repair_topk_ids_maps_and_counts(update_logical_expert_counter, tokens):
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
        torch.full_like(logical_experts, 2),
        torch.ones_like(logical_experts),
    )
    has_local_replica = (logical_experts % 5 == 0).to(torch.int32)
    logical_to_physical = torch.stack(
        (
            replica_counts,
            has_local_replica,
            logical_experts,
            torch.where(replica_counts == 2, logical_experts + experts, logical_experts),
        ),
        dim=1,
    )
    counter = torch.zeros((experts,), dtype=torch.int64, device="cuda")
    expected_counter = torch.zeros_like(counter)

    logical_ids_long = logical_ids.to(torch.long)
    token_indices = torch.arange(tokens, device="cuda", dtype=torch.int64).unsqueeze(1)
    replica_indices = (
        (((token_indices * 2654435769) & 0xFFFFFFFF) + ((logical_ids.to(torch.int64) * 2246822519) & 0xFFFFFFFF))
        & 0xFFFFFFFF
    ) % replica_counts[logical_ids_long].to(torch.int64)
    replica_indices = torch.where(has_local_replica[logical_ids_long] != 0, 0, replica_indices)
    expected_ids = logical_to_physical[logical_ids_long, replica_indices + 2]
    if update_logical_expert_counter:
        expected_counter.scatter_add_(
            0,
            logical_ids.reshape(-1).to(torch.long),
            torch.ones(logical_ids.numel(), dtype=torch.int64, device="cuda"),
        )

    physical_ids = eplb_repair_topk_ids(
        logical_topk_ids=logical_ids,
        logical_to_physical_map=logical_to_physical,
        logical_expert_counter=counter,
        update_logical_expert_counter=update_logical_expert_counter,
    )
    torch.cuda.synchronize()

    assert torch.equal(logical_ids, original_logical_ids)
    assert torch.equal(physical_ids, expected_ids)
    assert torch.equal(counter, expected_counter)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required for the Triton EPLB kernel")
def test_eplb_repair_topk_ids_empty_input_skips_kernel():
    from lightllm.common.basemodel.triton_kernel.fused_moe.eplb_topk_ids import (
        eplb_repair_topk_ids,
    )

    experts = 64
    logical_ids = torch.empty((0, 4), dtype=torch.int32, device="cuda")
    counter = torch.zeros((experts,), dtype=torch.int64, device="cuda")
    logical_to_physical = torch.stack(
        (
            torch.ones((experts,), dtype=torch.int32, device="cuda"),
            torch.ones((experts,), dtype=torch.int32, device="cuda"),
            torch.arange(experts, dtype=torch.int32, device="cuda"),
        ),
        dim=1,
    )
    physical_ids = eplb_repair_topk_ids(
        logical_topk_ids=logical_ids,
        logical_to_physical_map=logical_to_physical,
        logical_expert_counter=counter,
        update_logical_expert_counter=True,
    )

    assert physical_ids.shape == (0, 4)
    assert physical_ids.dtype is torch.int32
    assert torch.equal(counter, torch.zeros_like(counter))
