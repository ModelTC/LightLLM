import threading
import time
from contextlib import nullcontext
from types import SimpleNamespace

import pytest
import torch

from lightllm.common.basemodel.layer_weights.meta_weights.fused_moe.eplb_placement import (
    build_initial_local_expert_ids,
    build_logical_to_physical_map,
    build_logical_to_physical_maps_for_layers,
    _estimate_rank_load,
    plan_redundant_experts,
    select_improving_placements,
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
from lightllm.server.router.model_infer.mode_backend.base_backend import ModeBackend
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
    PinnedMemoryEPLBTransfer,
    TransferStep,
    _commit_staging_rows,
    align_target_placement,
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


def _manual_runtime_rank_load(source_load, placement, alignment):
    """Reference the committed runtime logical-to-physical maps on CPU."""
    samples, layers, source_ranks, num_logical_experts = source_load.shape
    ranks, redundant = placement.shape[1:]
    assert source_ranks == ranks
    num_experts_per_rank = num_logical_experts // ranks
    num_physical_experts_per_rank = num_experts_per_rank + redundant
    raw = torch.zeros((samples, layers, ranks, num_logical_experts), dtype=torch.float64)
    for layer in range(layers):
        for current_rank in range(ranks):
            logical_to_physical = build_logical_to_physical_map(
                _rank_to_logic_expert_ids(placement[layer].tolist(), num_logical_experts),
                num_logical_experts,
                current_rank=current_rank,
            )
            for expert, packed_row in enumerate(logical_to_physical):
                num_replicas = packed_row[0]
                physical_expert_ids = packed_row[2 : num_replicas + 2]
                if packed_row[1]:
                    physical_expert_ids = physical_expert_ids[:1]
                for physical_id in physical_expert_ids:
                    rank = physical_id // num_physical_experts_per_rank
                    raw[:, layer, rank, expert] += source_load[:, layer, current_rank, expert] / len(
                        physical_expert_ids
                    )
    return (torch.ceil(raw / alignment) * alignment).sum(dim=3)


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


def test_plan_redundant_experts_never_uses_owner_or_duplicate_rank():
    expert_load = (
        torch.tensor(
            [
                [100, 90, 80, 70, 60, 50, 40, 30],
                [30, 40, 50, 60, 70, 80, 90, 100],
            ]
        )
        .unsqueeze(0)
        .unsqueeze(2)
    )
    placement = plan_redundant_experts(expert_load, num_ranks=4, num_redundant_experts_per_rank=2)

    for layer_placement in placement:
        for rank, expert_ids in enumerate(layer_placement.tolist()):
            assert len(expert_ids) == len(set(expert_ids))
            assert all(expert_id // 2 != rank for expert_id in expert_ids)


def test_plan_redundant_experts_minimizes_samplewise_aligned_critical_load():
    samples = torch.tensor([[[300, 20, 20, 200]], [[100, 300, 40, 160]]]).unsqueeze(2)
    placement = plan_redundant_experts(samples, num_ranks=2, num_redundant_experts_per_rank=1, expert_alignment=128)
    candidates = [torch.tensor([[[left], [right]]]) for left in (2, 3) for right in (0, 1)]

    def critical(candidate):
        return _estimate_rank_load(samples, candidate, expert_alignment=128).max(dim=2).values.sum()

    assert torch.equal(placement, torch.tensor([[[3], [0]]]))
    assert critical(placement) == min(critical(candidate) for candidate in candidates)


def test_select_improving_placements_rejects_regressing_layer():
    expert_load = torch.tensor([[8649, 5740, 5002, 3441]]).unsqueeze(0).unsqueeze(2)
    current = torch.tensor([[[2], [0]]])
    regressing_candidate = torch.tensor([[[1], [0]]])

    selected, improved, _metrics, _before_load, _after_load = select_improving_placements(
        expert_load, current, regressing_candidate, rebalance_gain_threshold=0.05
    )

    current_ratio = _estimate_rank_load(expert_load, current).max() / _estimate_rank_load(expert_load, current).mean()
    candidate_ratio = (
        _estimate_rank_load(expert_load, regressing_candidate).max()
        / _estimate_rank_load(expert_load, regressing_candidate).mean()
    )
    assert current_ratio.item() == pytest.approx(1.1007, abs=1e-4)
    assert candidate_ratio.item() == pytest.approx(1.1184, abs=1e-4)
    assert not improved.item()
    assert torch.equal(selected, current)


def test_select_improving_placements_rejects_near_balance_when_gain_is_below_threshold():
    expert_load = torch.tensor([[1, 2, 1, 17]]).unsqueeze(0).unsqueeze(2)
    current = torch.tensor([[[3], [0]]])
    candidate = torch.tensor([[[3], [1]]])

    selected, improved, _metrics, _before_load, _after_load = select_improving_placements(
        expert_load, current, candidate, rebalance_gain_threshold=0.05
    )

    current_ratio = _estimate_rank_load(expert_load, current).max() / _estimate_rank_load(expert_load, current).mean()
    candidate_ratio = (
        _estimate_rank_load(expert_load, candidate).max() / _estimate_rank_load(expert_load, candidate).mean()
    )
    assert current_ratio.item() == pytest.approx(1.047619, abs=1e-6)
    assert candidate_ratio.item() == pytest.approx(1.0)
    assert not improved.item()
    assert torch.equal(selected, current)


def test_select_improving_placements_accepts_alignment_aware_gain_even_when_current_ranks_are_balanced():
    expert_load = torch.tensor([[100, 129, 100, 129]]).unsqueeze(0).unsqueeze(2)
    current = torch.tensor([[[2], [0]]])
    candidate = torch.tensor([[[3], [1]]])

    selected, improved, metrics, _before_load, _after_load = select_improving_placements(
        expert_load,
        current,
        candidate,
        rebalance_gain_threshold=0.05,
        expert_alignment=128,
    )

    assert metrics["model_imbalance_ratio"] == pytest.approx(1.0)
    assert metrics["candidate_rebalance_gain"] == pytest.approx(0.25)
    assert improved.item()
    assert torch.equal(selected, candidate)


def test_select_improving_placements_rejects_insufficient_rebalance_gain():
    expert_load = torch.tensor([[1, 1, 6, 7]]).unsqueeze(0).unsqueeze(2)
    current = torch.tensor([[[2], [0]]])
    candidate = torch.tensor([[[3], [0]]])

    selected, improved, _metrics, _before_load, _after_load = select_improving_placements(
        expert_load, current, candidate, rebalance_gain_threshold=0.05
    )

    current_ratio = _estimate_rank_load(expert_load, current).max() / _estimate_rank_load(expert_load, current).mean()
    candidate_ratio = (
        _estimate_rank_load(expert_load, candidate).max() / _estimate_rank_load(expert_load, candidate).mean()
    )
    relative_improvement = (current_ratio - candidate_ratio) / current_ratio
    assert current_ratio.item() == pytest.approx(1.4)
    assert candidate_ratio.item() == pytest.approx(1.333333, abs=1e-6)
    assert relative_improvement.item() == pytest.approx(0.047619, abs=1e-6)
    assert not improved.item()
    assert torch.equal(selected, current)

    selected, improved, _metrics, _before_load, _after_load = select_improving_placements(
        expert_load,
        current,
        candidate,
        rebalance_gain_threshold=0.04,
    )

    assert improved.item()
    assert torch.equal(selected, candidate)


@pytest.mark.parametrize("rebalance_gain_threshold", [-0.01, 1.01, float("nan"), float("inf")])
def test_select_improving_placements_rejects_invalid_rebalance_gain_threshold(
    rebalance_gain_threshold,
):
    expert_load = torch.tensor([[1, 1, 1, 2]]).unsqueeze(0).unsqueeze(2)
    current = torch.tensor([[[2], [0]]])
    candidate = torch.tensor([[[3], [0]]])

    with pytest.raises(ValueError, match="rebalance_gain_threshold"):
        select_improving_placements(
            expert_load,
            current,
            candidate,
            rebalance_gain_threshold=rebalance_gain_threshold,
        )


def test_select_improving_placements_accepts_sufficient_rebalance_gain():
    expert_load = torch.tensor([[1, 1, 1, 2]]).unsqueeze(0).unsqueeze(2)
    current = torch.tensor([[[2], [0]]])
    candidate = torch.tensor([[[3], [0]]])

    selected, improved, _metrics, _before_load, _after_load = select_improving_placements(
        expert_load, current, candidate, rebalance_gain_threshold=0.05
    )

    current_ratio = _estimate_rank_load(expert_load, current).max() / _estimate_rank_load(expert_load, current).mean()
    candidate_ratio = (
        _estimate_rank_load(expert_load, candidate).max() / _estimate_rank_load(expert_load, candidate).mean()
    )
    relative_improvement = (current_ratio - candidate_ratio) / current_ratio
    assert current_ratio.item() == pytest.approx(1.2)
    assert candidate_ratio.item() == pytest.approx(1.0)
    assert relative_improvement.item() == pytest.approx(1 / 6)
    assert improved.item()
    assert torch.equal(selected, candidate)


def test_select_improving_placements_rejects_raw_improvement_that_does_not_improve_aligned_compute():
    expert_load = torch.tensor([[1, 1, 1, 8]]).unsqueeze(0).unsqueeze(2)
    current = torch.tensor([[[2], [0]]])
    raw_improving_candidate = torch.tensor([[[3], [0]]])

    _, raw_improved, _metrics, _before_load, _after_load = select_improving_placements(
        expert_load, current, raw_improving_candidate, rebalance_gain_threshold=0.05
    )
    selected, aligned_improved, _metrics, _before_load, _after_load = select_improving_placements(
        expert_load,
        current,
        raw_improving_candidate,
        rebalance_gain_threshold=0.05,
        expert_alignment=128,
    )

    assert raw_improved.item()
    assert not aligned_improved.item()
    assert torch.equal(selected, current)


def test_estimate_rank_load_aligns_each_sample_before_accumulation():
    samples = torch.tensor([[[20, 0, 0, 0]], [[20, 0, 0, 0]]]).unsqueeze(2)
    placement = torch.tensor([[[2], [0]]])

    per_sample = _estimate_rank_load(samples, placement, expert_alignment=128)
    accumulated = _estimate_rank_load(samples.sum(dim=0, keepdim=True), placement, expert_alignment=128)

    assert torch.equal(per_sample[:, 0], torch.tensor([[128.0, 128.0], [128.0, 128.0]]))
    assert torch.equal(per_sample.sum(dim=0)[0], torch.tensor([256.0, 256.0]))
    assert torch.equal(accumulated[0, 0], torch.tensor([128.0, 128.0]))


def test_select_improving_placements_rejects_lower_ratio_when_critical_is_unchanged():
    samples = torch.tensor(
        [
            [[255, 220, 226, 254]],
            [[172, 278, 51, 238]],
            [[249, 291, 284, 183]],
        ]
    ).unsqueeze(2)
    current = torch.tensor([[[2], [0]]])
    mean_inflating_candidate = torch.tensor([[[2], [1]]])

    selected, improved, _metrics, _before_load, _after_load = select_improving_placements(
        samples,
        current,
        mean_inflating_candidate,
        rebalance_gain_threshold=0.05,
        expert_alignment=128,
    )
    current_load = _estimate_rank_load(samples, current, expert_alignment=128)
    candidate_load = _estimate_rank_load(samples, mean_inflating_candidate, expert_alignment=128)
    current_critical = current_load.max(dim=2).values.sum()
    candidate_critical = candidate_load.max(dim=2).values.sum()

    assert candidate_load.mean(dim=2).sum() > current_load.mean(dim=2).sum()
    assert current_critical == candidate_critical
    assert not improved.item()
    assert torch.equal(selected, current)


def test_select_improving_placements_accepts_five_percent_critical_reduction():
    samples = torch.tensor(
        [
            [[13, 352, 348, 141]],
            [[287, 175, 236, 179]],
            [[316, 99, 266, 353]],
        ]
    ).unsqueeze(2)
    current = torch.tensor([[[2], [0]]])
    candidate = torch.tensor([[[2], [1]]])

    selected, improved, _metrics, _before_load, _after_load = select_improving_placements(
        samples,
        current,
        candidate,
        rebalance_gain_threshold=0.05,
        expert_alignment=128,
    )

    assert improved.item()
    assert torch.equal(selected, candidate)


def test_select_improving_placements_rejects_single_layer_gain_below_model_threshold():
    # Layer 0 becomes better, but layer 1 dominates model critical load.  The
    # aggregate estimated critical-load reduction gain is below 5%, so neither layer may be changed.
    expert_load = torch.tensor([[1, 1, 1, 2], [0, 0, 0, 10]]).unsqueeze(0).unsqueeze(2)
    current = torch.tensor([[[2], [0]], [[2], [0]]])
    candidate = torch.tensor([[[3], [0]], [[2], [0]]])

    selected, improved, metrics, _before_load, _after_load = select_improving_placements(
        expert_load, current, candidate, rebalance_gain_threshold=0.05
    )

    assert not torch.any(improved)
    assert torch.equal(selected, current)
    assert metrics["candidate_rebalance_gain"] == pytest.approx(0.5 / 13)
    assert metrics["candidate_changed_layer_count"] == 1


def test_select_improving_placements_accepts_only_when_model_gain_reaches_threshold():
    expert_load = torch.tensor([[1, 1, 1, 2], [0, 0, 0, 5]]).unsqueeze(0).unsqueeze(2)
    current = torch.tensor([[[2], [0]], [[2], [0]]])
    candidate = torch.tensor([[[3], [0]], [[2], [0]]])

    selected, improved, _metrics, _before_load, _after_load = select_improving_placements(
        expert_load, current, candidate, rebalance_gain_threshold=0.05
    )

    assert torch.equal(improved, torch.tensor([True, False]))
    assert torch.equal(selected, candidate)


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


def test_plan_redundant_experts_prefers_current_rank_load_relief():
    source_load = torch.zeros((1, 1, 4, 8), dtype=torch.int64)
    source_load[0, 0, 1, 0] = 1024
    placement = plan_redundant_experts(
        source_load,
        num_ranks=4,
        num_redundant_experts_per_rank=1,
        expert_alignment=128,
    )
    assert placement[0, 1, 0] == 0
    predicted = _estimate_rank_load(source_load, placement, expert_alignment=128)
    assert torch.equal(
        predicted,
        _manual_runtime_rank_load(source_load, placement, alignment=128),
    )


def test_plan_redundant_experts_accepts_aggregated_load():
    load = torch.tensor([[1000, 900, 800, 700, 600, 500, 400, 300]]).unsqueeze(0).unsqueeze(2)
    placement = plan_redundant_experts(load, num_ranks=4, num_redundant_experts_per_rank=1)
    assert placement.shape == (1, 4, 1)


def test_current_rank_estimate_matches_local_first_runtime_replica_sharing():
    placement = torch.tensor([[[4], [5], [0], [1]]], dtype=torch.int64)
    source_load = torch.zeros((1, 1, 4, 8), dtype=torch.int64)
    source_load[0, 0, 0, 0] = 256
    source_load[0, 0, 2, 0] = 128

    predicted = _estimate_rank_load(source_load, placement, expert_alignment=128)
    runtime = _manual_runtime_rank_load(source_load, placement, alignment=128)

    assert torch.equal(predicted, runtime)
    assert torch.equal(predicted[0, 0], torch.tensor([256.0, 0.0, 128.0, 0.0]))


def test_current_rank_planner_constraints_and_real_critical_improvement():
    load_by_node = torch.tensor(
        [
            [
                [
                    [697, 451, 383, 536, 349, 404, 854, 425],
                    [861, 103, 166, 612, 444, 263, 910, 392],
                ]
            ],
            [
                [
                    [944, 457, 338, 108, 63, 525, 48, 216],
                    [439, 117, 837, 550, 833, 201, 729, 5],
                ]
            ],
            [
                [
                    [749, 159, 18, 723, 12, 700, 419, 51],
                    [112, 135, 8, 840, 40, 970, 90, 683],
                ]
            ],
        ],
        dtype=torch.int64,
    )
    source_load = torch.zeros((3, 1, 4, 8), dtype=torch.int64)
    source_load[:, :, 0] = load_by_node[:, :, 0]
    source_load[:, :, 2] = load_by_node[:, :, 1]
    initial = torch.tensor([[[2], [4], [6], [0]]], dtype=torch.int64)
    planned = plan_redundant_experts(source_load, 4, 1, expert_alignment=128)

    for rank, experts in enumerate(planned[0].tolist()):
        assert len(experts) == len(set(experts)) == 1
        assert experts[0] // 2 != rank

    before = _estimate_rank_load(source_load, initial, expert_alignment=128)
    after = _estimate_rank_load(source_load, planned, expert_alignment=128)
    manual_before = _manual_runtime_rank_load(source_load, initial, 128)
    manual_after = _manual_runtime_rank_load(source_load, planned, 128)
    assert torch.equal(before, manual_before)
    assert torch.equal(after, manual_after)
    assert after.max(dim=2).values.sum() < before.max(dim=2).values.sum()


def test_current_rank_select_uses_the_same_runtime_critical_prediction():
    source_load = torch.zeros((2, 1, 4, 8), dtype=torch.int64)
    source_load[:, 0, 1, 0] = torch.tensor([1024, 768])
    source_load[:, 0, 0, 6] = torch.tensor([896, 1024])
    current = torch.tensor([[[2], [4], [6], [0]]], dtype=torch.int64)
    candidate = plan_redundant_experts(source_load, 4, 1, expert_alignment=128)
    selected, _improved, _metrics, _before_load, _after_load = select_improving_placements(
        source_load,
        current,
        candidate,
        rebalance_gain_threshold=0.05,
        expert_alignment=128,
    )
    assert torch.equal(
        _estimate_rank_load(source_load, selected, 128),
        _manual_runtime_rank_load(source_load, selected, 128),
    )


def _count_moved_slots(current: torch.Tensor, target: torch.Tensor) -> int:
    """Count rank rows gaining an expert: one migrated row per new expert id."""
    moved = 0
    for layer in range(current.shape[0]):
        for rank in range(current.shape[1]):
            moved += len(set(target[layer, rank].tolist()) - set(current[layer, rank].tolist()))
    return moved


def test_sticky_plan_reproduces_current_when_load_unchanged():
    generator = torch.Generator().manual_seed(7)
    load = torch.randint(1, 1000, (3, 16, 32), generator=generator).unsqueeze(2)
    placement = plan_redundant_experts(load, num_ranks=4, num_redundant_experts_per_rank=2)

    replanned = plan_redundant_experts(
        load,
        num_ranks=4,
        num_redundant_experts_per_rank=2,
        current_placement=placement,
        stickiness=0.1,
    )

    assert torch.equal(replanned, placement)
    for layer in range(placement.shape[0]):
        assert build_transfer_plan(placement[layer], replanned[layer], 32, 4, 4) == []


def test_sticky_plan_bounded_moves_under_small_perturbation():
    generator = torch.Generator().manual_seed(11)
    load = torch.randint(100, 1000, (4, 16, 32), generator=generator).unsqueeze(2)
    placement = plan_redundant_experts(load, num_ranks=4, num_redundant_experts_per_rank=2)
    noise = torch.rand((4, 16, 32), generator=generator).unsqueeze(2) * 0.1 + 0.95
    perturbed = (load.double() * noise).round().to(torch.int64)

    sticky = plan_redundant_experts(perturbed, 4, 2, current_placement=placement, stickiness=0.1)
    free = plan_redundant_experts(perturbed, 4, 2)

    sticky_moves = _count_moved_slots(placement, sticky)
    free_moves = _count_moved_slots(placement, free)
    assert sticky_moves <= placement.numel() // 4
    assert sticky_moves < free_moves

    def critical(candidate):
        return _estimate_rank_load(perturbed, candidate).max(dim=2).values.sum()

    assert critical(sticky) <= critical(free) * 1.1


def test_sticky_plan_still_churns_under_phase_shift():
    layers, experts = 8, 32
    before = torch.full((layers, experts), 10, dtype=torch.int64)
    after = torch.full((layers, experts), 10, dtype=torch.int64)
    offsets = torch.arange(4)
    for layer in range(layers):
        before[layer, (4 * layer + offsets) % experts] = 5000
        after[layer, (4 * layer + 16 + offsets) % experts] = 5000
    before = before.unsqueeze(0).unsqueeze(2)
    after = after.unsqueeze(0).unsqueeze(2)
    placement = plan_redundant_experts(before, num_ranks=4, num_redundant_experts_per_rank=2)

    replanned = plan_redundant_experts(
        after,
        num_ranks=4,
        num_redundant_experts_per_rank=2,
        current_placement=placement,
        stickiness=0.1,
    )

    assert _count_moved_slots(placement, replanned) > placement.numel() // 2


def test_transfer_plan_slot_permutation_is_free():
    current = torch.tensor([[4, 5], [6, 7], [0, 1], [2, 3]])
    target = torch.tensor([[5, 4], [7, 6], [1, 0], [3, 2]])

    assert torch.equal(align_target_placement(current, target), current)
    assert build_transfer_plan(current, target, num_logical_experts=8, world_size=4, node_world_size=2) == []


def test_align_target_placement_keeps_retained_experts_in_live_slots():
    current = torch.tensor([[4, 5], [6, 7], [0, 1], [2, 3]])
    target = torch.tensor([[5, 6], [7, 0], [1, 0], [3, 2]])

    canonical = align_target_placement(current, target)

    assert torch.equal(canonical, torch.tensor([[6, 5], [0, 7], [0, 1], [2, 3]]))


def test_align_target_placement_replaces_duplicate_current_replicas():
    current = torch.tensor([[1, 1], [3, 3]])
    target = torch.tensor([[1, 2], [3, 0]])

    canonical = align_target_placement(current, target)

    assert torch.equal(canonical, target)
    assert build_transfer_plan(current, target, num_logical_experts=4, world_size=2, node_world_size=2)


def test_canonical_placement_keeps_transfer_rows_and_published_map_consistent():
    num_logical_experts = 8
    world_size = 4
    num_redundant_slots_per_rank = 2
    num_experts_per_rank = num_logical_experts // world_size
    num_physical_experts_per_rank = num_experts_per_rank + num_redundant_slots_per_rank
    current = torch.tensor([[4, 5], [6, 7], [0, 1], [2, 3]])
    target = torch.tensor([[5, 6], [7, 0], [1, 0], [3, 2]])
    canonical = align_target_placement(current, target)
    plan = build_transfer_plan(current, canonical, num_logical_experts, world_size, node_world_size=2)

    # Label every current physical row by its resident logical expert, then
    # apply the transfer plan from a frozen source snapshot just as staging
    # copies do before the destination rows are published.
    source_rows = [
        list(range(rank * num_experts_per_rank, (rank + 1) * num_experts_per_rank)) + current[rank].tolist()
        for rank in range(world_size)
    ]
    live_rows = [row.copy() for row in source_rows]
    for step in plan:
        live_rows[step.dst_rank][num_experts_per_rank + step.dst_slot] = source_rows[step.src_rank][step.src_local_row]

    for current_rank in range(world_size):
        logical_to_physical = build_logical_to_physical_map(
            _rank_to_logic_expert_ids(canonical.tolist(), num_logical_experts),
            num_logical_experts,
            current_rank=current_rank,
        )
        for logical_expert, packed_row in enumerate(logical_to_physical):
            count = packed_row[0]
            for physical_id in packed_row[2 : count + 2]:
                rank, row = divmod(physical_id, num_physical_experts_per_rank)
                assert live_rows[rank][row] == logical_expert


def test_plan_and_broadcast_publishes_canonical_placement(monkeypatch):
    manager = manager_module.EPLBManager.__new__(manager_module.EPLBManager)
    manager.global_rank = 0
    manager.world_size = 4
    manager.node_world_size = 2
    manager.num_logical_experts = 8
    manager.num_redundant_experts_per_rank = 2
    manager.current_placement = torch.tensor([[[4, 5], [6, 7], [0, 1], [2, 3]]])
    manager.placement_stickiness = 0.1
    manager.rebalance_gain_threshold = 0.05
    manager.evaluation_group = object()
    candidate = torch.tensor([[[5, 4], [7, 6], [1, 0], [3, 2]]])
    broadcasts = []

    def fixed_selector(*_args, **_kwargs):
        rank_load = torch.full((1, 1, 4), 100.0)
        return candidate.clone(), torch.tensor([True]), {}, rank_load, rank_load

    def record_broadcast(result_list, **_kwargs):
        broadcasts.append(result_list[0])

    monkeypatch.setattr(
        manager_module,
        "plan_redundant_experts",
        lambda *_args, **_kwargs: candidate.clone(),
    )
    monkeypatch.setattr(manager_module, "select_improving_placements", fixed_selector)
    monkeypatch.setattr(manager_module.dist, "broadcast_object_list", record_broadcast)

    result = manager._plan_and_broadcast(torch.full((1, 1, 2, 8), 100, dtype=torch.int64))

    assert torch.equal(result["placement"], manager.current_placement)
    assert broadcasts and torch.equal(broadcasts[0]["placement"], manager.current_placement)


def test_stickiness_zero_matches_unbiased_plan():
    generator = torch.Generator().manual_seed(17)
    load = torch.randint(1, 1000, (2, 8, 16), generator=generator).unsqueeze(2)
    unbiased = plan_redundant_experts(load, num_ranks=4, num_redundant_experts_per_rank=2)
    unrelated = _initial_extra_expert_placement(16, 4, 2).unsqueeze(0).expand(8, -1, -1).clone()

    replanned = plan_redundant_experts(
        load,
        num_ranks=4,
        num_redundant_experts_per_rank=2,
        current_placement=unrelated,
        stickiness=0.0,
    )

    assert torch.equal(replanned, unbiased)


def test_plan_and_broadcast_propagates_rank_zero_error_after_existing_broadcast(
    monkeypatch,
):
    manager = manager_module.EPLBManager.__new__(manager_module.EPLBManager)
    manager.global_rank = 0
    manager.world_size = 2
    manager.node_world_size = 2
    manager.num_logical_experts = 8
    manager.num_redundant_experts_per_rank = 2
    manager.current_placement = torch.tensor([[[4, 5], [6, 7], [0, 1], [2, 3]]])
    manager.placement_stickiness = 0.1
    manager.rebalance_gain_threshold = 0.05
    manager.evaluation_group = object()
    broadcasted = []

    def broken_planner(*_args, **_kwargs):
        raise RuntimeError("planner boom")

    def record_broadcast(result_list, **_kwargs):
        broadcasted.append(result_list[0])

    monkeypatch.setattr(manager_module, "plan_redundant_experts", broken_planner)
    monkeypatch.setattr(manager_module.dist, "broadcast_object_list", record_broadcast)

    with pytest.raises(RuntimeError, match="EPLB planner failed on rank zero") as exc_info:
        manager._plan_and_broadcast(torch.full((1, 1, 1, 8), 100, dtype=torch.int64))

    assert isinstance(exc_info.value.__cause__, RuntimeError)
    assert str(exc_info.value.__cause__) == "planner boom"
    assert broadcasted == [{"kind": "error", "message": "RuntimeError: planner boom"}]


def test_steady_state_sparse_sampling_records_steps_sixteen_to_nineteen_and_evaluates_step_twenty(
    monkeypatch,
):
    manager = manager_module.EPLBManager.__new__(manager_module.EPLBManager)
    counter = torch.tensor([10, 20, 30, 40], dtype=torch.int64)
    manager.weights = [
        type(
            "Weight",
            (),
            {
                "fuse_moe_impl": _test_moe_impl(
                    eplb=True,
                    route_counter=counter,
                    recording=False,
                    num_logical_experts=4,
                    world_size=1,
                ),
            },
        )()
    ]
    manager.in_flight = False
    manager.prefill_steps = 15
    manager.step_interval = 20
    manager.sampling_interval = manager.step_interval
    manager.evaluation_group = object()
    manager.num_logical_experts = 4
    manager.global_rank = 1
    manager.evaluation_in_flight = False
    manager._steady_collection_end_step = None
    manager._continuous_collection_start_step = None
    manager._continuous_collection_end_step = None

    recordings, resets, started = [], [], []
    manager._set_recording = lambda enabled: recordings.append(enabled)
    manager._reset_route_counters = lambda: resets.append(True)
    monkeypatch.setattr(manager, "_start_evaluation", lambda: started.append(True))
    monkeypatch.setattr(
        manager_module.torch.cuda,
        "synchronize",
        lambda: pytest.fail("step must not synchronize CUDA"),
    )

    manager.step()
    assert manager.prefill_steps == 16
    assert recordings == [True]
    assert resets == [True]
    assert manager._steady_collection_end_step == 20

    for _ in range(3):
        manager.step()
    assert manager.prefill_steps == 19
    assert recordings == [True]
    assert started == []

    manager.step()
    assert manager.prefill_steps == 20
    assert manager._steady_collection_end_step is None
    assert started == [True]


def test_steady_sampling_window_clamps_to_short_interval_without_moving_boundary(
    monkeypatch,
):
    manager = manager_module.EPLBManager.__new__(manager_module.EPLBManager)
    manager.in_flight = False
    manager.evaluation_in_flight = False
    manager.prefill_steps = 0
    manager.step_interval = 20
    manager.sampling_interval = 3
    manager._steady_collection_end_step = None
    manager._continuous_collection_start_step = None
    manager._continuous_collection_end_step = None
    recordings, resets, started = [], [], []
    manager._set_recording = lambda enabled: recordings.append((manager.prefill_steps, enabled))
    manager._reset_route_counters = lambda: resets.append(manager.prefill_steps)
    monkeypatch.setattr(manager, "_start_evaluation", lambda: started.append(manager.prefill_steps))

    manager._prepare_next_sampling_window()
    assert resets == [0]
    assert recordings == [(0, True)]
    assert manager._steady_collection_end_step == 3

    manager.step()
    manager.step()
    assert started == []
    manager.step()
    assert started == [3]


def test_eplb_step_does_not_start_a_second_evaluation_while_one_is_pending(monkeypatch):
    manager = manager_module.EPLBManager.__new__(manager_module.EPLBManager)
    counter = torch.tensor([10, 20, 30, 40], dtype=torch.int64)
    manager.weights = [
        type(
            "Weight",
            (),
            {
                "fuse_moe_impl": _test_moe_impl(
                    eplb=True,
                    route_counter=counter,
                    recording=False,
                    num_logical_experts=4,
                    world_size=1,
                ),
            },
        )()
    ]
    manager.in_flight = False
    manager.prefill_steps = 1
    manager.step_interval = 2
    manager.sampling_interval = manager.step_interval
    manager.evaluation_group = object()
    manager.num_logical_experts = 4
    manager.global_rank = 1
    manager.evaluation_in_flight = True

    started = []
    monkeypatch.setattr(manager, "_poll_evaluation", lambda: True)
    monkeypatch.setattr(manager, "_start_evaluation", lambda: started.append(True))

    manager.step()

    assert manager.prefill_steps == 1
    assert started == []


def test_evaluation_no_improvement_logs_model_fields_without_reopening_interval_window(
    monkeypatch,
):
    class DoneThread:
        def join(self):
            pass

    manager = manager_module.EPLBManager.__new__(manager_module.EPLBManager)
    manager.evaluation_in_flight = True
    manager._evaluation_lock = threading.Lock()
    manager._evaluation_error = None
    manager._evaluation_thread = DoneThread()
    manager._evaluation_result = {
        "kind": "no_improvement",
        "model_imbalance_ratio": 1.2,
        "candidate_model_imbalance_ratio": 1.1,
        "candidate_rebalance_gain": 0.01,
        "candidate_changed_layer_count": 2,
    }
    manager.global_rank = 0
    manager.step_interval = 20
    manager.sampling_interval = 20
    manager.weights = []
    manager._eplb_impls = []
    recordings, logs = [], []
    manager._set_recording = lambda enabled: recordings.append(enabled)
    monkeypatch.setattr(manager_module.logger, "info", lambda *args: logs.append(args))

    assert not manager._poll_evaluation()
    assert recordings == [False]
    assert "model_imbalance_ratio" in logs[0][0]
    assert "candidate_rebalance_gain" in logs[0][0]
    assert "candidate_changed_layer_count" in logs[0][0]
    assert "actual_changed_layer_count" in logs[0][0]
    assert "next_sampling_interval" in logs[0][0]
    assert manager.sampling_interval == 80


def test_interval_one_rearms_after_evaluation_but_never_evaluates_empty_counter(
    monkeypatch,
):
    class DoneThread:
        def join(self):
            pass

    manager = manager_module.EPLBManager.__new__(manager_module.EPLBManager)
    manager.in_flight = False
    manager.prefill_steps = 1
    manager.step_interval = 1
    manager.sampling_interval = 1
    manager._steady_collection_end_step = None
    manager.evaluation_in_flight = True
    manager._evaluation_lock = threading.Lock()
    manager._evaluation_error = None
    manager._evaluation_thread = DoneThread()
    manager._evaluation_result = {
        "kind": "no_improvement",
        "model_imbalance_ratio": 1.2,
        "candidate_model_imbalance_ratio": 1.1,
        "candidate_rebalance_gain": 0.01,
        "candidate_changed_layer_count": 2,
    }
    manager.global_rank = 1
    manager.weights = []
    manager._eplb_impls = []
    recordings, starts = [], []
    manager._set_recording = lambda enabled: recordings.append(enabled)
    manager._start_evaluation = lambda: starts.append(True)
    manager._evaluation_ready_on_all_ranks = lambda: True

    manager.poll()

    # The no-improvement backoff changes interval 1 to 4.  The clamped
    # steady window arms immediately but still waits for boundary step 5.
    assert recordings == [True]
    assert manager._steady_collection_end_step is not None
    assert manager._steady_collection_end_step == 5
    assert starts == []
    assert manager.prefill_steps == 1

    for _ in range(3):
        manager.step()
    assert starts == []
    manager.step()
    assert starts == [True]


def test_evaluation_worker_error_is_raised_by_main_thread():
    class DoneThread:
        def join(self):
            pass

    manager = manager_module.EPLBManager.__new__(manager_module.EPLBManager)
    manager.evaluation_in_flight = True
    manager._evaluation_lock = threading.Lock()
    manager._evaluation_error = RuntimeError("planner failed")
    manager._evaluation_result = None
    manager._evaluation_thread = DoneThread()

    with pytest.raises(RuntimeError, match="planner failed"):
        manager._poll_evaluation()


def test_evaluation_state_is_cleared_before_second_round(monkeypatch):
    class DoneThread:
        def join(self):
            pass

    class PendingThread:
        def __init__(self, **_kwargs):
            self.started = False

        def start(self):
            self.started = True

        def join(self):
            pytest.fail("pending worker must not be joined")

    class Event:
        def record(self, _stream):
            pass

    manager = manager_module.EPLBManager.__new__(manager_module.EPLBManager)
    manager.evaluation_in_flight = True
    manager._evaluation_lock = threading.Lock()
    manager._evaluation_error = None
    manager._evaluation_thread = DoneThread()
    manager._evaluation_result = {
        "kind": "no_improvement",
        "model_imbalance_ratio": 1.2,
        "candidate_model_imbalance_ratio": 1.1,
        "candidate_rebalance_gain": 0.01,
        "candidate_changed_layer_count": 1,
    }
    manager.global_rank = 1
    manager.prefill_steps = 0
    manager.step_interval = 1
    manager.sampling_interval = 1
    manager.weights = []
    manager._eplb_impls = []
    manager._set_recording = lambda _enabled: None
    monkeypatch.setattr(manager_module.threading, "Thread", PendingThread)
    monkeypatch.setattr(manager_module.torch.cuda, "Event", Event)
    monkeypatch.setattr(manager_module.torch.cuda, "current_stream", lambda: object())

    assert not manager._poll_evaluation()
    assert manager._evaluation_result is None
    assert manager._evaluation_error is None

    manager._start_evaluation()
    assert manager.evaluation_in_flight
    assert manager._evaluation_result is None
    assert manager._poll_evaluation()  # New worker has not produced a result.


def test_manager_collects_aggregated_route_counters():
    counters = [
        torch.tensor([10, 11], dtype=torch.int64),
        torch.tensor([40, 41], dtype=torch.int64),
    ]
    manager = manager_module.EPLBManager.__new__(manager_module.EPLBManager)
    manager.weights = [
        type(
            "Weight",
            (),
            {
                "fuse_moe_impl": _test_moe_impl(
                    eplb=True,
                    route_counter=counter,
                    recording=False,
                    num_logical_experts=2,
                    world_size=1,
                ),
            },
        )()
        for counter in counters
    ]
    manager._eplb_impls = [weight.fuse_moe_impl for weight in manager.weights]
    manager.num_logical_experts = 2

    samples = manager._collect_local_samples()

    assert torch.equal(
        samples,
        torch.tensor([[[10, 11], [40, 41]]], dtype=torch.int64),
    )


def test_expert_load_imbalance_ratio_averages_layer_ratios():
    global_load = torch.tensor(
        [
            [
                [[1, 2, 3], [1, 2, 3]],
                [[4, 5, 6], [6, 5, 4]],
            ]
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

    manager._publish_expert_load_metrics(
        {
            "expert_imbalance_ratio": 1.25,
        }
    )

    assert calls == [
        (manager_module.EPLB_EXPERT_IMBALANCE_RATIO_METRIC, 1.25),
    ]


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


def test_steady_sampling_resets_aggregated_route_counter():
    counter = torch.ones((4,), dtype=torch.int64)
    manager = manager_module.EPLBManager.__new__(manager_module.EPLBManager)
    impl = _test_moe_impl(
        eplb=True,
        route_counter=counter,
        recording=False,
        num_logical_experts=4,
        world_size=1,
    )
    manager._eplb_impls = [impl]

    manager._reset_route_counters()
    manager._reset_route_counters()

    assert impl.route_counter.shape == (4,)
    assert torch.count_nonzero(impl.route_counter) == 0


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


def test_manager_evaluation_collective_preserves_current_rank_axis(monkeypatch):
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
    manager.node_world_size = 2
    manager.step_interval = 20
    manager.sampling_interval = 20
    manager.num_logical_experts = 4
    manager.num_redundant_experts_per_rank = 1
    manager.current_placement = _initial_extra_expert_placement(4, 4, 1).unsqueeze(0)
    manager.evaluation_group = object()
    manager._evaluation_lock = threading.Lock()
    manager._evaluation_result = None
    manager._evaluation_error = None
    manager._continuous_collection_end_step = None
    local = torch.full((1, 1, 4), 100, dtype=torch.int64)
    manager._collect_local_samples = lambda: local
    seen = {}

    def all_reduce(tensor, **kwargs):
        seen["before"] = tensor.clone()
        seen["group"] = kwargs["group"]
        # Simulate current rank 0's contribution from another process.
        tensor[:, :, 0].fill_(100)

    monkeypatch.setattr(manager_module.dist, "all_reduce", all_reduce)
    monkeypatch.setattr(manager_module.torch.cuda, "set_device", lambda _device: None)

    def plan_and_broadcast(global_load):
        seen["global_load"] = global_load.clone()
        return {"kind": "insufficient"}

    manager._plan_and_broadcast = plan_and_broadcast

    manager._evaluate_after_event(type("Event", (), {"synchronize": lambda self: None})())

    assert seen["group"] is manager.evaluation_group
    expected_local = local
    assert seen["before"].shape == (1, 1, 4, 4)
    assert torch.equal(seen["before"][:, :, 0], torch.zeros_like(expected_local))
    assert torch.equal(seen["before"][:, :, 1], torch.zeros_like(expected_local))
    assert torch.equal(seen["before"][:, :, 2], expected_local)
    assert torch.equal(seen["before"][:, :, 3], torch.zeros_like(expected_local))
    assert torch.equal(seen["global_load"][:, :, 0], torch.full_like(expected_local, 100))
    assert torch.equal(seen["global_load"][:, :, 1], torch.zeros_like(expected_local))
    assert torch.equal(seen["global_load"][:, :, 2], expected_local)
    assert torch.equal(seen["global_load"][:, :, 3], torch.zeros_like(expected_local))
    assert manager._evaluation_error is None
    assert manager._evaluation_result["expert_imbalance_ratio"] == 1.0
    assert manager._evaluation_result["sample_window_steps"] == 4


def test_manager_planned_evaluation_builds_improved_metadata_in_one_multilayer_call(
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
    manager.global_rank = 1
    manager.world_size = 4
    manager.node_world_size = 2
    manager.step_interval = 20
    manager.sampling_interval = 20
    manager.num_logical_experts = 4
    manager.num_redundant_experts_per_rank = 1
    manager.current_placement = _initial_extra_expert_placement(4, 4, 1).unsqueeze(0).expand(3, -1, -1).clone()
    manager.evaluation_group = object()
    manager._evaluation_lock = threading.Lock()
    manager._evaluation_result = None
    manager._evaluation_error = None
    manager._continuous_collection_end_step = None
    manager._collect_local_samples = lambda: torch.full((1, 3, 4), 100, dtype=torch.int64)
    planned_placement = torch.tensor(
        [
            [[3], [0], [1], [2]],
            [[2], [3], [0], [1]],
            [[1], [2], [3], [0]],
        ],
        dtype=torch.int64,
    )
    manager._plan_and_broadcast = lambda _global_load: {
        "kind": "planned",
        "placement": planned_placement,
        "improved": torch.tensor([True, False, True]),
    }
    monkeypatch.setattr(manager_module.dist, "all_reduce", lambda _tensor, **_kwargs: None)
    monkeypatch.setattr(manager_module.torch.cuda, "set_device", lambda _device: None)
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

    manager._evaluate_after_event(type("Event", (), {"synchronize": lambda self: None})())

    assert manager._evaluation_error is None
    assert calls == [(2, 4, 2)]
    metadata = manager._evaluation_result["metadata"]
    assert metadata[1] is None
    assert [layer_index for layer_index, _plan in manager._evaluation_result["layer_plans"]] == [0, 2]
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


def test_transfer_plan_uses_existing_rows_and_prefers_local_node_replicas():
    current = torch.tensor([[4, 5], [6, 7], [0, 1], [2, 3]])
    target = current.clone()
    target[0, 0] = 6  # primary r3, but r1 replica is on r0's node.
    target[2, 1] = 4  # primary r2 is local to destination r2.
    plan = build_transfer_plan(current, target, num_logical_experts=8, world_size=4, node_world_size=2)
    by_dst = {(step.dst_rank, step.dst_slot): step for step in plan}
    assert len(by_dst) == 2
    assert by_dst[0, 0] == TransferStep(0, 0, 1, 2)
    assert by_dst[2, 1] == TransferStep(2, 1, 2, 0)


def test_transfer_plan_cross_node_and_stable_source_load_tie_break():
    current = torch.tensor([[0, 1], [2, 3], [4, 5], [4, 7]])
    target = current.clone()
    target[0, 0] = 4
    target[0, 1] = 4
    first = build_transfer_plan(current, target, 8, 4, 2)
    second = build_transfer_plan(current, target, 8, 4, 2)
    assert first == second
    selected = [step for step in first if step.dst_rank == 0]
    assert [(step.src_rank, step.src_local_row) for step in selected] == [
        (2, 0),
        (3, 2),
    ]


def test_extract_expert_tensors_includes_weight_and_scale_in_order():
    class Pack:
        def __init__(self, offset, scale=True):
            self.weight = torch.full((3, 2), offset)
            self.weight_scale = torch.full((3, 1), offset + 1) if scale else None

    weight = type("Weight", (), {"w13": Pack(1), "w2": Pack(10, scale=False)})()
    tensors = extract_eplb_expert_tensors(weight)
    assert [name for name, _ in tensors] == [
        "w13.weight",
        "w13.weight_scale",
        "w2.weight",
    ]


def test_commit_staging_rows_only_overwrites_redundant_rows():
    live = torch.arange(20).reshape(5, 4)
    staging = torch.full((2, 4), -1)
    _commit_staging_rows(
        live,
        staging,
        num_experts_per_rank=3,
        changed_dst_slots=(0, 1),
    )
    assert torch.equal(live[:3], torch.arange(12).reshape(3, 4))
    assert torch.equal(live[3:], staging)


def test_commit_staging_rows_preserves_unchanged_destination_slots():
    live = torch.arange(28).reshape(7, 4)
    staging = torch.tensor([[-1, -1, -1, -1], [-2, -2, -2, -2], [-3, -3, -3, -3], [-4, -4, -4, -4]])
    original = live.clone()

    _commit_staging_rows(
        live,
        staging,
        num_experts_per_rank=3,
        changed_dst_slots=(3, 1),
    )

    assert torch.equal(live[:4], original[:4])
    assert torch.equal(live[4], staging[1])
    assert torch.equal(live[5], original[5])
    assert torch.equal(live[6], staging[3])


def test_commit_staging_rows_merges_contiguous_changed_slots():
    copies = []

    class View:
        def __init__(self, owner, start, length):
            self.owner = owner
            self.start = start
            self.length = length

        def copy_(self, source, **_kwargs):
            copies.append(
                (
                    self.owner,
                    self.start,
                    self.length,
                    source.owner,
                    source.start,
                    source.length,
                )
            )

    class Tensor:
        def __init__(self, owner, rows):
            self.owner = owner
            self.shape = (rows,)

        def narrow(self, _dim, start, length):
            return View(self.owner, start, length)

    _commit_staging_rows(
        Tensor("live", 20),
        Tensor("staging", 4),
        num_experts_per_rank=10,
        changed_dst_slots=(3, 1, 2),
    )

    assert copies == [("live", 11, 3, "staging", 1, 3)]


def test_manager_inflight_ready_gate_commits_one_layer_and_propagates_worker_error(monkeypatch):
    class Transfer:
        def __init__(self):
            self.ready = 0
            self.commits = []
            self.finished = 0

        def ready_layer(self):
            return self.ready

        def commit(self, layer, post_copy=None):
            assert self.ready == layer
            self.ready = None
            self.commits.append(layer)
            if post_copy is not None:
                post_copy()

        def finish(self):
            self.finished += 1

    manager = manager_module.EPLBManager.__new__(manager_module.EPLBManager)
    manager.transfer = Transfer()
    manager.control_group = object()
    manager._control_ready_count = torch.empty(1, dtype=torch.int32)
    manager.in_flight_layers = [0, 1]
    committed, finished = [], []
    manager._commit_layer_metadata = committed.append
    manager._finish_rebalance = lambda: finished.append(True)
    operations = []

    class CurrentStream:
        def wait_stream(self, stream):
            operations.append(("wait", stream))

    overlap_stream = object()
    monkeypatch.setattr(manager_module.torch.cuda, "current_stream", lambda: CurrentStream())
    monkeypatch.setattr(g_infer_context, "get_overlap_stream", lambda: overlap_stream)

    def set_global_ready(count):
        return lambda tensor, **_kwargs: tensor.fill_(count)

    monkeypatch.setattr(manager_module.dist, "all_reduce", set_global_ready(0))
    manager._poll_in_flight()
    assert manager.transfer.commits == []

    monkeypatch.setattr(manager_module.dist, "all_reduce", set_global_ready(1))
    manager._poll_in_flight()
    assert manager.transfer.commits == [0]
    assert committed == [0]
    assert operations == [("wait", overlap_stream)]
    assert not finished

    manager.transfer.ready = 1
    manager._poll_in_flight()
    assert manager.transfer.commits == [0, 1]
    assert committed == [0, 1]
    assert finished == [True]
    assert manager.transfer.finished == 1

    manager.in_flight_layers = [2]
    manager.transfer.ready = 9
    with pytest.raises(RuntimeError, match="does not match expected"):
        manager._poll_in_flight()

    class BrokenTransfer:
        def ready_layer(self):
            raise RuntimeError("boom")

    manager.transfer = BrokenTransfer()
    encoded_statuses = []

    def retain_local_error(tensor, **_kwargs):
        encoded_statuses.append(int(tensor.item()))

    monkeypatch.setattr(manager_module.dist, "all_reduce", retain_local_error)
    with pytest.raises(RuntimeError, match="EPLB transfer worker failed on this rank") as exc_info:
        manager._poll_in_flight()
    assert isinstance(exc_info.value.__cause__, RuntimeError)
    assert str(exc_info.value.__cause__) == "boom"
    assert encoded_statuses == [manager_module.EPLB_CONTROL_ERROR]


def test_manager_inflight_remote_worker_error_does_not_commit(monkeypatch):
    class Transfer:
        def __init__(self):
            self.commits = []

        def ready_layer(self):
            return 0

        def commit(self, *args):
            self.commits.append(args)

    manager = manager_module.EPLBManager.__new__(manager_module.EPLBManager)
    manager.transfer = Transfer()
    manager.control_group = object()
    manager._control_ready_count = torch.empty(1, dtype=torch.int32)
    manager.in_flight_layers = [0]
    manager._commit_layer_metadata = lambda _layer: None
    manager._finish_rebalance = lambda: None
    statuses = []

    def remote_error(tensor, **_kwargs):
        statuses.append(int(tensor.item()))
        tensor.fill_(manager_module.EPLB_CONTROL_ERROR)

    monkeypatch.setattr(manager_module.dist, "all_reduce", remote_error)

    with pytest.raises(RuntimeError, match="EPLB transfer worker failed on another rank"):
        manager._poll_in_flight()
    assert statuses == [1]
    assert manager.transfer.commits == []


def test_evaluation_ready_gate_propagates_local_and_remote_errors(monkeypatch):
    manager = manager_module.EPLBManager.__new__(manager_module.EPLBManager)
    manager._evaluation_lock = threading.Lock()
    manager.control_group = object()
    manager._control_ready_count = torch.empty(1, dtype=torch.int32)
    manager._evaluation_error = RuntimeError("evaluation boom")
    manager._evaluation_result = None
    statuses = []

    def retain_local_error(tensor, **_kwargs):
        statuses.append(int(tensor.item()))

    monkeypatch.setattr(manager_module.dist, "all_reduce", retain_local_error)
    with pytest.raises(RuntimeError, match="EPLB evaluation failed on this rank") as exc_info:
        manager._evaluation_ready_on_all_ranks()
    assert isinstance(exc_info.value.__cause__, RuntimeError)
    assert str(exc_info.value.__cause__) == "evaluation boom"
    assert statuses == [manager_module.EPLB_CONTROL_ERROR]

    manager._evaluation_error = None
    manager._evaluation_result = {"kind": "no_improvement"}
    statuses.clear()

    def remote_error(tensor, **_kwargs):
        statuses.append(int(tensor.item()))
        tensor.fill_(manager_module.EPLB_CONTROL_ERROR)

    monkeypatch.setattr(manager_module.dist, "all_reduce", remote_error)
    with pytest.raises(RuntimeError, match="EPLB evaluation failed on another rank"):
        manager._evaluation_ready_on_all_ranks()
    assert statuses == [1]


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_manager_inflight_commit_orders_live_weights_between_overlap_forwards(
    monkeypatch,
):
    class Transfer:
        def __init__(self, live, staging):
            self.live = live
            self.staging = staging
            self.ready = 0

        def ready_layer(self):
            return self.ready

        def commit(self, layer, post_copy=None):
            assert self.ready == layer
            self.ready = None
            self.live.copy_(self.staging, non_blocking=True)
            if post_copy is not None:
                post_copy()

        def finish(self):
            pass

    live = torch.tensor([1.0], device="cuda")
    staging = torch.tensor([2.0], device="cuda")
    previous_read = torch.empty_like(live)
    next_read = torch.empty_like(live)
    source_stream = torch.cuda.Stream(device=live.device)
    destination_stream = torch.cuda.Stream(device=live.device)
    initial_stream = torch.cuda.current_stream(device=live.device)
    original_overlap_stream = g_infer_context.overlap_stream

    manager = manager_module.EPLBManager.__new__(manager_module.EPLBManager)
    manager.transfer = Transfer(live, staging)
    manager.control_group = object()
    manager._control_ready_count = torch.empty(1, dtype=torch.int32)
    manager.in_flight_layers = [0]
    manager._commit_layer_metadata = lambda _layer: None
    manager._finish_rebalance = lambda: None
    monkeypatch.setattr(manager_module.dist, "all_reduce", lambda tensor, **_kwargs: tensor.fill_(1))

    try:
        g_infer_context.overlap_stream = source_stream
        with torch.cuda.stream(source_stream):
            source_stream.wait_stream(initial_stream)
            torch.cuda._sleep(20_000_000)
            previous_read.copy_(live, non_blocking=True)
        with torch.cuda.stream(destination_stream):
            manager._poll_in_flight()
        with torch.cuda.stream(source_stream):
            source_stream.wait_stream(destination_stream)
            next_read.copy_(live, non_blocking=True)
        source_stream.synchronize()

        assert previous_read.item() == 1.0
        assert next_read.item() == 2.0
    finally:
        g_infer_context.overlap_stream = original_overlap_stream


def test_manager_rearms_after_rebalance_for_interval_one():
    recording_calls = []
    manager = manager_module.EPLBManager.__new__(manager_module.EPLBManager)
    manager.step_interval = 1
    manager.sampling_interval = 1
    manager._steady_collection_end_step = None
    manager._continuous_collection_start_step = 0
    manager.weights = []
    manager._eplb_impls = []
    manager.target_placement = torch.zeros(1)
    manager.in_flight_started_at = 0
    manager.global_rank = 1
    manager._set_recording = lambda enabled: recording_calls.append(enabled)
    manager._finish_rebalance()
    assert manager.in_flight is False
    assert recording_calls == [True]
    assert manager._steady_collection_end_step is None
    assert manager._continuous_collection_start_step is None


def test_manager_sparse_insufficient_schedules_bounded_fresh_window(monkeypatch):
    class DoneThread:
        def join(self):
            pass

    manager = manager_module.EPLBManager.__new__(manager_module.EPLBManager)
    manager.evaluation_in_flight = True
    manager._evaluation_lock = threading.Lock()
    manager._evaluation_error = None
    manager._evaluation_thread = DoneThread()
    manager._evaluation_result = {
        "kind": "insufficient",
        "minimum_layer_samples": 1,
        "minimum": 2,
    }
    manager.global_rank = 0
    manager.step_interval = 20
    manager.sampling_interval = 20
    manager.prefill_steps = 37
    manager.weights = []
    manager._eplb_impls = []
    manager._continuous_collection_start_step = None
    manager._continuous_collection_end_step = None
    recordings, resets, logs = [], [], []
    manager._set_recording = lambda enabled: recordings.append(enabled)
    manager._reset_route_counters = lambda: resets.append(True)
    monkeypatch.setattr(manager_module.logger, "info", lambda *args: logs.append(args))

    assert not manager._poll_evaluation()
    assert not hasattr(manager, "_retained_local_samples")
    assert manager._continuous_collection_start_step == 40
    assert manager._continuous_collection_end_step == 60
    assert recordings == [False]
    assert resets == [True]
    assert manager.sampling_interval == 20
    assert "insufficient samples" in logs[0][0]
    assert "scheduled_fresh_window" in logs[0][0]

    manager.in_flight = False
    manager.evaluation_in_flight = False
    starts = []
    monkeypatch.setattr(manager, "_start_evaluation", lambda: starts.append(manager.prefill_steps))
    manager.step()
    manager.step()
    assert manager.prefill_steps == 39
    assert starts == []
    manager.step()
    assert manager.prefill_steps == 40
    assert starts == []
    for _ in range(19):
        manager.step()
    assert manager.prefill_steps == 59
    assert starts == []
    manager.step()
    assert starts == [60]


def test_manager_full_window_insufficient_clears_and_backs_off():
    class DoneThread:
        def join(self):
            pass

    manager = manager_module.EPLBManager.__new__(manager_module.EPLBManager)
    manager.evaluation_in_flight = True
    manager._evaluation_lock = threading.Lock()
    manager._evaluation_error = None
    manager._evaluation_thread = DoneThread()
    manager._evaluation_result = {
        "kind": "insufficient",
        "minimum_layer_samples": 1,
        "minimum": 2,
    }
    manager.global_rank = 1
    manager.step_interval = 20
    manager.sampling_interval = 20
    manager.prefill_steps = 60
    manager.weights = []
    manager._eplb_impls = []
    manager._continuous_collection_start_step = 40
    manager._continuous_collection_end_step = 60
    recordings, resets = [], []
    manager._set_recording = lambda enabled: recordings.append(enabled)
    manager._reset_route_counters = lambda: resets.append(True)

    assert not manager._poll_evaluation()
    assert not hasattr(manager, "_retained_local_samples")
    assert manager._continuous_collection_start_step is None
    assert manager._continuous_collection_end_step is None
    assert manager.sampling_interval == 80
    assert recordings == [False]
    assert resets == [True]


def test_begin_continuous_collection_uses_full_window_at_fixed_boundary(monkeypatch):
    manager = manager_module.EPLBManager.__new__(manager_module.EPLBManager)
    manager.in_flight = False
    manager.evaluation_in_flight = False
    manager.prefill_steps = 36
    manager.step_interval = 20
    manager.sampling_interval = 20
    manager._steady_collection_end_step = manager.prefill_steps + 1
    recordings, resets, starts = [], [], []
    manager._set_recording = lambda enabled: recordings.append(enabled)
    manager._reset_route_counters = lambda: resets.append(True)
    monkeypatch.setattr(manager, "_start_evaluation", lambda: starts.append(True))

    # The window is never truncated to the next boundary: it waits until 40,
    # records a full 20 fresh steps, then evaluates at the boundary at 60.
    manager._begin_continuous_collection()
    assert manager._continuous_collection_start_step == 40
    assert manager._continuous_collection_end_step == 60
    assert recordings == [False]
    assert resets == [True]
    assert manager._steady_collection_end_step is None

    for _ in range(4):
        manager.step()
    assert manager.prefill_steps == 40
    assert recordings == [False, True]
    assert starts == []
    for _ in range(19):
        manager.step()
    assert starts == []
    manager.step()  # 60: the full window ends and triggers the evaluation.
    assert starts == [True]


def test_begin_continuous_collection_preserves_full_window_at_sparse_boundary():
    manager = manager_module.EPLBManager.__new__(manager_module.EPLBManager)
    manager.prefill_steps = 80
    manager.step_interval = 20
    manager.sampling_interval = 80
    manager._steady_collection_end_step = None
    recordings = []
    manager._set_recording = lambda enabled: recordings.append(enabled)
    manager._reset_route_counters = lambda: None

    manager._begin_continuous_collection()
    assert manager._continuous_collection_start_step == 140
    assert manager._continuous_collection_end_step == 160
    assert recordings == [False]


def test_first_no_improvement_switches_to_sparse_sampling_window(monkeypatch):
    class DoneThread:
        def join(self):
            pass

    manager = manager_module.EPLBManager.__new__(manager_module.EPLBManager)
    manager.evaluation_in_flight = True
    manager._evaluation_lock = threading.Lock()
    manager._evaluation_error = None
    manager._evaluation_thread = DoneThread()
    manager._evaluation_result = {
        "kind": "no_improvement",
        "model_imbalance_ratio": 1.2,
        "candidate_model_imbalance_ratio": 1.1,
        "candidate_rebalance_gain": 0.01,
        "candidate_changed_layer_count": 2,
    }
    manager.global_rank = 1
    manager.step_interval = 20
    manager.sampling_interval = 20
    manager._continuous_collection_start_step = 0
    manager.weights = []
    manager._eplb_impls = []
    recordings = []
    manager._set_recording = lambda enabled: recordings.append(enabled)

    assert not manager._poll_evaluation()
    assert manager._continuous_collection_start_step is None
    assert recordings == [False]
    assert manager.sampling_interval == 80


def test_continuous_collection_evaluates_only_after_one_full_base_window(monkeypatch):
    manager = manager_module.EPLBManager.__new__(manager_module.EPLBManager)
    manager.in_flight = False
    manager._continuous_collection_start_step = 0
    manager._continuous_collection_end_step = 20
    manager.prefill_steps = 0
    manager.step_interval = 20
    manager.sampling_interval = 320
    manager._steady_collection_end_step = None
    manager.evaluation_in_flight = False
    started = []
    monkeypatch.setattr(manager, "_start_evaluation", lambda: started.append(True))

    for _ in range(19):
        manager.step()
    assert manager.prefill_steps == 19
    assert started == []

    manager.step()
    assert manager.prefill_steps == 20
    assert started == [True]


def test_no_improvement_exponentially_backs_off_sampling_interval_at_cap():
    class DoneThread:
        def join(self):
            pass

    manager = manager_module.EPLBManager.__new__(manager_module.EPLBManager)
    manager._evaluation_lock = threading.Lock()
    manager._set_recording = lambda _enabled: None
    manager.global_rank = 1
    manager.step_interval = 20
    manager.sampling_interval = 20
    manager.weights = []
    manager._eplb_impls = []

    for expected_interval in (80, 320, 320):
        manager.evaluation_in_flight = True
        manager._evaluation_error = None
        manager._evaluation_thread = DoneThread()
        manager._evaluation_result = {
            "kind": "no_improvement",
            "model_imbalance_ratio": 1.2,
            "candidate_model_imbalance_ratio": 1.1,
            "candidate_rebalance_gain": 0.01,
            "candidate_changed_layer_count": 2,
        }
        assert not manager._poll_evaluation()
        assert manager.sampling_interval == expected_interval


def test_sparse_backoff_arms_and_evaluates_only_at_new_interval_boundary(monkeypatch):
    manager = manager_module.EPLBManager.__new__(manager_module.EPLBManager)
    manager.in_flight = False
    manager.prefill_steps = 18
    manager.step_interval = 20
    manager.sampling_interval = 80
    manager._steady_collection_end_step = None
    manager._continuous_collection_start_step = None
    manager._continuous_collection_end_step = None
    manager.evaluation_in_flight = False
    recordings, resets, starts = [], [], []
    manager._set_recording = lambda enabled: recordings.append(enabled)
    manager._reset_route_counters = lambda: resets.append(True)
    monkeypatch.setattr(manager, "_start_evaluation", lambda: starts.append(True))

    manager.step()
    manager.step()
    assert manager.prefill_steps == 20
    assert recordings == []
    assert starts == []

    for _ in range(55):
        manager.step()
    assert manager.prefill_steps == 75
    assert recordings == []
    assert starts == []

    manager.step()
    assert manager.prefill_steps == 76
    assert recordings == [True]
    assert resets == [True]
    assert manager._steady_collection_end_step is not None

    for _ in range(4):
        manager.step()
    assert manager.prefill_steps == 80
    assert starts == [True]
    assert manager._steady_collection_end_step is None


def test_planned_rebalance_resets_sampling_interval_to_base(monkeypatch):
    manager = manager_module.EPLBManager.__new__(manager_module.EPLBManager)
    manager.current_placement = torch.zeros((1, 1, 1), dtype=torch.int64)
    manager.num_logical_experts = 1
    manager.world_size = 1
    manager.node_world_size = 1
    manager.step_interval = 20
    manager.sampling_interval = 320
    manager._continuous_collection_start_step = 0
    manager.global_rank = 1
    manager.transfer = type(
        "Transfer",
        (),
        {"start": lambda self, plans: setattr(self, "started", plans)},
    )()
    manager._reset_route_counters = lambda: None

    manager._start_rebalance(
        {
            "placement": torch.zeros((1, 1, 1), dtype=torch.int64),
            "improved": torch.tensor([True]),
            "metadata": [None],
            "layer_plans": [(0, object())],
            "before": {"max": 1.0, "p95": 1.0},
            "after": {"max": 1.0, "p95": 1.0},
            "model_imbalance_ratio": 1.0,
            "candidate_model_imbalance_ratio": 1.0,
            "candidate_rebalance_gain": 0.1,
            "candidate_changed_layer_count": 1,
        }
    )

    assert manager.sampling_interval == 20
    assert manager.in_flight
    assert manager._continuous_collection_start_step is None
    assert len(manager.transfer.started) == 1


def test_first_rebalance_completion_switches_to_four_step_sparse_window(monkeypatch):
    manager = manager_module.EPLBManager.__new__(manager_module.EPLBManager)
    manager.step_interval = 20
    manager.sampling_interval = manager.step_interval
    manager._steady_collection_end_step = None
    manager.weights = []
    manager._eplb_impls = []
    manager.target_placement = torch.zeros(1)
    manager.in_flight_started_at = 0
    manager.global_rank = 1
    recordings, starts = [], []
    manager._set_recording = lambda enabled: recordings.append(enabled)
    manager._finish_rebalance()
    assert recordings == [False]

    manager.in_flight = False
    manager.prefill_steps = 38
    manager.evaluation_in_flight = False
    monkeypatch.setattr(manager, "_start_evaluation", lambda: starts.append(True))
    manager.prefill_steps = 35
    manager.step()
    assert recordings == [False, True]
    for _ in range(4):
        manager.step()
    assert starts == [True]


def test_manager_inflight_step_does_not_poll():
    manager = manager_module.EPLBManager.__new__(manager_module.EPLBManager)
    manager.in_flight = True
    calls = []
    manager._poll_in_flight = lambda: calls.append("poll")
    manager.step()
    assert calls == []


def test_manager_poll_advances_inflight_before_evaluation():
    manager = manager_module.EPLBManager.__new__(manager_module.EPLBManager)
    manager.in_flight = True
    manager.evaluation_in_flight = True
    calls = []
    manager._poll_in_flight = lambda: calls.append("inflight")
    manager._poll_evaluation = lambda: calls.append("evaluation")

    manager.poll()

    assert calls == ["inflight"]


def test_manager_poll_waits_for_all_evaluation_results(monkeypatch):
    manager = manager_module.EPLBManager.__new__(manager_module.EPLBManager)
    manager.in_flight = False
    manager.evaluation_in_flight = True
    manager._evaluation_lock = threading.Lock()
    manager._evaluation_result = None
    manager._evaluation_error = None
    manager.control_group = object()
    manager._control_ready_count = torch.empty(1, dtype=torch.int32)
    calls = []
    manager._poll_evaluation = lambda: calls.append("evaluation")

    monkeypatch.setattr(manager_module.dist, "all_reduce", lambda tensor, **_kwargs: tensor.fill_(0))
    manager.poll()
    assert calls == []

    manager._evaluation_result = {"kind": "no_improvement"}
    monkeypatch.setattr(manager_module.dist, "all_reduce", lambda tensor, **_kwargs: tensor.fill_(1))
    manager.poll()
    assert calls == ["evaluation"]


def test_mode_backend_owns_eplb_poll_and_prefill_step():
    backend = object.__new__(ModeBackend)
    calls = []
    backend.eplb_manager = SimpleNamespace(
        poll=lambda: calls.append("poll"),
        step=lambda: calls.append("step"),
    )
    backend.prefill = lambda **_kwargs: calls.append("prefill")

    backend._poll_eplb()
    backend._run_prefill(event_pack=object(), prefill_reqs=[])

    assert calls == ["poll", "prefill", "step"]


def test_mode_backend_eplb_hooks_are_noops_when_disabled():
    backend = object.__new__(ModeBackend)
    backend.eplb_manager = None
    calls = []
    backend.prefill = lambda **_kwargs: calls.append("prefill")

    backend._poll_eplb()
    backend._run_prefill(event_pack=object(), prefill_reqs=[])

    assert calls == ["prefill"]


def test_pinned_transfer_groups_sources_in_collective_order():
    plan = [
        TransferStep(0, 2, 1, 3),
        TransferStep(0, 0, 0, 1),
        TransferStep(1, 1, 1, 3),
    ]

    grouped = PinnedMemoryEPLBTransfer._group_steps_by_source(plan)

    assert [source for source, _steps in grouped] == [(0, 1), (1, 3)]
    assert grouped[1][1] == [plan[0], plan[2]]


def test_pinned_transfer_copies_source_row_through_cpu_buffer(monkeypatch):
    class Stream:
        def __init__(self):
            self.synchronize_count = 0

        def synchronize(self):
            self.synchronize_count += 1

    transfer = object.__new__(PinnedMemoryEPLBTransfer)
    transfer.global_rank = 0
    transfer.transfer_group = object()
    transfer._copy_stream = Stream()
    transfer.live = [[("weight", torch.tensor([[1.0, 2.0], [3.0, 4.0]]))]]
    transfer.pinned_rows = [("weight", torch.empty(2))]
    transfer.staging = [("weight", torch.zeros((2, 2)))]
    broadcasts = []
    monkeypatch.setattr(transfer_module.torch.cuda, "stream", lambda _stream: nullcontext())
    monkeypatch.setattr(
        transfer_module.dist,
        "broadcast",
        lambda tensor, src, group: broadcasts.append((tensor.clone(), src, group)),
    )

    transfer._copy_layer(
        0,
        [
            TransferStep(dst_rank=0, dst_slot=1, src_rank=0, src_local_row=1),
            TransferStep(dst_rank=1, dst_slot=0, src_rank=0, src_local_row=1),
        ],
    )

    assert len(broadcasts) == 1
    assert torch.equal(broadcasts[0][0], torch.tensor([3.0, 4.0]))
    assert broadcasts[0][1:] == (0, transfer.transfer_group)
    assert torch.equal(transfer.staging[0][1][1], torch.tensor([3.0, 4.0]))
    assert torch.count_nonzero(transfer.staging[0][1][0]) == 0
    assert transfer._copy_stream.synchronize_count == 2


def test_pinned_transfer_waits_for_each_layer_commit_before_reusing_staging(monkeypatch):
    class Event:
        def __init__(self):
            self.synchronize_count = 0

        def synchronize(self):
            self.synchronize_count += 1

        def record(self, _stream):
            pass

    transfer = object.__new__(PinnedMemoryEPLBTransfer)
    transfer.device = "cuda:0"
    transfer.global_rank = 0
    transfer.live = [[], []]
    transfer.staging = []
    transfer.num_experts_per_rank = 0
    transfer._release = threading.Event()
    transfer._release.set()
    transfer._consumed_event = Event()
    transfer._consumed_recorded = False
    transfer._ready = None
    transfer._ready_lock = threading.Lock()
    transfer._error = None
    transfer._thread = None
    copied = []
    transfer._copy_layer = lambda layer_index, _plan: copied.append(layer_index)
    monkeypatch.setattr(transfer_module.torch.cuda, "set_device", lambda _device: None)
    monkeypatch.setattr(transfer_module.torch.cuda, "current_stream", lambda: object())

    transfer.start([(0, []), (1, [])])
    deadline = time.monotonic() + 2
    while transfer.ready_layer() is None and time.monotonic() < deadline:
        time.sleep(0.001)
    assert transfer.ready_layer() == 0
    assert copied == [0]

    transfer.commit(0)
    deadline = time.monotonic() + 2
    while transfer.ready_layer() is None and time.monotonic() < deadline:
        time.sleep(0.001)
    assert transfer.ready_layer() == 1
    assert copied == [0, 1]
    assert transfer._consumed_event.synchronize_count == 1
    transfer.commit(1)
    transfer.finish()


def test_manager_constructs_pinned_memory_transfer(monkeypatch):
    weight = type(
        "Weight",
        (),
        {
            "n_routed_experts": 4,
            "fuse_moe_impl": _test_moe_impl(
                eplb=True,
                num_logical_experts=4,
                world_size=2,
                num_redundant_experts_per_rank=2,
                route_counter=torch.zeros((4,), dtype=torch.int64),
            ),
        },
    )()
    transfer = object()
    groups = [object(), object(), object()]
    new_group_calls = []
    monkeypatch.setattr(manager_module, "_find_fused_moe_weights", lambda model: [weight])
    monkeypatch.setattr(manager_module, "get_global_rank", lambda: 0)
    monkeypatch.setattr(manager_module, "get_global_world_size", lambda: 2)
    monkeypatch.setattr(manager_module, "get_node_world_size", lambda: 2)
    monkeypatch.setattr(manager_module, "get_prefill_eplb_step_interval", lambda: 20)
    monkeypatch.setattr(manager_module, "get_eplb_rebalance_gain_threshold", lambda: 0.07)

    def new_group(*args, **kwargs):
        new_group_calls.append((args, kwargs))
        return groups[len(new_group_calls) - 1]

    monkeypatch.setattr(manager_module.dist, "new_group", new_group)
    transfer_calls = []
    monkeypatch.setattr(
        manager_module,
        "PinnedMemoryEPLBTransfer",
        lambda weights, group, rank: (transfer_calls.append((weights, group, rank)) or transfer),
    )
    logs = []
    monkeypatch.setattr(manager_module.logger, "info", lambda message: logs.append(message))
    manager = manager_module.EPLBManager(type("Model", (), {})())
    assert manager.transfer is transfer
    assert (
        manager.evaluation_group,
        manager.control_group,
        manager.transfer_group,
    ) == tuple(groups)
    assert new_group_calls == [(([0, 1],), {"backend": "gloo"})] * 3
    assert transfer_calls == [([weight], groups[2], 0)]
    assert manager.rebalance_gain_threshold == 0.07
    assert "rebalance_gain_threshold=0.0700" in logs[0]
    assert manager._continuous_collection_start_step is None
    assert manager._continuous_collection_end_step == manager.step_interval
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
