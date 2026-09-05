import threading
import time
from collections import deque
from contextlib import contextmanager
from types import SimpleNamespace

import pytest
import torch

from lightllm.common.basemodel.layer_weights.meta_weights.fused_moe.eplb_placement import (
    build_initial_redundant_expert_ids,
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
from lightllm.common.basemodel.layer_weights.meta_weights.fused_moe.impl import (
    deepgemm_impl as deepgemm_module,
)
from lightllm.common.basemodel.layer_weights.meta_weights.fused_moe.expert_parallel_state import (
    EPLBState,
    ExpertParallelState,
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
from lightllm.common.basemodel.layer_weights.meta_weights.fused_moe.expert_parallel_state import (
    disable_eplb_model_init,
    is_eplb_model_init_disabled,
)
from lightllm.common.cuda_batch_memcpy import (
    CudaBatchMemcpy,
    CudaBatchMemcpyUnavailable,
)
from lightllm.server.router.model_infer.mode_backend.eplb_transfer import (
    TransferStep,
    align_target_placement,
    build_transfer_plan,
    commit_staging_rows,
    extract_expert_tensors,
)


def _test_parallel_state(
    *,
    eplb=False,
    num_logical_experts=128,
    world_size=16,
    num_redundant_experts_per_rank=1,
    route_counter=None,
    recording=False,
    recorded_sample_count=0,
):
    eplb_state = None
    if eplb:
        if route_counter is None:
            route_counter = torch.zeros((2, num_logical_experts), dtype=torch.int64)
        initial_layout_world_size = max(world_size, 2)
        eplb_state = EPLBState(
            num_redundant_experts_per_rank=num_redundant_experts_per_rank,
            initial_redundant_expert_ids_by_rank=build_initial_redundant_expert_ids(
                num_logical_experts,
                initial_layout_world_size,
                num_redundant_experts_per_rank,
            ),
            logical_to_physical_map=torch.zeros((num_logical_experts, 2), dtype=torch.int32),
            logical_replica_count=torch.ones(num_logical_experts, dtype=torch.int32),
            route_counter=route_counter,
            recording=recording,
            recorded_sample_count=recorded_sample_count,
        )
    return ExpertParallelState(
        num_logical_experts=num_logical_experts,
        world_size=world_size,
        eplb=eplb_state,
    )


def _validated_expert_parallel_state(
    *,
    eplb=True,
    n_routed_experts=4,
    world_size=2,
    num_redundant_experts_per_rank=1,
    device="cpu",
):
    runtime = None
    if eplb:
        runtime = EPLBState(
            num_redundant_experts_per_rank=num_redundant_experts_per_rank,
            initial_redundant_expert_ids_by_rank=build_initial_redundant_expert_ids(
                n_routed_experts,
                world_size,
                num_redundant_experts_per_rank,
            ),
            logical_to_physical_map=torch.empty((n_routed_experts, 2), dtype=torch.int32, device=device),
            logical_replica_count=torch.ones(n_routed_experts, dtype=torch.int32, device=device),
            route_counter=torch.zeros((2, n_routed_experts), dtype=torch.int64, device=device),
        )
    return ExpertParallelState(
        num_logical_experts=n_routed_experts,
        world_size=world_size,
        eplb=runtime,
    )


def _set_expert_parallel_state(impl, state):
    impl.expert_parallel_state = state
    impl.eplb = state.eplb


def _manual_runtime_rank_load(source_load, placement, node_world_size, alignment):
    """Reference the committed runtime logical-to-physical maps on CPU."""
    samples, layers, nodes, num_logical_experts = source_load.shape
    ranks, redundant = placement.shape[1:]
    num_experts_per_rank = num_logical_experts // ranks
    num_physical_experts_per_rank = num_experts_per_rank + redundant
    raw = torch.zeros((samples, layers, ranks, num_logical_experts), dtype=torch.float64)
    for layer in range(layers):
        for source_node in range(nodes):
            logical_to_physical, replica_count = build_logical_to_physical_map(
                placement[layer],
                num_logical_experts,
                source_rank=source_node * node_world_size,
                node_world_size=node_world_size,
            )
            for expert in range(num_logical_experts):
                count = int(replica_count[expert].item())
                for physical_id in logical_to_physical[expert, :count].tolist():
                    rank = physical_id // num_physical_experts_per_rank
                    raw[:, layer, rank, expert] += source_load[:, layer, source_node, expert] / count
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
            shared_expert_gate=None,
            is_prefill=None,
            preserve_logical_ids=False,
        ):
            seen["select"] = {"preserve_logical_ids": preserve_logical_ids}
            return "weights", "physical_ids", "logical_ids"

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
    assert seen["select"]["preserve_logical_ids"]
    assert seen["fused"]["topk_ids"] == "physical_ids"


def test_parallel_state_derives_expert_layout():
    state = _validated_expert_parallel_state(eplb=True)
    assert state.num_primary_experts_per_rank == 2
    assert state.num_total_physical_experts == 6


def test_factory_selects_all_paths_and_requires_ep_state():
    plain_quant = SimpleNamespace(method_name="none")
    marlin_quant = SimpleNamespace(method_name="awq_marlin")
    state = _validated_expert_parallel_state(eplb=False)
    ep_impl = create_fuse_moe_impl(
        n_routed_experts=4,
        num_fused_shared_experts=0,
        routed_scaling_factor=1.0,
        quant_method=plain_quant,
        expert_parallel_state=state,
    )
    assert isinstance(ep_impl, deepgemm_module.FuseMoeDeepGEMM)
    assert ep_impl.expert_parallel_state is state
    assert state.eplb is None
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


def test_eplb_redundant_experts_defaults_per_ep_rank():
    parser = make_argument_parser()

    assert parser.parse_args([]).eplb_num_redundant_experts_per_rank == 2
    assert parser.parse_args(["--eplb_num_redundant_experts_per_rank", "3"]).eplb_num_redundant_experts_per_rank == 3
    assert StartArgs().eplb_num_redundant_experts_per_rank == 2


@pytest.mark.parametrize(
    ("num_logical_experts", "num_ranks", "num_redundant_experts_per_rank", "expected"),
    [
        (8, 4, 2, [[2, 3], [4, 5], [6, 7], [0, 1]]),
        (6, 3, 4, [[2, 3, 4, 5], [4, 5, 0, 1], [0, 1, 2, 3]]),
    ],
)
def test_build_initial_redundant_expert_ids(
    num_logical_experts,
    num_ranks,
    num_redundant_experts_per_rank,
    expected,
):
    actual = build_initial_redundant_expert_ids(
        num_logical_experts,
        num_ranks,
        num_redundant_experts_per_rank,
    )

    assert actual.dtype == torch.int64
    assert actual.shape == (num_ranks, num_redundant_experts_per_rank)
    assert torch.equal(actual, torch.tensor(expected, dtype=torch.int64))


def test_plan_redundant_experts_never_uses_owner_or_duplicate_rank():
    expert_load = torch.tensor(
        [
            [100, 90, 80, 70, 60, 50, 40, 30],
            [30, 40, 50, 60, 70, 80, 90, 100],
        ]
    )
    placement = plan_redundant_experts(expert_load, num_ranks=4, num_redundant_experts_per_rank=2)

    for layer_placement in placement:
        for rank, expert_ids in enumerate(layer_placement.tolist()):
            assert len(expert_ids) == len(set(expert_ids))
            assert all(expert_id // 2 != rank for expert_id in expert_ids)


def test_plan_redundant_experts_minimizes_samplewise_aligned_critical_load():
    samples = torch.tensor([[[300, 20, 20, 200]], [[100, 300, 40, 160]]])
    placement = plan_redundant_experts(samples, num_ranks=2, num_redundant_experts_per_rank=1, expert_alignment=128)
    candidates = [torch.tensor([[[left], [right]]]) for left in (2, 3) for right in (0, 1)]

    def critical(candidate):
        return _estimate_rank_load(samples, candidate, expert_alignment=128).max(dim=2).values.sum()

    assert torch.equal(placement, torch.tensor([[[3], [0]]]))
    assert critical(placement) == min(critical(candidate) for candidate in candidates)


def test_select_improving_placements_rejects_regressing_layer():
    expert_load = torch.tensor([[8649, 5740, 5002, 3441]])
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
    expert_load = torch.tensor([[1, 2, 1, 17]])
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
    expert_load = torch.tensor([[100, 129, 100, 129]])
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
    expert_load = torch.tensor([[1, 1, 6, 7]])
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
    expert_load = torch.tensor([[1, 1, 1, 2]])
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
    expert_load = torch.tensor([[1, 1, 1, 2]])
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
    expert_load = torch.tensor([[1, 1, 1, 8]])
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
    samples = torch.tensor([[[20, 0, 0, 0]], [[20, 0, 0, 0]]])
    placement = torch.tensor([[[2], [0]]])

    per_sample = _estimate_rank_load(samples, placement, expert_alignment=128)
    accumulated = _estimate_rank_load(samples.sum(dim=0), placement, expert_alignment=128)

    assert torch.equal(per_sample[:, 0], torch.tensor([[128.0, 128.0], [128.0, 128.0]]))
    assert torch.equal(per_sample.sum(dim=0)[0], torch.tensor([256.0, 256.0]))
    assert torch.equal(accumulated[0], torch.tensor([128.0, 128.0]))


def test_select_improving_placements_rejects_lower_ratio_when_critical_is_unchanged():
    samples = torch.tensor(
        [
            [[255, 220, 226, 254]],
            [[172, 278, 51, 238]],
            [[249, 291, 284, 183]],
        ]
    )
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
    )
    current = torch.tensor([[[2], [0]]])
    candidate = torch.tensor([[[2], [1]]])

    selected, improved, _metrics, _before_load, _after_load = select_improving_placements(
        samples, current, candidate, rebalance_gain_threshold=0.05, expert_alignment=128
    )

    assert improved.item()
    assert torch.equal(selected, candidate)


def test_select_improving_placements_rejects_single_layer_gain_below_model_threshold():
    # Layer 0 becomes better, but layer 1 dominates model critical load.  The
    # aggregate estimated critical-load reduction gain is below 5%, so neither layer may be changed.
    expert_load = torch.tensor([[1, 1, 1, 2], [0, 0, 0, 10]])
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
    expert_load = torch.tensor([[1, 1, 1, 2], [0, 0, 0, 5]])
    current = torch.tensor([[[2], [0]], [[2], [0]]])
    candidate = torch.tensor([[[3], [0]], [[2], [0]]])

    selected, improved, _metrics, _before_load, _after_load = select_improving_placements(
        expert_load, current, candidate, rebalance_gain_threshold=0.05
    )

    assert torch.equal(improved, torch.tensor([True, False]))
    assert torch.equal(selected, candidate)


def test_logical_to_physical_map_has_at_most_one_slot_per_rank():
    redundant_expert_ids = torch.tensor([[2, 3], [0, 1]])
    logical_to_physical, replica_count = build_logical_to_physical_map(redundant_expert_ids, num_logical_experts=4)

    assert logical_to_physical.shape == (4, 2)
    assert torch.equal(replica_count, torch.full((4,), 2, dtype=torch.int64))


def test_logical_to_physical_map_requires_expert_count_divisible_by_rank_count_without_source_rank():
    redundant_expert_ids = torch.tensor([[0], [1]])

    with pytest.raises(AssertionError):
        build_logical_to_physical_map(redundant_expert_ids, num_logical_experts=5)


def test_logical_to_physical_map_prefers_source_node_replicas():
    # Four ranks, two ranks per node, two primary experts/rank and one
    # redundant slot/rank.  Expert 0 is primary on rank 0 and replicated on
    # rank 2 (the other node).
    redundant = torch.tensor([[4], [5], [0], [1]], dtype=torch.int64)
    rank0_map, rank0_count = build_logical_to_physical_map(
        redundant, num_logical_experts=8, source_rank=0, node_world_size=2
    )
    rank1_map, rank1_count = build_logical_to_physical_map(
        redundant, num_logical_experts=8, source_rank=1, node_world_size=2
    )
    fallback_redundant = torch.tensor([[4], [5], [0], [1], [2], [3]], dtype=torch.int64)
    rank4_map, rank4_count = build_logical_to_physical_map(fallback_redundant, 12, source_rank=4, node_world_size=2)

    assert rank0_count[0].item() == rank1_count[0].item() == 1
    assert torch.equal(rank0_map[0, :1], torch.tensor([0]))
    assert torch.equal(rank1_map[0, :1], torch.tensor([0]))
    assert rank4_count[0].item() == 2
    assert set(rank4_map[0, :2].tolist()) == {0, 8}


def test_source_node_local_maps_fall_back_to_global_replicas():
    redundant = torch.tensor([[4], [5], [0], [1]], dtype=torch.int64)
    maps = [build_logical_to_physical_map(redundant, 8, source_rank=rank, node_world_size=2) for rank in range(4)]
    assert maps[0][1][0].item() == maps[1][1][0].item() == 1
    assert maps[2][1][0].item() == maps[3][1][0].item() == 1
    assert maps[0][0][0, 0].item() == maps[1][0][0, 0].item() == 0
    assert maps[2][0][0, 0].item() == maps[3][0][0, 0].item() == 8


def test_source_rank_rotates_selected_replica_order_without_changing_copies():
    # Expert 0 is primary on rank 0 and redundant on rank 1, so both ranks
    # on node 0 have the same two local copies.  Their source-rank phases
    # must differ while their selected set/count remain identical.
    redundant = torch.tensor([[1], [0], [3], [2]], dtype=torch.int64)
    rank0_map, rank0_count = build_logical_to_physical_map(redundant, 4, source_rank=0, node_world_size=2)
    rank1_map, rank1_count = build_logical_to_physical_map(redundant, 4, source_rank=1, node_world_size=2)

    assert rank0_count[0].item() == rank1_count[0].item() == 2
    assert set(rank0_map[0, :2].tolist()) == set(rank1_map[0, :2].tolist()) == {0, 3}
    assert torch.equal(rank1_map[0, :2], torch.tensor([3, 0], dtype=torch.int32))


@pytest.mark.parametrize(
    "source_rank,node_world_size",
    [(None, None), (0, 2), (1, 2), (2, 2), (3, 2)],
)
def test_logical_to_physical_maps_for_layers_match_single_layer_api(source_rank, node_world_size):
    placements_by_layer = torch.tensor(
        [
            [[4, 5], [0, 1], [0, 1], [2, 3]],
            [[6, 7], [0, 1], [0, 1], [2, 3]],
            [[4, 5], [0, 1], [0, 1], [2, 3]],
        ],
        dtype=torch.int64,
    )

    maps_by_layer, counts_by_layer = build_logical_to_physical_maps_for_layers(
        placements_by_layer,
        num_logical_experts=8,
        source_rank=source_rank,
        node_world_size=node_world_size,
    )
    expected_by_layer = [
        build_logical_to_physical_map(
            placement,
            num_logical_experts=8,
            source_rank=source_rank,
            node_world_size=node_world_size,
        )
        for placement in placements_by_layer
    ]

    assert maps_by_layer.shape == (3, 8, 4)
    assert counts_by_layer.shape == (3, 8)
    assert maps_by_layer.dtype == counts_by_layer.dtype == torch.int32
    assert torch.equal(maps_by_layer, torch.stack([item[0] for item in expected_by_layer]))
    assert torch.equal(counts_by_layer, torch.stack([item[1] for item in expected_by_layer]))
    if source_rank is None:
        # expert 0 的主副本在 rank 0，且 rank 1、2 都有一个冗余副本；第 1、2
        # 个冗余副本必须分别写入映射表的第 1、2 列，而不能互相覆盖。
        assert torch.equal(counts_by_layer[:, 0], torch.tensor([3, 3, 3], dtype=torch.int32))
        assert torch.equal(
            maps_by_layer[:, 0, :3],
            torch.tensor([[0, 6, 10], [0, 6, 10], [0, 6, 10]], dtype=torch.int32),
        )
    positions = torch.arange(maps_by_layer.shape[-1]).view(1, 1, -1)
    valid = positions < counts_by_layer.unsqueeze(-1)
    assert torch.all(maps_by_layer[valid] >= 0)
    assert torch.all(maps_by_layer[~valid] == -1)


def test_plan_redundant_experts_prefers_first_replica_on_new_node():
    # One redundant slot per rank leaves legal alternatives on both nodes;
    # topology preference therefore puts every first replica away from its
    # primary node before considering same-node duplicates.
    placement = plan_redundant_experts(
        torch.tensor([[1000, 900, 800, 700, 600, 500, 400, 300]]),
        num_ranks=4,
        num_redundant_experts_per_rank=1,
        node_world_size=2,
    )
    for rank, expert in enumerate(placement[0, :, 0].tolist()):
        assert expert // 2 // 2 != rank // 2


def test_plan_redundant_experts_single_node_matches_default_behavior():
    load = torch.tensor([[1000, 900, 800, 700, 600, 500, 400, 300]])
    default = plan_redundant_experts(load, num_ranks=4, num_redundant_experts_per_rank=1)
    single_node = plan_redundant_experts(load, num_ranks=4, num_redundant_experts_per_rank=1, node_world_size=4)
    assert torch.equal(single_node, default)


def test_source_node_estimate_matches_local_first_runtime_replica_sharing():
    # Expert 0 has a copy on each node.  Node 0 and node 1 issue unequal
    # traffic, so collapsing them before planning produces the wrong result.
    placement = torch.tensor([[[4], [5], [0], [1]]], dtype=torch.int64)
    source_load = torch.zeros((1, 1, 2, 8), dtype=torch.int64)
    source_load[0, 0, 0, 0] = 256
    source_load[0, 0, 1, 0] = 128

    predicted = _estimate_rank_load(source_load, placement, expert_alignment=128, node_world_size=2)
    runtime = _manual_runtime_rank_load(source_load, placement, node_world_size=2, alignment=128)
    collapsed_global = _estimate_rank_load(source_load.sum(dim=2), placement, expert_alignment=128)

    assert torch.equal(predicted, runtime)
    assert torch.equal(predicted[0, 0], torch.tensor([256.0, 0.0, 128.0, 0.0]))
    assert not torch.equal(predicted, collapsed_global)


def test_source_node_planner_constraints_and_real_critical_improvement():
    source_load = torch.tensor(
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
    initial = build_initial_redundant_expert_ids(8, 4, 1).unsqueeze(0)
    planned = plan_redundant_experts(source_load, 4, 1, expert_alignment=128, node_world_size=2)

    for rank, experts in enumerate(planned[0].tolist()):
        assert len(experts) == len(set(experts)) == 1
        assert experts[0] // 2 != rank

    before = _estimate_rank_load(source_load, initial, expert_alignment=128, node_world_size=2)
    after = _estimate_rank_load(source_load, planned, expert_alignment=128, node_world_size=2)
    manual_before = _manual_runtime_rank_load(source_load, initial, 2, 128)
    manual_after = _manual_runtime_rank_load(source_load, planned, 2, 128)
    assert torch.equal(before, manual_before)
    assert torch.equal(after, manual_after)
    assert after.max(dim=2).values.sum() < before.max(dim=2).values.sum()


def test_source_node_select_uses_the_same_runtime_critical_prediction():
    source_load = torch.zeros((2, 1, 2, 8), dtype=torch.int64)
    source_load[:, 0, 0, 0] = torch.tensor([1024, 768])
    source_load[:, 0, 1, 6] = torch.tensor([896, 1024])
    current = torch.tensor([[[2], [4], [6], [0]]], dtype=torch.int64)
    candidate = plan_redundant_experts(source_load, 4, 1, expert_alignment=128, node_world_size=2)
    selected, _improved, _metrics, _before_load, _after_load = select_improving_placements(
        source_load,
        current,
        candidate,
        rebalance_gain_threshold=0.05,
        expert_alignment=128,
        node_world_size=2,
    )
    assert torch.equal(
        _estimate_rank_load(source_load, selected, 128, 2),
        _manual_runtime_rank_load(source_load, selected, 2, 128),
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
    load = torch.randint(1, 1000, (3, 16, 32), generator=generator)
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
    load = torch.randint(100, 1000, (4, 16, 32), generator=generator)
    placement = plan_redundant_experts(load, num_ranks=4, num_redundant_experts_per_rank=2)
    noise = torch.rand((4, 16, 32), generator=generator) * 0.1 + 0.95
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

    logical_to_physical, replica_count = build_logical_to_physical_map(canonical, num_logical_experts)
    for logical_expert, count in enumerate(replica_count.tolist()):
        for physical_id in logical_to_physical[logical_expert, :count].tolist():
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
        rank_load = torch.full((1, 4), 100.0)
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


def test_stickiness_zero_matches_legacy():
    generator = torch.Generator().manual_seed(17)
    load = torch.randint(1, 1000, (2, 8, 16), generator=generator)
    legacy = plan_redundant_experts(load, num_ranks=4, num_redundant_experts_per_rank=2)
    unrelated = build_initial_redundant_expert_ids(16, 4, 2).unsqueeze(0).expand(8, -1, -1).clone()

    replanned = plan_redundant_experts(
        load,
        num_ranks=4,
        num_redundant_experts_per_rank=2,
        current_placement=unrelated,
        stickiness=0.0,
    )

    assert torch.equal(replanned, legacy)


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
    counter = torch.tensor([[10, 20, 30, 40], [0, 0, 0, 0]], dtype=torch.int64)
    manager.weights = [
        type(
            "Weight",
            (),
            {
                "expert_parallel_state": _test_parallel_state(
                    eplb=True,
                    route_counter=counter,
                    recording=False,
                    recorded_sample_count=1,
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
    manager._sampling_pending = False
    manager._steady_collection_end_step = None
    manager._continuous_collection_start_step = None
    manager._continuous_collection_end_step = None

    recordings, resets, started = [], [], []
    manager._set_recording = lambda enabled: recordings.append(enabled)
    manager._reset_recorded_samples = lambda: resets.append(True)
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
    assert manager._sampling_pending
    assert manager._steady_collection_end_step == 20

    for _ in range(3):
        manager.step()
    assert manager.prefill_steps == 19
    assert recordings == [True]
    assert started == []

    manager.step()
    assert manager.prefill_steps == 20
    assert not manager._sampling_pending
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
    manager._sampling_pending = False
    manager._steady_collection_end_step = None
    manager._continuous_collection_start_step = None
    manager._continuous_collection_end_step = None
    recordings, resets, started = [], [], []
    manager._set_recording = lambda enabled: recordings.append((manager.prefill_steps, enabled))
    manager._reset_recorded_samples = lambda: resets.append(manager.prefill_steps)
    monkeypatch.setattr(manager, "_start_evaluation", lambda: started.append(manager.prefill_steps))

    manager._prepare_next_sampling_window()
    assert resets == [0]
    assert recordings == [(0, True)]
    assert manager._steady_collection_end_step == 3
    assert manager._sampling_pending

    manager.step()
    manager.step()
    assert started == []
    manager.step()
    assert started == [3]


def test_eplb_step_does_not_start_a_second_evaluation_while_one_is_pending(monkeypatch):
    manager = manager_module.EPLBManager.__new__(manager_module.EPLBManager)
    counter = torch.tensor([[10, 20, 30, 40], [0, 0, 0, 0]], dtype=torch.int64)
    manager.weights = [
        type(
            "Weight",
            (),
            {
                "expert_parallel_state": _test_parallel_state(
                    eplb=True,
                    route_counter=counter,
                    recording=False,
                    recorded_sample_count=1,
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
    manager._eplb_states = []
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
    manager._sampling_pending = False
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
    manager._eplb_states = []
    recordings, starts = [], []
    manager._set_recording = lambda enabled: recordings.append(enabled)
    manager._start_evaluation = lambda: starts.append(True)
    manager._evaluation_ready_on_all_ranks = lambda: True

    manager.poll()

    # The no-improvement backoff changes interval 1 to 4.  The clamped
    # steady window arms immediately but still waits for boundary step 5.
    assert recordings == [True]
    assert manager._sampling_pending
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
    manager._eplb_states = []
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


def test_manager_collects_recent_ring_samples_in_chronological_order(monkeypatch):
    counters = [
        torch.tensor([[10, 11], [20, 21], [30, 31]], dtype=torch.int64),
        torch.tensor([[40, 41], [50, 51], [60, 61]], dtype=torch.int64),
    ]
    manager = manager_module.EPLBManager.__new__(manager_module.EPLBManager)
    manager.weights = [
        type(
            "Weight",
            (),
            {
                "expert_parallel_state": _test_parallel_state(
                    eplb=True,
                    route_counter=counter,
                    recording=False,
                    recorded_sample_count=5,
                    num_logical_experts=2,
                    world_size=1,
                ),
            },
        )()
        for counter in counters
    ]
    manager._eplb_states = [weight.expert_parallel_state.eplb for weight in manager.weights]
    manager.evaluation_group = object()
    metadata_sizes = []

    def all_reduce(metadata, **_kwargs):
        metadata_sizes.append(metadata.numel())

    monkeypatch.setattr(manager_module.dist, "all_reduce", all_reduce)

    samples = manager._collect_local_samples()

    assert metadata_sizes == [4]
    assert torch.equal(
        samples,
        torch.tensor(
            [
                [[30, 31], [60, 61]],
                [[10, 11], [40, 41]],
                [[20, 21], [50, 51]],
            ],
            dtype=torch.int64,
        ),
    )


def test_manager_collects_only_two_recent_sparse_samples(monkeypatch):
    counters = [
        torch.tensor([[10, 11], [20, 21], [30, 31], [40, 41]], dtype=torch.int64),
        torch.tensor([[50, 51], [60, 61], [70, 71], [80, 81]], dtype=torch.int64),
    ]
    manager = manager_module.EPLBManager.__new__(manager_module.EPLBManager)
    manager.weights = [
        type(
            "Weight",
            (),
            {
                "expert_parallel_state": _test_parallel_state(
                    eplb=True,
                    route_counter=counter,
                    recording=False,
                    recorded_sample_count=2,
                    num_logical_experts=2,
                    world_size=1,
                ),
            },
        )()
        for counter in counters
    ]
    manager._eplb_states = [weight.expert_parallel_state.eplb for weight in manager.weights]
    manager.evaluation_group = object()
    monkeypatch.setattr(manager_module.dist, "all_reduce", lambda tensor, **kwargs: None)

    samples = manager._collect_local_samples()

    assert torch.equal(
        samples,
        torch.tensor([[[10, 11], [50, 51]], [[20, 21], [60, 61]]], dtype=torch.int64),
    )


def test_eplb_counter_capacity_covers_default_dense_interval(monkeypatch):
    args = type(
        "Args",
        (),
        {"enable_prefill_eplb": True, "eplb_num_redundant_experts_per_rank": 2},
    )()
    weight = object.__new__(fused_weight_module.FusedMoeWeight)
    weight.n_routed_experts = 4
    weight.global_world_size = 2
    weight.global_rank_ = 0
    weight.enable_ep_moe = True
    monkeypatch.setattr(fused_weight_module, "get_env_start_args", lambda: args)
    monkeypatch.setattr(fused_weight_module, "get_prefill_eplb_step_interval", lambda: 20)
    monkeypatch.setattr(fused_weight_module, "get_node_world_size", lambda: 2)
    monkeypatch.setattr(torch.Tensor, "cuda", lambda tensor: tensor)
    original_zeros = torch.zeros

    def cpu_zeros(*shape, **kwargs):
        kwargs.pop("device", None)
        return original_zeros(*shape, **kwargs)

    monkeypatch.setattr(fused_weight_module.torch, "zeros", cpu_zeros)

    weight._init_expert_parallel_state()

    assert weight.expert_parallel_state.eplb.route_counter.shape == (40, 4)


def test_steady_sampling_resets_fixed_ring_without_retained_history():
    counter = torch.ones((8, 4), dtype=torch.int64)
    manager = manager_module.EPLBManager.__new__(manager_module.EPLBManager)
    state = _test_parallel_state(
        eplb=True,
        route_counter=counter,
        recording=False,
        recorded_sample_count=99,
        num_logical_experts=4,
        world_size=1,
    ).eplb
    manager._eplb_states = [state]

    manager._reset_recorded_samples()
    manager._reset_recorded_samples()

    assert state.route_counter.shape == (8, 4)
    assert torch.count_nonzero(state.route_counter) == 0
    assert state.recorded_sample_count == 0
    assert not hasattr(manager, "_retained_local_samples")
    assert not hasattr(manager, "_sample_history")


def test_ep_without_eplb_creates_layout_without_eplb_runtime_state(monkeypatch):
    args = type(
        "Args",
        (),
        {"enable_prefill_eplb": False, "eplb_num_redundant_experts_per_rank": 2},
    )()
    weight = object.__new__(fused_weight_module.FusedMoeWeight)
    weight.n_routed_experts = 4
    weight.global_world_size = 2
    weight.global_rank_ = 0
    weight.enable_ep_moe = True
    monkeypatch.setattr(fused_weight_module, "get_env_start_args", lambda: args)

    weight._init_expert_parallel_state()

    assert weight.expert_parallel_state is not None
    assert weight.expert_parallel_state.eplb is None
    assert weight.expert_parallel_state.num_total_physical_experts == weight.n_routed_experts
    assert weight._initial_redundant_expert_ids == []
    assert not hasattr(weight, "route_counter")
    assert not hasattr(weight, "routed_expert_counter_tensor")


def test_disable_eplb_model_init_skips_eplb_state(monkeypatch):
    args = type(
        "Args",
        (),
        {"enable_prefill_eplb": True, "eplb_num_redundant_experts_per_rank": 2},
    )()
    weight = object.__new__(fused_weight_module.FusedMoeWeight)
    weight.n_routed_experts = 4
    weight.global_world_size = 2
    weight.global_rank_ = 0
    weight.enable_ep_moe = True
    monkeypatch.setattr(fused_weight_module, "get_env_start_args", lambda: args)
    monkeypatch.setattr(
        fused_weight_module,
        "build_initial_redundant_expert_ids",
        lambda *args, **kwargs: pytest.fail("disabled scope must not initialize EPLB"),
    )

    with disable_eplb_model_init():
        weight._init_expert_parallel_state()

    assert weight.expert_parallel_state.eplb is None
    assert weight.expert_parallel_state.num_total_physical_experts == weight.n_routed_experts
    assert weight._initial_redundant_expert_ids == []


def test_disable_eplb_model_init_scope_restores_after_exception():
    assert not is_eplb_model_init_disabled()
    with disable_eplb_model_init():
        assert is_eplb_model_init_disabled()
        with disable_eplb_model_init():
            assert is_eplb_model_init_disabled()
        assert is_eplb_model_init_disabled()

    with pytest.raises(RuntimeError):
        with disable_eplb_model_init():
            raise RuntimeError
    assert not is_eplb_model_init_disabled()


def test_manager_evaluation_collective_preserves_source_node_axis(monkeypatch):
    manager = manager_module.EPLBManager.__new__(manager_module.EPLBManager)
    manager.weights = [
        type(
            "Weight",
            (),
            {
                "expert_parallel_state": _test_parallel_state(
                    eplb=True,
                    route_counter=torch.zeros((1, 4), dtype=torch.int64),
                    num_logical_experts=4,
                    world_size=1,
                )
            },
        )()
    ]
    manager._eplb_states = [manager.weights[0].expert_parallel_state.eplb]
    manager.global_rank = 2
    manager.world_size = 4
    manager.node_world_size = 2
    manager.step_interval = 20
    manager.sampling_interval = 20
    manager.num_logical_experts = 4
    manager.num_redundant_experts_per_rank = 1
    manager.current_placement = build_initial_redundant_expert_ids(4, 4, 1).unsqueeze(0)
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
        # Simulate source node 0's contribution from the other ranks.
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
    assert seen["before"].shape == (1, 1, 2, 4)
    assert torch.equal(seen["before"][:, :, 0], torch.zeros_like(expected_local))
    assert torch.equal(seen["before"][:, :, 1], expected_local)
    assert torch.equal(seen["global_load"][:, :, 0], torch.full_like(expected_local, 100))
    assert torch.equal(seen["global_load"][:, :, 1], expected_local)
    assert manager._evaluation_error is None
    assert manager._evaluation_result["recorded_sample_count"] == 1
    assert manager._evaluation_result["sample_window_steps"] == 4


def test_manager_planned_evaluation_builds_improved_metadata_in_one_multilayer_call(monkeypatch):
    manager = manager_module.EPLBManager.__new__(manager_module.EPLBManager)
    manager.weights = [
        type(
            "Weight",
            (),
            {
                "expert_parallel_state": _test_parallel_state(
                    eplb=True,
                    route_counter=torch.zeros((1, 4), dtype=torch.int64),
                    num_logical_experts=4,
                    world_size=4,
                )
            },
        )()
        for _ in range(3)
    ]
    manager._eplb_states = [weight.expert_parallel_state.eplb for weight in manager.weights]
    manager.global_rank = 1
    manager.world_size = 4
    manager.node_world_size = 2
    manager.step_interval = 20
    manager.sampling_interval = 20
    manager.num_logical_experts = 4
    manager.num_redundant_experts_per_rank = 1
    manager.current_placement = build_initial_redundant_expert_ids(4, 4, 1).unsqueeze(0).expand(3, -1, -1).clone()
    manager.evaluation_group = object()
    manager._evaluation_lock = threading.Lock()
    manager._evaluation_result = None
    manager._evaluation_error = None
    manager._continuous_collection_end_step = None
    prepare_calls = []
    prepared_batches = object()

    def prepare_transfer(layer_plans):
        prepare_calls.append(layer_plans)
        return prepared_batches

    manager.transfer = SimpleNamespace(prepare_transfer=prepare_transfer)
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
        calls.append(args[0].shape)
        return original_build_maps_for_layers(*args, **kwargs)

    monkeypatch.setattr(
        manager_module,
        "build_logical_to_physical_maps_for_layers",
        build_maps_for_layers,
    )

    manager._evaluate_after_event(type("Event", (), {"synchronize": lambda self: None})())

    assert manager._evaluation_error is None
    assert calls == [torch.Size([2, 4, 1])]
    metadata = manager._evaluation_result["metadata"]
    assert metadata[1] is None
    assert [layer_index for layer_index, _plan in manager._evaluation_result["layer_plans"]] == [0, 2]
    assert prepare_calls == [manager._evaluation_result["layer_plans"]]
    assert manager._evaluation_result["prepared_batches"] is prepared_batches
    for layer_index in (0, 2):
        item = metadata[layer_index]
        expected = build_logical_to_physical_map(
            planned_placement[layer_index],
            4,
            source_rank=manager.global_rank,
            node_world_size=manager.node_world_size,
        )
        assert torch.equal(item[0], expected[0])
        assert torch.equal(item[1], expected[1])


def test_manager_preparation_error_is_saved_as_evaluation_error(monkeypatch):
    manager = manager_module.EPLBManager.__new__(manager_module.EPLBManager)
    manager.weights = [
        type(
            "Weight",
            (),
            {
                "expert_parallel_state": _test_parallel_state(
                    eplb=True,
                    route_counter=torch.zeros((1, 2), dtype=torch.int64),
                    num_logical_experts=2,
                    world_size=1,
                )
            },
        )()
    ]
    manager._eplb_states = [manager.weights[0].expert_parallel_state.eplb]
    manager.global_rank = 0
    manager.world_size = 1
    manager.node_world_size = 1
    manager.step_interval = 20
    manager.sampling_interval = 20
    manager.num_logical_experts = 2
    manager.current_placement = torch.tensor([[[0], [1]]], dtype=torch.int64)
    manager.evaluation_group = object()
    manager._evaluation_lock = threading.Lock()
    manager._evaluation_result = None
    manager._evaluation_error = None
    manager._continuous_collection_end_step = None
    manager._collect_local_samples = lambda: torch.ones((1, 1, 2), dtype=torch.int64)
    manager._plan_and_broadcast = lambda _global_load: {
        "kind": "planned",
        "placement": torch.tensor([[[0], [1]]], dtype=torch.int64),
        "improved": torch.tensor([True]),
    }

    def fail_prepare(_layer_plans):
        raise RuntimeError("prepare failed")

    manager.transfer = SimpleNamespace(prepare_transfer=fail_prepare)
    monkeypatch.setattr(manager_module.dist, "all_reduce", lambda _tensor, **_kwargs: None)
    monkeypatch.setattr(manager_module.torch.cuda, "set_device", lambda _device: None)
    monkeypatch.setattr(manager_module, "build_transfer_plan", lambda *_args, **_kwargs: [])

    manager._evaluate_after_event(type("Event", (), {"synchronize": lambda self: None})())

    assert manager._evaluation_result is None
    assert isinstance(manager._evaluation_error, RuntimeError)
    assert str(manager._evaluation_error) == "prepare failed"


def test_decode_dispatch_keeps_logical_ids_and_uses_logical_expert_count(monkeypatch):
    class Buffer:
        def low_latency_dispatch(self, **kwargs):
            calls.append(kwargs)
            return "recv", "masked", "handle", "event", "hook"

    impl = object.__new__(deepgemm_module.FuseMoeDeepGEMM)
    impl.quant_method = type("Quant", (), {"method_name": "fp8"})()
    impl.n_routed_experts = 128
    _set_expert_parallel_state(impl, _test_parallel_state(eplb=True))
    impl._select_experts = lambda **_kwargs: (
        torch.ones((1, 2)),
        torch.tensor([[0, 127]], dtype=torch.int32),
        torch.tensor([[0, 127]], dtype=torch.int32),
    )
    calls = []
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

    assert result[2].tolist() == [[0, 127]]
    assert calls[0]["num_experts"] == 128


def test_decode_select_does_not_clone_or_map_eplb_topk_ids(monkeypatch):
    from lightllm.common.basemodel.triton_kernel.fused_moe import topk_select

    impl = object.__new__(deepgemm_module.FuseMoeDeepGEMM)
    impl.routed_scaling_factor = 1.0
    _set_expert_parallel_state(impl, _test_parallel_state(eplb=True))
    topk_ids = torch.tensor([[3, 127]], dtype=torch.int32)
    monkeypatch.setattr(topk_select, "select_experts", lambda **_kwargs: (torch.ones((1, 2)), topk_ids))
    _, selected, origin = impl._select_experts(
        torch.empty((1, 4)),
        torch.empty((1, 128)),
        None,
        2,
        False,
        False,
        0,
        0,
        "softmax",
        is_prefill=False,
    )

    assert selected.data_ptr() == origin.data_ptr()


def test_eplb_prefill_uses_single_fused_path_for_global_and_scale(monkeypatch):
    from lightllm.common.basemodel.triton_kernel.fused_moe import grouped_topk

    impl = object.__new__(deepgemm_module.FuseMoeDeepGEMM)
    impl.routed_scaling_factor = 1.0
    impl.quant_method = object()
    _set_expert_parallel_state(impl, _test_parallel_state(eplb=True))
    physical_ids = torch.tensor([[130, 131]], dtype=torch.long)
    calls = []

    def fused_topk(**kwargs):
        calls.append(kwargs)
        return torch.ones((1, 2)), physical_ids, None

    monkeypatch.setattr(grouped_topk, "triton_grouped_topk_eplb", fused_topk)
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
        per_expert_scale=torch.tensor([2.0]),
    )

    assert weights.tolist() == [[1.0, 1.0]]
    assert topk_idx is physical_ids
    assert topk_idx.dtype is torch.long
    assert qinput == "qinput"
    assert not calls[0]["use_grouped_topk"]
    assert calls[0]["per_expert_scale"].tolist() == [2.0]
    assert not calls[0]["return_logical_ids"]


def test_eplb_prefill_dispatch_consumes_physical_ids_and_event(monkeypatch):
    from lightllm.common.basemodel.triton_kernel.fused_moe import grouped_topk

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
    state = _test_parallel_state(
        eplb=True,
        route_counter=torch.zeros((3, 128), dtype=torch.int64),
        recording=True,
    )
    _set_expert_parallel_state(impl, state)
    impl.ep_balance_counters = None
    calls, fused_calls = [], []
    physical_ids = torch.tensor([[130, 131]], dtype=torch.long)

    def fused_topk(**kwargs):
        fused_calls.append(kwargs)
        return torch.ones((1, 2)), physical_ids, None

    monkeypatch.setattr(grouped_topk, "triton_grouped_topk_eplb", fused_topk)
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
    assert len(fused_calls) == 1
    assert fused_calls[0]["sample_index"] == 0
    assert fused_calls[0]["record_load"]
    assert state.eplb.recorded_sample_count == 1
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
    _set_expert_parallel_state(impl, _test_parallel_state(eplb=True))
    impl.ep_balance_counters = None
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


def test_deepgemm_constructor_configures_eplb():
    state = _validated_expert_parallel_state()
    impl = deepgemm_module.FuseMoeDeepGEMM(4, 0, 1.0, SimpleNamespace(), expert_parallel_state=state)
    assert impl.expert_parallel_state is state


def test_prefill_eplb_returns_requested_logical_ids(monkeypatch):
    from lightllm.common.basemodel.triton_kernel.fused_moe import grouped_topk

    impl = object.__new__(deepgemm_module.FuseMoeDeepGEMM)
    impl.routed_scaling_factor = 1.0
    _set_expert_parallel_state(impl, _test_parallel_state(eplb=True))

    def fused_topk(**kwargs):
        assert kwargs["return_logical_ids"]
        return (
            torch.ones((1, 2)),
            torch.tensor([[13, 14]], dtype=torch.int32),
            torch.tensor([[3, 4]], dtype=torch.int32),
        )

    monkeypatch.setattr(grouped_topk, "triton_grouped_topk_eplb", fused_topk)
    _, physical_ids, logical_ids = impl._select_experts(
        torch.empty((1, 4)),
        torch.empty((1, 128)),
        None,
        2,
        False,
        False,
        0,
        0,
        "softmax",
        is_prefill=True,
        preserve_logical_ids=True,
    )

    assert physical_ids.tolist() == [[13, 14]]
    assert logical_ids.tolist() == [[3, 4]]


def test_decode_masked_group_gemm_uses_primary_rows_only_when_eplb_is_enabled(
    monkeypatch,
):
    impl = object.__new__(deepgemm_module.FuseMoeDeepGEMM)
    _set_expert_parallel_state(impl, _test_parallel_state(eplb=True, num_logical_experts=8, world_size=1))
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
    assert captured["w13"].shape[0] == captured["w2"].shape[0] == 8
    assert captured["w13_scale"].shape[0] == captured["w2_scale"].shape[0] == 8


def test_decode_fused_experts_uses_cached_primary_weight_packs_and_logical_experts(
    monkeypatch,
):
    impl = object.__new__(deepgemm_module.FuseMoeDeepGEMM)
    impl.n_routed_experts = 128
    _set_expert_parallel_state(
        impl,
        _test_parallel_state(eplb=True, num_logical_experts=128, world_size=16, num_redundant_experts_per_rank=2),
    )
    impl.quant_method = object()
    impl.ep_balance_counters = None
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

    assert [call["num_experts"] for call in captured] == [128, 128]
    assert all(call["w13"].weight.shape[0] == call["w2"].weight.shape[0] == 8 for call in captured)
    assert captured[0]["w13"] is captured[1]["w13"]
    assert captured[0]["w2"] is captured[1]["w2"]


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


def test_extract_expert_tensors_includes_weight_scale_and_zero_point_in_order():
    class Pack:
        def __init__(self, offset, scale=True, zero=True):
            self.weight = torch.full((3, 2), offset)
            self.weight_scale = torch.full((3, 1), offset + 1) if scale else None
            self.weight_zero_point = torch.full((3, 1), offset + 2) if zero else None

    weight = type("Weight", (), {"w13": Pack(1), "w2": Pack(10, scale=False, zero=False)})()
    tensors = extract_expert_tensors(weight)
    assert [name for name, _ in tensors] == [
        "w13.weight",
        "w13.weight_scale",
        "w13.weight_zero_point",
        "w2.weight",
    ]


def test_commit_staging_rows_only_overwrites_redundant_rows():
    live = torch.arange(20).reshape(5, 4)
    staging = torch.full((2, 4), -1)
    commit_staging_rows(
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

    commit_staging_rows(
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

    commit_staging_rows(
        Tensor("live", 20),
        Tensor("staging", 4),
        num_experts_per_rank=10,
        changed_dst_slots=(3, 1, 2),
    )

    assert copies == [("live", 11, 3, "staging", 1, 3)]


def test_manager_inflight_ready_gate_commits_ordered_prefix_and_propagates_worker_error(
    monkeypatch,
):
    class Transfer:
        def __init__(self):
            self.pending = [(0, 0), (1, 1), (2, 2)]
            self.commits = []
            self.finished = 0

        def pending_layers(self):
            return self.pending

        def commit(self, layer, buffer_index, post_copy=None):
            assert self.pending.pop(0) == (layer, buffer_index)
            self.commits.append((layer, buffer_index))
            if post_copy is not None:
                post_copy()

        def finish(self):
            self.finished += 1

    manager = manager_module.EPLBManager.__new__(manager_module.EPLBManager)
    manager.transfer = Transfer()
    manager.in_flight = True
    manager.world_size = 2
    manager.control_group = object()
    manager._control_ready_count = torch.empty(1, dtype=torch.int32)
    manager.in_flight_layers = [0, 1, 2]
    manager._commit_layer_metadata = lambda layer: committed.append(layer)
    manager._finish_rebalance = lambda: finished.append(True)
    committed, finished = [], []
    operations = []
    current_stream_calls = []

    class CurrentStream:
        def wait_stream(self, stream):
            operations.append(("wait", stream))

    overlap_stream = object()

    def current_stream():
        current_stream_calls.append(True)
        return CurrentStream()

    monkeypatch.setattr(manager_module.torch.cuda, "current_stream", current_stream)
    monkeypatch.setattr(g_infer_context, "get_overlap_stream", lambda: overlap_stream)

    original_commit = manager.transfer.commit

    def record_commit(*args, **kwargs):
        operations.append(("commit", args[0]))
        return original_commit(*args, **kwargs)

    manager.transfer.commit = record_commit

    def set_global_ready(count):
        return lambda tensor, **kwargs: tensor.fill_(count)

    monkeypatch.setattr(manager_module.dist, "all_reduce", set_global_ready(0))
    manager._poll_in_flight()
    assert manager.transfer.commits == []
    assert operations == []
    assert current_stream_calls == []

    # Local rank has three prefetched layers, but global MIN-ready only permits two.
    monkeypatch.setattr(manager_module.dist, "all_reduce", set_global_ready(2))
    manager._poll_in_flight()
    assert manager.transfer.commits == [(0, 0), (1, 1)]
    assert committed == [0, 1]
    assert not finished
    assert manager.transfer.finished == 0
    assert operations == [("wait", overlap_stream), ("commit", 0), ("commit", 1)]
    assert current_stream_calls == [True]

    monkeypatch.setattr(manager_module.dist, "all_reduce", set_global_ready(1))
    manager._poll_in_flight()
    assert finished == [True]
    assert manager.transfer.finished == 1
    assert operations == [
        ("wait", overlap_stream),
        ("commit", 0),
        ("commit", 1),
        ("wait", overlap_stream),
        ("commit", 2),
    ]
    assert current_stream_calls == [True, True]

    manager.in_flight_layers = [3]
    manager.transfer.pending = [(9, 0)]
    monkeypatch.setattr(manager_module.dist, "all_reduce", set_global_ready(1))
    with pytest.raises(RuntimeError, match="does not match expected"):
        manager._poll_in_flight()

    class BrokenTransfer:
        def pending_layers(self):
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

        def pending_layers(self):
            return [(0, 0)]

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
            self.pending = [(0, 0)]

        def pending_layers(self):
            return self.pending

        def commit(self, layer, buffer_index, post_copy=None):
            assert self.pending.pop(0) == (layer, buffer_index)
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


def test_transfer_ring_reuses_a_buffer_only_after_commit_and_consumption(monkeypatch):
    operations = []

    class Event:
        def __init__(self):
            self.recorded = 0
            self.synchronized = 0

        def record(self, stream):
            self.recorded += 1

        def synchronize(self):
            self.synchronized += 1
            operations.append("consumed synchronize")

    transfer = object.__new__(transfer_module._EPLBTransferBase)
    transfer.backend = "test"
    transfer.device = torch.device("cuda", 0)
    transfer.staging_depth = 2
    transfer.staging = [[], []]
    transfer.live = [[], [], []]
    transfer.num_experts_per_rank = 0
    transfer._release = [threading.Event(), threading.Event()]
    for release in transfer._release:
        release.set()
    transfer._consumed_events = [Event(), Event()]
    transfer._consumed_recorded = [False, False]
    transfer._changed_dst_slots = [(), ()]
    transfer._pending = deque()
    transfer._pending_lock = threading.Lock()
    transfer._error = None
    transfer._thread = None
    transfer._needs_staging_reuse_barrier = True
    transfer.transfer_group = "transfer-group"
    copied = []

    def copy_layer(layer, _plan, _staging):
        copied.append(layer)
        operations.append(("copy", layer))

    transfer._copy_layer = copy_layer
    monkeypatch.setattr(transfer_module.torch.cuda, "set_device", lambda device: None)
    monkeypatch.setattr(transfer_module.torch.cuda, "Event", Event)
    monkeypatch.setattr(transfer_module.torch.cuda, "current_stream", lambda: object())
    monkeypatch.setattr(
        transfer_module.dist,
        "barrier",
        lambda **kwargs: operations.append(("barrier", kwargs["group"])),
    )

    transfer.start([(0, []), (1, []), (2, [])])
    deadline = time.monotonic() + 2
    while len(transfer.pending_layers()) < 2 and time.monotonic() < deadline:
        time.sleep(0.001)
    assert transfer.pending_layers() == [(0, 0), (1, 1)]
    assert copied == [0, 1]
    assert operations == [("copy", 0), ("copy", 1)]

    transfer.commit(0, 0)
    while len(transfer.pending_layers()) < 2 and time.monotonic() < deadline:
        time.sleep(0.001)
    assert transfer.pending_layers() == [(1, 1), (2, 0)]
    assert copied == [0, 1, 2]
    assert transfer._consumed_events[0].synchronized == 1
    assert operations == [
        ("copy", 0),
        ("copy", 1),
        "consumed synchronize",
        ("barrier", "transfer-group"),
        ("copy", 2),
    ]
    transfer.commit(1, 1)
    transfer.commit(2, 0)
    transfer.finish()


def test_transfer_finalization_failure_stays_in_worker_and_success_finalizes_once(
    monkeypatch,
):
    def make_transfer(finalize):
        transfer = object.__new__(transfer_module._EPLBTransferBase)
        transfer.backend = "test"
        transfer.device = torch.device("cuda", 0)
        transfer.global_rank = 0
        transfer.staging_depth = 1
        transfer.staging = [[]]
        transfer._release = [threading.Event()]
        transfer._release[0].set()
        transfer._consumed_events = [object()]
        transfer._consumed_recorded = [False]
        transfer._changed_dst_slots = [()]
        transfer._pending = deque()
        transfer._pending_lock = threading.Lock()
        transfer._error = None
        transfer._thread = None
        transfer._needs_staging_reuse_barrier = False
        transfer._copy_batch = lambda _batch, _prepared_batch: None
        transfer._finish_transfer_generation = finalize
        return transfer

    monkeypatch.setattr(transfer_module.torch.cuda, "set_device", lambda _device: None)

    finalized_before_publish = []
    success = make_transfer(lambda: finalized_before_publish.append(len(success._pending)))
    success.start([(0, [])])
    success.finish()
    assert finalized_before_publish == [0]
    assert success.pending_layers() == [(0, 0)]

    failed = make_transfer(lambda: (_ for _ in ()).throw(RuntimeError("cache boom")))
    failed.start([(0, [])])
    failed._thread.join()
    assert list(failed._pending) == []
    with pytest.raises(RuntimeError, match="EPLB migration worker failed") as exc_info:
        failed.pending_layers()
    assert isinstance(exc_info.value.__cause__, RuntimeError)
    assert str(exc_info.value.__cause__) == "cache boom"
    with pytest.raises(RuntimeError, match="EPLB migration worker failed"):
        failed.finish()
    assert failed._thread is None


def test_manager_rearms_after_rebalance_for_interval_one():
    recording_calls = []
    manager = manager_module.EPLBManager.__new__(manager_module.EPLBManager)
    manager.step_interval = 1
    manager.sampling_interval = 1
    manager._sampling_pending = False
    manager._continuous_collection_start_step = 0
    manager.weights = []
    manager._eplb_states = []
    manager.target_placement = torch.zeros(1)
    manager.in_flight_started_at = 0
    manager.global_rank = 1
    manager._set_recording = lambda enabled: recording_calls.append(enabled)
    manager._finish_rebalance()
    assert manager.in_flight is False
    assert recording_calls == [True]
    assert not manager._sampling_pending
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
    manager._eplb_states = []
    manager._continuous_collection_start_step = None
    manager._continuous_collection_end_step = None
    recordings, resets, logs = [], [], []
    manager._set_recording = lambda enabled: recordings.append(enabled)
    manager._reset_recorded_samples = lambda: resets.append(True)
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
    manager._eplb_states = []
    manager._continuous_collection_start_step = 40
    manager._continuous_collection_end_step = 60
    recordings, resets = [], []
    manager._set_recording = lambda enabled: recordings.append(enabled)
    manager._reset_recorded_samples = lambda: resets.append(True)

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
    manager._sampling_pending = True
    recordings, resets, starts = [], [], []
    manager._set_recording = lambda enabled: recordings.append(enabled)
    manager._reset_recorded_samples = lambda: resets.append(True)
    monkeypatch.setattr(manager, "_start_evaluation", lambda: starts.append(True))

    # The window is never truncated to the next boundary: it waits until 40,
    # records a full 20 fresh steps, then evaluates at the boundary at 60.
    manager._begin_continuous_collection()
    assert manager._continuous_collection_start_step == 40
    assert manager._continuous_collection_end_step == 60
    assert recordings == [False]
    assert resets == [True]
    assert not manager._sampling_pending

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
    manager._sampling_pending = False
    recordings = []
    manager._set_recording = lambda enabled: recordings.append(enabled)
    manager._reset_recorded_samples = lambda: None

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
    manager._eplb_states = []
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
    manager._sampling_pending = False
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
    manager._eplb_states = []

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
    manager._sampling_pending = False
    manager._continuous_collection_start_step = None
    manager._continuous_collection_end_step = None
    manager.evaluation_in_flight = False
    recordings, resets, starts = [], [], []
    manager._set_recording = lambda enabled: recordings.append(enabled)
    manager._reset_recorded_samples = lambda: resets.append(True)
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
    assert manager._sampling_pending

    for _ in range(4):
        manager.step()
    assert manager.prefill_steps == 80
    assert starts == [True]
    assert not manager._sampling_pending


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
        {"start": lambda self, plans, prepared_batches: setattr(self, "started", (plans, prepared_batches))},
    )()
    manager._reset_recorded_samples = lambda: None

    prepared_batches = [object()]
    manager._start_rebalance(
        {
            "placement": torch.zeros((1, 1, 1), dtype=torch.int64),
            "improved": torch.tensor([True]),
            "metadata": [None],
            "layer_plans": [(0, object())],
            "prepared_batches": prepared_batches,
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
    assert len(manager.transfer.started[0]) == 1
    assert manager.transfer.started[1] is prepared_batches


def test_transfer_start_rejects_prepared_batches_with_wrong_batch_count():
    transfer = object.__new__(transfer_module._EPLBTransferBase)
    transfer._thread = None
    transfer.staging_depth = 2
    transfer.staging = [object(), object()]

    with pytest.raises(ValueError, match="prepared batch count"):
        transfer.start([(0, []), (1, []), (2, [])], prepared_batches=[object()])


def test_first_rebalance_completion_switches_to_four_step_sparse_window(monkeypatch):
    manager = manager_module.EPLBManager.__new__(manager_module.EPLBManager)
    manager.step_interval = 20
    manager.sampling_interval = manager.step_interval
    manager._sampling_pending = False
    manager.weights = []
    manager._eplb_states = []
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


def test_nixl_descriptor_runs_merge_only_jointly_contiguous_source_and_destination_rows():
    steps = [
        TransferStep(0, 2, 1, 3),
        TransferStep(0, 0, 1, 1),
        TransferStep(0, 1, 1, 2),
        TransferStep(0, 4, 1, 7),
    ]
    runs = transfer_module.NixlEPLBTransfer._contiguous_runs(steps)
    assert [[(step.dst_slot, step.src_local_row) for step in run] for run in runs] == [
        [(0, 1), (1, 2), (2, 3)],
        [(4, 7)],
    ]


def test_nixl_prepare_batch_compiles_hot_path_without_tensor_views(monkeypatch):
    @contextmanager
    def use_stream(stream):
        yield

    class BatchMemcpy:
        def __init__(self):
            self.prepared = []
            self.enqueued = []

        def prepare(self, descriptors):
            descriptor = tuple(descriptors)
            self.prepared.append(descriptor)
            return descriptor

        def enqueue(self, descriptor, stream):
            self.enqueued.append((descriptor, stream))

    stream = SimpleNamespace(cuda_stream=123, synchronize=lambda: None)
    batch_memcpy = BatchMemcpy()
    transfer = object.__new__(transfer_module.NixlEPLBTransfer)
    transfer._push_stream = stream
    transfer._batch_memcpy = batch_memcpy
    transfer.global_rank = 0
    transfer._same_node_ranks = {0, 1, 2}
    transfer._live_row_layout = [
        [
            ("w13.weight", 1000, 32),
            ("w13.weight_scale", 2000, 32),
            ("w13.weight_zero_point", 3000, 32),
        ]
    ]
    transfer._push_staging_row_layout = {
        1: [[("w13.weight", 4000, 32), ("w13.weight_scale", 5000, 32), ("w13.weight_zero_point", 6000, 32)]],
        2: [[("w13.weight", 7000, 32), ("w13.weight_scale", 8000, 32), ("w13.weight_zero_point", 9000, 32)]],
    }
    transfer._get_remote_read = lambda *_args: None
    transfer._wait_xfers = lambda _xfers: None
    monkeypatch.setattr(transfer_module.torch.cuda, "stream", use_stream)

    run_a = [TransferStep(1, 3, 0, 5), TransferStep(1, 4, 0, 6)]
    run_b = [TransferStep(2, 1, 0, 2)]
    batch = [(0, run_a + run_b, 0, object())]
    prepared = transfer._prepare_batch(batch)
    monkeypatch.setattr(transfer, "_prepare_batch", lambda _batch: pytest.fail("hot path must not prepare descriptors"))
    transfer._copy_batch(batch, prepared)

    expected = (
        (1160, 4096, 64),
        (2160, 5096, 64),
        (3160, 6096, 64),
        (1064, 7032, 32),
        (2064, 8032, 32),
        (3064, 9032, 32),
    )
    assert batch_memcpy.prepared == [expected]
    assert batch_memcpy.enqueued == [(expected, 123)]


def test_nixl_prepare_transfer_batches_match_staging_depth():
    transfer = object.__new__(transfer_module.NixlEPLBTransfer)
    transfer.staging_depth = 2
    transfer.staging = ["staging-0", "staging-1"]
    seen_batches = []

    def prepare_batch(batch):
        seen_batches.append(batch)
        return f"prepared-{len(seen_batches)}"

    transfer._prepare_batch = prepare_batch
    layer_plans = [(3, "plan-3"), (4, "plan-4"), (5, "plan-5")]

    prepared_batches = transfer.prepare_transfer(layer_plans)

    assert prepared_batches == ["prepared-1", "prepared-2"]
    assert [[(layer, buffer) for layer, _plan, buffer, _staging in batch] for batch in seen_batches] == [
        [(3, 0), (4, 1)],
        [(5, 0)],
    ]


def test_cuda_batch_memcpy_cuda13_abi_and_descriptor_layout():
    class Function:
        def __init__(self, callback):
            self.callback = callback
            self.restype = None
            self.argtypes = None

        def __call__(self, *args):
            return self.callback(*args)

    class Library:
        def __init__(self):
            def get_version(pointer):
                ctypes.cast(pointer, ctypes.POINTER(ctypes.c_int))[0] = 13000
                return 0

            self.cudaRuntimeGetVersion = Function(get_version)
            self.cudaMemcpyBatchAsync = Function(lambda *args: self.calls.append(args) or 0)
            self.cudaGetErrorString = Function(lambda _result: b"fake cuda error")
            self.calls = []

    import ctypes

    library = Library()
    batch_memcpy = CudaBatchMemcpy(library)
    prepared = batch_memcpy.prepare(((101, 201, 64), (102, 202, 128)))
    batch_memcpy.enqueue(prepared, 777)

    assert len(library.cudaMemcpyBatchAsync.argtypes) == 8
    assert library.calls[0][3] == 2
    assert library.calls[0][6] == 1
    assert library.calls[0][7].value == 777
    assert [pointer for pointer in library.calls[0][0]] == [201, 202]
    assert [pointer for pointer in library.calls[0][1]] == [101, 102]
    assert list(library.calls[0][2]) == [64, 128]
    attrs = library.calls[0][4]._obj
    assert attrs.srcAccessOrder == 1
    assert attrs.srcLocHint.type == attrs.srcLocHint.id == 0
    assert attrs.dstLocHint.type == attrs.dstLocHint.id == 0
    assert attrs.flags == 1


def test_cuda_batch_memcpy_rejects_unsupported_runtime_and_invalid_descriptors():
    class Function:
        def __init__(self, callback):
            self.callback = callback
            self.restype = None
            self.argtypes = None

        def __call__(self, *args):
            return self.callback(*args)

    class OldRuntimeLibrary:
        def __init__(self):
            def get_version(pointer):
                ctypes.cast(pointer, ctypes.POINTER(ctypes.c_int))[0] = 12080
                return 0

            self.cudaRuntimeGetVersion = Function(get_version)
            self.cudaMemcpyBatchAsync = Function(lambda *_args: 0)
            self.cudaGetErrorString = Function(lambda _result: b"fake cuda error")

    import ctypes

    with pytest.raises(CudaBatchMemcpyUnavailable, match="13.0"):
        CudaBatchMemcpy(OldRuntimeLibrary())

    class FutureRuntimeLibrary:
        def __init__(self):
            def get_version(pointer):
                ctypes.cast(pointer, ctypes.POINTER(ctypes.c_int))[0] = 14000
                return 0

            self.cudaRuntimeGetVersion = Function(get_version)
            self.cudaMemcpyBatchAsync = Function(lambda *_args: 0)
            self.cudaGetErrorString = Function(lambda _result: b"fake cuda error")

    with pytest.raises(CudaBatchMemcpyUnavailable, match="13.x"):
        CudaBatchMemcpy(FutureRuntimeLibrary())

    class MissingBatchSymbolLibrary:
        def __init__(self):
            def get_version(pointer):
                ctypes.cast(pointer, ctypes.POINTER(ctypes.c_int))[0] = 13000
                return 0

            self.cudaRuntimeGetVersion = Function(get_version)
            self.cudaGetErrorString = Function(lambda _result: b"fake cuda error")

    with pytest.raises(CudaBatchMemcpyUnavailable, match="cudaMemcpyBatchAsync"):
        CudaBatchMemcpy(MissingBatchSymbolLibrary())
    with pytest.raises(ValueError, match="at least one"):
        CudaBatchMemcpy.prepare(())
    with pytest.raises(ValueError, match="positive"):
        CudaBatchMemcpy.prepare(((1, 2, 0),))


def test_nixl_transfer_fails_fast_without_cuda13_batch_memcpy(monkeypatch):
    def unavailable():
        raise CudaBatchMemcpyUnavailable("missing cudaMemcpyBatchAsync")

    monkeypatch.setattr(transfer_module, "CudaBatchMemcpy", unavailable)
    with pytest.raises(RuntimeError, match="CUDA Runtime 13.x"):
        transfer_module.NixlEPLBTransfer._require_batch_memcpy()


@pytest.mark.parametrize(("layers", "expected_depth"), [(1, 1), (8, 8), (9, 8), (43, 8)])
def test_nixl_transfer_bounds_staging_depth(monkeypatch, layers, expected_depth):
    def base_init(self, *_args):
        self.device = "mock-device"

    monkeypatch.setattr(transfer_module._EPLBTransferBase, "__init__", base_init)
    monkeypatch.setattr(transfer_module.torch.cuda, "Stream", lambda device: ("stream", device))
    monkeypatch.setattr(transfer_module.NixlEPLBTransfer, "_require_batch_memcpy", staticmethod(lambda: object()))
    monkeypatch.setattr(transfer_module.NixlEPLBTransfer, "_init_ipc_metadata", lambda self: None)
    monkeypatch.setattr(transfer_module.NixlEPLBTransfer, "_init_push_layouts", lambda self: None)

    transfer = transfer_module.NixlEPLBTransfer([object()] * layers, object(), 0, 1)

    assert transfer.staging_depth == expected_depth


def test_nixl_remote_read_cache_reuses_exact_batch_key_and_releases_on_shutdown():
    class Tensor:
        nbytes = 8

        def __init__(self, pointer):
            self.pointer = pointer

        def data_ptr(self):
            return self.pointer

        def get_device(self):
            return 0

        def __getitem__(self, _):
            return self

    class Agent:
        def __init__(self):
            self.prepared = 0
            self.made = 0
            self.released_xfers = 0
            self.released_dlists = 0
            self.removed_agents = []

        def get_xfer_descs(self, descriptors, _):
            return descriptors

        def prep_xfer_dlist(self, *_args, **_kwargs):
            self.prepared += 1
            return f"dlist-{self.prepared}"

        def make_prepped_xfer(self, *_args, **_kwargs):
            self.made += 1
            return f"xfer-{self.made}"

        def query_xfer_backend(self, _):
            return "UCX"

        def release_xfer_handle(self, _):
            self.released_xfers += 1

        def release_dlist_handle(self, _):
            self.released_dlists += 1

        def remove_remote_agent(self, remote_name):
            self.removed_agents.append(remote_name)

    agent = Agent()
    transfer = object.__new__(transfer_module.NixlEPLBTransfer)
    transfer._nixl_agent = agent
    transfer._xfer_cache = {}
    transfer._used_xfer_cache_keys = set()
    transfer._remote_agents = {1: "remote-1"}
    transfer._remote_layouts = {1: [[("weight", 1000, 0, 8)]]}
    transfer._registered_descs = None
    transfer.live = [[("weight", Tensor(100))]]
    staging = [("weight", Tensor(200))]
    first_entries = [(0, [TransferStep(0, 0, 1, 0)], staging)]

    first = transfer._get_remote_read(1, first_entries)
    assert transfer._get_remote_read(1, first_entries) is first
    assert agent.made == 1

    changed_entries = [(0, [TransferStep(0, 1, 1, 0)], staging)]
    transfer._get_remote_read(1, changed_entries)
    assert agent.made == 2

    transfer.shutdown()
    assert agent.released_xfers == 2
    assert agent.released_dlists == 4
    assert agent.removed_agents == ["remote-1"]


def test_nixl_remote_read_cache_is_bounded_to_the_current_transfer_generation(
    monkeypatch,
):
    class Tensor:
        nbytes = 8

        def __init__(self, pointer):
            self.pointer = pointer

        def data_ptr(self):
            return self.pointer

        def get_device(self):
            return 0

        def __getitem__(self, _):
            return self

    class Agent:
        def __init__(self):
            self.made = 0
            self.released_xfers = 0
            self.released_dlists = 0

        def get_xfer_descs(self, descriptors, _):
            return descriptors

        def prep_xfer_dlist(self, *_args, **_kwargs):
            return object()

        def make_prepped_xfer(self, *_args, **_kwargs):
            self.made += 1
            return object()

        def query_xfer_backend(self, _):
            return "UCX"

        def release_xfer_handle(self, _):
            self.released_xfers += 1

        def release_dlist_handle(self, _):
            self.released_dlists += 1

        def remove_remote_agent(self, _):
            pass

    agent = Agent()
    transfer = object.__new__(transfer_module.NixlEPLBTransfer)
    transfer._nixl_agent = agent
    transfer._xfer_cache = {}
    transfer._used_xfer_cache_keys = set()
    transfer._remote_agents = {1: "remote-1"}
    transfer._remote_layouts = {1: [[("weight", 1000, 0, 8)]]}
    transfer._registered_descs = None
    transfer._ipc_staging = {}
    transfer.live = [[("weight", Tensor(100))]]
    transfer.device = torch.device("cuda", 0)
    transfer.transfer_group = object()
    transfer.global_rank = 0
    transfer.world_size = 2
    transfer.staging_depth = 1
    transfer.staging = [[]]
    transfer.num_experts_per_rank = 0
    transfer._release = [threading.Event()]
    transfer._release[0].set()
    transfer._consumed_events = [object()]
    transfer._consumed_recorded = [False]
    transfer._changed_dst_slots = [()]
    transfer._pending = deque()
    transfer._pending_lock = threading.Lock()
    transfer._error = None
    transfer._thread = None
    transfer._needs_staging_reuse_barrier = False

    staging = [("weight", Tensor(200))]
    entries_a = [(0, [TransferStep(0, 0, 1, 0)], staging)]
    entries_b = [(0, [TransferStep(0, 1, 1, 0)], staging)]
    generation = [entries_a]

    def copy_batch(_batch, _prepared_batch):
        transfer._get_remote_read(1, generation[0])

    transfer._copy_batch = copy_batch
    monkeypatch.setattr(transfer_module.torch.cuda, "set_device", lambda _device: None)

    transfer.start([(0, [])], [None])
    transfer.finish()
    assert agent.made == 1
    assert agent.released_xfers == agent.released_dlists == 0
    assert len(transfer._xfer_cache) == 1

    # The real manager releases each staging buffer through commit(). This
    # focused cache test has no commits, so model that hand-off before the
    # next generation reuses buffer zero.
    transfer._release[0].set()
    transfer.start([(0, [])], [None])
    transfer.finish()
    assert agent.made == 1
    assert agent.released_xfers == agent.released_dlists == 0
    assert len(transfer._xfer_cache) == 1

    generation[:] = [entries_b]
    transfer._release[0].set()
    transfer.start([(0, [])], [None])
    transfer.finish()
    assert agent.made == 2
    assert agent.released_xfers == 1
    assert agent.released_dlists == 2
    assert len(transfer._xfer_cache) == 1
    transfer.shutdown()


def test_nixl_ipc_metadata_exports_staging_per_local_target(monkeypatch):
    from lightllm.server.router.model_infer.mode_backend.pd import p2p_fix

    class Tensor:
        shape = (4, 2)
        dtype = torch.float16
        device = torch.device("cuda", 0)
        nbytes = 16

        def __init__(self, label):
            self.label = label

        def numel(self):
            return 3

        def __getitem__(self, _index):
            return self

    transfer = object.__new__(transfer_module.NixlEPLBTransfer)
    transfer.transfer_group = object()
    transfer.global_rank = 0
    transfer.world_size = 3
    transfer.device = torch.device("cuda", 0)
    transfer.live = [[("w13.weight", Tensor("local-w13")), ("w2.weight", Tensor("local-w2"))]]
    transfer.staging_depth = 8
    transfer.staging = [[("w13.weight", Tensor(f"local-staging-{index}"))] for index in range(8)]
    transfer._ipc_staging = {}

    reduce_calls, rebuild_calls, gathers = [], [], []

    def reduce_tensor(tensor):
        reduce_calls.append(tensor.label)
        return None, (f"export-{tensor.label}",)

    def rebuild_tensor(export):
        rebuild_calls.append(export)
        return Tensor(export)

    source_one = [[("w13.weight", (4, 2), torch.float16, (f"rank1-staging-{index}",))] for index in range(8)]

    def all_gather(output, value, **_kwargs):
        gathers.append(value)
        if len(gathers) == 1:
            output[:] = ["node-a", "node-a", "node-b"]
        else:
            output[:] = [value, {0: {"staging": source_one}}, {}]

    monkeypatch.setattr(p2p_fix, "reduce_tensor", reduce_tensor)
    monkeypatch.setattr(p2p_fix, "p2p_fix_rebuild_cuda_tensor", rebuild_tensor)
    monkeypatch.setattr(transfer_module.dist, "all_gather_object", all_gather)
    monkeypatch.setattr(transfer_module.torch.cuda, "set_device", lambda _device: None)

    transfer._init_ipc_metadata()

    assert reduce_calls == [*(f"local-staging-{index}" for index in range(8))]
    assert rebuild_calls == [*(f"rank1-staging-{index}" for index in range(8))]
    assert transfer._same_node_ranks == {0, 1}
    assert transfer._cross_node_ranks == {2}
    assert transfer._needs_staging_reuse_barrier
    assert [name for name, _ in transfer._ipc_staging[1][0]] == ["w13.weight"]


def test_nixl_copy_batch_source_pushes_local_rows_and_keeps_remote_ucx_reads(
    monkeypatch,
):
    class Stream:
        def __init__(self):
            self.synchronized = 0
            self.cuda_stream = 123

        def synchronize(self):
            self.synchronized += 1

    @contextmanager
    def use_stream(_stream):
        yield

    stream = Stream()
    transfer = object.__new__(transfer_module.NixlEPLBTransfer)
    transfer._push_stream = stream
    transfer._batch_memcpy = SimpleNamespace(enqueue=lambda descriptor, stream: enqueued.append((descriptor, stream)))
    remote_reads, waited_xfers, enqueued = [], [], []
    transfer._get_remote_read = lambda src_rank, entries: remote_reads.append((src_rank, entries)) or (
        None,
        None,
        "xfer",
    )
    transfer._wait_xfers = waited_xfers.extend

    monkeypatch.setattr(transfer_module.torch.cuda, "stream", use_stream)
    prepared_push = transfer_module.NixlEPLBTransfer._PreparedBatch({}, "push")
    transfer._copy_batch([], prepared_push)
    assert enqueued == [("push", stream.cuda_stream)]

    remote_step = TransferStep(0, 1, 2, 0)
    prepared_remote = transfer_module.NixlEPLBTransfer._PreparedBatch({2: [(0, [remote_step], [])]}, None)
    transfer._copy_batch([], prepared_remote)
    assert [rank for rank, _ in remote_reads] == [2]
    assert waited_xfers == [(None, None, "xfer")]


def test_nixl_copy_batch_self_only_rank_pushes_and_synchronizes(monkeypatch):
    class Stream:
        def __init__(self):
            self.synchronized = 0
            self.cuda_stream = 456

        def synchronize(self):
            self.synchronized += 1

    @contextmanager
    def use_stream(_stream):
        yield

    transfer = object.__new__(transfer_module.NixlEPLBTransfer)
    transfer._push_stream = Stream()
    enqueued = []
    transfer._batch_memcpy = SimpleNamespace(enqueue=lambda descriptor, stream: enqueued.append((descriptor, stream)))
    transfer._wait_xfers = lambda _xfers: None
    monkeypatch.setattr(transfer_module.torch.cuda, "stream", use_stream)

    prepared = transfer_module.NixlEPLBTransfer._PreparedBatch({}, "push")
    transfer._copy_batch([], prepared)

    assert enqueued == [("push", transfer._push_stream.cuda_stream)]
    assert transfer._push_stream.synchronized == 1


def test_manager_constructs_nixl_transfer(monkeypatch):
    weight = type(
        "Weight",
        (),
        {
            "n_routed_experts": 4,
            "expert_parallel_state": _test_parallel_state(
                eplb=True,
                num_logical_experts=4,
                world_size=2,
                num_redundant_experts_per_rank=2,
                route_counter=torch.zeros((2, 4), dtype=torch.int64),
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
        "NixlEPLBTransfer",
        lambda weights, group, rank, world_size: (
            transfer_calls.append((weights, group, rank, world_size)) or transfer
        ),
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
    assert transfer_calls == [([weight], groups[2], 0, 2)]
    assert manager.rebalance_gain_threshold == 0.07
    assert "rebalance_gain_threshold=0.0700" in logs[0]
    assert manager._continuous_collection_start_step is None
    assert manager._continuous_collection_end_step == manager.step_interval
    assert weight.expert_parallel_state.eplb.recording
    assert manager._eplb_states[0] is weight.expert_parallel_state.eplb
    assert not hasattr(weight.expert_parallel_state.eplb, "record_load")


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required for the Triton EPLB kernel")
@pytest.mark.parametrize("record_load", [False, True])
@pytest.mark.parametrize("tokens", [1, 32])
@pytest.mark.parametrize("scoring_func", ["sigmoid", "softmax"])
@pytest.mark.parametrize("renormalize", [False, True])
def test_grouped_topk_eplb_matches_topk_mapping_and_counting(record_load, tokens, scoring_func, renormalize):
    from lightllm.common.basemodel.triton_kernel.fused_moe.grouped_topk import (
        triton_grouped_topk,
        triton_grouped_topk_eplb,
    )

    torch.manual_seed(1234)
    topk = 8
    experts = 256
    num_expert_group = 8
    gating_output = torch.randn((tokens, experts), dtype=torch.bfloat16, device="cuda")
    correction_bias = torch.randn((experts,), dtype=torch.float32, device="cuda")
    hidden_states = torch.empty((tokens, 1), dtype=torch.bfloat16, device="cuda")
    logical_to_physical = torch.stack(
        (
            torch.arange(experts, dtype=torch.int32, device="cuda"),
            torch.arange(experts, dtype=torch.int32, device="cuda") + experts,
        ),
        dim=1,
    )
    logical_replica_count = torch.where(
        torch.arange(experts, device="cuda") % 3 == 0,
        torch.full((experts,), 2, dtype=torch.int32, device="cuda"),
        torch.ones((experts,), dtype=torch.int32, device="cuda"),
    )
    expected_counter = torch.zeros((2, experts), dtype=torch.int64, device="cuda")
    fused_counter = torch.zeros_like(expected_counter)

    expected_weights, logical_ids = triton_grouped_topk(
        hidden_states,
        gating_output,
        correction_bias,
        topk,
        renormalize,
        num_expert_group,
        4,
        scoring_func,
        2,
    )
    if tokens == 1:
        replica_indices = torch.zeros_like(logical_ids)
    else:
        token_indices = torch.arange(tokens, device="cuda", dtype=torch.int64).unsqueeze(1)
        replica_indices = (
            (((token_indices * 2654435769) & 0xFFFFFFFF) + ((logical_ids.to(torch.int64) * 2246822519) & 0xFFFFFFFF))
            & 0xFFFFFFFF
        ) % logical_replica_count[logical_ids.to(torch.long)].to(torch.int64)
    expected_ids = logical_to_physical[logical_ids.to(torch.long), replica_indices.to(torch.long)].to(torch.long)
    if record_load:
        expected_counter[1].scatter_add_(
            0,
            logical_ids.reshape(-1).to(torch.long),
            torch.ones(logical_ids.numel(), dtype=torch.int64, device="cuda"),
        )
    fused_weights, fused_ids, fused_logical_ids = triton_grouped_topk_eplb(
        hidden_states,
        gating_output,
        correction_bias,
        topk,
        renormalize,
        num_expert_group,
        4,
        scoring_func,
        logical_to_physical,
        logical_replica_count,
        fused_counter,
        sample_index=1,
        record_load=record_load,
        use_grouped_topk=True,
        group_score_used_topk_num=2,
    )
    torch.cuda.synchronize()

    torch.testing.assert_close(fused_weights, expected_weights, rtol=1e-5, atol=1e-6)
    assert torch.equal(fused_ids, expected_ids)
    assert fused_logical_ids is None
    assert torch.equal(fused_counter, expected_counter)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required for the Triton EPLB kernel")
@pytest.mark.parametrize("record_load", [False, True])
@pytest.mark.parametrize("tokens", [1, 32])
def test_global_topk_eplb_supports_scale_logical_ids_and_counting(record_load, tokens):
    from lightllm.common.basemodel.triton_kernel.fused_moe.grouped_topk import triton_grouped_topk_eplb

    torch.manual_seed(1234)
    topk = 4
    experts = 64
    gating_output = torch.randn((tokens, experts), dtype=torch.float32, device="cuda")
    scale = torch.linspace(0.5, 1.5, experts, dtype=torch.float32, device="cuda")
    logical_to_physical = torch.stack(
        (
            torch.arange(experts, dtype=torch.int32, device="cuda"),
            torch.arange(experts, dtype=torch.int32, device="cuda") + experts,
        ),
        dim=1,
    )
    logical_replica_count = torch.where(
        torch.arange(experts, device="cuda") % 3 == 0,
        torch.full((experts,), 2, dtype=torch.int32, device="cuda"),
        torch.ones((experts,), dtype=torch.int32, device="cuda"),
    )
    expected_counter = torch.zeros((2, experts), dtype=torch.int64, device="cuda")
    fused_counter = torch.zeros_like(expected_counter)
    expected_weights, expected_logical_ids = torch.softmax(gating_output, dim=-1).topk(topk, dim=-1)
    if tokens == 1:
        replica_indices = torch.zeros_like(expected_logical_ids)
    else:
        token_indices = torch.arange(tokens, device="cuda", dtype=torch.int64).unsqueeze(1)
        replica_indices = (
            (
                ((token_indices * 2654435769) & 0xFFFFFFFF)
                + ((expected_logical_ids.to(torch.int64) * 2246822519) & 0xFFFFFFFF)
            )
            & 0xFFFFFFFF
        ) % logical_replica_count[expected_logical_ids].to(torch.int64)
    expected_ids = logical_to_physical[expected_logical_ids, replica_indices.to(torch.long)].to(torch.long)
    if record_load:
        expected_counter[1].scatter_add_(
            0,
            expected_logical_ids.reshape(-1),
            torch.ones(expected_logical_ids.numel(), dtype=torch.int64, device="cuda"),
        )
    expected_weights = expected_weights / expected_weights.sum(dim=-1, keepdim=True)
    expected_weights = expected_weights * scale[expected_logical_ids]

    weights, physical_ids, logical_ids = triton_grouped_topk_eplb(
        hidden_states=torch.empty((tokens, 1), dtype=torch.float32, device="cuda"),
        gating_output=gating_output,
        correction_bias=torch.randn((experts,), dtype=torch.float32, device="cuda"),
        topk=topk,
        renormalize=True,
        num_expert_group=8,
        topk_group=4,
        scoring_func="sigmoid",
        logical_to_physical_map=logical_to_physical,
        logical_replica_count=logical_replica_count,
        expert_counter=fused_counter,
        sample_index=1,
        record_load=record_load,
        use_grouped_topk=False,
        per_expert_scale=scale,
        return_logical_ids=True,
    )
    torch.cuda.synchronize()

    torch.testing.assert_close(weights, expected_weights, rtol=1e-5, atol=1e-6)
    assert torch.equal(physical_ids, expected_ids)
    assert torch.equal(logical_ids, expected_logical_ids)
    assert torch.equal(fused_counter, expected_counter)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required for the Triton EPLB kernel")
def test_triton_grouped_topk_eplb_empty_tokens_skips_kernel():
    from lightllm.common.basemodel.triton_kernel.fused_moe.grouped_topk import triton_grouped_topk_eplb

    experts = 64
    counter = torch.zeros((1, experts), dtype=torch.int64, device="cuda")
    weights, physical_ids, logical_ids = triton_grouped_topk_eplb(
        hidden_states=torch.empty((0, 1), device="cuda"),
        gating_output=torch.empty((0, experts), device="cuda"),
        correction_bias=None,
        topk=4,
        renormalize=False,
        num_expert_group=8,
        topk_group=4,
        scoring_func="softmax",
        logical_to_physical_map=torch.zeros((experts, 1), dtype=torch.int32, device="cuda"),
        logical_replica_count=torch.ones((experts,), dtype=torch.int32, device="cuda"),
        expert_counter=counter,
        sample_index=0,
        record_load=True,
        use_grouped_topk=False,
        return_logical_ids=True,
    )

    assert weights.shape == physical_ids.shape == logical_ids.shape == (0, 4)
    assert physical_ids.dtype is logical_ids.dtype is torch.long
    assert torch.equal(counter, torch.zeros_like(counter))
