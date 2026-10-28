import builtins
import io
import threading
import time
from collections import deque
from contextlib import contextmanager
from types import SimpleNamespace

import pytest
import torch

from lightllm.common.cuda_wrapper import CudaBatchMemcpy, find_loaded_library


def _replica_index_reference(token, expert, counts):
    value = token ^ (((expert.to(torch.int64) + 1) * 0x9E3779B9) & 0xFFFFFFFF)
    value = ((value ^ (value >> 16)) * 0x7FEB352D) & 0xFFFFFFFF
    value = ((value ^ (value >> 15)) * 0x846CA68B) & 0xFFFFFFFF
    return (value ^ (value >> 16)) % counts


from lightllm.common.basemodel.layer_weights.meta_weights.fused_moe.eplb_placement import (
    build_initial_redundant_expert_ids,
    expand_redundant_placement,
    build_logical_to_physical_map,
    build_logical_to_physical_maps_for_layers,
    _estimate_rank_load,
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
from lightllm.utils import device_utils, envs_utils, dist_utils
from lightllm.common.eplb_utils import extract_eplb_expert_tensors
from lightllm.common.basemodel.triton_kernel.fused_moe import grouped_topk
from lightllm.common.basemodel.layer_weights.meta_weights.fused_moe import eplb_planner as eplb_planner_module
from lightllm.common.basemodel.layer_weights.meta_weights.fused_moe import eplb_placement as placement_module
from lightllm.server.router.model_infer.mode_backend.eplb_transfer import (
    TransferStep,
    _commit_staging_rows,
    align_target_placement,
    build_transfer_plan,
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
            logical_to_physical_map=torch.zeros((num_logical_experts, 2), dtype=torch.int32),
            logical_replica_count=torch.ones(num_logical_experts, dtype=torch.int32),
            route_counter=route_counter,
            physical_to_logical=expand_redundant_placement(
                build_initial_redundant_expert_ids(
                    num_logical_experts, initial_layout_world_size, num_redundant_experts_per_rank
                ),
                num_logical_experts,
            ),
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
            logical_to_physical_map=torch.empty((n_routed_experts, 2), dtype=torch.int32, device=device),
            logical_replica_count=torch.ones(n_routed_experts, dtype=torch.int32, device=device),
            route_counter=torch.zeros((2, n_routed_experts), dtype=torch.int64, device=device),
            physical_to_logical=expand_redundant_placement(
                build_initial_redundant_expert_ids(n_routed_experts, world_size, num_redundant_experts_per_rank),
                n_routed_experts,
            ),
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
    num_physical_experts_per_rank = redundant
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


@pytest.mark.parametrize("alpha,limit,clamp_up_add_one", [(None, None, True), (1.0, 10.0, False), (1.702, 7.0, True)])
def test_base_call_template_forwards_selection_and_capture_callback(alpha, limit, clamp_up_add_one):
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
            alpha=None,
            limit=None,
            clamp_up_add_one=True,
            dist_group=None,
        ):
            seen["fused"] = {
                "topk_ids": topk_ids,
                "activation": (alpha, limit, clamp_up_add_one),
                "dist_group": dist_group,
            }
            return "output"

    seen, captured = {}, []
    dist_group = object()
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
        alpha=alpha,
        limit=limit,
        clamp_up_add_one=clamp_up_add_one,
        dist_group=dist_group,
    )
    assert result == "output"
    assert captured == ["logical_ids"]
    assert seen["select"]["preserve_logical_ids"]
    assert seen["fused"]["topk_ids"] == "physical_ids"
    assert seen["fused"]["activation"] == (alpha, limit, clamp_up_add_one)
    assert seen["fused"]["dist_group"] is dist_group


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

    monkeypatch.setattr(fused_weight_module, "FusedMoeWeight", FakeFusedMoeWeight)
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

    assert manager_module.find_ep_moe_weights(model) == [alternate, aliased, first]


def test_eplb_redundant_experts_defaults_per_ep_rank():
    parser = make_argument_parser()

    assert parser.parse_args([]).eplb_num_redundant_experts_per_rank == 2
    assert parser.parse_args(["--eplb_num_redundant_experts_per_rank", "3"]).eplb_num_redundant_experts_per_rank == 3
    assert StartArgs().eplb_num_redundant_experts_per_rank == 2
    assert parser.parse_args(["--eplb_num_redundant_experts_per_rank", "0"]).eplb_num_redundant_experts_per_rank == 0
    assert not hasattr(StartArgs(), "eplb_placement_mode")
    with pytest.raises(SystemExit):
        parser.parse_args(["--eplb_placement_mode", "full"])


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


def test_select_improving_placements_rejects_regressing_layer():
    expert_load = torch.tensor([[8649, 5740, 5002, 3441]])
    current = expand_redundant_placement(torch.tensor([[[2], [0]]]), (expert_load).shape[-1])
    regressing_candidate = expand_redundant_placement(torch.tensor([[[1], [0]]]), (expert_load).shape[-1])

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
    current = expand_redundant_placement(torch.tensor([[[3], [0]]]), (expert_load).shape[-1])
    candidate = expand_redundant_placement(torch.tensor([[[3], [1]]]), (expert_load).shape[-1])

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
    current = expand_redundant_placement(torch.tensor([[[2], [0]]]), (expert_load).shape[-1])
    candidate = expand_redundant_placement(torch.tensor([[[3], [1]]]), (expert_load).shape[-1])

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
    current = expand_redundant_placement(torch.tensor([[[2], [0]]]), (expert_load).shape[-1])
    candidate = expand_redundant_placement(torch.tensor([[[3], [0]]]), (expert_load).shape[-1])

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
    current = expand_redundant_placement(torch.tensor([[[2], [0]]]), (expert_load).shape[-1])
    candidate = expand_redundant_placement(torch.tensor([[[3], [0]]]), (expert_load).shape[-1])

    with pytest.raises(ValueError, match="rebalance_gain_threshold"):
        select_improving_placements(
            expert_load,
            current,
            candidate,
            rebalance_gain_threshold=rebalance_gain_threshold,
        )


def test_select_improving_placements_accepts_sufficient_rebalance_gain():
    expert_load = torch.tensor([[1, 1, 1, 2]])
    current = expand_redundant_placement(torch.tensor([[[2], [0]]]), (expert_load).shape[-1])
    candidate = expand_redundant_placement(torch.tensor([[[3], [0]]]), (expert_load).shape[-1])

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
    current = expand_redundant_placement(torch.tensor([[[2], [0]]]), (expert_load).shape[-1])
    raw_improving_candidate = expand_redundant_placement(torch.tensor([[[3], [0]]]), (expert_load).shape[-1])

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
    placement = expand_redundant_placement(torch.tensor([[[2], [0]]]), (samples.sum(dim=0)).shape[-1])

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
    current = expand_redundant_placement(torch.tensor([[[2], [0]]]), (samples).shape[-1])
    mean_inflating_candidate = expand_redundant_placement(torch.tensor([[[2], [1]]]), (samples).shape[-1])

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
    current = expand_redundant_placement(torch.tensor([[[2], [0]]]), (samples).shape[-1])
    candidate = expand_redundant_placement(torch.tensor([[[2], [1]]]), (samples).shape[-1])

    selected, improved, _metrics, _before_load, _after_load = select_improving_placements(
        samples, current, candidate, rebalance_gain_threshold=0.05, expert_alignment=128
    )

    assert improved.item()
    assert torch.equal(selected, candidate)


def test_select_improving_placements_rejects_single_layer_gain_below_model_threshold():
    # Layer 0 becomes better, but layer 1 dominates model critical load.  The
    # aggregate estimated critical-load reduction gain is below 5%, so neither layer may be changed.
    expert_load = torch.tensor([[1, 1, 1, 2], [0, 0, 0, 10]])
    current = expand_redundant_placement(torch.tensor([[[2], [0]], [[2], [0]]]), (expert_load).shape[-1])
    candidate = expand_redundant_placement(torch.tensor([[[3], [0]], [[2], [0]]]), (expert_load).shape[-1])

    selected, improved, metrics, _before_load, _after_load = select_improving_placements(
        expert_load, current, candidate, rebalance_gain_threshold=0.05
    )

    assert not torch.any(improved)
    assert torch.equal(selected, current)
    assert metrics["candidate_rebalance_gain"] == pytest.approx(0.5 / 13)
    assert metrics["candidate_changed_layer_count"] == 1


def test_select_improving_placements_accepts_only_when_model_gain_reaches_threshold():
    expert_load = torch.tensor([[1, 1, 1, 2], [0, 0, 0, 5]])
    current = expand_redundant_placement(torch.tensor([[[2], [0]], [[2], [0]]]), (expert_load).shape[-1])
    candidate = expand_redundant_placement(torch.tensor([[[3], [0]], [[2], [0]]]), (expert_load).shape[-1])

    selected, improved, _metrics, _before_load, _after_load = select_improving_placements(
        expert_load, current, candidate, rebalance_gain_threshold=0.05
    )

    assert torch.equal(improved, torch.tensor([True, False]))
    assert torch.equal(selected, candidate)


def test_logical_to_physical_map_has_at_most_one_slot_per_rank():
    redundant_expert_ids = expand_redundant_placement(torch.tensor([[2, 3], [0, 1]]), 4)
    logical_to_physical, replica_count = build_logical_to_physical_map(redundant_expert_ids, num_logical_experts=4)

    assert logical_to_physical.shape == (4, 2)
    assert torch.equal(replica_count, torch.full((4,), 2, dtype=torch.int64))


def test_logical_to_physical_map_requires_expert_count_divisible_by_rank_count_without_source_rank():
    redundant_expert_ids = torch.tensor([[0, 1], [2, 3]])

    with pytest.raises(AssertionError):
        build_logical_to_physical_map(redundant_expert_ids, num_logical_experts=5)


def test_logical_to_physical_map_prefers_source_node_replicas():
    # Four ranks, two ranks per node, two primary experts/rank and one
    # redundant slot/rank.  Expert 0 is primary on rank 0 and replicated on
    # rank 2 (the other node).
    redundant = expand_redundant_placement(torch.tensor([[4], [5], [0], [1]], dtype=torch.int64), 8)
    rank0_map, rank0_count = build_logical_to_physical_map(
        redundant, num_logical_experts=8, source_rank=0, node_world_size=2
    )
    rank1_map, rank1_count = build_logical_to_physical_map(
        redundant, num_logical_experts=8, source_rank=1, node_world_size=2
    )
    fallback_redundant = expand_redundant_placement(torch.tensor([[4], [5], [0], [1], [2], [3]], dtype=torch.int64), 12)
    rank4_map, rank4_count = build_logical_to_physical_map(fallback_redundant, 12, source_rank=4, node_world_size=2)

    assert rank0_count[0].item() == rank1_count[0].item() == 1
    assert torch.equal(rank0_map[0, :1], torch.tensor([0]))
    assert torch.equal(rank1_map[0, :1], torch.tensor([0]))
    assert rank4_count[0].item() == 2
    assert set(rank4_map[0, :2].tolist()) == {0, 8}


def test_source_node_local_maps_fall_back_to_global_replicas():
    redundant = expand_redundant_placement(torch.tensor([[4], [5], [0], [1]], dtype=torch.int64), 8)
    maps = [build_logical_to_physical_map(redundant, 8, source_rank=rank, node_world_size=2) for rank in range(4)]
    assert maps[0][1][0].item() == maps[1][1][0].item() == 1
    assert maps[2][1][0].item() == maps[3][1][0].item() == 1
    assert maps[0][0][0, 0].item() == maps[1][0][0, 0].item() == 0
    assert maps[2][0][0, 0].item() == maps[3][0][0, 0].item() == 8


def test_source_rank_rotates_selected_replica_order_without_changing_copies():
    # Expert 0 is primary on rank 0 and redundant on rank 1, so both ranks
    # on node 0 have the same two local copies.  Their source-rank phases
    # must differ while their selected set/count remain identical.
    redundant = expand_redundant_placement(torch.tensor([[1], [0], [3], [2]], dtype=torch.int64), 4)
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
    placements_by_layer = expand_redundant_placement(
        torch.tensor(
            [
                [[4, 5], [0, 1], [0, 1], [2, 3]],
                [[6, 7], [0, 1], [0, 1], [2, 3]],
                [[4, 5], [0, 1], [0, 1], [2, 3]],
            ],
            dtype=torch.int64,
        ),
        8,
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


def test_source_node_estimate_matches_local_first_runtime_replica_sharing():
    # Expert 0 has a copy on each node.  Node 0 and node 1 issue unequal
    # traffic, so collapsing them before planning produces the wrong result.
    placement = expand_redundant_placement(torch.tensor([[[4], [5], [0], [1]]], dtype=torch.int64), 8)
    source_load = torch.zeros((1, 1, 2, 8), dtype=torch.int64)
    source_load[0, 0, 0, 0] = 256
    source_load[0, 0, 1, 0] = 128

    predicted = _estimate_rank_load(source_load, placement, expert_alignment=128, node_world_size=2)
    runtime = _manual_runtime_rank_load(source_load, placement, node_world_size=2, alignment=128)
    collapsed_global = _estimate_rank_load(source_load.sum(dim=2), placement, expert_alignment=128)

    assert torch.equal(predicted, runtime)
    assert torch.equal(predicted[0, 0], torch.tensor([256.0, 0.0, 128.0, 0.0]))
    assert not torch.equal(predicted, collapsed_global)


def _count_moved_slots(current: torch.Tensor, target: torch.Tensor) -> int:
    """Count rank rows gaining an expert: one migrated row per new expert id."""
    moved = 0
    for layer in range(current.shape[0]):
        for rank in range(current.shape[1]):
            moved += len(set(target[layer, rank].tolist()) - set(current[layer, rank].tolist()))
    return moved


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
    current = torch.tensor([[0, 1, 2], [2, 3, 4], [4, 5, 0]])
    target = current.roll(1, dims=0).flip(dims=(1,))
    canonical = align_target_placement(current, target)
    plan = build_transfer_plan(current, canonical, 6, 3, 3)
    live = current.clone()
    for step in plan:
        live[step.dst_rank, step.dst_slot] = current[step.src_rank, step.src_local_row]
    assert torch.equal(live, canonical)
    maps, counts = build_logical_to_physical_map(canonical, 6)
    for expert, count in enumerate(counts.tolist()):
        for physical_id in maps[expert, :count].tolist():
            rank, row = divmod(physical_id, current.shape[1])
            assert live[rank, row] == expert


def test_plan_and_broadcast_publishes_canonical_placement(monkeypatch):
    manager = manager_module.EPLBManager.__new__(manager_module.EPLBManager)
    manager.global_rank = 0
    manager.world_size = 4
    manager.node_world_size = 2
    manager.num_logical_experts = 8
    manager.num_redundant_experts_per_rank = 2
    manager.current_placement = torch.tensor([[[4, 5], [6, 7], [0, 1], [2, 3]]])
    manager._initial_plan_complete = False
    manager._evaluation_snapshot = {"placement": manager.current_placement}
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

    monkeypatch.setattr(manager_module, "plan_eplb_candidate", lambda *_args, **_kwargs: candidate.clone())
    monkeypatch.setattr(manager_module, "select_improving_placements", fixed_selector)
    monkeypatch.setattr(manager_module.dist, "broadcast_object_list", record_broadcast)

    result = manager._plan_and_broadcast(torch.full((1, 1, 2, 8), 100, dtype=torch.int64))

    assert torch.equal(result["placement"], manager.current_placement)
    assert broadcasts and torch.equal(broadcasts[0]["placement"], manager.current_placement)


def test_plan_and_broadcast_propagates_rank_zero_error_after_existing_broadcast(monkeypatch):
    manager = manager_module.EPLBManager.__new__(manager_module.EPLBManager)
    manager.global_rank = 0
    manager.world_size = 2
    manager.node_world_size = 2
    manager.num_logical_experts = 8
    manager.num_redundant_experts_per_rank = 2
    manager.current_placement = torch.tensor([[[4, 5], [6, 7], [0, 1], [2, 3]]])
    manager._initial_plan_complete = False
    manager._evaluation_snapshot = {"placement": manager.current_placement}
    manager.placement_stickiness = 0.1
    manager.rebalance_gain_threshold = 0.05
    manager.evaluation_group = object()
    broadcasted = []

    def broken_planner(*_args, **_kwargs):
        raise RuntimeError("planner boom")

    def record_broadcast(result_list, **_kwargs):
        broadcasted.append(result_list[0])

    monkeypatch.setattr(manager_module, "plan_eplb_candidate", broken_planner)
    monkeypatch.setattr(manager_module.dist, "broadcast_object_list", record_broadcast)

    with pytest.raises(RuntimeError, match="EPLB planner failed on rank zero") as exc_info:
        manager._plan_and_broadcast(torch.full((1, 1, 1, 8), 100, dtype=torch.int64))

    assert isinstance(exc_info.value.__cause__, RuntimeError)
    assert str(exc_info.value.__cause__) == "planner boom"
    assert broadcasted == [{"kind": "error", "message": "RuntimeError: planner boom"}]


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
    manager.min_rebalance_interval = 0
    manager._last_evaluation_step = 0
    manager.weights = []
    manager._eplb_states = []
    recordings, logs = [], []
    manager._set_recording = lambda enabled: recordings.append(enabled)
    monkeypatch.setattr(manager_module.logger, "info", lambda *args: logs.append(args))

    assert not manager._poll_evaluation()
    assert recordings == [True]
    assert "model_imbalance_ratio" in logs[0][0]
    assert "candidate_rebalance_gain" in logs[0][0]
    assert "candidate_changed_layer_count" in logs[0][0]
    assert "actual_changed_layer_count" in logs[0][0]
    assert "next_sampling_interval" in logs[0][0]
    assert manager.sampling_interval == 20


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
    manager.min_rebalance_interval = 0
    manager._last_evaluation_step = 0
    manager._freeze_evaluation_samples = lambda: None
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


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
@pytest.mark.parametrize(
    "step_interval,steady_steps,capacity", [(1, 4, 8), (2, 4, 8), (3, 4, 8), (1, 1, 2), (20, 64, 40)]
)
def test_eplb_counter_preserves_each_dispatch_in_maximum_window(monkeypatch, step_interval, steady_steps, capacity):
    from lightllm.common.basemodel.triton_kernel.fused_moe.grouped_topk import triton_grouped_topk_eplb

    args = StartArgs(
        enable_ep_moe=True,
        enable_prefill_eplb=True,
        eplb_num_redundant_experts_per_rank=2,
        run_mode="prefill",
    )
    weight = object.__new__(fused_weight_module.FusedMoeWeight)
    weight.n_routed_experts = 4
    weight.global_world_size = 2
    weight.global_rank_ = 0
    weight.enable_ep_moe = True
    weight.tp_rank_, weight.tp_world_size_ = 0, 1
    monkeypatch.setattr(envs_utils, "get_env_start_args", lambda: args)
    monkeypatch.setattr(envs_utils, "get_prefill_eplb_step_interval", lambda: step_interval)
    monkeypatch.setattr(envs_utils, "get_prefill_eplb_steady_sample_steps", lambda: steady_steps)
    monkeypatch.setattr(dist_utils, "get_node_world_size", lambda: 2)
    monkeypatch.setattr(torch.Tensor, "cuda", lambda tensor: tensor)
    original_zeros = torch.zeros

    def cpu_zeros(*shape, **kwargs):
        kwargs.pop("device", None)
        return original_zeros(*shape, **kwargs)

    monkeypatch.setattr(fused_weight_module.torch, "zeros", cpu_zeros)

    weight._init_expert_parallel_state({})

    state = weight.expert_parallel_state.eplb
    assert state.route_counter.shape == (capacity, 4, 4)
    state.route_counter = state.route_counter.to("cuda")
    manager = manager_module.EPLBManager.__new__(manager_module.EPLBManager)
    manager._eplb_states = [state]
    manager.step_interval, manager.steady_sample_steps = step_interval, steady_steps
    manager.sampling_interval = 16 * step_interval
    steps = max(step_interval, manager._steady_sample_window_steps())
    manager._use_full_window = False
    manager.current_placement = state.physical_to_logical.unsqueeze(0)
    manager.prefill_steps = 0
    manager._sample_step_counts = deque([0], maxlen=max(step_interval, 4) + 1)
    manager.in_flight = True
    state.recording = True
    expected = []
    for dispatch in range(2 * steps):
        if dispatch % 2 == 0:
            manager.before_prefill()
        # Move the hotspot between rank-local expert pairs across dispatches.
        expert = 2 * (dispatch % 2)
        scores = torch.full((256, 4), -10.0, device="cuda")
        scores[:128, expert] = 10
        scores[128:, expert + 1] = 10
        triton_grouped_topk_eplb(
            hidden_states=scores,
            gating_output=scores,
            correction_bias=None,
            topk=1,
            renormalize=True,
            num_expert_group=1,
            topk_group=1,
            scoring_func="softmax",
            logical_to_physical_map=state.logical_to_physical_map.to("cuda"),
            logical_replica_count=state.logical_replica_count.to("cuda"),
            expert_counter=state.route_counter,
            sample_index=state.next_sample_index(),
            record_load=state.recording,
            use_grouped_topk=False,
        )
        expected.append([128, 128, 0, 0] if expert == 0 else [0, 0, 128, 128])
        if dispatch % 2 == 1:
            manager.step()
    manager._freeze_evaluation_samples()
    samples = manager._evaluation_snapshot["samples"][:, 0].cpu()
    assert torch.equal(samples[..., 0], torch.tensor(expected, dtype=torch.int64))
    for copies in (1, 2):
        start = 1 + copies * (copies - 1) // 2
        assert torch.equal(samples[..., start : start + copies].sum(-1), samples[..., 0])


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


@pytest.mark.parametrize(
    "enabled,run_mode,nnodes", [(False, "prefill", 1), (False, "normal", 1), (True, "decode", 1), (True, "decode", 2)]
)
@pytest.mark.parametrize("draft", [False, True])
def test_inactive_eplb_creates_logical_layout_without_runtime_state(monkeypatch, enabled, run_mode, nnodes, draft):
    args = type(
        "Args",
        (),
        {
            "enable_prefill_eplb": enabled,
            "eplb_num_redundant_experts_per_rank": 2,
            "run_mode": run_mode,
            "nnodes": nnodes,
        },
    )()
    weight = object.__new__(fused_weight_module.FusedMoeWeight)
    weight.n_routed_experts = 4
    weight.global_world_size = 2
    weight.global_rank_ = 0
    weight.enable_ep_moe = True
    weight.tp_rank_, weight.tp_world_size_ = 0, 1
    monkeypatch.setattr(envs_utils, "get_env_start_args", lambda: args)
    monkeypatch.setattr(
        placement_module,
        "build_initial_redundant_expert_ids",
        lambda *args: pytest.fail("inactive EPLB must not initialize physical replicas"),
    )

    weight._init_expert_parallel_state({"is_mtp_draft_model": draft})

    assert weight.expert_parallel_state is not None
    assert weight.expert_parallel_state.eplb is None
    assert weight.expert_parallel_state.num_total_physical_experts == weight.n_routed_experts
    assert not hasattr(weight, "_initial_redundant_expert_ids")
    assert not hasattr(weight, "route_counter")
    assert not hasattr(weight, "routed_expert_counter_tensor")


@pytest.mark.parametrize("run_mode", ["prefill", "normal"])
@pytest.mark.parametrize("transport_experts", [4, 8])
def test_draft_keeps_static_unrecorded_layout_in_shared_expert_space(monkeypatch, run_mode, transport_experts):
    # This checks initialization contracts; actual CUDA storage is covered by
    # the checkpoint-loading and graph routing tests.
    monkeypatch.setattr(device_utils, "is_sm100_gpu", lambda: False)
    real_zeros = torch.zeros
    monkeypatch.setattr(torch.Tensor, "cuda", lambda tensor, *args, **kwargs: tensor)
    monkeypatch.setattr(torch, "zeros", lambda *args, **kwargs: real_zeros(*args, **{**kwargs, "device": "cpu"}))
    args = StartArgs(
        enable_ep_moe=True,
        enable_prefill_eplb=True,
        eplb_num_redundant_experts_per_rank=2,
        run_mode=run_mode,
    )
    weight = object.__new__(fused_weight_module.FusedMoeWeight)
    weight.n_routed_experts = 4
    weight.global_world_size = 2
    weight.global_rank_ = 0
    weight.enable_ep_moe = True
    weight.tp_rank_, weight.tp_world_size_ = 0, 1
    monkeypatch.setattr(envs_utils, "get_env_start_args", lambda: args)
    monkeypatch.setattr(dist_utils, "get_node_world_size", lambda: 2)
    monkeypatch.setattr(deepgemm_module.dist_group_manager, "ll_num_experts", transport_experts, raising=False)
    if transport_experts != 8:
        monkeypatch.setattr(
            placement_module,
            "build_initial_redundant_expert_ids",
            lambda *args: pytest.fail("reject incompatible transport before creating draft weights"),
        )
        with pytest.raises(ValueError, match="same DeepEP expert space"):
            weight._init_expert_parallel_state({"is_mtp_draft_model": True})
        return

    weight._init_expert_parallel_state({"is_mtp_draft_model": True})

    state = weight.expert_parallel_state.eplb
    assert weight.expert_parallel_state.num_total_physical_experts == 8
    assert not state.recording
    assert state.next_sample_index() == state.recorded_sample_count == 0
    assert state.route_counter.shape[0] == 2
    assert state.migration_poll is None
    assert state.placement_generation == 0
    assert not hasattr(weight, "_initial_redundant_expert_ids")


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
    manager._evaluation_snapshot = {"steps": 4}
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


@pytest.mark.parametrize("phase", [0, 1])
def test_manager_planned_evaluation_builds_improved_metadata_in_one_multilayer_call(monkeypatch, phase):
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
    for state in manager._eplb_states:
        state.replica_rotation_rank = phase
    manager.global_rank = 1
    manager.world_size = 4
    manager.node_world_size = 2
    manager.step_interval = 20
    manager.sampling_interval = 20
    manager.num_logical_experts = 4
    manager.num_redundant_experts_per_rank = 1
    manager.current_placement = (
        expand_redundant_placement(build_initial_redundant_expert_ids(4, 4, 1), 4)
        .unsqueeze(0)
        .expand(3, -1, -1)
        .clone()
    )
    manager.evaluation_group = object()
    manager._evaluation_lock = threading.Lock()
    manager._evaluation_result = None
    manager._evaluation_error = None
    manager._evaluation_snapshot = {"steps": 4}
    prepare_calls = []
    prepared_batches = object()

    def prepare_transfer(layer_plans):
        prepare_calls.append(layer_plans)
        return prepared_batches

    manager.transfer = SimpleNamespace(prepare_transfer=prepare_transfer)
    manager._collect_local_samples = lambda: torch.full((1, 3, 4), 100, dtype=torch.int64)
    planned_placement = torch.tensor(
        [
            [[3, 0], [0, 1], [1, 2], [2, 3]],
            [[2, 3], [3, 0], [0, 1], [1, 2]],
            [[1, 2], [2, 3], [3, 0], [0, 1]],
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
    assert calls == [torch.Size([2, 4, 2])]
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
            replica_rotation_rank=phase,
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
    manager._evaluation_snapshot = {"steps": 4}
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


def test_decode_dispatch_uses_published_physical_ids_and_expert_count(monkeypatch):
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
        torch.tensor([[0, 143]], dtype=torch.int32),
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

    assert result[2].tolist() == [[0, 143]]
    assert calls[0]["num_experts"] == 144


def test_decode_select_maps_experts_without_sampling_or_polling(monkeypatch):
    impl = object.__new__(deepgemm_module.FuseMoeDeepGEMM)
    impl.routed_scaling_factor = 1.0
    _set_expert_parallel_state(impl, _test_parallel_state(eplb=True))
    logical_ids = torch.tensor([[3, 127]], dtype=torch.int64)
    physical_ids = torch.tensor([[13, 143]], dtype=torch.int64)
    counter = impl.eplb.route_counter.clone()
    impl.eplb.recording = True

    def forbidden():
        pytest.fail("decode must not sample or poll EPLB")

    impl.eplb.next_sample_index = forbidden
    impl.eplb.migration_poll = forbidden

    def select(**kwargs):
        assert kwargs["record_load"] is False
        assert kwargs["use_replicas"] is False
        assert kwargs["sample_index"] == 0
        assert kwargs["logical_to_physical_map"] is impl.eplb.logical_to_physical_map
        return torch.ones((1, 2)), physical_ids, logical_ids

    monkeypatch.setattr(grouped_topk, "triton_grouped_topk_eplb", select)
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

    assert selected is physical_ids
    assert origin is logical_ids
    assert torch.equal(impl.eplb.route_counter, counter)
    assert impl.eplb.recorded_sample_count == 0


def test_eplb_prefill_uses_single_fused_path_for_global_topk(monkeypatch):
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
    )

    assert weights.tolist() == [[1.0, 1.0]]
    assert topk_idx is physical_ids
    assert topk_idx.dtype is torch.long
    assert qinput == "qinput"
    assert not calls[0]["use_grouped_topk"]
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


def test_decode_masked_group_gemm_uses_all_published_physical_rows(
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
    assert captured["w13"].shape[0] == captured["w2"].shape[0] == 10
    assert captured["w13_scale"].shape[0] == captured["w2_scale"].shape[0] == 10


def test_decode_fused_experts_uses_original_weight_packs_and_physical_experts(
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

    assert [call["num_experts"] for call in captured] == [160, 160]
    assert all(call["w13"].weight.shape[0] == call["w2"].weight.shape[0] == 10 for call in captured)
    assert captured[0]["w13"] is captured[1]["w13"]
    assert captured[0]["w2"] is captured[1]["w2"]


def test_extract_expert_tensors_includes_weight_scale_and_zero_point_in_order():
    class Pack:
        def __init__(self, offset, scale=True, zero=True):
            self.weight = torch.full((3, 2), offset)
            self.weight_scale = torch.full((3, 1), offset + 1) if scale else None
            self.weight_zero_point = torch.full((3, 1), offset + 2) if zero else None

    weight = type("Weight", (), {"w13": Pack(1), "w2": Pack(10, scale=False, zero=False)})()
    tensors = extract_eplb_expert_tensors(weight)
    assert [name for name, _ in tensors] == [
        "w13.weight",
        "w13.weight_scale",
        "w13.weight_zero_point",
        "w2.weight",
    ]


def test_commit_staging_rows_only_overwrites_changed_absolute_rows():
    live = torch.arange(20).reshape(5, 4)
    staging = torch.full((5, 4), -1)
    _commit_staging_rows(
        live,
        staging,
        changed_dst_slots=(3, 4),
    )
    assert torch.equal(live[:3], torch.arange(12).reshape(3, 4))
    assert torch.equal(live[3:], staging[3:])


def test_commit_staging_rows_preserves_unchanged_destination_slots():
    live = torch.arange(28).reshape(7, 4)
    staging = -torch.arange(28).reshape(7, 4)
    original = live.clone()

    _commit_staging_rows(
        live,
        staging,
        changed_dst_slots=(6, 4),
    )

    assert torch.equal(live[:4], original[:4])
    assert torch.equal(live[4], staging[4])
    assert torch.equal(live[5], original[5])
    assert torch.equal(live[6], staging[6])


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
        Tensor("staging", 20),
        changed_dst_slots=(13, 11, 12),
    )

    assert copies == [("live", 11, 3, "staging", 11, 3)]


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
        def record(self, stream):
            operations.append("consumed record")

        def synchronize(self):
            operations.append("consumed synchronize")

    transfer = object.__new__(transfer_module.NixlEPLBTransfer)
    transfer.backend, transfer.device, transfer.global_rank = "test", "cuda:0", 0
    transfer.staging = [[]]
    transfer._release = [threading.Event()]
    transfer._release[0].set()
    transfer._consumed_events, transfer._consumed_recorded = [Event()], [False]
    transfer._pending, transfer._pending_lock = deque(), threading.Lock()
    transfer._error, transfer._thread = None, None
    transfer._needs_staging_reuse_barrier, transfer.transfer_group = True, object()
    transfer._copy_part = lambda part: operations.append(("copy", part[0]))
    transfer._prepare_part = lambda layer, plan, part: (layer, part)
    transfer._start_transfer_generation = lambda: None
    transfer._finish_transfer_generation = lambda: None
    transfer._commit_full_parts = lambda *args: None
    monkeypatch.setattr(transfer_module.torch.cuda, "set_device", lambda device: None)
    monkeypatch.setattr(transfer_module.torch.cuda, "current_stream", lambda: object())
    monkeypatch.setattr(transfer_module.dist, "barrier", lambda **kwargs: operations.append("barrier"))
    plans = [(0, []), (1, []), (2, [])]
    prepared = transfer.prepare_transfer(plans)
    monkeypatch.setattr(transfer, "prepare_transfer", lambda plans: pytest.fail("must reuse prepared layers"))
    transfer.start(plans, prepared)
    for layer in range(3):
        deadline = time.monotonic() + 2
        while not transfer.pending_layers() and time.monotonic() < deadline:
            time.sleep(0.001)
        assert transfer.pending_layers() == [(layer, 0)]
        assert operations[-1] == ("copy", layer)
        transfer.commit(layer, 0)
    transfer.finish()
    assert operations == [
        ("copy", 0),
        "consumed record",
        "consumed synchronize",
        "barrier",
        ("copy", 1),
        "consumed record",
        "consumed synchronize",
        "barrier",
        ("copy", 2),
        "consumed record",
    ]


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_complete_layer_commit_queues_weights_and_mapping_without_waiting_on_host(monkeypatch):
    live = torch.ones((2, 8), device="cuda")
    staging = torch.full_like(live, 2)
    mapping = torch.ones(1, device="cuda")
    old_read = torch.empty_like(live)
    new_read = torch.empty_like(live)
    stream = torch.cuda.Stream()
    current = torch.cuda.current_stream()
    original_overlap = g_infer_context.overlap_stream
    consumed = torch.cuda.Event()
    transfer = object.__new__(transfer_module.NixlEPLBTransfer)
    transfer.live, transfer.staging = [[("weight", live)]], [[("weight", staging)]]
    transfer._prepared_layer = SimpleNamespace(layer_index=0, parts=(None,), changed_dst_slots=(0,))
    transfer._pending, transfer._pending_lock = deque([(0, 0)]), threading.Lock()
    transfer._release = [threading.Event()]
    transfer._consumed_events, transfer._consumed_recorded = [consumed], [False]
    transfer._error = None
    manager = object.__new__(manager_module.EPLBManager)
    manager.transfer, manager.in_flight_layers = transfer, [0]
    manager.control_group = object()
    manager._control_ready_count = torch.empty(1, dtype=torch.int32)
    manager._commit_layer_metadata = lambda _layer: mapping.fill_(2)
    transfer.transfer_group = object()

    def reduce_ready(tensor, **kwargs):
        if kwargs["group"] is manager.control_group:
            tensor.fill_(1)

    monkeypatch.setattr(manager_module.dist, "all_reduce", reduce_ready)
    try:
        g_infer_context.overlap_stream = stream
        with torch.cuda.stream(stream):
            stream.wait_stream(current)
            torch.cuda._sleep(100_000_000)
            old_read.copy_(live)
        manager._poll_in_flight(finish_generation=False)
        assert transfer._consumed_recorded == [True]
        assert not consumed.query(), "commit waited for the preceding GPU work on the host"
        with torch.cuda.stream(stream):
            stream.wait_stream(current)
            new_read.copy_(live * mapping)
        stream.synchronize()
        torch.testing.assert_close(old_read, torch.ones_like(live))
        torch.testing.assert_close(new_read[0], torch.full_like(new_read[0], 4))
        torch.testing.assert_close(new_read[1], torch.full_like(new_read[1], 2))
    finally:
        g_infer_context.overlap_stream = original_overlap


def test_transfer_finalizes_after_join_and_propagates_finalization_failure(monkeypatch):
    def make_transfer(finalize):
        transfer = object.__new__(transfer_module.NixlEPLBTransfer)
        transfer.backend = "test"
        transfer.device = torch.device("cuda", 0)
        transfer.global_rank = 0
        transfer.staging_depth = 1
        transfer.staging = [[]]
        transfer._release = [threading.Event()]
        transfer._release[0].set()
        transfer._consumed_events = [object()]
        transfer._consumed_recorded = [False]
        transfer._pending = deque()
        transfer._pending_lock = threading.Lock()
        transfer._error = None
        transfer._thread = None
        transfer._needs_staging_reuse_barrier = False
        transfer._copy_part = lambda part: None
        transfer._prepare_part = lambda layer, plan, part: None
        transfer._start_transfer_generation = lambda: None
        transfer._finish_transfer_generation = finalize
        return transfer

    monkeypatch.setattr(transfer_module.torch.cuda, "set_device", lambda _device: None)

    finalized_before_publish = []
    success = make_transfer(lambda: finalized_before_publish.append(len(success._pending)))
    success.start([(0, [])], success.prepare_transfer([(0, [])]))
    success.finish()
    assert finalized_before_publish == [1]
    assert success.pending_layers() == [(0, 0)]

    failed = make_transfer(lambda: (_ for _ in ()).throw(RuntimeError("cache boom")))
    failed.start([(0, [])], failed.prepare_transfer([(0, [])]))
    failed._thread.join()
    assert list(failed._pending) == [(0, 0)]
    with pytest.raises(RuntimeError, match="cache boom"):
        failed.finish()
    assert failed._thread is None


def test_manager_rearms_after_rebalance_for_interval_one():
    recording_calls = []
    manager = manager_module.EPLBManager.__new__(manager_module.EPLBManager)
    manager.step_interval = 1
    manager.sampling_interval = 1
    manager._last_evaluation_step = 1
    manager.weights = []
    manager._eplb_states = []
    manager.target_placement = torch.zeros(1)
    manager.in_flight_started_at = 0
    manager.global_rank = 1
    manager._set_recording = lambda enabled: recording_calls.append(enabled)
    manager._finish_rebalance()
    assert manager.in_flight is False
    assert recording_calls == [True]
    assert manager._next_evaluation_step == 2


def test_planned_rebalance_resets_sampling_interval_to_base(monkeypatch):
    manager = manager_module.EPLBManager.__new__(manager_module.EPLBManager)
    manager.current_placement = torch.zeros((1, 1, 1), dtype=torch.int64)
    manager.num_logical_experts = 1
    manager.world_size = 1
    manager.node_world_size = 1
    manager.step_interval = 20
    manager.sampling_interval = 320
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
    assert manager._initial_plan_complete
    assert not manager._use_full_window
    assert len(manager.transfer.started[0]) == 1
    assert manager.transfer.started[1] is prepared_batches


def test_transfer_start_rejects_prepared_layers_with_wrong_count():
    transfer = object.__new__(transfer_module.NixlEPLBTransfer)
    transfer._thread = None
    transfer.staging_depth = 2
    transfer.staging = [object(), object()]

    with pytest.raises(ValueError, match="prepared layer count"):
        transfer.start([(0, []), (1, []), (2, [])], prepared_layers=[object()])


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


def test_nixl_prepare_part_compiles_hot_path_without_tensor_views(monkeypatch):
    class BatchMemcpy:
        def __init__(self):
            self.prepared = []
            self.enqueued = []

        def prepare_memcpy_batch(self, descriptors):
            descriptor = tuple(descriptors)
            self.prepared.append(descriptor)
            return descriptor

        def cudaMemcpyBatchAsync(self, descriptor, stream):
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

    run_a = [TransferStep(1, 3, 0, 5), TransferStep(1, 4, 0, 6)]
    run_b = [TransferStep(2, 1, 0, 2)]
    local_inbound = TransferStep(0, 0, 1, 0)
    remote_inbound = TransferStep(0, 1, 3, 2)
    staging = object()
    transfer.staging = [staging]
    prepared = transfer._prepare_part(0, run_a + run_b + [local_inbound, remote_inbound], part_index=0)
    monkeypatch.setattr(transfer, "_prepare_part", lambda *args: pytest.fail("hot path must not prepare descriptors"))
    monkeypatch.setattr(transfer_module.torch.cuda, "stream", lambda _stream: pytest.fail("must not switch streams"))
    transfer._copy_part(prepared)

    expected = (
        (1160, 4096, 32),
        (2160, 5096, 32),
        (3160, 6096, 32),
        (1192, 4128, 32),
        (2192, 5128, 32),
        (3192, 6128, 32),
        (1064, 7032, 32),
        (2064, 8032, 32),
        (3064, 9032, 32),
    )
    assert batch_memcpy.prepared == [expected]
    assert batch_memcpy.enqueued == [(expected, 123)]
    assert prepared.remote_entries == {3: [(0, [remote_inbound], staging)]}


def test_nixl_prepare_transfer_builds_complete_layers_before_transfer():
    transfer = object.__new__(transfer_module.NixlEPLBTransfer)
    transfer.staging_depth, transfer.staging = 1, ["staging"]
    seen = []

    def prepare_part(layer_index, plan, part_index):
        seen.append((layer_index, part_index))
        return f"prepared-{layer_index}-{part_index}"

    transfer.global_rank = 0
    transfer._prepare_part = prepare_part
    prepared = transfer.prepare_transfer([(3, []), (4, []), (5, [])])
    assert seen == [(3, 0), (4, 0), (5, 0)]
    assert [layer.layer_index for layer in prepared] == [3, 4, 5]
    assert [layer.parts for layer in prepared] == [
        ("prepared-3-0",),
        ("prepared-4-0",),
        ("prepared-5-0",),
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
    prepared = batch_memcpy.prepare_memcpy_batch(((101, 201, 64), (102, 202, 128)))
    batch_memcpy.cudaMemcpyBatchAsync(prepared, 777)

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

    with pytest.raises(RuntimeError, match="13.0"):
        CudaBatchMemcpy(OldRuntimeLibrary())

    class FutureRuntimeLibrary:
        def __init__(self):
            def get_version(pointer):
                ctypes.cast(pointer, ctypes.POINTER(ctypes.c_int))[0] = 14000
                return 0

            self.cudaRuntimeGetVersion = Function(get_version)
            self.cudaMemcpyBatchAsync = Function(lambda *_args: 0)
            self.cudaGetErrorString = Function(lambda _result: b"fake cuda error")

    with pytest.raises(RuntimeError, match="13.x"):
        CudaBatchMemcpy(FutureRuntimeLibrary())

    class MissingBatchSymbolLibrary:
        def __init__(self):
            def get_version(pointer):
                ctypes.cast(pointer, ctypes.POINTER(ctypes.c_int))[0] = 13000
                return 0

            self.cudaRuntimeGetVersion = Function(get_version)
            self.cudaGetErrorString = Function(lambda _result: b"fake cuda error")

    with pytest.raises(RuntimeError, match="cudaMemcpyBatchAsync"):
        CudaBatchMemcpy(MissingBatchSymbolLibrary())
    with pytest.raises(ValueError, match="at least one"):
        CudaBatchMemcpy.prepare_memcpy_batch(())
    with pytest.raises(ValueError, match="positive"):
        CudaBatchMemcpy.prepare_memcpy_batch(((1, 2, 0),))


def test_nixl_transfer_fails_fast_without_cuda13_batch_memcpy(monkeypatch):
    failure = RuntimeError("missing cudaMemcpyBatchAsync")

    def unavailable():
        raise failure

    monkeypatch.setattr(
        transfer_module.NixlEPLBTransfer, "_init_staging", lambda self, *_args: setattr(self, "device", "mock")
    )
    monkeypatch.setattr(transfer_module.torch.cuda, "Stream", lambda device: ("stream", device))
    monkeypatch.setattr(transfer_module, "CudaBatchMemcpy", unavailable)
    weight = SimpleNamespace(expert_parallel_state=_test_parallel_state(eplb=True))
    with pytest.raises(RuntimeError, match="missing cudaMemcpyBatchAsync") as exc_info:
        transfer_module.NixlEPLBTransfer([weight], object(), 0, 1)
    assert exc_info.value is failure


@pytest.mark.parametrize(
    ("maps", "open_raises", "expected"),
    [
        (
            "7f /cuda-12/libcudart.so.12.8\n"
            "7f /first/libcudart.so.13 (deleted)\n"
            "7f /second/libcudart.so.13\n"
            "7f /cuda-14/libcudart.so.14\n"
            "7f /cuda-130/libcudart.so.130\n",
            False,
            "/first/libcudart.so.13",
        ),
        ("7f /cuda-12/libcudart.so.12\n7f /cuda-130/libcudart.so.130\n", False, None),
        ("", True, None),
    ],
)
def test_cuda_batch_memcpy_finds_first_loaded_cuda13_runtime(monkeypatch, maps, open_raises, expected):
    def open_maps(_path):
        if open_raises:
            raise OSError("maps unavailable")
        return io.StringIO(maps)

    monkeypatch.setattr(builtins, "open", open_maps)
    assert find_loaded_library("libcudart", major_version=13) == expected


@pytest.mark.parametrize(("layers", "expected_depth"), [(1, 1), (8, 1), (9, 1), (43, 1)])
def test_nixl_transfer_bounds_staging_depth(monkeypatch, layers, expected_depth):
    def base_init(self, *_args):
        self.device = "mock-device"

    monkeypatch.setattr(transfer_module.NixlEPLBTransfer, "_init_staging", base_init)
    monkeypatch.setattr(transfer_module.torch.cuda, "Stream", lambda device: ("stream", device))
    monkeypatch.setattr(transfer_module, "CudaBatchMemcpy", lambda: object())
    monkeypatch.setattr(transfer_module.NixlEPLBTransfer, "_init_ipc_metadata", lambda self: None)
    monkeypatch.setattr(transfer_module.NixlEPLBTransfer, "_init_push_layouts", lambda self: None)

    weight = SimpleNamespace(expert_parallel_state=_test_parallel_state(eplb=True))
    transfer = transfer_module.NixlEPLBTransfer([weight] * layers, object(), 0, 1)

    assert transfer.staging_depth == expected_depth


def test_nixl_remote_read_cache_reuses_exact_batch_key_and_releases_on_shutdown():
    class Tensor:
        nbytes = 8

        def __init__(self, pointer, nbytes=8):
            self.pointer = pointer
            self.nbytes = nbytes

        def data_ptr(self):
            return self.pointer

        def get_device(self):
            return 0

        def __getitem__(self, index):
            return Tensor(self.pointer + index * self.nbytes, self.nbytes)

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
    staging = [("weight", Tensor(200, 4))]
    first_entries = [(0, [TransferStep(0, 0, 1, 0)], staging)]

    first = transfer._get_remote_read(1, first_entries, 0)
    assert transfer._get_remote_read(1, first_entries, 0) is first
    assert agent.made == 1

    changed_entries = [(0, [TransferStep(0, 1, 1, 0)], staging)]
    transfer._get_remote_read(1, changed_entries, 0)
    assert agent.made == 2

    # Cache identity must include both exact address ranges.
    transfer._remote_layouts[1][0][0] = ("weight", 1008, 0, 8)
    transfer._get_remote_read(1, first_entries, 0)
    assert agent.made == 3
    staging[0][1].pointer += 16
    transfer._get_remote_read(1, first_entries, 0)
    assert agent.made == 4

    transfer.shutdown()
    assert agent.released_xfers == 4
    assert agent.released_dlists == 8
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

        def __getitem__(self, index):
            return Tensor(self.pointer + index * self.nbytes)

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
    transfer._pending = deque()
    transfer._pending_lock = threading.Lock()
    transfer._error = None
    transfer._thread = None
    transfer._needs_staging_reuse_barrier = False

    staging = [("weight", Tensor(200))]
    entries_a = [(0, [TransferStep(0, 0, 1, 0)], staging)]
    entries_b = [(0, [TransferStep(0, 1, 1, 0)], staging)]
    generation = [entries_a]

    def copy_part(_prepared_part):
        transfer._get_remote_read(1, generation[0], 0)

    transfer._copy_part = copy_part
    monkeypatch.setattr(transfer_module.torch.cuda, "set_device", lambda _device: None)

    prepared_batches = [transfer._PreparedLayer(0, (), (None,))]
    transfer.start([(0, [])], prepared_batches)
    transfer.finish()
    assert agent.made == 1
    assert agent.released_xfers == agent.released_dlists == 0
    assert len(transfer._xfer_cache) == 1

    # The real manager releases each staging buffer through commit(). This
    # focused cache test has no commits, so model that hand-off before the
    # next generation reuses buffer zero.
    transfer._release[0].set()
    transfer.start([(0, [])], prepared_batches)
    transfer.finish()
    assert agent.made == 1
    assert agent.released_xfers == agent.released_dlists == 0
    assert len(transfer._xfer_cache) == 1

    generation[:] = [entries_b]
    transfer._release[0].set()
    transfer.start([(0, [])], prepared_batches)
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


def test_nixl_copy_part_source_pushes_local_rows_and_keeps_remote_ucx_reads(
    monkeypatch,
):
    class Stream:
        def __init__(self):
            self.synchronized = 0
            self.cuda_stream = 123

        def synchronize(self):
            self.synchronized += 1

    stream = Stream()
    transfer = object.__new__(transfer_module.NixlEPLBTransfer)
    transfer._push_stream = stream
    transfer._batch_memcpy = SimpleNamespace(
        cudaMemcpyBatchAsync=lambda descriptor, stream: enqueued.append((descriptor, stream))
    )
    remote_reads, waited_xfers, enqueued = [], [], []
    transfer._get_remote_read = lambda src_rank, entries, part_index=None: remote_reads.append((src_rank, entries)) or (
        None,
        None,
        "xfer",
    )
    transfer._wait_xfers = waited_xfers.extend

    monkeypatch.setattr(transfer_module.torch.cuda, "stream", lambda _stream: pytest.fail("must not switch streams"))
    prepared_push = transfer_module.NixlEPLBTransfer._PreparedPart({}, "push", 0)
    transfer._copy_part(prepared_push)
    assert enqueued == [("push", stream.cuda_stream)]

    remote_step = TransferStep(0, 1, 2, 0)
    prepared_remote = transfer_module.NixlEPLBTransfer._PreparedPart({2: [(0, [remote_step], [])]}, None, 0)
    transfer._copy_part(prepared_remote)
    assert [rank for rank, _ in remote_reads] == [2]
    assert waited_xfers == [(None, None, "xfer")]


def test_nixl_copy_part_self_only_rank_pushes_and_synchronizes(monkeypatch):
    class Stream:
        def __init__(self):
            self.synchronized = 0
            self.cuda_stream = 456

        def synchronize(self):
            self.synchronized += 1

    transfer = object.__new__(transfer_module.NixlEPLBTransfer)
    transfer._push_stream = Stream()
    enqueued = []
    transfer._batch_memcpy = SimpleNamespace(
        cudaMemcpyBatchAsync=lambda descriptor, stream: enqueued.append((descriptor, stream))
    )
    transfer._wait_xfers = lambda _xfers: None
    monkeypatch.setattr(transfer_module.torch.cuda, "stream", lambda _stream: pytest.fail("must not switch streams"))

    prepared = transfer_module.NixlEPLBTransfer._PreparedPart({}, "push", 0)
    transfer._copy_part(prepared)

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
    state = weight.expert_parallel_state.eplb
    state.route_counter.fill_(9)
    state.recorded_sample_count = 37
    transfer = object()
    groups = [object(), object(), object()]
    new_group_calls = []
    monkeypatch.setattr(manager_module, "find_ep_moe_weights", lambda model: [weight])
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
    assert manager._next_evaluation_step == manager.step_interval
    assert tuple(manager._sample_step_counts) == (0,)
    assert weight.expert_parallel_state.eplb.recording
    assert state.recorded_sample_count == 0
    assert torch.count_nonzero(state.route_counter) == 0
    assert manager._eplb_states[0] is weight.expert_parallel_state.eplb
    assert not hasattr(weight.expert_parallel_state.eplb, "record_load")


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required for physical decode routing")
@pytest.mark.parametrize("tokens", [1, 4, 16])
@pytest.mark.parametrize("capture", [False, True])
def test_decode_reads_republished_layout_without_writing_counters(tokens, capture):
    torch.manual_seed(57)
    experts, topk = 8, 2
    logits = torch.randn((tokens, experts), device="cuda")
    hidden = torch.empty((tokens, 1), device="cuda")
    placement = expand_redundant_placement(build_initial_redundant_expert_ids(experts, 2, 1), experts)
    mapping, replicas = build_logical_to_physical_map(placement, experts)
    mapping, replicas = mapping.cuda(), replicas.cuda()
    counter = torch.full((2, experts, 4), 17, dtype=torch.int64, device="cuda")
    draft_mapping, draft_replicas, draft_counter = mapping.clone(), replicas.clone(), counter.clone()

    def run(route_map=mapping, counts=replicas, load_counter=counter):
        return grouped_topk.triton_grouped_topk_eplb(
            hidden,
            logits,
            None,
            topk,
            True,
            1,
            1,
            "sigmoid",
            route_map,
            counts,
            load_counter,
            sample_index=0,
            record_load=False,
            use_grouped_topk=True,
            use_replicas=False,
        )

    run()
    run(draft_mapping, draft_replicas, draft_counter)
    if capture:
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            weights, physical_ids, _ = run()
        draft_graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(draft_graph):
            draft_weights, draft_ids, _ = run(draft_mapping, draft_replicas, draft_counter)
    for layout in (placement, placement.roll(1, dims=0), placement.roll(2, dims=1)):
        next_mapping, next_replicas = build_logical_to_physical_map(layout, experts)
        mapping.copy_(next_mapping)
        replicas.copy_(next_replicas)
        if capture:
            graph.replay()
        else:
            weights, physical_ids, _ = run()
        logical_ids = layout.flatten().cuda()[physical_ids]
        reference_scores, reference_ids = torch.sigmoid(logits).topk(topk, dim=-1)
        reference_weights = reference_scores / reference_scores.sum(dim=-1, keepdim=True)
        torch.testing.assert_close(physical_ids, next_mapping[:, 0].cuda().long()[reference_ids])
        torch.testing.assert_close(logical_ids, reference_ids)
        torch.testing.assert_close(weights, reference_weights)
        # An expert-specific output catches stale layouts even if all IDs stay in bounds.
        actual = (weights * (logical_ids + 1).square()).sum(dim=-1)
        expected = (reference_weights * (reference_ids + 1).square()).sum(dim=-1)
        torch.testing.assert_close(actual, expected)
        if capture:
            draft_graph.replay()
        else:
            draft_weights, draft_ids, _ = run(draft_mapping, draft_replicas, draft_counter)
        draft_logical_ids = placement.flatten().cuda()[draft_ids]
        torch.testing.assert_close(draft_logical_ids, reference_ids)
        torch.testing.assert_close(draft_weights, reference_weights)
        if capture:
            graph.replay()
        else:
            weights, physical_ids, _ = run()
        torch.testing.assert_close(layout.flatten().cuda()[physical_ids], reference_ids)
        assert torch.all(counter == 17)
        assert torch.all(draft_counter == 17)


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
        replica_indices = _replica_index_reference(
            token_indices, logical_ids, logical_replica_count[logical_ids.to(torch.long)].to(torch.int64)
        )
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
@pytest.mark.parametrize("experts,groups,group_topk", [(288, 1, 1), (256, 8, 4), (160, 8, 3)])
@pytest.mark.parametrize("tokens", [1, 32])
@pytest.mark.parametrize("strided", [False, True])
@pytest.mark.parametrize("scores", ["equal", "quantized", "saturated"])
def test_grouped_topk_eplb_matches_logical_selection_at_ties(experts, groups, group_topk, tokens, scores, strided):
    from lightllm.common.basemodel.triton_kernel.fused_moe.grouped_topk import (
        get_group_score_used_topk_num,
        triton_grouped_topk,
        triton_grouped_topk_eplb,
    )

    generator = torch.Generator(device="cuda").manual_seed(20261022)
    logits = torch.randn((tokens, experts), device="cuda", generator=generator).round()
    bias = (torch.arange(experts, device="cuda") % 11 / 64).to(torch.bfloat16)
    if scores == "equal":
        logits.zero_()
        bias.zero_()
    elif scores == "quantized":
        # Exactly representable sigmoid scores isolate tie policy from libm rounding.
        logits = (logits > 0).float() * 32
    elif scores == "saturated":
        logits.add_(32)
    if strided:
        logits = torch.stack((logits, torch.empty_like(logits)), dim=-1)[..., 0]
    common = dict(
        hidden_states=torch.empty((tokens, 1), device="cuda", dtype=torch.bfloat16),
        gating_output=logits,
        correction_bias=bias,
        topk=8,
        renormalize=True,
        num_expert_group=groups,
        topk_group=group_topk,
        scoring_func="sigmoid",
        group_score_used_topk_num=get_group_score_used_topk_num(group_topk, groups, 8),
    )
    expected_weights, expected_ids = triton_grouped_topk(**common)
    # Both paths must match the routing formula, including deterministic ties.
    reference_scores = logits.sigmoid()
    choice_scores = reference_scores + bias
    group_values = (
        choice_scores.reshape(tokens, groups, -1).topk(common["group_score_used_topk_num"], dim=-1).values.sum(-1)
    )
    threshold = group_values.topk(group_topk, dim=-1).values[:, -1:]
    allowed = (group_values >= threshold).repeat_interleave(experts // groups, dim=-1)
    reference_ids = choice_scores.masked_fill(~allowed, -float("inf")).argsort(descending=True, stable=True)[:, :8]
    torch.testing.assert_close(expected_ids, reference_ids, rtol=0, atol=0)
    reference_weights = reference_scores.gather(1, reference_ids)
    reference_weights /= reference_weights.sum(-1, keepdim=True)
    torch.testing.assert_close(expected_weights, reference_weights, rtol=1e-6, atol=1e-7)
    mapping = torch.arange(experts - 1, -1, -1, dtype=torch.int32, device="cuda")[:, None]
    counter = torch.zeros((1, experts), dtype=torch.int64, device="cuda")
    weights, physical_ids, logical_ids = triton_grouped_topk_eplb(
        **common,
        logical_to_physical_map=mapping,
        logical_replica_count=torch.ones(experts, dtype=torch.int32, device="cuda"),
        expert_counter=counter,
        sample_index=0,
        record_load=True,
        use_grouped_topk=True,
        return_logical_ids=True,
    )
    assert torch.equal(logical_ids, expected_ids)
    assert torch.equal(physical_ids, mapping[expected_ids, 0].long())
    torch.testing.assert_close(weights, expected_weights, rtol=1e-6, atol=1e-7)
    expected_counts = torch.bincount(expected_ids.flatten(), minlength=experts)
    assert torch.equal(counter[0], expected_counts)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required for the Triton EPLB kernel")
@pytest.mark.parametrize("record_load", [False, True])
@pytest.mark.parametrize("tokens", [1, 32])
def test_global_topk_eplb_supports_logical_ids_and_counting(record_load, tokens):
    from lightllm.common.basemodel.triton_kernel.fused_moe.grouped_topk import triton_grouped_topk_eplb

    torch.manual_seed(1234)
    topk = 4
    experts = 64
    gating_output = torch.randn((tokens, experts), dtype=torch.float32, device="cuda")
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
        replica_indices = _replica_index_reference(
            token_indices, expected_logical_ids, logical_replica_count[expected_logical_ids].to(torch.int64)
        )
    expected_ids = logical_to_physical[expected_logical_ids, replica_indices.to(torch.long)].to(torch.long)
    if record_load:
        expected_counter[1].scatter_add_(
            0,
            expected_logical_ids.reshape(-1),
            torch.ones(expected_logical_ids.numel(), dtype=torch.int64, device="cuda"),
        )
    expected_weights = expected_weights / expected_weights.sum(dim=-1, keepdim=True)

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


def test_last_layer_commit_finishes_at_outer_boundary_and_preserves_samples(monkeypatch):
    manager = manager_module.EPLBManager.__new__(manager_module.EPLBManager)
    state = SimpleNamespace(recording=True, recorded_sample_count=7)
    operations = []

    class Transfer:
        pending = [(0, 0)]

        def pending_layers(self):
            return self.pending[:]

        def commit(self, layer, buffer, post_copy):
            assert self.pending.pop(0) == (layer, buffer)
            post_copy()

        def finish(self):
            operations.append("finish")

    manager.transfer = Transfer()
    manager.in_flight = True
    manager.in_flight_layers = [0]
    manager.control_group = object()
    manager._control_ready_count = torch.empty(1, dtype=torch.int32)
    manager._inflight_timing = dict(
        control_poll_calls=0, control_cpu_ms=0.0, commit_calls=0, commit_cpu_ms=0.0, commit_max_ms=0.0
    )
    manager._commit_layer_metadata = lambda layer: operations.append(("metadata", layer))

    manager._eplb_states = [state]
    manager.target_placement = torch.zeros(1)
    manager.target_metadata = None
    manager.global_rank = 1
    manager.sampling_interval = 4
    manager._last_evaluation_step = 4
    monkeypatch.setattr(manager_module.dist, "all_reduce", lambda *args, **kwargs: None)
    monkeypatch.setattr(
        manager_module.torch.cuda, "current_stream", lambda: SimpleNamespace(wait_stream=lambda s: None)
    )
    monkeypatch.setattr(g_infer_context, "get_overlap_stream", lambda: object())
    manager.poll_migration_at_layer_boundary()
    assert operations == [("metadata", 0)]
    assert manager.in_flight and manager.in_flight_layers == []
    assert state.recording and state.recorded_sample_count == 7
    manager.poll_migration_at_layer_boundary()
    assert operations == [("metadata", 0)]
    manager.poll()
    assert operations == [("metadata", 0), "finish"]
    assert not manager.in_flight and state.recording and state.recorded_sample_count == 7
    assert manager._next_evaluation_step == 8


def test_layer_hook_does_not_start_an_evaluation():
    manager = manager_module.EPLBManager.__new__(manager_module.EPLBManager)
    manager.in_flight = False
    manager.evaluation_in_flight = True
    manager.poll_migration_at_layer_boundary()


def test_migration_metadata_is_committed_before_selecting_experts(monkeypatch):
    events = []
    state = EPLBState(
        num_redundant_experts_per_rank=1,
        logical_to_physical_map=torch.zeros((4, 2), dtype=torch.int32),
        logical_replica_count=torch.ones(4, dtype=torch.int32),
        route_counter=torch.zeros((4, 4), dtype=torch.int64),
        recording=True,
    )
    state.migration_poll = lambda: events.append(("migration", state.recorded_sample_count))
    impl = deepgemm_module.FuseMoeDeepGEMM.__new__(deepgemm_module.FuseMoeDeepGEMM)
    impl.eplb, impl.routed_scaling_factor = state, 1.0

    def topk(**kwargs):
        events.append(("topk", kwargs["sample_index"]))
        return torch.ones((2, 2)), torch.zeros((2, 2), dtype=torch.int64), None

    monkeypatch.setattr(grouped_topk, "triton_grouped_topk_eplb", topk)
    impl._select_experts(
        input_tensor=torch.zeros((2, 4)),
        router_logits=torch.zeros((2, 4)),
        correction_bias=None,
        top_k=2,
        renormalize=True,
        use_grouped_topk=True,
        topk_group=1,
        num_expert_group=1,
        scoring_func="sigmoid",
        is_prefill=True,
    )
    assert events == [("migration", 0), ("topk", 0)]
    assert state.recorded_sample_count == 1


def test_eplb_expert_scaling_uses_logical_ids_instead_of_physical_rows(monkeypatch):
    state = EPLBState(
        num_redundant_experts_per_rank=1,
        logical_to_physical_map=torch.zeros((4, 2), dtype=torch.int32),
        logical_replica_count=torch.ones(4, dtype=torch.int32),
        route_counter=torch.zeros((4, 4), dtype=torch.int64),
    )
    impl = deepgemm_module.FuseMoeDeepGEMM.__new__(deepgemm_module.FuseMoeDeepGEMM)
    impl.eplb, impl.routed_scaling_factor = state, 2.5

    def topk(**kwargs):
        assert kwargs["return_logical_ids"]
        return torch.ones((1, 2)), torch.tensor([[7, 8]]), torch.tensor([[1, 3]])

    monkeypatch.setattr(grouped_topk, "triton_grouped_topk_eplb", topk)
    weights, physical, logical = impl._select_experts(
        input_tensor=torch.zeros((1, 4)),
        router_logits=torch.zeros((1, 4)),
        correction_bias=None,
        top_k=2,
        renormalize=True,
        use_grouped_topk=True,
        topk_group=1,
        num_expert_group=1,
        scoring_func="sigmoid",
        is_prefill=True,
        per_expert_scale=torch.tensor([10.0, 20.0, 30.0, 40.0]),
    )
    assert weights.tolist() == [[50.0, 100.0]]
    assert physical.tolist() == [[7, 8]] and logical.tolist() == [[1, 3]]
