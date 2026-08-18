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
    estimate_rank_load,
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
from lightllm.server.router.model_infer.mode_backend.eplb_transfer import (
    TransferStep,
    build_transfer_plan,
    commit_staging_rows,
    extract_expert_tensors,
)
from lightllm.utils import envs_utils


def _test_parallel_state(
    *,
    eplb=False,
    logical_experts=128,
    world_size=16,
    redundant_experts_per_rank=1,
    route_counter=None,
    recording=False,
    recorded_sample_count=0,
):
    eplb_state = None
    if eplb:
        if route_counter is None:
            route_counter = torch.zeros((2, logical_experts), dtype=torch.int64)
        eplb_state = EPLBState(
            redundant_experts_per_rank=redundant_experts_per_rank,
            logical_to_physical_map=torch.zeros((logical_experts, 2), dtype=torch.int32),
            logical_replica_count=torch.ones(logical_experts, dtype=torch.int32),
            route_counter=route_counter,
            recording=recording,
            recorded_sample_count=recorded_sample_count,
        )
    return ExpertParallelState(
        logical_experts=logical_experts,
        world_size=world_size,
        eplb=eplb_state,
    )


def _validated_expert_parallel_state(
    *,
    eplb=True,
    n_routed_experts=4,
    world_size=2,
    redundant_experts_per_rank=1,
    device="cpu",
):
    runtime = None
    if eplb:
        runtime = EPLBState(
            redundant_experts_per_rank=redundant_experts_per_rank,
            logical_to_physical_map=torch.empty((n_routed_experts, 2), dtype=torch.int32, device=device),
            logical_replica_count=torch.ones(n_routed_experts, dtype=torch.int32, device=device),
            route_counter=torch.zeros((2, n_routed_experts), dtype=torch.int64, device=device),
        )
    return ExpertParallelState(
        logical_experts=n_routed_experts,
        world_size=world_size,
        eplb=runtime,
    )


def _set_expert_parallel_state(impl, state):
    impl.expert_parallel_state = state
    impl.eplb_state = state.eplb


def _manual_runtime_rank_load(source_load, placement, node_world_size, alignment):
    """Reference the committed runtime logical-to-physical maps on CPU."""
    samples, layers, nodes, logical_experts = source_load.shape
    ranks, redundant = placement.shape[1:]
    experts_per_rank = logical_experts // ranks
    physical_experts_per_rank = experts_per_rank + redundant
    raw = torch.zeros((samples, layers, ranks, logical_experts), dtype=torch.float64)
    for layer in range(layers):
        for source_node in range(nodes):
            logical_to_physical, replica_count = build_logical_to_physical_map(
                placement[layer],
                logical_experts,
                source_rank=source_node * node_world_size,
                node_world_size=node_world_size,
            )
            for expert in range(logical_experts):
                count = int(replica_count[expert].item())
                for physical_id in logical_to_physical[expert, :count].tolist():
                    rank = physical_id // physical_experts_per_rank
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
    assert state.primary_experts_per_rank == 2
    assert state.total_physical_experts == 6


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


def test_eplb_rebalance_once_disabled_by_default(monkeypatch):
    monkeypatch.delenv("LIGHTLLM_EPLB_REBALANCE_ONCE", raising=False)
    envs_utils.enable_eplb_rebalance_once.cache_clear()
    envs_utils.enable_env_vars.cache_clear()

    try:
        assert not envs_utils.enable_eplb_rebalance_once()
    finally:
        envs_utils.enable_eplb_rebalance_once.cache_clear()
        envs_utils.enable_env_vars.cache_clear()


def test_eplb_rebalance_once_can_be_enabled(monkeypatch):
    monkeypatch.setenv("LIGHTLLM_EPLB_REBALANCE_ONCE", "TRUE")
    envs_utils.enable_eplb_rebalance_once.cache_clear()
    envs_utils.enable_env_vars.cache_clear()

    try:
        assert envs_utils.enable_eplb_rebalance_once()
    finally:
        envs_utils.enable_eplb_rebalance_once.cache_clear()
        envs_utils.enable_env_vars.cache_clear()


def test_get_eplb_rebalance_gain_threshold_defaults_to_five_percent(monkeypatch):
    monkeypatch.delenv("LIGHTLLM_EPLB_REBALANCE_GAIN_THRESHOLD", raising=False)
    envs_utils.get_eplb_rebalance_gain_threshold.cache_clear()

    try:
        assert envs_utils.get_eplb_rebalance_gain_threshold() == 0.05
    finally:
        envs_utils.get_eplb_rebalance_gain_threshold.cache_clear()


@pytest.mark.parametrize(("configured", "expected"), [("0", 0.0), (".04", 0.04), ("1", 1.0)])
def test_get_eplb_rebalance_gain_threshold_reads_valid_values(monkeypatch, configured, expected):
    monkeypatch.setenv("LIGHTLLM_EPLB_REBALANCE_GAIN_THRESHOLD", configured)
    envs_utils.get_eplb_rebalance_gain_threshold.cache_clear()

    try:
        assert envs_utils.get_eplb_rebalance_gain_threshold() == expected
    finally:
        envs_utils.get_eplb_rebalance_gain_threshold.cache_clear()


@pytest.mark.parametrize("configured", ["-0.01", "1.01", "nan", "inf"])
def test_get_eplb_rebalance_gain_threshold_rejects_invalid_values(monkeypatch, configured):
    env_name = "LIGHTLLM_EPLB_REBALANCE_GAIN_THRESHOLD"
    monkeypatch.setenv(env_name, configured)
    envs_utils.get_eplb_rebalance_gain_threshold.cache_clear()

    try:
        with pytest.raises(ValueError, match=env_name):
            envs_utils.get_eplb_rebalance_gain_threshold()
    finally:
        envs_utils.get_eplb_rebalance_gain_threshold.cache_clear()
        monkeypatch.delenv(env_name, raising=False)


def test_eplb_redundant_experts_defaults_per_ep_rank():
    parser = make_argument_parser()

    assert parser.parse_args([]).eplb_num_redundant_experts_per_rank == 2
    assert parser.parse_args(["--eplb_num_redundant_experts_per_rank", "3"]).eplb_num_redundant_experts_per_rank == 3
    assert StartArgs().eplb_num_redundant_experts_per_rank == 2


@pytest.mark.parametrize(
    ("num_logical_experts", "num_ranks", "redundant_experts_per_rank", "expected"),
    [
        (8, 4, 2, [[2, 3], [4, 5], [6, 7], [0, 1]]),
        (6, 3, 4, [[2, 3, 4, 5], [4, 5, 0, 1], [0, 1, 2, 3]]),
    ],
)
def test_build_initial_redundant_expert_ids(
    num_logical_experts,
    num_ranks,
    redundant_experts_per_rank,
    expected,
):
    actual = build_initial_redundant_expert_ids(
        num_logical_experts,
        num_ranks,
        redundant_experts_per_rank,
    )

    assert actual.dtype == torch.int64
    assert actual.shape == (num_ranks, redundant_experts_per_rank)
    assert torch.equal(actual, torch.tensor(expected, dtype=torch.int64))


def test_plan_redundant_experts_never_uses_owner_or_duplicate_rank():
    expert_load = torch.tensor(
        [
            [100, 90, 80, 70, 60, 50, 40, 30],
            [30, 40, 50, 60, 70, 80, 90, 100],
        ]
    )
    placement = plan_redundant_experts(expert_load, num_ranks=4, redundant_experts_per_rank=2)

    for layer_placement in placement:
        for rank, expert_ids in enumerate(layer_placement.tolist()):
            assert len(expert_ids) == len(set(expert_ids))
            assert all(expert_id // 2 != rank for expert_id in expert_ids)


def test_plan_redundant_experts_minimizes_samplewise_aligned_critical_load():
    samples = torch.tensor([[[300, 20, 20, 200]], [[100, 300, 40, 160]]])
    placement = plan_redundant_experts(samples, num_ranks=2, redundant_experts_per_rank=1, expert_alignment=128)
    candidates = [torch.tensor([[[left], [right]]]) for left in (2, 3) for right in (0, 1)]

    def critical(candidate):
        return estimate_rank_load(samples, candidate, expert_alignment=128).max(dim=2).values.sum()

    assert torch.equal(placement, torch.tensor([[[3], [0]]]))
    assert critical(placement) == min(critical(candidate) for candidate in candidates)


def test_select_improving_placements_rejects_regressing_layer():
    expert_load = torch.tensor([[8649, 5740, 5002, 3441]])
    current = torch.tensor([[[2], [0]]])
    regressing_candidate = torch.tensor([[[1], [0]]])

    selected, improved, _ = select_improving_placements(expert_load, current, regressing_candidate)

    current_ratio = estimate_rank_load(expert_load, current).max() / estimate_rank_load(expert_load, current).mean()
    candidate_ratio = (
        estimate_rank_load(expert_load, regressing_candidate).max()
        / estimate_rank_load(expert_load, regressing_candidate).mean()
    )
    assert current_ratio.item() == pytest.approx(1.1007, abs=1e-4)
    assert candidate_ratio.item() == pytest.approx(1.1184, abs=1e-4)
    assert not improved.item()
    assert torch.equal(selected, current)


def test_select_improving_placements_rejects_low_imbalance():
    expert_load = torch.tensor([[1, 2, 1, 17]])
    current = torch.tensor([[[3], [0]]])
    candidate = torch.tensor([[[3], [1]]])

    selected, improved, _ = select_improving_placements(expert_load, current, candidate)

    current_ratio = estimate_rank_load(expert_load, current).max() / estimate_rank_load(expert_load, current).mean()
    candidate_ratio = (
        estimate_rank_load(expert_load, candidate).max() / estimate_rank_load(expert_load, candidate).mean()
    )
    assert current_ratio.item() == pytest.approx(1.047619, abs=1e-6)
    assert candidate_ratio.item() == pytest.approx(1.0)
    assert not improved.item()
    assert torch.equal(selected, current)


def test_select_improving_placements_rejects_insufficient_rebalance_gain():
    expert_load = torch.tensor([[1, 1, 6, 7]])
    current = torch.tensor([[[2], [0]]])
    candidate = torch.tensor([[[3], [0]]])

    selected, improved, _ = select_improving_placements(expert_load, current, candidate)

    current_ratio = estimate_rank_load(expert_load, current).max() / estimate_rank_load(expert_load, current).mean()
    candidate_ratio = (
        estimate_rank_load(expert_load, candidate).max() / estimate_rank_load(expert_load, candidate).mean()
    )
    relative_improvement = (current_ratio - candidate_ratio) / current_ratio
    assert current_ratio.item() == pytest.approx(1.4)
    assert candidate_ratio.item() == pytest.approx(1.333333, abs=1e-6)
    assert relative_improvement.item() == pytest.approx(0.047619, abs=1e-6)
    assert not improved.item()
    assert torch.equal(selected, current)

    selected, improved, _ = select_improving_placements(
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

    selected, improved, _ = select_improving_placements(expert_load, current, candidate)

    current_ratio = estimate_rank_load(expert_load, current).max() / estimate_rank_load(expert_load, current).mean()
    candidate_ratio = (
        estimate_rank_load(expert_load, candidate).max() / estimate_rank_load(expert_load, candidate).mean()
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

    _, raw_improved, _ = select_improving_placements(expert_load, current, raw_improving_candidate)
    selected, aligned_improved, _ = select_improving_placements(
        expert_load, current, raw_improving_candidate, expert_alignment=128
    )

    assert raw_improved.item()
    assert not aligned_improved.item()
    assert torch.equal(selected, current)


def test_estimate_rank_load_aligns_each_sample_before_accumulation():
    samples = torch.tensor([[[20, 0, 0, 0]], [[20, 0, 0, 0]]])
    placement = torch.tensor([[[2], [0]]])

    per_sample = estimate_rank_load(samples, placement, expert_alignment=128)
    accumulated = estimate_rank_load(samples.sum(dim=0), placement, expert_alignment=128)

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

    selected, improved, _ = select_improving_placements(
        samples, current, mean_inflating_candidate, expert_alignment=128
    )
    current_load = estimate_rank_load(samples, current, expert_alignment=128)
    candidate_load = estimate_rank_load(samples, mean_inflating_candidate, expert_alignment=128)
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

    selected, improved, _ = select_improving_placements(samples, current, candidate, expert_alignment=128)

    assert improved.item()
    assert torch.equal(selected, candidate)


def test_select_improving_placements_rejects_single_layer_gain_below_model_threshold():
    # Layer 0 becomes better, but layer 1 dominates model critical load.  The
    # aggregate estimated critical-load reduction gain is below 5%, so neither layer may be changed.
    expert_load = torch.tensor([[1, 1, 1, 2], [0, 0, 0, 10]])
    current = torch.tensor([[[2], [0]], [[2], [0]]])
    candidate = torch.tensor([[[3], [0]], [[2], [0]]])

    selected, improved, metrics = select_improving_placements(expert_load, current, candidate)

    assert not torch.any(improved)
    assert torch.equal(selected, current)
    assert metrics["candidate_rebalance_gain"] == pytest.approx(0.5 / 13)
    assert metrics["candidate_changed_layer_count"] == 1


def test_select_improving_placements_accepts_only_when_model_gain_reaches_threshold():
    expert_load = torch.tensor([[1, 1, 1, 2], [0, 0, 0, 5]])
    current = torch.tensor([[[2], [0]], [[2], [0]]])
    candidate = torch.tensor([[[3], [0]], [[2], [0]]])

    selected, improved, _ = select_improving_placements(expert_load, current, candidate)

    assert torch.equal(improved, torch.tensor([True, False]))
    assert torch.equal(selected, candidate)


def test_logical_to_physical_map_has_at_most_one_slot_per_rank():
    redundant_expert_ids = torch.tensor([[2, 3], [0, 1]])
    logical_to_physical, replica_count = build_logical_to_physical_map(redundant_expert_ids, num_logical_experts=4)

    assert logical_to_physical.shape == (4, 2)
    assert torch.equal(replica_count, torch.full((4,), 2, dtype=torch.int64))


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


def test_plan_redundant_experts_prefers_first_replica_on_new_node():
    # One redundant slot per rank leaves legal alternatives on both nodes;
    # topology preference therefore puts every first replica away from its
    # primary node before considering same-node duplicates.
    placement = plan_redundant_experts(
        torch.tensor([[1000, 900, 800, 700, 600, 500, 400, 300]]),
        num_ranks=4,
        redundant_experts_per_rank=1,
        node_world_size=2,
    )
    for rank, expert in enumerate(placement[0, :, 0].tolist()):
        assert expert // 2 // 2 != rank // 2


def test_plan_redundant_experts_single_node_matches_default_behavior():
    load = torch.tensor([[1000, 900, 800, 700, 600, 500, 400, 300]])
    default = plan_redundant_experts(load, num_ranks=4, redundant_experts_per_rank=1)
    single_node = plan_redundant_experts(load, num_ranks=4, redundant_experts_per_rank=1, node_world_size=4)
    assert torch.equal(single_node, default)


def test_source_node_estimate_matches_local_first_runtime_replica_sharing():
    # Expert 0 has a copy on each node.  Node 0 and node 1 issue unequal
    # traffic, so collapsing them before planning produces the wrong result.
    placement = torch.tensor([[[4], [5], [0], [1]]], dtype=torch.int64)
    source_load = torch.zeros((1, 1, 2, 8), dtype=torch.int64)
    source_load[0, 0, 0, 0] = 256
    source_load[0, 0, 1, 0] = 128

    predicted = estimate_rank_load(source_load, placement, expert_alignment=128, node_world_size=2)
    runtime = _manual_runtime_rank_load(source_load, placement, node_world_size=2, alignment=128)
    collapsed_global = estimate_rank_load(source_load.sum(dim=2), placement, expert_alignment=128)

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

    before = estimate_rank_load(source_load, initial, expert_alignment=128, node_world_size=2)
    after = estimate_rank_load(source_load, planned, expert_alignment=128, node_world_size=2)
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
    selected, _improved, _metrics = select_improving_placements(
        source_load, current, candidate, expert_alignment=128, node_world_size=2
    )
    assert torch.equal(
        estimate_rank_load(source_load, selected, 128, 2),
        _manual_runtime_rank_load(source_load, selected, 2, 128),
    )


def test_steady_state_sparse_sampling_arms_step_nineteen_and_evaluates_step_twenty(
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
                    logical_experts=4,
                    world_size=1,
                ),
            },
        )()
    ]
    manager.rebalanced = True
    manager.initial_sampling_complete = True
    manager.rebalance_once = False
    manager.in_flight = False
    manager.prefill_steps = 18
    manager.step_interval = 20
    manager.sampling_interval = manager.step_interval
    manager.evaluation_group = object()
    manager.num_logical_experts = 4
    manager.global_rank = 1
    manager.evaluation_in_flight = False
    manager._sampling_pending = False

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
    assert manager.prefill_steps == 19
    assert recordings == [True]
    assert resets == [True]
    assert manager._sampling_pending

    manager.step()
    assert manager.prefill_steps == 20
    assert not manager._sampling_pending
    assert started == [True]


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
                    logical_experts=4,
                    world_size=1,
                ),
            },
        )()
    ]
    manager.rebalanced = False
    manager.rebalance_once = False
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
    manager.rebalance_once = False
    manager.rebalanced = True
    manager.initial_sampling_complete = True
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
    manager.rebalanced = False
    manager.initial_sampling_complete = False
    manager.rebalance_once = False
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

    assert recordings == [False]
    assert not manager._sampling_pending
    assert starts == []
    assert manager.prefill_steps == 1

    manager.step()
    assert starts == []
    manager.step()
    assert recordings == [False, True]
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
    manager.step_interval = 1
    manager.sampling_interval = 1
    manager.rebalance_once = False
    manager.rebalanced = False
    manager.initial_sampling_complete = True
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
                    logical_experts=2,
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
                    logical_experts=2,
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
    assert weight.expert_parallel_state.total_physical_experts == weight.n_routed_experts
    assert weight._initial_redundant_expert_ids == []
    assert not hasattr(weight, "route_counter")
    assert not hasattr(weight, "routed_expert_counter_tensor")


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
                    logical_experts=4,
                    world_size=1,
                )
            },
        )()
    ]
    manager._eplb_states = [manager.weights[0].expert_parallel_state.eplb]
    manager.global_rank = 2
    manager.world_size = 4
    manager.node_world_size = 2
    manager.num_logical_experts = 4
    manager.redundant_experts_per_rank = 1
    manager.current_placement = build_initial_redundant_expert_ids(4, 4, 1).unsqueeze(0)
    manager.evaluation_group = object()
    manager._evaluation_lock = threading.Lock()
    manager._evaluation_result = None
    manager._evaluation_error = None
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
    assert seen["before"].shape == (1, 1, 2, 4)
    assert torch.equal(seen["before"][:, :, 0], torch.zeros_like(local))
    assert torch.equal(seen["before"][:, :, 1], local)
    assert torch.equal(seen["global_load"][:, :, 0], torch.full_like(local, 100))
    assert torch.equal(seen["global_load"][:, :, 1], local)
    assert manager._evaluation_error is None


def test_eplb_step_stops_after_completed_rebalance_in_once_mode():
    manager = manager_module.EPLBManager.__new__(manager_module.EPLBManager)
    manager.in_flight = False
    manager.evaluation_in_flight = False
    manager.rebalanced = True
    manager.initial_sampling_complete = True
    manager.rebalance_once = True
    manager.prefill_steps = 0

    manager.step()

    assert manager.prefill_steps == 0


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
    monkeypatch.setattr(
        deepgemm_module,
        "eplb_map_fast",
        lambda *_args, **_kwargs: pytest.fail("decode must not map"),
    )

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


def test_select_experts_and_quant_input_keeps_prefill_ids_logical(monkeypatch):
    from lightllm.common.basemodel.triton_kernel.fused_moe import topk_select

    impl = object.__new__(deepgemm_module.FuseMoeDeepGEMM)
    impl.routed_scaling_factor = 1.0
    impl.quant_method = object()
    _set_expert_parallel_state(impl, _test_parallel_state(eplb=True))
    logical_ids = torch.tensor([[3, 127]], dtype=torch.int32)
    monkeypatch.setattr(
        topk_select,
        "select_experts",
        lambda **_kwargs: (torch.ones((1, 2)), logical_ids),
    )
    monkeypatch.setattr(
        deepgemm_module,
        "eplb_map_fast",
        lambda *_args, **_kwargs: pytest.fail("selection must not map"),
    )
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
    assert topk_idx.dtype is torch.long
    assert topk_idx.tolist() == [[3, 127]]
    assert qinput == "qinput"


def test_prepare_prefill_topk_ids_returns_identity_without_eplb(monkeypatch):
    impl = object.__new__(deepgemm_module.FuseMoeDeepGEMM)
    _set_expert_parallel_state(impl, _test_parallel_state())
    logical_ids = torch.tensor([[1, 2]], dtype=torch.int64)
    monkeypatch.setattr(
        deepgemm_module,
        "eplb_map_fast",
        lambda *_args, **_kwargs: pytest.fail("must not map"),
    )

    assert impl._prepare_prefill_topk_ids(logical_ids) is logical_ids


def test_prefill_dispatch_without_routing_policy_preserves_event_and_avoids_routing_stream(
    monkeypatch,
):
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
    _set_expert_parallel_state(impl, _test_parallel_state())
    impl.ep_balance_counters = None
    calls = []
    caller_event = object()
    monkeypatch.setattr(
        deepgemm_module,
        "_get_prefill_routing_stream",
        lambda _device: pytest.fail("must not create routing stream"),
    )
    monkeypatch.setattr(
        deepgemm_module.ElasticBuffer,
        "capture",
        lambda: pytest.fail("must not capture routing event"),
    )
    monkeypatch.setattr(deepgemm_module.dist_group_manager, "ep_buffer", Buffer())
    monkeypatch.setattr(
        deepgemm_module,
        "get_deepep_num_max_dispatch_tokens_per_rank_prefill",
        lambda: 16,
    )
    monkeypatch.setattr(deepgemm_module, "get_ep_num_sms", lambda: 8)

    impl.dispatch(
        "qinput",
        torch.tensor([[1, 2]], dtype=torch.int32),
        torch.ones((1, 2)),
        caller_event,
    )

    assert calls[0]["previous_event"] is caller_event
    assert calls[0]["topk_idx"].dtype is torch.long


def test_prefill_dispatch_routes_external_logical_ids_once(monkeypatch):
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
    _set_expert_parallel_state(
        impl,
        _test_parallel_state(
            eplb=True,
            logical_experts=128,
            world_size=2,
            redundant_experts_per_rank=1,
            route_counter=torch.zeros((3, 128), dtype=torch.int64),
        ),
    )
    impl.ep_balance_counters = None
    mapped = []
    calls = []

    def map_in_place(topk_ids, *_args, **_kwargs):
        mapped.append(topk_ids)
        assert topk_ids.dtype is torch.int32
        assert topk_ids.is_contiguous()
        order.append("map")
        topk_ids.add_(10)

    monkeypatch.setattr(deepgemm_module, "eplb_map_fast", map_in_place)
    monkeypatch.setattr(deepgemm_module.dist_group_manager, "ep_buffer", Buffer())
    monkeypatch.setattr(
        deepgemm_module,
        "get_deepep_num_max_dispatch_tokens_per_rank_prefill",
        lambda: 16,
    )
    monkeypatch.setattr(deepgemm_module, "get_ep_num_sms", lambda: 8)
    order = []
    source_event = SimpleNamespace(current_stream_wait=lambda: order.append("source_wait"))
    dispatch_event = object()

    @contextmanager
    def fake_routing_stream(_stream):
        yield

    original_to = torch.Tensor.to

    def observe_long_conversion(tensor, *args, **kwargs):
        dtype = kwargs.get("dtype", args[0] if args else None)
        if dtype is torch.long:
            order.append("long")
        return original_to(tensor, *args, **kwargs)

    monkeypatch.setattr(deepgemm_module, "_get_prefill_routing_stream", lambda _device: object())
    monkeypatch.setattr(torch.cuda, "stream", fake_routing_stream)
    monkeypatch.setattr(torch.Tensor, "to", observe_long_conversion)
    monkeypatch.setattr(
        deepgemm_module.ElasticBuffer,
        "capture",
        lambda: order.append("capture") or dispatch_event,
    )
    logical_ids = torch.tensor([[1, 2]], dtype=torch.int64)

    result = impl.dispatch("qinput", logical_ids, torch.ones((1, 2)), source_event)

    assert len(mapped) == 1
    assert order == ["source_wait", "map", "long", "capture"]
    assert calls[0]["topk_idx"].dtype is torch.long
    assert calls[0]["topk_idx"].tolist() == [[11, 12]]
    assert calls[0]["previous_event"] is dispatch_event
    assert calls[0]["num_experts"] == 130
    assert result[3] == [4]


def test_deepgemm_constructor_configures_eplb():
    state = _validated_expert_parallel_state()
    impl = deepgemm_module.FuseMoeDeepGEMM(4, 0, 1.0, SimpleNamespace(), expert_parallel_state=state)
    assert impl.expert_parallel_state is state


def test_prefill_eplb_clones_logical_ids_only_for_metadata_capture(monkeypatch):
    from lightllm.common.basemodel.triton_kernel.fused_moe import topk_select

    impl = object.__new__(deepgemm_module.FuseMoeDeepGEMM)
    impl.routed_scaling_factor = 1.0
    _set_expert_parallel_state(
        impl,
        _test_parallel_state(
            eplb=True,
            route_counter=torch.zeros((2, 128), dtype=torch.int64),
        ),
    )
    impl._fused_experts = lambda **kwargs: kwargs["topk_ids"]

    def select(**_kwargs):
        return torch.ones((1, 2)), torch.tensor([[3, 4]], dtype=torch.int32)

    def map_in_place(topk_ids, *_args, **_kwargs):
        topk_ids.add_(10)

    monkeypatch.setattr(topk_select, "select_experts", select)
    monkeypatch.setattr(deepgemm_module, "eplb_map_fast", map_in_place)
    clone_calls = []
    original_clone = torch.Tensor.clone

    def clone_spy(tensor, *args, **kwargs):
        clone_calls.append(tensor.data_ptr())
        return original_clone(tensor, *args, **kwargs)

    monkeypatch.setattr(torch.Tensor, "clone", clone_spy)
    result = impl(
        torch.empty((1, 4)),
        torch.empty((1, 128)),
        None,
        None,
        None,
        "softmax",
        2,
        False,
        False,
        0,
        0,
        is_prefill=True,
    )
    assert result.tolist() == [[13, 14]]
    assert clone_calls == []

    captured = []
    result = impl(
        torch.empty((1, 4)),
        torch.empty((1, 128)),
        None,
        None,
        None,
        "softmax",
        2,
        False,
        False,
        0,
        0,
        is_prefill=True,
        moe_capture_callback=lambda ids: captured.append(ids.clone()),
    )
    assert result.tolist() == [[13, 14]]
    assert captured[0].tolist() == [[3, 4]]
    assert len(clone_calls) == 2  # EPLB preserves logical IDs, then the test captures a stable copy.


def test_fused_experts_with_topk_prefill_eplb_maps_and_records_in_place(monkeypatch):
    impl = object.__new__(deepgemm_module.FuseMoeDeepGEMM)
    _set_expert_parallel_state(
        impl,
        _test_parallel_state(
            eplb=True,
            logical_experts=4,
            world_size=1,
            route_counter=torch.zeros((3, 4), dtype=torch.int64),
            recording=True,
            recorded_sample_count=4,
        ),
    )
    calls, fused_calls = [], []

    def map_in_place(topk_ids, logical_map, replica_count, counter, sample_index, *, record_load):
        calls.append((topk_ids, logical_map, replica_count, counter, sample_index, record_load))
        topk_ids.add_(10)

    monkeypatch.setattr(deepgemm_module, "eplb_map_fast", map_in_place)
    impl._fused_experts = lambda **kwargs: fused_calls.append(kwargs) or "out"
    input_tensor, w13, w2 = object(), object(), object()
    topk_weights = torch.ones((1, 2))
    topk_ids = torch.tensor([[1, 2]], dtype=torch.int32)

    assert impl.fused_experts_with_topk(input_tensor, w13, w2, topk_weights, topk_ids, is_prefill=True) == "out"
    assert calls == [
        (
            topk_ids,
            impl.eplb_state.logical_to_physical_map,
            impl.eplb_state.logical_replica_count,
            impl.eplb_state.route_counter,
            1,
            True,
        )
    ]
    assert impl.eplb_state.recorded_sample_count == 5
    assert topk_ids.tolist() == [[11, 12]]
    assert fused_calls == [
        {
            "input_tensor": input_tensor,
            "w13": w13,
            "w2": w2,
            "topk_weights": topk_weights,
            "topk_ids": topk_ids,
            "is_prefill": True,
        }
    ]


def test_fused_experts_with_topk_prefill_eplb_maps_without_recording(monkeypatch):
    impl = object.__new__(deepgemm_module.FuseMoeDeepGEMM)
    _set_expert_parallel_state(
        impl,
        _test_parallel_state(
            eplb=True,
            logical_experts=4,
            world_size=1,
            route_counter=torch.zeros((3, 4), dtype=torch.int64),
            recorded_sample_count=4,
        ),
    )
    calls = []
    monkeypatch.setattr(
        deepgemm_module,
        "eplb_map_fast",
        lambda *args, **kwargs: calls.append((args, kwargs)) or args[0].add_(10),
    )
    impl._fused_experts = lambda **kwargs: kwargs["topk_ids"]
    topk_ids = torch.tensor([[1, 2]], dtype=torch.int32)

    assert (
        impl.fused_experts_with_topk(object(), object(), object(), torch.ones((1, 2)), topk_ids, is_prefill=True)
        is topk_ids
    )
    assert calls[0][0][4] == 0
    assert calls[0][1] == {"record_load": False}
    assert impl.eplb_state.recorded_sample_count == 4
    assert topk_ids.tolist() == [[11, 12]]


def test_decode_masked_group_gemm_uses_primary_rows_only_when_eplb_is_enabled(
    monkeypatch,
):
    impl = object.__new__(deepgemm_module.FuseMoeDeepGEMM)
    _set_expert_parallel_state(impl, _test_parallel_state(eplb=True, logical_experts=8, world_size=1))
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
        _test_parallel_state(eplb=True, logical_experts=128, world_size=16, redundant_experts_per_rank=2),
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
    commit_staging_rows(live, staging, experts_per_rank=3, changed_dst_slots=(0, 1))
    assert torch.equal(live[:3], torch.arange(12).reshape(3, 4))
    assert torch.equal(live[3:], staging)


def test_commit_staging_rows_preserves_unchanged_destination_slots():
    live = torch.arange(28).reshape(7, 4)
    staging = torch.tensor([[-1, -1, -1, -1], [-2, -2, -2, -2], [-3, -3, -3, -3], [-4, -4, -4, -4]])
    original = live.clone()

    commit_staging_rows(live, staging, experts_per_rank=3, changed_dst_slots=(3, 1))

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
        experts_per_rank=10,
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
    with pytest.raises(RuntimeError, match="boom"):
        manager._poll_in_flight()


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
    transfer.experts_per_rank = 0
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


@pytest.mark.parametrize(("rebalance_once", "expected_calls"), [(True, [False]), (False, [True])])
def test_manager_rearms_after_rebalance_only_for_continuous_interval_one(rebalance_once, expected_calls):
    recording_calls = []
    manager = manager_module.EPLBManager.__new__(manager_module.EPLBManager)
    manager.rebalance_once = rebalance_once
    manager.rebalanced = False
    manager.initial_sampling_complete = True
    manager.step_interval = 1
    manager.sampling_interval = 1
    manager._sampling_pending = False
    manager.weights = []
    manager._eplb_states = []
    manager.target_placement = torch.zeros(1)
    manager.in_flight_started_at = 0
    manager.global_rank = 1
    manager._set_recording = lambda enabled: recording_calls.append(enabled)
    manager._finish_rebalance()
    assert manager.in_flight is False
    assert manager.rebalanced
    assert recording_calls == expected_calls
    assert not manager._sampling_pending


def test_manager_keeps_recording_closed_after_insufficient_interval_window(monkeypatch):
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
    manager.rebalance_once = False
    manager.rebalanced = True
    manager.initial_sampling_complete = True
    manager.weights = []
    manager._eplb_states = []
    recordings = []
    manager._set_recording = lambda enabled: recordings.append(enabled)

    assert not manager._poll_evaluation()
    assert recordings == [False]
    assert manager.sampling_interval == 20


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
    manager.rebalance_once = False
    manager.rebalanced = False
    manager.initial_sampling_complete = False
    manager.weights = []
    manager._eplb_states = []
    recordings = []
    manager._set_recording = lambda enabled: recordings.append(enabled)

    assert not manager._poll_evaluation()
    assert manager.initial_sampling_complete
    assert recordings == [False]
    assert manager.sampling_interval == 80


def test_dense_initial_window_evaluates_only_at_step_twenty(monkeypatch):
    manager = manager_module.EPLBManager.__new__(manager_module.EPLBManager)
    manager.in_flight = False
    manager.rebalanced = False
    manager.initial_sampling_complete = False
    manager.rebalance_once = False
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
    manager.rebalance_once = False
    manager.rebalanced = True
    manager.initial_sampling_complete = True
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
    manager.rebalanced = True
    manager.initial_sampling_complete = True
    manager.rebalance_once = False
    manager.prefill_steps = 18
    manager.step_interval = 20
    manager.sampling_interval = 80
    manager._sampling_pending = False
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

    manager.prefill_steps = 78
    manager.step()
    assert manager.prefill_steps == 79
    assert recordings == [True]
    assert resets == [True]
    assert manager._sampling_pending

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
    manager.initial_sampling_complete = True
    manager.global_rank = 1
    manager.transfer = type("Transfer", (), {"start": lambda self, plans: setattr(self, "plans", plans)})()
    manager._reset_recorded_samples = lambda: None
    monkeypatch.setattr(manager_module, "build_transfer_plan", lambda *args: object())

    manager._start_rebalance(
        {
            "placement": torch.zeros((1, 1, 1), dtype=torch.int64),
            "improved": torch.tensor([True]),
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


def test_first_rebalance_completion_switches_to_sparse_step_nineteen_arm(monkeypatch):
    manager = manager_module.EPLBManager.__new__(manager_module.EPLBManager)
    manager.rebalance_once = False
    manager.rebalanced = False
    manager.initial_sampling_complete = True
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
    manager.step()
    assert recordings == [False, True]
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


def test_nixl_descriptor_caches_are_bounded_to_the_current_transfer_generation(
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
    transfer._push_descriptor_cache = {}
    transfer._used_push_descriptor_cache_keys = set()
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
    transfer.experts_per_rank = 0
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
    source_a, destination_a = Tensor(300), Tensor(400)
    source_b, destination_b = Tensor(500), Tensor(600)
    copies_a = [(destination_a, source_a)]
    copies_b = [(destination_b, source_b)]
    key_a = ((source_a.data_ptr(), destination_a.data_ptr()),)
    key_b = ((source_b.data_ptr(), destination_b.data_ptr()),)
    stale_key = ((700, 800),)
    transfer._push_descriptor_cache = {
        key_a: (object(), object()),
        stale_key: (object(), object()),
    }
    generation = [entries_a, copies_a]

    def copy_batch(_batch):
        transfer._get_remote_read(1, generation[0])
        transfer._cached_descriptor_tensors(generation[1])

    transfer._copy_batch = copy_batch
    monkeypatch.setattr(transfer_module.torch.cuda, "set_device", lambda _device: None)

    transfer.start([(0, [])])
    transfer.finish()
    assert agent.made == 1
    assert agent.released_xfers == agent.released_dlists == 0
    assert set(transfer._push_descriptor_cache) == {key_a}
    assert len(transfer._xfer_cache) == 1

    # The real manager releases each staging buffer through commit(). This
    # focused cache test has no commits, so model that hand-off before the
    # next generation reuses buffer zero.
    transfer._release[0].set()
    transfer.start([(0, [])])
    transfer.finish()
    assert agent.made == 1
    assert agent.released_xfers == agent.released_dlists == 0
    assert set(transfer._push_descriptor_cache) == {key_a}
    assert len(transfer._xfer_cache) == 1

    transfer._push_descriptor_cache[key_b] = (object(), object())
    generation[:] = [entries_b, copies_b]
    transfer._release[0].set()
    transfer.start([(0, [])])
    transfer.finish()
    assert agent.made == 2
    assert agent.released_xfers == 1
    assert agent.released_dlists == 2
    assert set(transfer._push_descriptor_cache) == {key_b}
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

        def synchronize(self):
            self.synchronized += 1

    @contextmanager
    def use_stream(_stream):
        yield

    stream = Stream()
    transfer = object.__new__(transfer_module.NixlEPLBTransfer)
    transfer.global_rank = 0
    transfer.experts_per_rank = 2
    transfer._same_node_ranks = {0, 1}
    transfer._push_stream = stream
    remote_reads, pushed, waited_xfers = [], [], []
    transfer._push_same_node = lambda dst_rank, entries: pushed.append((dst_rank, entries))
    transfer._get_remote_read = lambda src_rank, entries: remote_reads.append((src_rank, entries)) or (
        None,
        None,
        "xfer",
    )
    transfer._wait_xfers = waited_xfers.extend

    monkeypatch.setattr(transfer_module.torch.cuda, "stream", use_stream)
    staging = []
    local_step = TransferStep(1, 0, 0, 0)
    remote_step = TransferStep(0, 1, 2, 0)

    transfer._copy_batch([(0, [local_step], 0, staging)])
    assert pushed == [(1, [(0, [local_step], 0)])]
    assert remote_reads == []

    transfer._copy_batch([(0, [remote_step], 0, staging)])
    assert [rank for rank, _ in remote_reads] == [2]
    assert waited_xfers == [(None, None, "xfer")]


def test_nixl_copy_batch_self_only_rank_pushes_and_synchronizes(monkeypatch):
    class Stream:
        def __init__(self):
            self.synchronized = 0

        def synchronize(self):
            self.synchronized += 1

    @contextmanager
    def use_stream(_stream):
        yield

    transfer = object.__new__(transfer_module.NixlEPLBTransfer)
    transfer.global_rank = 0
    transfer.experts_per_rank = 2
    transfer._same_node_ranks = {0}
    transfer._push_stream = Stream()
    pushed = []
    transfer._push_same_node = lambda dst_rank, entries: pushed.append((dst_rank, entries))
    transfer._wait_xfers = lambda _xfers: None
    monkeypatch.setattr(transfer_module.torch.cuda, "stream", use_stream)

    self_step = TransferStep(0, 0, 0, 0)
    transfer._copy_batch([(0, [self_step], 0, [])])

    assert pushed == [(0, [(0, [self_step], 0)])]
    assert transfer._push_stream.synchronized == 1


def test_manager_constructs_nixl_transfer(monkeypatch):
    weight = type(
        "Weight",
        (),
        {
            "n_routed_experts": 4,
            "expert_parallel_state": _test_parallel_state(
                eplb=True,
                logical_experts=4,
                world_size=2,
                redundant_experts_per_rank=2,
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
    monkeypatch.setattr(manager_module, "enable_eplb_rebalance_once", lambda: False)

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
    assert not manager.initial_sampling_complete
    assert weight.expert_parallel_state.eplb.recording
    assert manager._eplb_states[0] is weight.expert_parallel_state.eplb
    assert not hasattr(weight.expert_parallel_state.eplb, "record_load")


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required for the Triton EPLB kernel")
def test_eplb_record_flag_controls_counter_without_changing_mapping():
    from lightllm.common.basemodel.triton_kernel.fused_moe.eplb_kernels import eplb_map

    topk_ids = torch.tensor([[0, 1], [0, 2]], dtype=torch.int64, device="cuda")
    logical_to_physical = torch.tensor([[0, 4], [1, -1], [2, -1], [3, -1]], dtype=torch.int64, device="cuda")
    replica_count = torch.tensor([2, 1, 1, 1], dtype=torch.int64, device="cuda")
    expert_counter = torch.zeros((2, 4), dtype=torch.int64, device="cuda")
    record_load = torch.ones((), dtype=torch.int32, device="cuda")

    eplb_map(
        topk_ids,
        logical_to_physical,
        replica_count,
        expert_counter,
        record_load,
        sample_index=0,
    )
    torch.cuda.synchronize()
    assert torch.equal(topk_ids.cpu(), torch.tensor([[0, 1], [4, 2]]))
    assert torch.equal(expert_counter.cpu(), torch.tensor([[2, 1, 1, 0], [0, 0, 0, 0]]))

    logical_topk_ids = torch.tensor([[0, 1], [0, 2]], dtype=torch.int64, device="cuda")
    graph_topk_ids = torch.empty_like(logical_topk_ids)
    expert_counter.zero_()
    record_load.fill_(1)
    graph_topk_ids.copy_(logical_topk_ids)
    eplb_map(
        graph_topk_ids,
        logical_to_physical,
        replica_count,
        expert_counter,
        record_load,
        sample_index=1,
    )
    torch.cuda.synchronize()

    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        graph_topk_ids.copy_(logical_topk_ids)
        eplb_map(
            graph_topk_ids,
            logical_to_physical,
            replica_count,
            expert_counter,
            record_load,
            sample_index=1,
        )

    expert_counter.zero_()
    graph.replay()
    torch.cuda.synchronize()
    assert torch.equal(graph_topk_ids.cpu(), torch.tensor([[0, 1], [4, 2]]))
    assert torch.equal(expert_counter.cpu(), torch.tensor([[0, 0, 0, 0], [2, 1, 1, 0]]))

    expert_counter.zero_()
    record_load.fill_(0)
    graph.replay()
    torch.cuda.synchronize()
    assert torch.equal(graph_topk_ids.cpu(), torch.tensor([[0, 1], [4, 2]]))
    assert torch.equal(expert_counter.cpu(), torch.zeros((2, 4), dtype=torch.int64))

    topk_ids.copy_(torch.tensor([[0, 1], [0, 2]], dtype=torch.int64, device="cuda"))
    record_load.fill_(0)
    eplb_map(
        topk_ids,
        logical_to_physical,
        replica_count,
        expert_counter,
        record_load,
        sample_index=0,
    )
    torch.cuda.synchronize()
    assert torch.equal(topk_ids.cpu(), torch.tensor([[0, 1], [4, 2]]))
    assert torch.equal(expert_counter.cpu(), torch.zeros((2, 4), dtype=torch.int64))


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required for the Triton EPLB kernel")
def test_eplb_map_uses_prebuilt_source_node_local_map():
    from lightllm.common.basemodel.triton_kernel.fused_moe.eplb_kernels import eplb_map

    redundant = torch.tensor([[4], [5], [0], [1]], dtype=torch.int64)
    rank2_map, rank2_count = build_logical_to_physical_map(
        redundant, num_logical_experts=8, source_rank=2, node_world_size=2
    )
    topk_ids = torch.tensor([[0], [0]], dtype=torch.int64, device="cuda")
    counter = torch.zeros((1, 8), dtype=torch.int64, device="cuda")
    eplb_map(
        topk_ids,
        rank2_map.cuda(),
        rank2_count.cuda(),
        counter,
        torch.zeros((), dtype=torch.int32, device="cuda"),
        sample_index=0,
    )
    torch.cuda.synchronize()
    assert torch.equal(topk_ids.cpu(), torch.tensor([[8], [8]], dtype=torch.int64))
