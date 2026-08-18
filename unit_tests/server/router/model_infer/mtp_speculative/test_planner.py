from types import SimpleNamespace

import numpy as np
import torch

from lightllm.server.router.model_infer.mtp_speculative.engine import SpecEngine
from lightllm.server.router.model_infer.mtp_speculative.planner import (
    DSparkPlanner,
    FixedSpecPlanner,
    LightSpecPlanner,
    SpecDecodePlan,
    _InferCostMsTable,
)
from lightllm.server.router.model_infer.mtp_speculative.proposers.base import SpecProposal
from lightllm.server.router.model_infer.mtp_speculative.proposers.dflash import DFlashProposer
from lightllm.server.router.model_infer.mtp_speculative.proposers.dspark import DSparkProposer
from lightllm.server.router.model_infer.mtp_speculative.proposers.eagle3 import Eagle3Proposer
from lightllm.server.router.model_infer.mtp_speculative.proposers.eagle_mtp import (
    AutoregressiveEagleProposer,
    EagleMTPProposer,
)
from lightllm.server.router.model_infer.mtp_speculative.proposers.parallel_block import ParallelBlockProposer
from lightllm.server.router.model_infer.mtp_speculative.proposers.vanilla_mtp import VanillaMTPProposer


def build_draft_cost_provider(proposer_class, max_draft_step: int = 3, block_size: int = 3):
    backend = SimpleNamespace(
        max_draft_step=max_draft_step,
        draft_models=[SimpleNamespace(block_size=block_size)],
    )
    return proposer_class(backend=backend, enable_dynmaic_mtp=True)


def build_lightspec_planner(
    max_draft_step: int = 3,
    proposer_class=VanillaMTPProposer,
    block_size: int = 3,
):
    return LightSpecPlanner(
        max_draft_step=max_draft_step,
        draft_cost_provider=build_draft_cost_provider(
            proposer_class=proposer_class,
            max_draft_step=max_draft_step,
            block_size=block_size,
        ),
    )


def build_dspark_planner(max_draft_step: int = 3, block_size: int = 3):
    return DSparkPlanner(
        max_draft_step=max_draft_step,
        draft_cost_provider=build_draft_cost_provider(
            proposer_class=DSparkProposer,
            max_draft_step=max_draft_step,
            block_size=block_size,
        ),
    )


def build_planner(spec_mode: str, enable_dynmaic_mtp: bool = True):
    engine = SpecEngine.__new__(SpecEngine)
    engine.spec_mode = spec_mode
    engine.enable_dynmaic_mtp = enable_dynmaic_mtp
    engine.backend = SimpleNamespace(
        max_draft_step=3,
        draft_models=[SimpleNamespace(block_size=3)],
    )
    proposer_class = {
        "dspark": DSparkProposer,
        "dflash": DFlashProposer,
        "eagle3": Eagle3Proposer,
    }.get(spec_mode, VanillaMTPProposer)
    engine.proposer = build_draft_cost_provider(proposer_class)
    return engine._build_decode_planner()


def test_fixed_planner_returns_static_plan():
    plan = FixedSpecPlanner(max_draft_step=3).plan(req_num=4, original_batch_size=16)

    assert not plan.is_dynamic
    assert plan.dynamic_batch_size is None
    assert plan.draft_step == plan.pre_draft_step == 3
    assert not plan.skip_verify_sync


def test_infer_cost_candidates_include_feasible_boundaries():
    costs = _InferCostMsTable()
    costs.update(batch_size=4, infer_cost_ms=1.0)
    costs.update(batch_size=8, infer_cost_ms=2.0)

    assert costs.get_batch_size_keys_between(5, 10) == [5, 8, 10]


def test_engine_routes_only_dspark_to_the_confidence_planner():
    assert isinstance(build_planner("eagle3", enable_dynmaic_mtp=False), FixedSpecPlanner)
    assert isinstance(build_planner("dspark"), DSparkPlanner)

    dflash_planner = build_planner("dflash")
    assert isinstance(dflash_planner, LightSpecPlanner)
    assert dflash_planner.draft_steps == (3,)

    eagle_planner = build_planner("eagle3")
    assert isinstance(eagle_planner, LightSpecPlanner)
    assert eagle_planner.draft_steps == (1, 2, 3)


def test_proposer_families_use_drafter_standard_abstractions():
    assert issubclass(EagleMTPProposer, AutoregressiveEagleProposer)
    assert issubclass(Eagle3Proposer, AutoregressiveEagleProposer)
    assert issubclass(DFlashProposer, ParallelBlockProposer)
    assert issubclass(DSparkProposer, ParallelBlockProposer)
    assert not issubclass(DSparkProposer, DFlashProposer)


def test_dynamic_plan_filters_selected_rows():
    plan = SpecDecodePlan(dynamic_batch_size=2, draft_step=3, pre_draft_step=3)
    reqs = ["req0", "req0", "req1", "req1"]

    selected_reqs = plan.filter_reqs(
        reqs=reqs,
        selected_row_mask_cpu=torch.tensor([1, 0, 1, 0], dtype=torch.int32),
    )

    assert selected_reqs == ["req0", "req1"]


def test_dynamic_decode_frees_unselected_and_rejected_rows():
    freed = []
    engine = SpecEngine.__new__(SpecEngine)
    engine.backend = SimpleNamespace(
        model=SimpleNamespace(
            req_manager=SimpleNamespace(mem_manager=SimpleNamespace(free=lambda indexes: freed.append(indexes.clone())))
        )
    )
    model_input = SimpleNamespace(mem_indexes_cpu=torch.tensor([10, 11, 12, 13]))

    engine.free_unused_decode_mem(
        model_input=model_input,
        selected_row_mask_cpu=torch.tensor([1, 0, 1, 0], dtype=torch.bool),
        accepted_index_cpu=torch.tensor([1, 0], dtype=torch.int32),
        extra_mem_indexes_cpu=torch.tensor([20]),
    )

    assert len(freed) == 1
    assert freed[0].tolist() == [11, 12, 13, 20]


def test_engine_records_request_spec_metrics_in_one_pass():
    class MetricReq:
        def __init__(self, req_idx: int, mtp_step: int):
            self.req_idx = req_idx
            self.mtp_step = mtp_step
            self.accepted = 0
            self.verified = 0
            self.verify_steps = 0

        def update_mtp_accepted_token_num(self, accept_token_num: int):
            self.accepted += accept_token_num

        def update_mtp_verify_token_num(self, verify_token_num: int):
            self.verified += verify_token_num

        def update_mtp_verify_step_num(self, verify_step_num: int):
            self.verify_steps += verify_step_num

    engine = SpecEngine.__new__(SpecEngine)
    engine.backend = SimpleNamespace(is_master_in_dp=True)
    req0 = MetricReq(req_idx=10, mtp_step=3)
    req1 = MetricReq(req_idx=11, mtp_step=3)

    engine.record_request_spec_metrics(
        decode_reqs=[req0, req1],
        accept_lengths_cpu=torch.tensor([2, 1], dtype=torch.int32),
        verified_row_reqs=[req0, req0, req1],
    )

    assert (req0.accepted, req0.verified, req0.verify_steps) == (1, 2, 1)
    assert (req1.accepted, req1.verified, req1.verify_steps) == (0, 1, 1)

    fixed_req = MetricReq(req_idx=12, mtp_step=3)
    engine.record_request_spec_metrics(
        decode_reqs=[fixed_req],
        accept_lengths_cpu=torch.tensor([3], dtype=torch.int32),
    )
    assert (fixed_req.accepted, fixed_req.verified, fixed_req.verify_steps) == (2, 4, 1)


def test_lightspec_stays_full_width_until_costs_are_profiled():
    plan = build_lightspec_planner().plan(
        req_num=2,
        original_batch_size=8,
        proposal_req_num=2,
    )

    assert plan.dynamic_batch_size == 8
    assert plan.draft_step == plan.pre_draft_step == 3


def test_lightspec_collects_full_width_progress_before_adapting():
    planner = build_lightspec_planner(proposer_class=EagleMTPProposer)
    for batch_size in (2, 4, 8):
        planner.update_infer_cost(batch_size, infer_cost_ms=float(batch_size), is_draft_model=False)
        planner.update_infer_cost(batch_size, infer_cost_ms=float(batch_size), is_draft_model=True)

    plan = planner.plan(req_num=2, original_batch_size=8, proposal_req_num=2)

    assert plan.dynamic_batch_size == 8
    assert plan.draft_step == plan.pre_draft_step == 3


def test_lightspec_selects_eagle_draft_depth_and_verify_capacity():
    planner = build_lightspec_planner(proposer_class=EagleMTPProposer)
    for batch_size, target_cost in ((2, 1.0), (4, 1.1), (8, 10.0)):
        planner.update_infer_cost(batch_size, target_cost, is_draft_model=False)
    for batch_size, draft_cost in ((2, 0.1), (4, 0.1), (8, 0.2)):
        planner.update_infer_cost(batch_size, draft_cost, is_draft_model=True)
    for _ in range(8):
        planner.update_verified_batch(
            accept_lengths=[4, 4],
            req_num=2,
            dynamic_batch_size=8,
            verified_draft_step=3,
        )

    plan = planner.plan(req_num=2, original_batch_size=8, proposal_req_num=2)

    assert plan.dynamic_batch_size == 4
    assert plan.pre_draft_step == 3
    assert plan.draft_step == 1


def test_lightspec_compacts_block_verify_without_changing_draft_shape():
    planner = build_lightspec_planner(
        max_draft_step=7,
        proposer_class=DFlashProposer,
        block_size=7,
    )
    for batch_size, target_cost in ((2, 1.0), (4, 1.1), (8, 3.0), (16, 8.0)):
        planner.update_infer_cost(batch_size, target_cost, is_draft_model=False)
    planner.update_infer_cost(batch_size=14, infer_cost_ms=0.5, is_draft_model=True)
    for _ in range(8):
        planner.update_verified_batch(
            accept_lengths=[3, 3],
            req_num=2,
            dynamic_batch_size=16,
            verified_draft_step=7,
        )

    plan = planner.plan(req_num=2, original_batch_size=16, proposal_req_num=2)

    assert plan.dynamic_batch_size < 16
    assert plan.draft_step == plan.pre_draft_step == 7


def test_lightspec_bounds_verify_to_existing_proposals():
    planner = build_lightspec_planner()

    cold_start = planner.plan(req_num=2, original_batch_size=8, proposal_req_num=0)
    mixed_batch = planner.plan(req_num=2, original_batch_size=8, proposal_req_num=1)
    ready_batch = planner.plan(req_num=2, original_batch_size=8, proposal_req_num=2)

    assert cold_start.dynamic_batch_size == 2
    assert mixed_batch.dynamic_batch_size == 5
    assert ready_batch.dynamic_batch_size == 8
    assert not cold_start.all_reqs_have_proposals
    assert not mixed_batch.all_reqs_have_proposals
    assert ready_batch.all_reqs_have_proposals


def test_engine_counts_requests_with_a_previous_proposal():
    engine = SpecEngine.__new__(SpecEngine)
    engine.planner = build_lightspec_planner()
    model_input = SimpleNamespace(batch_size=8)

    mixed_plan = engine.plan_decode(
        model_input=model_input,
        decode_reqs=[SimpleNamespace(cur_output_len=1), SimpleNamespace(cur_output_len=2)],
    )
    ready_plan = engine.plan_decode(
        model_input=model_input,
        decode_reqs=[SimpleNamespace(cur_output_len=2), SimpleNamespace(cur_output_len=2)],
    )

    assert mixed_plan.dynamic_batch_size == 5
    assert not mixed_plan.all_reqs_have_proposals
    assert ready_plan.dynamic_batch_size == 8
    assert ready_plan.all_reqs_have_proposals


def test_lightspec_eagle_draft_always_keeps_the_extend_candidate():
    planner = build_lightspec_planner(proposer_class=EagleMTPProposer)
    for batch_size in (2, 4, 8):
        planner.update_infer_cost(batch_size, infer_cost_ms=float(batch_size), is_draft_model=False)
        planner.update_infer_cost(batch_size, infer_cost_ms=float(batch_size), is_draft_model=True)
    planner.update_verified_batch(
        accept_lengths=[2, 2],
        req_num=2,
        dynamic_batch_size=8,
        verified_draft_step=3,
    )

    plan = planner.plan(req_num=2, original_batch_size=8, proposal_req_num=2)

    assert planner.draft_steps == (1, 2, 3)
    assert plan.draft_step >= 1


def test_vanilla_cost_provider_prices_each_selected_mtp_module():
    planner = build_lightspec_planner()
    planner.update_infer_cost(batch_size=8, infer_cost_ms=0.25, is_draft_model=True)

    draft_cost_ms = planner.draft_cost_provider.get_draft_cost_ms(
        draft_infer_costs=planner.draft_infer_costs,
        req_num=2,
        verify_batch_size=8,
        draft_step=3,
    )

    assert draft_cost_ms == 0.75


def test_eagle_cost_provider_prices_extend_and_decode_separately():
    planner = build_lightspec_planner(proposer_class=EagleMTPProposer)
    planner.update_infer_cost(batch_size=2, infer_cost_ms=0.25, is_draft_model=True)

    draft_cost_ms = planner.draft_cost_provider.get_draft_cost_ms(
        draft_infer_costs=planner.draft_infer_costs,
        req_num=2,
        verify_batch_size=8,
        draft_step=3,
    )

    assert draft_cost_ms == 1.5


def test_autoregressive_eagle_cost_provider_prices_extend_and_decode_rows():
    for proposer_class in (EagleMTPProposer, Eagle3Proposer):
        planner = build_lightspec_planner(
            max_draft_step=7,
            proposer_class=proposer_class,
        )
        for batch_size in (2, 4, 8, 16):
            planner.update_infer_cost(batch_size, infer_cost_ms=float(batch_size), is_draft_model=True)

        draft_cost_ms = planner.draft_cost_provider.get_draft_cost_ms(
            draft_infer_costs=planner.draft_infer_costs,
            req_num=8,
            verify_batch_size=16,
            draft_step=7,
        )

        assert draft_cost_ms == 64.0


def test_block_cost_provider_prices_commit_and_complete_block():
    planner = build_lightspec_planner(
        max_draft_step=7,
        proposer_class=DFlashProposer,
        block_size=7,
    )
    planner.update_infer_cost(batch_size=14, infer_cost_ms=0.7, is_draft_model=True)

    draft_cost_ms = planner.draft_cost_provider.get_draft_cost_ms(
        draft_infer_costs=planner.draft_infer_costs,
        req_num=2,
        verify_batch_size=16,
        draft_step=7,
    )

    assert np.isclose(draft_cost_ms, 1.5)


def test_lightspec_records_one_batch_observation_per_configuration():
    planner = build_lightspec_planner()

    planner.update_verified_batch(
        accept_lengths=[2, 2],
        req_num=2,
        dynamic_batch_size=4,
        verified_draft_step=1,
    )
    planner.update_verified_batch(
        accept_lengths=[1, 1],
        req_num=2,
        dynamic_batch_size=4,
        verified_draft_step=3,
    )

    assert planner.progress_ema_by_config[(2, 4, 1)].get() == 1.0
    assert planner.progress_ema_by_config[(2, 4, 3)].get() == 0.5
    assert planner.progress_ema_by_config[(2, 4, 1)].get_count() == 1
    assert planner.progress_ema_by_config[(2, 4, 3)].get_count() == 1


def test_lightspec_high_concurrency_does_not_multiply_ema_updates():
    planner = build_lightspec_planner()

    planner.update_verified_batch(
        accept_lengths=[1] * 128,
        req_num=128,
        dynamic_batch_size=128,
        verified_draft_step=1,
    )

    assert planner.progress_ema_by_config[(128, 128, 1)].get_count() == 1


def test_lightspec_estimates_unseen_shapes_from_prefix_survival():
    planner = build_lightspec_planner()
    planner.update_verified_batch(
        accept_lengths=[2, 1],
        req_num=2,
        dynamic_batch_size=8,
        verified_draft_step=3,
    )

    assert planner._estimate_progress(req_num=2, dynamic_batch_size=4, draft_step=3) == 0.75
    assert planner._estimate_progress(req_num=2, dynamic_batch_size=2, draft_step=3) == 1.0
    assert planner._estimate_progress(req_num=2, dynamic_batch_size=4, draft_step=2) == 0.75


def test_lightspec_does_not_transfer_deep_progress_to_short_drafts():
    planner = build_lightspec_planner()
    planner.update_verified_batch(
        accept_lengths=[4, 2],
        req_num=2,
        dynamic_batch_size=8,
        verified_draft_step=3,
    )

    assert np.isclose(planner._estimate_progress(req_num=2, dynamic_batch_size=6, draft_step=2), 5 / 6)


def test_lightspec_short_current_proposal_can_recover_to_a_deeper_draft():
    planner = build_lightspec_planner(proposer_class=EagleMTPProposer)
    for batch_size, target_cost in ((2, 1.0), (4, 1.1), (6, 1.2), (8, 1.3)):
        planner.update_infer_cost(batch_size, target_cost, is_draft_model=False)
        planner.update_infer_cost(batch_size, 0.01, is_draft_model=True)
    planner.update_verified_batch(
        accept_lengths=[4, 4],
        req_num=2,
        dynamic_batch_size=8,
        verified_draft_step=3,
    )
    planner.pre_draft_step = 1

    plan = planner.plan(req_num=2, original_batch_size=8, proposal_req_num=2)

    assert plan.dynamic_batch_size <= 4
    assert plan.pre_draft_step == 1
    assert plan.draft_step == 3


def test_engine_skips_feedback_for_a_mixed_proposal_batch():
    engine = SpecEngine.__new__(SpecEngine)
    engine.enable_dynmaic_mtp = True
    engine.planner = build_lightspec_planner()
    plan = SpecDecodePlan(
        dynamic_batch_size=5,
        draft_step=3,
        pre_draft_step=3,
        all_reqs_have_proposals=False,
    )

    engine.update_planner_feedback(
        plan=plan,
        proposal=SpecProposal(
            token_ids=torch.empty((0,), dtype=torch.int64),
            extra_mem_indexes_cpu=None,
        ),
        req_num=2,
        accept_lengths_cpu=torch.tensor([1, 2], dtype=torch.int32),
    )

    assert not engine.planner.progress_ema_by_config


def test_dspark_applies_confidence_capacity_after_two_step_delay():
    planner = build_dspark_planner()
    for batch_size, target_cost in ((2, 1.0), (4, 1.1), (8, 10.0)):
        planner.update_infer_cost(batch_size, target_cost, is_draft_model=False)
    planner.update_infer_cost(batch_size=6, infer_cost_ms=0.5, is_draft_model=True)
    confidence_probs = np.asarray([[0.9, 0.9, 0.9]] * 2, dtype=np.float64)
    plan = SpecDecodePlan(dynamic_batch_size=8, draft_step=3, pre_draft_step=3)
    proposal = SpecProposal(
        token_ids=torch.empty((0,), dtype=torch.int64),
        extra_mem_indexes_cpu=None,
        schedule_scores_cpu=torch.from_numpy(confidence_probs),
    )
    engine = SpecEngine.__new__(SpecEngine)
    engine.enable_dynmaic_mtp = True
    engine.planner = planner

    engine.update_planner_feedback(
        plan=plan,
        proposal=proposal,
        req_num=2,
        accept_lengths_cpu=torch.tensor([1, 1], dtype=torch.int32),
    )
    first_plan = planner.plan(req_num=2, original_batch_size=8)
    engine.update_planner_feedback(
        plan=plan,
        proposal=proposal,
        req_num=2,
        accept_lengths_cpu=torch.tensor([1, 1], dtype=torch.int32),
    )
    second_plan = planner.plan(req_num=2, original_batch_size=8)

    assert first_plan.dynamic_batch_size == 8
    assert second_plan.dynamic_batch_size == 4
    assert second_plan.draft_step == second_plan.pre_draft_step == 3


def test_dspark_uses_confidence_scores_without_acceptance_ema():
    planner = build_dspark_planner()
    for batch_size, target_cost in ((2, 1.0), (4, 1.2), (8, 10.0)):
        planner.update_infer_cost(batch_size, target_cost, is_draft_model=False)
    planner.update_infer_cost(batch_size=6, infer_cost_ms=0.5, is_draft_model=True)

    low_confidence = np.cumprod(np.full((2, 3), 0.1), axis=1)
    high_confidence = np.cumprod(np.full((2, 3), 0.9), axis=1)

    assert planner._select_dynamic_batch_size_from_survival_scores(2, low_confidence) == 2
    assert planner._select_dynamic_batch_size_from_survival_scores(2, high_confidence) == 4


def test_topk_prefix_sums_only_computes_requested_counts():
    result = DSparkPlanner._topk_prefix_sums(
        values=np.asarray([0.1, 0.9, 0.4, 0.7]),
        counts=[0, 2, 4],
    )

    assert set(result) == {0, 2, 4}
    np.testing.assert_allclose([result[0], result[2], result[4]], [0.0, 1.6, 2.1])
