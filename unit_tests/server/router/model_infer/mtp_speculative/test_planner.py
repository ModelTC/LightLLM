from types import SimpleNamespace

import numpy as np
import pytest
import torch

from lightllm.server.router.model_infer.mtp_speculative.engine import SpecEngine
from lightllm.server.router.model_infer.mtp_speculative.dp_planner import (
    BaseDpPlanner,
    FixedDpPlanner,
    build_dp_planner,
)
from lightllm.server.router.model_infer.mtp_speculative.dp_overlap_planner import (
    BaseDpOverlapPlanner,
    FixedDpOverlapPlanner,
    build_dp_overlap_planner,
)
from lightllm.server.router.model_infer.mtp_speculative.dp_proposers import build_dp_spec_proposer
from lightllm.server.router.model_infer.mtp_speculative.dp_proposers.base import BaseDpProposer
from lightllm.server.router.model_infer.mtp_speculative.dp_proposers.eagle3 import DpEagle3Proposer
from lightllm.server.router.model_infer.mtp_speculative.dp_proposers.eagle_no_att import DpEagleNoAttProposer
from lightllm.server.router.model_infer.mtp_speculative.dp_proposers.eagle_with_att import DpEagleWithAttProposer
from lightllm.server.router.model_infer.mtp_speculative.dp_proposers.vanilla_no_att import DpVanillaNoAttProposer
from lightllm.server.router.model_infer.mtp_speculative.dp_proposers.vanilla_with_att import (
    DpVanillaWithAttProposer,
)
from lightllm.server.router.model_infer.mtp_speculative.dp_overlap_proposers import (
    build_dp_overlap_spec_proposer,
)
from lightllm.server.router.model_infer.mtp_speculative.dp_overlap_proposers.base import BaseDpOverlapProposer
from lightllm.server.router.model_infer.mtp_speculative.dp_overlap_proposers.eagle3 import DpOverlapEagle3Proposer
from lightllm.server.router.model_infer.mtp_speculative.dp_overlap_proposers.eagle_no_att import (
    DpOverlapEagleNoAttProposer,
)
from lightllm.server.router.model_infer.mtp_speculative.dp_overlap_proposers.eagle_with_att import (
    DpOverlapEagleWithAttProposer,
)
from lightllm.server.router.model_infer.mtp_speculative.dp_overlap_proposers.vanilla_no_att import (
    DpOverlapVanillaNoAttProposer,
)
from lightllm.server.router.model_infer.mtp_speculative.dp_overlap_proposers.vanilla_with_att import (
    DpOverlapVanillaWithAttProposer,
)
from lightllm.server.router.model_infer.mtp_speculative.planner import (
    BaseMtpPlanner,
    DSparkPlanner,
    FixedSpecPlanner,
    LightSpecPlanner,
    SpecDecodePlan,
)
from lightllm.server.router.model_infer.mtp_speculative.planner.base import _InferCostMsTable
from lightllm.server.router.model_infer.mtp_speculative.proposers import build_spec_proposer
from lightllm.server.router.model_infer.mtp_speculative.proposers.base import BaseSpecProposer, SpecProposal
from lightllm.server.router.model_infer.mtp_speculative.proposers.dflash import DFlashProposer, DFlashSpecProposal
from lightllm.server.router.model_infer.mtp_speculative.proposers.dspark import DSparkProposer, DSparkSpecProposal
from lightllm.server.router.model_infer.mtp_speculative.proposers.eagle3 import Eagle3Proposer
from lightllm.server.router.model_infer.mtp_speculative.proposers.eagle_no_att import EagleNoAttProposer
from lightllm.server.router.model_infer.mtp_speculative.proposers.eagle_utils import EagleSpecProposal
from lightllm.server.router.model_infer.mtp_speculative.proposers.eagle_with_att import EagleWithAttProposer
from lightllm.server.router.model_infer.mtp_speculative.proposers.vanilla_no_att import VanillaNoAttProposer
from lightllm.server.router.model_infer.mtp_speculative.proposers.vanilla_utils import VanillaSpecProposal
from lightllm.server.router.model_infer.mtp_speculative.proposers.vanilla_with_att import VanillaWithAttProposer
from lightllm.server.router.model_infer.mtp_speculative import utils as mtp_utils


def build_lightspec_planner(
    max_draft_step: int = 3,
    spec_mode: str = "vanilla_with_att",
    block_size: int = 3,
):
    backend = SimpleNamespace(
        max_draft_step=max_draft_step,
        model=SimpleNamespace(graph=None),
        draft_models=[SimpleNamespace(block_size=block_size, graph=None)],
    )
    return LightSpecPlanner(
        spec_mode=spec_mode,
        backend=backend,
    )


def build_dspark_planner(max_draft_step: int = 3, block_size: int = 3):
    backend = SimpleNamespace(
        max_draft_step=max_draft_step,
        model=SimpleNamespace(graph=None),
        draft_models=[SimpleNamespace(block_size=block_size, graph=None)],
    )
    return DSparkPlanner(backend=backend)


def build_decode_reqs(req_num: int, req_num_with_proposals: int | None = None):
    if req_num_with_proposals is None:
        req_num_with_proposals = req_num
    return [SimpleNamespace(cur_output_len=2)] * req_num_with_proposals + [SimpleNamespace(cur_output_len=1)] * (
        req_num - req_num_with_proposals
    )


def build_planner(spec_mode: str, enable_dynmaic_mtp: bool = True):
    engine = SpecEngine.__new__(SpecEngine)
    engine.backend = SimpleNamespace(
        max_draft_step=3,
        model=SimpleNamespace(graph=None),
        draft_models=[SimpleNamespace(block_size=3, graph=None)],
    )
    return engine._build_mtp_planner(
        spec_mode=spec_mode,
        enable_dynmaic_mtp=enable_dynmaic_mtp,
    )


def test_fixed_planner_returns_static_plan():
    planner = FixedSpecPlanner(max_draft_step=3)
    plan = planner.plan(decode_reqs=build_decode_reqs(4), origin_batch_size=16)

    assert isinstance(planner, BaseMtpPlanner)
    assert plan.origin_batch_size == 16
    assert plan.dynamic_batch_size == plan.origin_batch_size
    assert plan.draft_step == plan.pre_draft_step == 3
    assert not plan.skip_verify_sync


def test_non_dp_engine_rejects_empty_decode_batch():
    engine = SpecEngine.__new__(SpecEngine)

    with pytest.raises(AssertionError, match="requires at least one request"):
        engine.plan_decode(model_input=SimpleNamespace(batch_size=0), decode_reqs=[])


def test_spec_engine_only_exposes_planning_and_proposal_interfaces():
    public_methods = {
        name for name, value in SpecEngine.__dict__.items() if callable(value) and not name.startswith("_")
    }

    assert public_methods == {
        "build_draft_state_from_prefill",
        "plan_decode",
        "prepare_decode_model_input",
        "propose_next",
        "update_planner_statics",
    }


def test_mode_proposals_own_their_schedule_metadata():
    assert "schedule_scores" not in SpecProposal.__dataclass_fields__
    assert "schedule_scores_cpu" not in SpecProposal.__dataclass_fields__

    for proposal_type in (VanillaSpecProposal, EagleSpecProposal, DFlashSpecProposal, DSparkSpecProposal):
        assert "schedule_scores" in proposal_type.__dataclass_fields__
    assert "schedule_scores_cpu" not in VanillaSpecProposal.__dataclass_fields__
    assert "schedule_scores_cpu" not in EagleSpecProposal.__dataclass_fields__
    assert "schedule_scores_cpu" not in DFlashSpecProposal.__dataclass_fields__
    assert "schedule_scores_cpu" in DSparkSpecProposal.__dataclass_fields__


def test_scatter_mtp_next_tokens_consumes_mode_proposal(monkeypatch):
    scatter_args = {}
    monkeypatch.setattr(
        mtp_utils,
        "mtp_scatter_next_token_ids",
        lambda **kwargs: scatter_args.update(kwargs),
    )
    req_to_next_token_ids = torch.empty((4, 2), dtype=torch.int64)
    req_to_next_token_scores = torch.empty((4, 1), dtype=torch.float32)
    backend = SimpleNamespace(
        model=SimpleNamespace(
            req_manager=SimpleNamespace(
                req_sampling_params_manager=SimpleNamespace(
                    req_to_next_token_ids=req_to_next_token_ids,
                    req_to_next_token_scores=req_to_next_token_scores,
                )
            )
        )
    )
    proposal = DFlashSpecProposal(
        token_ids=torch.arange(6, dtype=torch.int64).view(3, 2),
        extra_mem_indexes_cpu=None,
        schedule_scores=torch.arange(3, dtype=torch.float32).view(3, 1),
    )

    mtp_utils.scatter_mtp_next_tokens(
        backend=backend,
        proposal=proposal,
        b_req_mtp_start_loc=torch.tensor([0, 1], dtype=torch.int32),
        b_req_idx=torch.tensor([0, 1], dtype=torch.int32),
        mtp_accept_len=torch.ones(2, dtype=torch.int32),
        valid_row_count=2,
    )

    assert scatter_args["req_to_next_token_ids"] is req_to_next_token_ids
    assert scatter_args["req_to_next_token_scores"] is req_to_next_token_scores
    assert torch.equal(scatter_args["all_next_token_ids"], proposal.token_ids[:2])
    assert torch.equal(scatter_args["schedule_scores"], proposal.schedule_scores[:2])


def test_dp_planner_returns_fixed_backend_draft_step():
    planner = build_dp_planner(backend=SimpleNamespace(max_draft_step=4))

    assert type(planner) is FixedDpPlanner
    assert isinstance(planner, BaseDpPlanner)
    assert planner.get_draft_step() == 4


def test_dp_overlap_planner_returns_fixed_backend_draft_step():
    planner = build_dp_overlap_planner(backend=SimpleNamespace(max_draft_step=5))

    assert type(planner) is FixedDpOverlapPlanner
    assert isinstance(planner, BaseDpOverlapPlanner)
    assert planner.get_draft_step() == 5


def test_infer_cost_candidates_include_feasible_boundaries():
    costs = _InferCostMsTable()
    costs.update(batch_size=4, infer_cost_ms=1.0)
    costs.update(batch_size=8, infer_cost_ms=2.0)

    assert costs.get_batch_size_keys_between(5, 10) == [5, 8, 10]


def test_engine_routes_only_dspark_to_the_confidence_planner():
    fixed_planner = build_planner("eagle3", enable_dynmaic_mtp=False)
    dspark_planner = build_planner("dspark")
    assert isinstance(fixed_planner, FixedSpecPlanner)
    assert isinstance(dspark_planner, DSparkPlanner)
    assert isinstance(fixed_planner, BaseMtpPlanner)
    assert isinstance(dspark_planner, BaseMtpPlanner)

    vanilla_planner = build_planner("vanilla_with_att")
    assert isinstance(vanilla_planner, LightSpecPlanner)
    assert isinstance(vanilla_planner, BaseMtpPlanner)
    assert vanilla_planner.draft_steps == (1, 2, 3)

    vanilla_no_att_planner = build_planner("vanilla_no_att")
    assert vanilla_no_att_planner.draft_steps == (0, 1, 2, 3)

    eagle_no_att_planner = build_planner("eagle_no_att")
    assert eagle_no_att_planner.draft_steps == (0, 1, 2, 3)

    dflash_planner = build_planner("dflash")
    assert isinstance(dflash_planner, LightSpecPlanner)
    assert dflash_planner.draft_steps == (3,)

    eagle_planner = build_planner("eagle3")
    assert isinstance(eagle_planner, LightSpecPlanner)
    assert eagle_planner.draft_steps == (1, 2, 3)


def test_dynamic_planner_registers_cuda_graph_costs_from_backend():
    backend = SimpleNamespace(
        max_draft_step=3,
        model=SimpleNamespace(graph=SimpleNamespace(infer_cost_ms_by_batch_size={4: 1.2})),
        draft_models=[
            SimpleNamespace(
                block_size=3,
                graph=SimpleNamespace(infer_cost_ms_by_batch_size={4: 0.3}),
            )
        ],
    )

    planner = LightSpecPlanner(spec_mode="vanilla_with_att", backend=backend)

    assert planner.target_infer_costs.estimate(4) == 1.2
    assert planner.draft_infer_costs.estimate(4) == 0.3


def test_each_mode_proposer_only_inherits_its_base_interface():
    proposer_types = (
        VanillaWithAttProposer,
        VanillaNoAttProposer,
        EagleWithAttProposer,
        EagleNoAttProposer,
        Eagle3Proposer,
        DFlashProposer,
        DSparkProposer,
    )
    dp_proposer_types = (
        DpVanillaWithAttProposer,
        DpVanillaNoAttProposer,
        DpEagleWithAttProposer,
        DpEagleNoAttProposer,
        DpEagle3Proposer,
    )
    dp_overlap_proposer_types = (
        DpOverlapVanillaWithAttProposer,
        DpOverlapVanillaNoAttProposer,
        DpOverlapEagleWithAttProposer,
        DpOverlapEagleNoAttProposer,
        DpOverlapEagle3Proposer,
    )

    for proposer_type in proposer_types:
        assert proposer_type.__bases__ == (BaseSpecProposer,)
    for proposer_type in dp_proposer_types:
        assert proposer_type.__bases__ == (BaseDpProposer,)
    for proposer_type in dp_overlap_proposer_types:
        assert proposer_type.__bases__ == (BaseDpOverlapProposer,)


def test_each_mtp_mode_builds_its_own_proposer():
    backend = SimpleNamespace()
    proposer_types = {
        "vanilla_with_att": VanillaWithAttProposer,
        "vanilla_no_att": VanillaNoAttProposer,
        "eagle_with_att": EagleWithAttProposer,
        "eagle_no_att": EagleNoAttProposer,
        "eagle3": Eagle3Proposer,
        "dflash": DFlashProposer,
        "dspark": DSparkProposer,
    }

    for spec_mode, proposer_type in proposer_types.items():
        proposer = build_spec_proposer(spec_mode=spec_mode, backend=backend, enable_dynmaic_mtp=False)
        assert type(proposer) is proposer_type


def test_each_supported_dp_mtp_mode_builds_its_own_dp_proposer():
    backend = SimpleNamespace()
    proposer_types = {
        "vanilla_with_att": DpVanillaWithAttProposer,
        "vanilla_no_att": DpVanillaNoAttProposer,
        "eagle_with_att": DpEagleWithAttProposer,
        "eagle_no_att": DpEagleNoAttProposer,
        "eagle3": DpEagle3Proposer,
    }

    for spec_mode, proposer_type in proposer_types.items():
        proposer = build_dp_spec_proposer(spec_mode=spec_mode, backend=backend, enable_dynmaic_mtp=False)
        assert type(proposer) is proposer_type
        assert isinstance(proposer, BaseDpProposer)


def test_each_supported_dp_overlap_mtp_mode_builds_its_own_proposer():
    backend = SimpleNamespace()
    proposer_types = {
        "vanilla_with_att": DpOverlapVanillaWithAttProposer,
        "vanilla_no_att": DpOverlapVanillaNoAttProposer,
        "eagle_with_att": DpOverlapEagleWithAttProposer,
        "eagle_no_att": DpOverlapEagleNoAttProposer,
        "eagle3": DpOverlapEagle3Proposer,
    }

    for spec_mode, proposer_type in proposer_types.items():
        proposer = build_dp_overlap_spec_proposer(spec_mode=spec_mode, backend=backend, enable_dynmaic_mtp=False)
        assert type(proposer) is proposer_type
        assert isinstance(proposer, BaseDpOverlapProposer)


def test_eagle_proposer_skips_draft_forward_for_zero_steps():
    proposer = EagleNoAttProposer(
        backend=SimpleNamespace(draft_models=[]),
        enable_dynmaic_mtp=True,
    )
    next_token_ids = torch.tensor([10, 11], dtype=torch.int64)

    proposal = proposer.propose_next(
        main_model_input=None,
        main_model_output=None,
        next_token_ids=next_token_ids,
        b_req_mtp_start_loc=torch.tensor([0, 1], dtype=torch.int32),
        draft_step=0,
    )

    assert isinstance(proposal, EagleSpecProposal)
    assert proposal.token_ids.tolist() == [[10], [11]]
    assert proposal.schedule_scores.shape == (2, 0)


def test_dynamic_decode_frees_unselected_and_rejected_rows():
    freed = []
    backend = SimpleNamespace(
        model=SimpleNamespace(
            req_manager=SimpleNamespace(mem_manager=SimpleNamespace(free=lambda indexes: freed.append(indexes.clone())))
        )
    )
    model_input = SimpleNamespace(mem_indexes_cpu=torch.tensor([10, 11, 12, 13]))

    mtp_utils.free_unused_mtp_decode_mem(
        backend=backend,
        proposal=SpecProposal(
            token_ids=torch.empty((0,), dtype=torch.int64),
            extra_mem_indexes_cpu=torch.tensor([20]),
        ),
        model_input=model_input,
        selected_row_mask_cpu=torch.tensor([1, 0, 1, 0], dtype=torch.bool),
        accepted_index_cpu=torch.tensor([1, 0], dtype=torch.int32),
    )

    assert len(freed) == 1
    assert freed[0].tolist() == [11, 12, 13, 20]


def test_records_request_mtp_metrics_in_one_pass():
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

    backend = SimpleNamespace(is_master_in_dp=True)
    req0 = MetricReq(req_idx=10, mtp_step=3)
    req1 = MetricReq(req_idx=11, mtp_step=3)

    mtp_utils.record_request_mtp_metrics(
        backend=backend,
        decode_reqs=[req0, req1],
        accept_lengths_cpu=torch.tensor([2, 1], dtype=torch.int32),
        verified_row_reqs=[req0, req0, req1],
    )

    assert (req0.accepted, req0.verified, req0.verify_steps) == (1, 2, 1)
    assert (req1.accepted, req1.verified, req1.verify_steps) == (0, 1, 1)

    fixed_req = MetricReq(req_idx=12, mtp_step=3)
    mtp_utils.record_request_mtp_metrics(
        backend=backend,
        decode_reqs=[fixed_req],
        accept_lengths_cpu=torch.tensor([3], dtype=torch.int32),
        verified_row_reqs=[fixed_req] * 4,
    )
    assert (fixed_req.accepted, fixed_req.verified, fixed_req.verify_steps) == (2, 4, 1)


def test_lightspec_stays_full_width_until_costs_are_profiled():
    plan = build_lightspec_planner().plan(
        decode_reqs=build_decode_reqs(2),
        origin_batch_size=8,
    )

    assert plan.origin_batch_size == 8
    assert plan.dynamic_batch_size == 8
    assert plan.draft_step == plan.pre_draft_step == 3


def test_lightspec_collects_full_width_progress_before_adapting():
    planner = build_lightspec_planner(spec_mode="eagle_with_att")
    for batch_size in (2, 4, 8):
        planner.target_infer_costs.update(batch_size=batch_size, infer_cost_ms=float(batch_size))
        planner.draft_infer_costs.update(batch_size=batch_size, infer_cost_ms=float(batch_size))

    plan = planner.plan(decode_reqs=build_decode_reqs(2), origin_batch_size=8)

    assert plan.dynamic_batch_size == 8
    assert plan.draft_step == plan.pre_draft_step == 3


def test_lightspec_selects_eagle_draft_depth_and_verify_capacity():
    planner = build_lightspec_planner(spec_mode="eagle_with_att")
    for batch_size, target_cost in ((2, 1.0), (4, 1.1), (8, 10.0)):
        planner.target_infer_costs.update(batch_size=batch_size, infer_cost_ms=target_cost)
    for batch_size, draft_cost in ((2, 0.1), (4, 0.1), (8, 0.2)):
        planner.draft_infer_costs.update(batch_size=batch_size, infer_cost_ms=draft_cost)
    for _ in range(8):
        planner._update_verified_batch(
            accept_lengths=[4, 4],
            req_num=2,
            dynamic_batch_size=8,
            verified_draft_step=3,
        )

    plan = planner.plan(decode_reqs=build_decode_reqs(2), origin_batch_size=8)

    assert plan.dynamic_batch_size == 4
    assert plan.pre_draft_step == 3
    assert plan.draft_step == 1


def test_lightspec_compacts_block_verify_without_changing_draft_shape():
    planner = build_lightspec_planner(
        max_draft_step=7,
        spec_mode="dflash",
        block_size=7,
    )
    for batch_size, target_cost in ((2, 1.0), (4, 1.1), (8, 3.0), (16, 8.0)):
        planner.target_infer_costs.update(batch_size=batch_size, infer_cost_ms=target_cost)
    planner.draft_infer_costs.update(batch_size=14, infer_cost_ms=0.5)
    for _ in range(8):
        planner._update_verified_batch(
            accept_lengths=[3, 3],
            req_num=2,
            dynamic_batch_size=16,
            verified_draft_step=7,
        )

    plan = planner.plan(decode_reqs=build_decode_reqs(2), origin_batch_size=16)

    assert plan.dynamic_batch_size < 16
    assert plan.draft_step == plan.pre_draft_step == 7


def test_lightspec_bounds_verify_to_existing_proposals():
    planner = build_lightspec_planner()

    cold_start = planner.plan(decode_reqs=build_decode_reqs(2, 0), origin_batch_size=8)
    mixed_batch = planner.plan(decode_reqs=build_decode_reqs(2, 1), origin_batch_size=8)
    ready_batch = planner.plan(decode_reqs=build_decode_reqs(2, 2), origin_batch_size=8)

    assert cold_start.dynamic_batch_size == 2
    assert mixed_batch.dynamic_batch_size == 5
    assert ready_batch.dynamic_batch_size == 8
    assert not cold_start.all_reqs_have_proposals
    assert not mixed_batch.all_reqs_have_proposals
    assert ready_batch.all_reqs_have_proposals


def test_engine_lets_planner_count_requests_with_a_previous_proposal():
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
    planner = build_lightspec_planner(spec_mode="eagle_with_att")
    for batch_size in (2, 4, 8):
        planner.target_infer_costs.update(batch_size=batch_size, infer_cost_ms=float(batch_size))
        planner.draft_infer_costs.update(batch_size=batch_size, infer_cost_ms=float(batch_size))
    planner._update_verified_batch(
        accept_lengths=[2, 2],
        req_num=2,
        dynamic_batch_size=8,
        verified_draft_step=3,
    )

    plan = planner.plan(decode_reqs=build_decode_reqs(2), origin_batch_size=8)

    assert planner.draft_steps == (1, 2, 3)
    assert plan.draft_step >= 1


def test_vanilla_with_attention_planner_prices_extend_then_normal_batches():
    planner = build_lightspec_planner(spec_mode="vanilla_with_att")
    planner.draft_infer_costs.update(batch_size=2, infer_cost_ms=0.1)

    draft_cost_ms = planner._get_draft_cost_ms(
        req_num=4,
        verify_batch_size=8,
        draft_step=3,
    )

    assert np.isclose(draft_cost_ms, 0.8)
    with pytest.raises(AssertionError, match="requires draft_step to be greater than 0"):
        planner._get_draft_cost_ms(req_num=4, verify_batch_size=8, draft_step=0)


def test_no_attention_planner_prices_only_normal_request_batches():
    for spec_mode in ("vanilla_no_att", "eagle_no_att"):
        planner = build_lightspec_planner(spec_mode=spec_mode)
        planner.draft_infer_costs.update(batch_size=2, infer_cost_ms=0.1)

        draft_cost_ms = planner._get_draft_cost_ms(
            req_num=4,
            verify_batch_size=8,
            draft_step=3,
        )

        assert np.isclose(draft_cost_ms, 0.6)


def test_eagle_planner_prices_extend_and_decode_separately():
    planner = build_lightspec_planner(spec_mode="eagle_with_att")
    planner.draft_infer_costs.update(batch_size=2, infer_cost_ms=0.25)

    draft_cost_ms = planner._get_draft_cost_ms(
        req_num=2,
        verify_batch_size=8,
        draft_step=3,
    )

    assert draft_cost_ms == 1.5


def test_autoregressive_eagle_planner_prices_extend_and_decode_rows():
    for spec_mode in ("eagle_with_att", "eagle3"):
        planner = build_lightspec_planner(
            max_draft_step=7,
            spec_mode=spec_mode,
        )
        for batch_size in (2, 4, 8, 16):
            planner.draft_infer_costs.update(batch_size=batch_size, infer_cost_ms=float(batch_size))

        draft_cost_ms = planner._get_draft_cost_ms(
            req_num=8,
            verify_batch_size=16,
            draft_step=7,
        )

        assert draft_cost_ms == 64.0


def test_block_planner_prices_commit_and_complete_block():
    planner = build_lightspec_planner(
        max_draft_step=7,
        spec_mode="dflash",
        block_size=7,
    )
    planner.draft_infer_costs.update(batch_size=8, infer_cost_ms=0.4)

    draft_cost_ms = planner._get_draft_cost_ms(
        req_num=2,
        verify_batch_size=16,
        draft_step=7,
    )

    assert np.isclose(draft_cost_ms, 1.5)


def test_dspark_planner_prices_commit_and_complete_block():
    planner = build_dspark_planner(max_draft_step=3, block_size=3)
    planner.draft_infer_costs.update(batch_size=8, infer_cost_ms=0.8)

    draft_cost_ms = planner._get_draft_cost_ms(
        req_num=4,
        verify_batch_size=8,
        draft_step=3,
    )

    assert np.isclose(draft_cost_ms, 2.0)


def test_lightspec_records_one_batch_observation_per_configuration():
    planner = build_lightspec_planner()

    planner._update_verified_batch(
        accept_lengths=[2, 2],
        req_num=2,
        dynamic_batch_size=4,
        verified_draft_step=1,
    )
    planner._update_verified_batch(
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

    planner._update_verified_batch(
        accept_lengths=[1] * 128,
        req_num=128,
        dynamic_batch_size=128,
        verified_draft_step=1,
    )

    assert planner.progress_ema_by_config[(128, 128, 1)].get_count() == 1


def test_lightspec_estimates_unseen_shapes_from_prefix_survival():
    planner = build_lightspec_planner()
    planner._update_verified_batch(
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
    planner._update_verified_batch(
        accept_lengths=[4, 2],
        req_num=2,
        dynamic_batch_size=8,
        verified_draft_step=3,
    )

    assert np.isclose(planner._estimate_progress(req_num=2, dynamic_batch_size=6, draft_step=2), 5 / 6)


def test_lightspec_short_current_proposal_can_recover_to_a_deeper_draft():
    planner = build_lightspec_planner(spec_mode="eagle_with_att")
    for batch_size, target_cost in ((2, 1.0), (4, 1.1), (6, 1.2), (8, 1.3)):
        planner.target_infer_costs.update(batch_size=batch_size, infer_cost_ms=target_cost)
        planner.draft_infer_costs.update(batch_size=batch_size, infer_cost_ms=0.01)
    planner._update_verified_batch(
        accept_lengths=[4, 4],
        req_num=2,
        dynamic_batch_size=8,
        verified_draft_step=3,
    )
    planner.pre_draft_step = 1

    plan = planner.plan(decode_reqs=build_decode_reqs(2), origin_batch_size=8)

    assert plan.dynamic_batch_size <= 4
    assert plan.pre_draft_step == 1
    assert plan.draft_step == 3


def test_engine_skips_feedback_for_a_mixed_proposal_batch():
    engine = SpecEngine.__new__(SpecEngine)
    engine.planner = build_lightspec_planner()
    plan = SpecDecodePlan(
        origin_batch_size=8,
        dynamic_batch_size=5,
        draft_step=3,
        pre_draft_step=3,
        all_reqs_have_proposals=False,
    )

    engine.update_planner_statics(
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
        planner.target_infer_costs.update(batch_size=batch_size, infer_cost_ms=target_cost)
    planner.draft_infer_costs.update(batch_size=6, infer_cost_ms=0.5)
    confidence_probs = np.asarray([[0.9, 0.9, 0.9]] * 2, dtype=np.float64)
    plan = SpecDecodePlan(origin_batch_size=8, dynamic_batch_size=8, draft_step=3, pre_draft_step=3)
    proposal = DSparkSpecProposal(
        token_ids=torch.empty((0,), dtype=torch.int64),
        extra_mem_indexes_cpu=None,
        schedule_scores_cpu=torch.from_numpy(confidence_probs),
    )
    engine = SpecEngine.__new__(SpecEngine)
    engine.planner = planner

    engine.update_planner_statics(
        plan=plan,
        proposal=proposal,
        req_num=2,
        accept_lengths_cpu=torch.tensor([1, 1], dtype=torch.int32),
    )
    first_plan = planner.plan(decode_reqs=build_decode_reqs(2), origin_batch_size=8)
    engine.update_planner_statics(
        plan=plan,
        proposal=proposal,
        req_num=2,
        accept_lengths_cpu=torch.tensor([1, 1], dtype=torch.int32),
    )
    second_plan = planner.plan(decode_reqs=build_decode_reqs(2), origin_batch_size=8)

    assert first_plan.dynamic_batch_size == 8
    assert second_plan.dynamic_batch_size == 4
    assert second_plan.draft_step == second_plan.pre_draft_step == 3


def test_dspark_uses_confidence_scores_without_acceptance_ema():
    planner = build_dspark_planner()
    for batch_size, target_cost in ((2, 1.0), (4, 1.2), (8, 10.0)):
        planner.target_infer_costs.update(batch_size=batch_size, infer_cost_ms=target_cost)
    planner.draft_infer_costs.update(batch_size=6, infer_cost_ms=0.5)

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
