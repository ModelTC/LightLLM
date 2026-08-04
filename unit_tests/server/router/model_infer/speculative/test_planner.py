import random

import numpy as np

from lightllm.server.router.model_infer.speculative.planner import (
    DSparkDynamicMTPPlanner,
    DynamicMTPPlanner,
    FixedMTPPlanner,
)


def test_fixed_planner_returns_static_plan():
    plan = FixedMTPPlanner(mtp_step=3).plan(req_num=4, original_batch_size=16)

    assert not plan.is_dynamic
    assert plan.dynamic_batch_size is None
    assert plan.draft_step == plan.pre_draft_step == 3
    assert plan.selection_mode == "none"
    assert not plan.skip_verify_sync


def test_dynamic_planner_stays_full_width_until_costs_are_profiled():
    plan = DynamicMTPPlanner(mtp_step=3, use_random_mode=False).plan(
        req_num=2, original_batch_size=8
    )

    assert plan.dynamic_batch_size == 8
    assert plan.draft_step == plan.pre_draft_step == 3
    assert plan.selection_mode == "observe"


def test_planner_does_not_reset_process_global_random_state():
    random.seed(2027)
    expected = random.random()
    random.seed(2027)

    DynamicMTPPlanner(mtp_step=3)

    assert random.random() == expected


def test_dspark_applies_confidence_capacity_after_two_step_delay():
    planner = DSparkDynamicMTPPlanner(mtp_step=3)
    planner.update_infer_cost(batch_size=2, infer_cost_ms=1.0, is_draft_model=False)
    planner.update_infer_cost(batch_size=4, infer_cost_ms=1.1, is_draft_model=False)
    planner.update_infer_cost(batch_size=8, infer_cost_ms=10.0, is_draft_model=False)
    schedule_probs = np.asarray([[1.0, 0.9, 0.9, 0.9]] * 2, dtype=np.float64)

    planner.update_predicted_schedule_probs(schedule_probs=schedule_probs, req_num=2)
    first_plan = planner.plan(req_num=2, original_batch_size=8)
    planner.update_predicted_schedule_probs(schedule_probs=schedule_probs, req_num=2)
    second_plan = planner.plan(req_num=2, original_batch_size=8)

    assert first_plan.dynamic_batch_size == 2
    assert second_plan.dynamic_batch_size == 4
    assert second_plan.draft_step == 3


def test_topk_prefix_sums_only_computes_requested_counts():
    result = DSparkDynamicMTPPlanner._topk_prefix_sums(
        values=np.asarray([0.1, 0.9, 0.4, 0.7]),
        counts=[0, 2, 4],
    )

    assert set(result) == {0, 2, 4}
    np.testing.assert_allclose([result[0], result[2], result[4]], [0.0, 1.6, 2.1])
