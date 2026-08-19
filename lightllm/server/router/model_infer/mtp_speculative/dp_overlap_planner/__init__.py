from lightllm.server.router.model_infer.mtp_speculative.dp_overlap_planner.base import BaseDpOverlapPlanner
from lightllm.server.router.model_infer.mtp_speculative.dp_overlap_planner.fixed import FixedDpOverlapPlanner


def build_dp_overlap_planner(*, backend) -> BaseDpOverlapPlanner:
    return FixedDpOverlapPlanner(draft_step=backend.max_draft_step)


__all__ = ["BaseDpOverlapPlanner", "FixedDpOverlapPlanner", "build_dp_overlap_planner"]
