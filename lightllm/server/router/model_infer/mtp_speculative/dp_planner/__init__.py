from lightllm.server.router.model_infer.mtp_speculative.dp_planner.base import BaseDpPlanner
from lightllm.server.router.model_infer.mtp_speculative.dp_planner.fixed import FixedDpPlanner


def build_dp_planner(*, backend) -> BaseDpPlanner:
    return FixedDpPlanner(draft_step=backend.max_draft_step)


__all__ = ["BaseDpPlanner", "FixedDpPlanner", "build_dp_planner"]
