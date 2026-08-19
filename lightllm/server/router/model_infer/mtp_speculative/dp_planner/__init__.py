from typing import TYPE_CHECKING

from lightllm.server.router.model_infer.mtp_speculative.dp_planner.base import BaseDpPlanner
from lightllm.server.router.model_infer.mtp_speculative.dp_planner.fixed import FixedDpPlanner

if TYPE_CHECKING:
    from lightllm.server.router.model_infer.mode_backend.base_backend import ModeBackend


def build_dp_planner(*, backend: "ModeBackend") -> BaseDpPlanner:
    return FixedDpPlanner(draft_step=backend.max_draft_step)


__all__ = ["BaseDpPlanner", "FixedDpPlanner", "build_dp_planner"]
