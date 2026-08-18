from __future__ import annotations

from lightllm.server.router.model_infer.mtp_speculative.planner.base import SpecDecodePlan


class FixedSpecPlanner:
    """Planner for fixed-width speculative decoding."""

    def __init__(self, max_draft_step: int) -> None:
        self.max_draft_step = int(max_draft_step)

    def plan(self, req_num: int | None = None, original_batch_size: int | None = None) -> SpecDecodePlan:
        return SpecDecodePlan(
            dynamic_batch_size=None,
            draft_step=self.max_draft_step,
            pre_draft_step=self.max_draft_step,
        )
