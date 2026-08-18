from __future__ import annotations

from typing import List

from lightllm.server.router.model_infer.mtp_speculative.planner.base import SpecDecodePlan


class FixedSpecPlanner:
    """Planner for fixed-width speculative decoding."""

    def __init__(self, max_draft_step: int) -> None:
        self.max_draft_step = int(max_draft_step)

    def plan(self, decode_reqs: List, original_batch_size: int) -> SpecDecodePlan:
        return SpecDecodePlan(
            dynamic_batch_size=None,
            draft_step=self.max_draft_step,
            pre_draft_step=self.max_draft_step,
        )
