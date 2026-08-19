from __future__ import annotations

from typing import List

from lightllm.server.router.model_infer.mtp_speculative.planner.base import BaseMtpPlanner, SpecDecodePlan


class FixedSpecPlanner(BaseMtpPlanner):
    """Planner for fixed-width speculative decoding."""

    def __init__(self, max_draft_step: int) -> None:
        self.max_draft_step = int(max_draft_step)

    def plan(self, decode_reqs: List, original_batch_size: int) -> SpecDecodePlan:
        return SpecDecodePlan(
            dynamic_batch_size=None,
            draft_step=self.max_draft_step,
            pre_draft_step=self.max_draft_step,
        )

    def update_feedback(
        self,
        plan: SpecDecodePlan,
        req_num: int,
        accept_lengths,
        schedule_scores=None,
    ) -> None:
        """固定规划不根据运行反馈调整 batch size 或 draft step。"""

        return
