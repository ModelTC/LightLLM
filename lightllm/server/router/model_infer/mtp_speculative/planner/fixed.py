from __future__ import annotations

from typing import TYPE_CHECKING, List

from lightllm.server.router.model_infer.mtp_speculative.planner.base import BaseMtpPlanner, SpecDecodePlan

if TYPE_CHECKING:
    from lightllm.server.router.model_infer.mtp_speculative.proposers.base import SpecProposal


class FixedSpecPlanner(BaseMtpPlanner):
    """Planner for fixed-width speculative decoding."""

    def __init__(self, max_draft_step: int) -> None:
        self.max_draft_step = int(max_draft_step)

    def plan(self, decode_reqs: List, origin_batch_size: int) -> SpecDecodePlan:
        return SpecDecodePlan(
            origin_batch_size=origin_batch_size,
            dynamic_batch_size=origin_batch_size,
            draft_step=self.max_draft_step,
            pre_draft_step=self.max_draft_step,
        )

    def update_statics(
        self,
        plan: SpecDecodePlan,
        proposal: SpecProposal,
        req_num: int,
        accept_lengths,
    ) -> None:
        """固定规划不根据运行反馈调整 batch size 或 draft step。"""

        return
