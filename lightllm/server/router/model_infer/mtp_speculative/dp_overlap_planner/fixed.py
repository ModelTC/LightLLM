from lightllm.server.router.model_infer.mtp_speculative.dp_overlap_planner.base import BaseDpOverlapPlanner


class FixedDpOverlapPlanner(BaseDpOverlapPlanner):
    """为 DP overlap 固定 shape decode 返回启动时配置的 draft step。"""

    def __init__(self, draft_step: int) -> None:
        self.draft_step = int(draft_step)

    def get_draft_step(self) -> int:
        return self.draft_step
