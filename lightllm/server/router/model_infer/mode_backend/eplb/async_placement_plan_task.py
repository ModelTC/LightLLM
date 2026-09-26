"""EPLB 专家布局的异步规划任务。"""

from typing import Optional

import torch

from .async_task import EPLBAsyncTask
from .placement import EPLBPlanner, ExpertPlacement


class EPLBPlanTask(EPLBAsyncTask):
    """在后台线程中根据全局专家负载生成新布局。"""

    def __init__(
        self,
        planner: EPLBPlanner,
        global_load: torch.Tensor,
        current_placement: ExpertPlacement,
    ) -> None:
        self.planner = planner
        self.global_load = global_load
        self.current_placement = current_placement
        self.result: Optional[ExpertPlacement] = None
        super().__init__(thread_name="eplb-plan")

    def execute(self) -> None:
        """根据全局 logical expert 负载生成目标布局。"""
        self.result = self.planner.plan(
            self.global_load.tolist(),
            self.current_placement,
        )
