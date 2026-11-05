"""EPLB 专家布局的异步规划任务。"""

from typing import Optional

import torch

from .async_task import EPLBAsyncTask
from .placement import EPLBPlanner, ExpertPlacement


class EPLBPlanTask(EPLBAsyncTask):
    """在后台线程中根据全局路由统计生成新布局。"""

    def __init__(
        self,
        planner: EPLBPlanner,
        route_statistics: torch.Tensor,
        current_placement: ExpertPlacement,
    ) -> None:
        self.planner = planner
        self.route_statistics = route_statistics
        self.current_placement = current_placement
        self.result: Optional[ExpertPlacement] = None
        super().__init__(thread_name="eplb-plan")

    def execute(self) -> None:
        """调用当前运行模式对应的 planner 生成目标布局。"""
        self.result = self.planner.plan(
            route_statistics=self.route_statistics,
            current_placement=self.current_placement,
        )
