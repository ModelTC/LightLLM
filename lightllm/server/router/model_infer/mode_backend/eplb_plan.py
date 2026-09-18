"""EPLB 专家布局的异步规划任务。"""

import os
import threading
from enum import Enum
from typing import Optional

import torch

from lightllm.common.basemodel.layer_weights.meta_weights.fused_moe.eplb_planner import (
    EPLBPlanner,
    ExpertPlacement,
)
from lightllm.utils.log_utils import init_logger

logger = init_logger(__name__)


class PlanTaskStatus(Enum):
    """异步规划任务的生命周期状态。"""

    IDLE = "idle"
    RUNNING = "running"
    SUCCEEDED = "succeeded"


class EPLBPlanTask:
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
        self.status = PlanTaskStatus.IDLE
        self.result: Optional[ExpertPlacement] = None
        self._thread = threading.Thread(
            target=self._run,
            name="eplb-plan",
            daemon=True,
        )

    def start(self) -> None:
        """启动异步规划。"""
        assert self.status is PlanTaskStatus.IDLE, "EPLB plan task has already been started"
        self.status = PlanTaskStatus.RUNNING
        self._thread.start()

    def is_finished(self) -> bool:
        """返回规划任务是否已经成功完成。"""
        return self.status is PlanTaskStatus.SUCCEEDED

    def _run(self) -> None:
        try:
            self.result = self.planner.plan(
                self.global_load.tolist(),
                self.current_placement,
            )
            self.status = PlanTaskStatus.SUCCEEDED
        except BaseException:
            logger.exception("EPLB planning failed")
            os._exit(1)
