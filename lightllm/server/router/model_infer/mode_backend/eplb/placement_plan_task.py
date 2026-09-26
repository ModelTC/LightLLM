"""EPLB 专家布局的异步规划任务。"""

import os
import threading
from typing import Optional

import torch

from lightllm.utils.log_utils import init_logger

from .placement import EPLBPlanner, ExpertPlacement

logger = init_logger(__name__)


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
        self.status = "idle"
        self.result: Optional[ExpertPlacement] = None
        self._thread = threading.Thread(
            target=self._run,
            name="eplb-plan",
            daemon=True,
        )

    def start(self) -> None:
        """启动异步规划。"""
        assert self.status == "idle", "EPLB plan task has already been started"
        self.status = "running"
        self._thread.start()

    def is_finished(self) -> bool:
        """返回规划任务是否已经成功完成。"""
        return self.status == "succeeded"

    def _run(self) -> None:
        try:
            self.result = self.planner.plan(
                self.global_load.tolist(),
                self.current_placement,
            )
            self.status = "succeeded"
        except BaseException:
            logger.exception("EPLB planning failed")
            os._exit(1)
