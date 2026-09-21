"""EPLB 专家传输计划的异步生成器。"""

import os
import threading
from enum import Enum
from typing import List, Optional

from lightllm.utils.log_utils import init_logger

from .expert_transfer import EPLBTransferInfo, build_transfer_plan
from .placement_planner import ExpertPlacement

logger = init_logger(__name__)


class TransferPlanStatus(Enum):
    """异步传输规划器的生命周期状态。"""

    IDLE = "idle"
    RUNNING = "running"
    SUCCEEDED = "succeeded"


class EPLBTransferPlanner:
    """在后台线程中逐层生成并按 layer 顺序拼接专家传输批次。

    输入布局在规划期间保持只读。每层独立调用 ``build_transfer_plan``，因此
    一个批次只包含同一层的任务，manager 可以在提交后立即发布该层的路由
    metadata。
    """

    def __init__(
        self,
        current_placement: ExpertPlacement,
        target_placement: ExpertPlacement,
        num_logical_experts: int,
        world_size: int,
    ) -> None:
        self.current_placement = current_placement
        self.target_placement = target_placement
        self.num_logical_experts = num_logical_experts
        self.world_size = world_size
        self.status = TransferPlanStatus.IDLE
        self.result: Optional[List[List[EPLBTransferInfo]]] = None
        self._thread = threading.Thread(
            target=self._run,
            name="eplb-transfer-plan",
            daemon=True,
        )

    def start(self) -> None:
        """启动异步传输规划。"""
        assert self.status is TransferPlanStatus.IDLE, "EPLB transfer planner has already been started"
        self.status = TransferPlanStatus.RUNNING
        self._thread.start()

    def is_finished(self) -> bool:
        """返回全部层的传输批次是否已经生成。"""
        return self.status is TransferPlanStatus.SUCCEEDED

    def _run(self) -> None:
        try:
            transfer_batches: List[List[EPLBTransferInfo]] = []
            layer_placements = zip(self.current_placement, self.target_placement)
            for layer_index, (current_layer, target_layer) in enumerate(layer_placements):
                layer_transfer_batches = build_transfer_plan(
                    current_layer,
                    target_layer,
                    layer_index,
                    self.num_logical_experts,
                    self.world_size,
                )
                transfer_batches.extend(layer_transfer_batches)
            self.result = transfer_batches
            self.status = TransferPlanStatus.SUCCEEDED
        except BaseException:
            logger.exception("EPLB transfer planning failed")
            os._exit(1)
