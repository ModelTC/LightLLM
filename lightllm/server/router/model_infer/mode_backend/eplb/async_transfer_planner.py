"""EPLB 专家传输计划的异步生成器。"""

from typing import List, Optional

from .async_task import EPLBAsyncTask
from .async_expert_transfer import EPLBTransferInfo, build_transfer_plan
from .placement import ExpertPlacement


class EPLBTransferPlanner(EPLBAsyncTask):
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
        self.result: Optional[List[List[EPLBTransferInfo]]] = None
        super().__init__(thread_name="eplb-transfer-plan")

    def execute(self) -> None:
        """逐层生成传输批次，并按 layer 顺序保存完整结果。"""
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
