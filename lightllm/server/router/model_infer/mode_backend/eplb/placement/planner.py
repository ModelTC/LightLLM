"""Abstract interface implemented by EPLB placement planners."""

from abc import ABC, abstractmethod

from .types import ExpertPlacement, LogicalExpertLoad


class EPLBPlanner(ABC):
    """根据逻辑专家负载生成完整物理布局。"""

    @abstractmethod
    def plan(
        self,
        logical_expert_load: LogicalExpertLoad,
        current_placement: ExpertPlacement,
    ) -> ExpertPlacement:
        """返回 ``[layer][rank][local physical expert]`` 专家布局。"""
