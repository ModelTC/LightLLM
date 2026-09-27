"""Abstract interface implemented by EPLB placement planners."""

from abc import ABC, abstractmethod

import torch

from .types import ExpertPlacement


class EPLBPlanner(ABC):
    """根据逻辑专家负载生成完整物理布局。"""

    @abstractmethod
    def plan(
        self,
        logical_expert_load_samples: torch.Tensor,
        current_placement: ExpertPlacement,
    ) -> ExpertPlacement:
        """根据 CPU 负载生成 ``[layer][rank][local physical expert]`` 布局。

        ``logical_expert_load_samples`` 的 shape 为
        ``[rank, layer, sample, logical_expert]``。具体 planner 决定如何聚合
        rank 和 sample 维度。
        """
