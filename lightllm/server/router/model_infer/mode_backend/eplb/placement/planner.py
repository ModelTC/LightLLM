"""Abstract interface implemented by EPLB placement planners."""

from abc import ABC, abstractmethod

import torch

from .types import ExpertPlacement


class EPLBPlanner(ABC):
    """根据逻辑专家负载生成完整物理布局。"""

    @abstractmethod
    def plan(
        self,
        route_statistics: torch.Tensor,
        current_placement: ExpertPlacement,
    ) -> ExpertPlacement:
        """根据 CPU 负载生成 ``[layer][rank][local physical expert]`` 布局。

        ``route_statistics`` 是 all-gather 后的 CPU 原始路由统计，
        根据 planner 所属的 run mode 使用不同 shape：

        - prefill: ``[rank, layer, sample, expert_num]``
        - decode: ``[rank, layer, expert_num, expert_num]``，最后两维保存包含
          主对角线的上三角共现矩阵

        每个具体 planner 只解释其所属 run mode 的输入；factory 负责校验
        plan mode 与 run mode 是否匹配。
        """
