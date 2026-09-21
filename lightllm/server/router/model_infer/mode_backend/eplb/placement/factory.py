"""EPLB placement planner selection."""

from typing import Callable, Dict

from .greedy import GreedyEPLBPlanner
from .planner import EPLBPlanner


def create_eplb_planner(
    plan_mode: str,
    num_ranks: int,
    num_redundant_experts_per_rank: int,
    expert_alignment: int,
) -> EPLBPlanner:
    """根据启动参数为当前推理进程创建布局规划器。"""
    planner_builders: Dict[str, Callable[[], EPLBPlanner]] = {
        "greedy": lambda: GreedyEPLBPlanner(
            num_ranks,
            num_redundant_experts_per_rank,
            expert_alignment=expert_alignment,
        ),
    }
    if plan_mode not in planner_builders:
        raise ValueError(f"unsupported EPLB plan mode {plan_mode!r}; expected one of {tuple(planner_builders)}")
    return planner_builders[plan_mode]()
