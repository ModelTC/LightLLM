"""EPLB placement planner selection."""

from typing import Callable, Dict

from .topology_aware import TopologyAwareEPLBPlanner
from .greedy import GreedyEPLBPlanner
from .planner import EPLBPlanner


def create_eplb_planner(
    plan_mode: str,
    num_ranks: int,
    num_redundant_experts_per_rank: int,
    expert_alignment: int,
    node_world_size: int,
) -> EPLBPlanner:
    """根据启动参数为当前推理进程创建布局规划器。"""
    planner_builders: Dict[str, Callable[[], EPLBPlanner]] = {
        "greedy": lambda: GreedyEPLBPlanner(
            num_ranks,
            num_redundant_experts_per_rank,
            expert_alignment=expert_alignment,
        ),
        "topology_aware": lambda: TopologyAwareEPLBPlanner(
            num_ranks,
            num_redundant_experts_per_rank,
            expert_alignment=expert_alignment,
            node_world_size=node_world_size,
        ),
    }
    assert plan_mode in planner_builders
    return planner_builders[plan_mode]()
