"""EPLB placement planner selection and run-mode validation."""

from typing import Callable, Dict, Tuple

from .topology_aware import TopologyAwareEPLBPlanner
from .global_balance import GlobalBalanceEPLBPlanner
from .planner import EPLBPlanner


def create_eplb_planner(
    plan_mode: str,
    run_mode: str,
    num_ranks: int,
    num_redundant_experts_per_rank: int,
    expert_alignment: int,
    node_world_size: int,
) -> EPLBPlanner:
    """校验 planner 的运行阶段，并创建对应的布局规划器。"""
    planner_configs: Dict[str, Tuple[str, Callable[[], EPLBPlanner]]] = {
        "global_balance": (
            "prefill",
            lambda: GlobalBalanceEPLBPlanner(
                num_ranks,
                num_redundant_experts_per_rank,
                expert_alignment=expert_alignment,
            ),
        ),
        "topology_aware": (
            "prefill",
            lambda: TopologyAwareEPLBPlanner(
                num_ranks,
                num_redundant_experts_per_rank,
                expert_alignment=expert_alignment,
                node_world_size=node_world_size,
            ),
        ),
    }

    assert plan_mode in planner_configs
    planner_run_mode, planner_builder = planner_configs[plan_mode]
    assert planner_run_mode == run_mode, (
        f"EPLB plan mode {plan_mode!r} is for {planner_run_mode!r}, " f"but run mode is {run_mode!r}"
    )
    return planner_builder()
