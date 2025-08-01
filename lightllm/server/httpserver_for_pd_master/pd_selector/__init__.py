from typing import List
from lightllm.server.httpserver_for_pd_master.manager import PD_Client_Obj
from .pd_selector import (
    PDSelector,
    RandomSelector,
    RoundRobinSelector,
    MemorySelector
)

__all__ = [
    "PDSelector",
    "RandomSelector", 
    "RoundRobinSelector",
    "MemorySelector"
]

def create_selector(selector_type: str, prefill_nodes: List[PD_Client_Obj], decode_nodes: List[PD_Client_Obj], pd_manager) -> PDSelector:
    if selector_type == "random":
        return RandomSelector(prefill_nodes, decode_nodes, pd_manager)
    elif selector_type == "round_robin":
        return RoundRobinSelector(prefill_nodes, decode_nodes, pd_manager)
    elif selector_type == "memory":
        return MemorySelector(prefill_nodes, decode_nodes, pd_manager)
    else:
        raise ValueError(f"Invalid selector type: {selector_type}")
